"""
RAG Service — Keyword Search (Vercel-compatible, no torch/sentence-transformers)

Uses existing ncert_chunks table: id, class_grade, subject, chapter, content, embedding
Vector search disabled for Vercel deployment (size limit).
Keyword search is accurate enough for NCERT structured content.

v3 changes:
  - Robust chapter name normalization: strips dashes, commas, parentheses, ampersands
  - Multi-strategy matching: exact → substring → word-overlap (fuzzy)
  - Handles: "Light - Reflection & Refraction" <-> "Light Reflection and Refraction"
  - Handles: "Acids, Bases & Salts" <-> "Acids Bases and Salts"
  - Handles: "Metals & Non-metals" <-> "Metals and Non-metals"
  
v4 changes:
  - FIX: Don't search for chapter names in content; use them only for chapter filtering
  - Two modes: keyword search (when topics provided) vs direct chapter fetch (no topics)
  - Fallback to chapter chunks when keyword search returns empty
"""

import logging
import re
from typing import List, Dict, Set

from app.core.config import settings
from app.core.database import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chapter Name Normalization — ROBUST
# ---------------------------------------------------------------------------
# Common "noise words" to ignore when doing word-overlap matching
_STOPWORDS: Set[str] = {
    "and", "of", "the", "in", "a", "an", "to", "for", "with", "on",
    "at", "by", "from", "its", "it", "is", "are", "be",
}


def _normalize_chapter_name(name: str) -> str:
    """
    Aggressively normalize chapter name for reliable matching.

    Handles:
      - All dash variants (- – —) → space
      - Colons, commas, semicolons → space
      - Parentheses & brackets → space
      - Ampersand (&) → "and"
      - Multiple spaces → single space
      - Case: lowercased
    """
    if not name:
        return ""
    text = name.strip().lower()

    # Normalize ampersand to "and" (with padding spaces so words don't merge)
    text = text.replace("&", " and ")

    # Replace all dashes, colons, commas, parens, brackets, slashes with space
    text = re.sub(r"[\u2013\u2014\-:,;()\[\]\/]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _word_set(normalized: str) -> Set[str]:
    """Return significant words (excluding stopwords) from a normalized string."""
    return {w for w in normalized.split() if w and w not in _STOPWORDS}


def _resolve_chapters(
    requested_chapters: List[str],
    subject: str,
    class_grade: str,
) -> List[str]:
    """
    Map frontend chapter names to actual DB chapter names.

    Strategy (per requested chapter):
      1. Exact normalized match
      2. Substring match (either direction)
      3. Word-overlap match (>= 2 significant words in common, or all-words-of-shorter-in-longer)
    """
    supabase = get_supabase()
    db_chapters: List[str] = []

    try:
        response = (
            supabase.table("ncert_chunks")
            .select("chapter")
            .ilike("subject", f"%{subject}%")
            .eq("class_grade", str(class_grade))
            .limit(1000)
            .execute()
        )
        db_chapters = list({
            row["chapter"]
            for row in (response.data or [])
            if row.get("chapter")
        })
        logger.info(
            f"DB has {len(db_chapters)} chapters for {subject} class {class_grade}"
        )
    except Exception as e:
        logger.error(f"Chapter resolution query failed: {e}")

    # Fallback: per-chapter ILIKE search if bulk fetch was thin
    if len(db_chapters) < 3:
        logger.warning("Few chapters from bulk query, trying per-chapter ILIKE...")
        for req_ch in requested_chapters:
            try:
                normalized = _normalize_chapter_name(req_ch)
                search_words = [
                    w for w in normalized.split()
                    if len(w) > 2 and w not in _STOPWORDS
                ][:3]
                if not search_words:
                    continue
                ilike_pattern = f"%{'%'.join(search_words)}%"
                resp = (
                    supabase.table("ncert_chunks")
                    .select("chapter")
                    .ilike("chapter", ilike_pattern)
                    .eq("class_grade", str(class_grade))
                    .limit(5)
                    .execute()
                )
                for row in (resp.data or []):
                    ch = row.get("chapter")
                    if ch and ch not in db_chapters:
                        db_chapters.append(ch)
            except Exception as e:
                logger.error(f"Per-chapter ILIKE failed for '{req_ch}': {e}")

    if not db_chapters:
        logger.warning(
            f"No chapters found in DB for {subject} class {class_grade}. "
            f"Returning requested chapters as-is."
        )
        return requested_chapters

    # Build lookup: normalized_name -> actual_db_name
    db_lookup: Dict[str, str] = {}
    for ch in db_chapters:
        db_lookup[_normalize_chapter_name(ch)] = ch

    matched: List[str] = []
    unmatched: List[str] = []

    for req_ch in requested_chapters:
        req_norm = _normalize_chapter_name(req_ch)
        req_words = _word_set(req_norm)

        # ── Strategy 1: Exact normalized match ──
        if req_norm in db_lookup:
            matched.append(db_lookup[req_norm])
            logger.debug(f"Exact match: '{req_ch}' -> '{db_lookup[req_norm]}'")
            continue

        # ── Strategy 2: Substring match (either direction) ──
        found = False
        for db_norm, db_actual in db_lookup.items():
            if req_norm and db_norm and (req_norm in db_norm or db_norm in req_norm):
                matched.append(db_actual)
                logger.debug(f"Substring match: '{req_ch}' -> '{db_actual}'")
                found = True
                break
        if found:
            continue

        # ── Strategy 3: Word-overlap match ──
        # Best match = highest overlap count; require either:
        #   (a) all significant words of shorter side contained in longer, OR
        #   (b) >= 2 significant words in common
        best_match: str = None
        best_score: int = 0
        for db_norm, db_actual in db_lookup.items():
            db_words = _word_set(db_norm)
            if not db_words or not req_words:
                continue
            overlap = req_words & db_words
            overlap_count = len(overlap)

            shorter = req_words if len(req_words) <= len(db_words) else db_words
            all_of_shorter_matched = shorter and shorter.issubset(req_words & db_words)

            if all_of_shorter_matched or overlap_count >= 2:
                if overlap_count > best_score:
                    best_score = overlap_count
                    best_match = db_actual

        if best_match:
            matched.append(best_match)
            logger.debug(
                f"Word-overlap match (score={best_score}): '{req_ch}' -> '{best_match}'"
            )
            continue

        unmatched.append(req_ch)

    if unmatched:
        logger.warning(
            f"Could not match chapters: {unmatched}. "
            f"DB has: {db_chapters}"
        )
    if matched:
        logger.info(f"Resolved chapters: {matched}")

    return matched if matched else requested_chapters


# ---------------------------------------------------------------------------
# Embedding stub (vector search disabled for Vercel)
# ---------------------------------------------------------------------------
def get_embedding(text: str) -> List[float]:
    """Disabled — returns empty. Vector search skipped on Vercel."""
    return []


# ---------------------------------------------------------------------------
# Vector Search (disabled)
# ---------------------------------------------------------------------------
def vector_search(
    query_embedding: List[float],
    subject: str,
    class_grade: str,
    chapters: List[str],
    limit: int = 8,
    threshold: float = 0.65,
) -> List[Dict]:
    """Disabled for Vercel deployment. Returns empty."""
    return []


# ---------------------------------------------------------------------------
# Keyword Search (FIXED - two modes)
# ---------------------------------------------------------------------------
def keyword_search(
    keywords: List[str],
    subject: str,
    class_grade: str,
    chapters: List[str],
    limit: int = 10,
) -> List[Dict]:
    """
    Search NCERT chunks. Two modes:
      1. If keywords provided → search those keywords IN content, filtered by chapter
      2. If no keywords → just return chunks from the chapter (no content filter)
    """
    supabase = get_supabase()

    try:
        results: List[Dict] = []
        seen_ids: Set = set()

        # ── Mode 2: No keywords → fetch chapter chunks directly ──
        if not keywords or all(not k.strip() for k in keywords):
            if chapters:
                logger.info(f"No keywords provided. Fetching chapter chunks directly.")
                query = (
                    supabase.table("ncert_chunks")
                    .select("id, chapter, subject, class_grade, content")
                    .ilike("subject", f"%{subject}%")
                    .eq("class_grade", str(class_grade))
                    .in_("chapter", chapters)
                )
                response = query.limit(limit).execute()
                rows = response.data or []
                for row in rows:
                    if row["id"] not in seen_ids:
                        row["similarity"] = 0.7
                        results.append(row)
                        seen_ids.add(row["id"])
                return results[:limit]
            return []

        # ── Mode 1: Keywords provided → search inside content ──
        for keyword in keywords[:8]:
            kw = (keyword or "").strip()
            if len(kw) < 3:
                continue

            # Skip keywords that look like full chapter names — they won't be
            # found verbatim in content. Heuristic: > 4 words = probably a title
            if len(kw.split()) > 4:
                continue

            query = (
                supabase.table("ncert_chunks")
                .select("id, chapter, subject, class_grade, content")
                .ilike("content", f"%{kw}%")
                .ilike("subject", f"%{subject}%")
                .eq("class_grade", str(class_grade))
            )
            if chapters:
                query = query.in_("chapter", chapters)

            response = query.limit(limit).execute()
            rows = response.data or []

            # Fallback: if chapter filter returned nothing, broaden
            if not rows and chapters:
                query_broad = (
                    supabase.table("ncert_chunks")
                    .select("id, chapter, subject, class_grade, content")
                    .ilike("content", f"%{kw}%")
                    .ilike("subject", f"%{subject}%")
                    .eq("class_grade", str(class_grade))
                )
                response = query_broad.limit(limit).execute()
                rows = response.data or []

            for row in rows:
                if row["id"] not in seen_ids:
                    row["similarity"] = 0.6
                    results.append(row)
                    seen_ids.add(row["id"])

        # ── Final fallback: if keyword search yielded nothing, return chapter chunks ──
        if not results and chapters:
            logger.info(f"Keyword search empty. Falling back to chapter chunks.")
            query = (
                supabase.table("ncert_chunks")
                .select("id, chapter, subject, class_grade, content")
                .ilike("subject", f"%{subject}%")
                .eq("class_grade", str(class_grade))
                .in_("chapter", chapters)
            )
            response = query.limit(limit).execute()
            rows = response.data or []
            for row in rows:
                if row["id"] not in seen_ids:
                    row["similarity"] = 0.5
                    results.append(row)
                    seen_ids.add(row["id"])

        return results[:limit]

    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Main Entry Point (FIXED)
# ---------------------------------------------------------------------------
def retrieve_context(
    chapters: List[str],
    topics: List[str],
    subject: str,
    class_grade: str,
    max_chunks: int = 10,
) -> List[Dict]:
    """
    Retrieval pipeline:
      1. Resolve chapter names against DB (fuzzy matching)
      2. Build keywords ONLY from real topics (not from chapter names!)
      3. If no real topics → fetch chapter chunks directly
      4. Return top chunks
    """
    # Resolve chapter names
    resolved_chapters = _resolve_chapters(chapters, subject, class_grade)
    if resolved_chapters != chapters:
        logger.info(f"Chapter resolution: {chapters} -> {resolved_chapters}")

    # ─────────────────────────────────────────────────────────────────
    # FIX: Don't add chapter names as content keywords.
    # Only use real topics (subtopics) as keywords for content search.
    # ─────────────────────────────────────────────────────────────────
    real_topics: List[str] = []
    for t in topics or []:
        if not t:
            continue
        # Skip if topic == chapter name (no actual topic info)
        if t in chapters or t in resolved_chapters:
            continue
        for part in t.split(","):
            word = part.strip()
            if len(word) >= 3:
                real_topics.append(word)

    query_text = " ".join(
        filter(None, [subject, str(class_grade)] + resolved_chapters + real_topics)
    )
    logger.info(
        f"RAG query: '{query_text[:80]}...' | real_topics={real_topics}"
    )

    # Keyword search with ONLY real topic keywords, NOT chapter names
    keyword_results = keyword_search(
        keywords=real_topics,  # ← only real topic keywords, NOT chapter names
        subject=subject,
        class_grade=str(class_grade),
        chapters=resolved_chapters,
        limit=max_chunks,
    )
    logger.info(f"Keyword search: {len(keyword_results)} results")

    final = keyword_results[:max_chunks]
    chapter_set = set(r.get("chapter", "?") for r in final)
    logger.info(f"Final: {len(final)} chunks from {len(chapter_set)} chapters")

    return final