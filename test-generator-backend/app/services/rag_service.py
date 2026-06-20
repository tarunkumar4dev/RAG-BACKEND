"""
RAG Service v5 — Per-chapter stride sampling for content diversity

v5 changes:
  - Mode 2 (no keywords) now samples PER CHAPTER instead of flat sample
  - Skips first 1-2 chunks (intro/history bias) and samples evenly from body
  - Multi-chapter scenarios: each chapter gets fair representation

v4 retained:
  - Don't search chapter names in content
  - Two modes: keyword vs direct chapter fetch
  - Fallback to chapter chunks when keyword search empty
"""

import logging
import re
from typing import List, Dict, Set

from app.core.config import settings
from app.core.database import get_supabase

logger = logging.getLogger(__name__)


_STOPWORDS: Set[str] = {
    "and", "of", "the", "in", "a", "an", "to", "for", "with", "on",
    "at", "by", "from", "its", "it", "is", "are", "be",
}


def _normalize_chapter_name(name: str) -> str:
    if not name:
        return ""
    text = name.strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2013\u2014\-:,;()\[\]\/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_set(normalized: str) -> Set[str]:
    return {w for w in normalized.split() if w and w not in _STOPWORDS}


def _resolve_chapters(requested_chapters, subject, class_grade):
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
        logger.info(f"DB has {len(db_chapters)} chapters for {subject} class {class_grade}")
    except Exception as e:
        logger.error(f"Chapter resolution query failed: {e}")

    if len(db_chapters) < 3:
        logger.warning("Few chapters from bulk query, trying per-chapter ILIKE...")
        for req_ch in requested_chapters:
            try:
                normalized = _normalize_chapter_name(req_ch)
                search_words = [w for w in normalized.split() if len(w) > 2 and w not in _STOPWORDS][:3]
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
        logger.warning(f"No chapters found in DB. Returning requested as-is.")
        return requested_chapters

    db_lookup: Dict[str, str] = {}
    for ch in db_chapters:
        db_lookup[_normalize_chapter_name(ch)] = ch

    matched: List[str] = []
    unmatched: List[str] = []

    for req_ch in requested_chapters:
        req_norm = _normalize_chapter_name(req_ch)
        req_words = _word_set(req_norm)

        # Strategy 1: Exact normalized match
        if req_norm in db_lookup:
            matched.append(db_lookup[req_norm])
            continue

        # Strategy 2: Substring match
        found = False
        for db_norm, db_actual in db_lookup.items():
            if req_norm and db_norm and (req_norm in db_norm or db_norm in req_norm):
                matched.append(db_actual)
                found = True
                break
        if found:
            continue

        # Strategy 3: Word-overlap match
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
            continue

        unmatched.append(req_ch)

    if unmatched:
        logger.warning(f"Could not match: {unmatched}. DB has: {db_chapters}")
    if matched:
        logger.info(f"Resolved chapters: {matched}")

    return matched if matched else requested_chapters


def get_embedding(text: str) -> List[float]:
    """Disabled — vector search skipped on Vercel."""
    return []


def vector_search(query_embedding, subject, class_grade, chapters, limit=8, threshold=0.65):
    """Disabled for Vercel."""
    return []


def keyword_search(keywords, subject, class_grade, chapters, limit=10):
    """
    Search NCERT chunks. Two modes:
      1. Keywords provided → search content, filtered by chapter
      2. No keywords → per-chapter stride sampling for diversity
    """
    supabase = get_supabase()

    try:
        results: List[Dict] = []
        seen_ids: Set = set()

        # ── Mode 2: No keywords → per-chapter stride sampling ──
        if not keywords or all(not k.strip() for k in keywords):
            if chapters:
                logger.info(f"No keywords. Per-chapter stride sampling for diversity.")
                num_chapters = len(chapters)
                per_chapter_quota = max(2, limit // num_chapters)

                for ch_name in chapters:
                    # Fetch all chunks for THIS chapter
                    query = (
                        supabase.table("ncert_chunks")
                        .select("id, chapter, subject, class_grade, content")
                        .ilike("subject", f"%{subject}%")
                        .eq("class_grade", str(class_grade))
                        .eq("chapter", ch_name)
                    )
                    response = query.limit(100).execute()
                    ch_rows = response.data or []

                    if not ch_rows:
                        continue

                    # Stride sample WITHIN this chapter, skipping first 1-2 (intro bias)
                    if len(ch_rows) > per_chapter_quota:
                        skip_intro = min(2, len(ch_rows) // 4)
                        body_rows = ch_rows[skip_intro:]
                        if len(body_rows) > per_chapter_quota:
                            stride = len(body_rows) / per_chapter_quota
                            sampled = [body_rows[int(i * stride)] for i in range(per_chapter_quota)]
                        else:
                            sampled = body_rows
                        logger.info(f"  '{ch_name}': stride-sampled {len(sampled)} from {len(ch_rows)} (skipped {skip_intro} intro)")
                    else:
                        sampled = ch_rows
                        logger.info(f"  '{ch_name}': took all {len(sampled)} chunks")

                    for row in sampled:
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

        # Final fallback: keyword search empty → chapter chunks
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


def retrieve_context(chapters, topics, subject, class_grade, max_chunks=10):
    """
    Retrieval pipeline:
      1. Resolve chapter names against DB
      2. Build keywords ONLY from real topics
      3. If no real topics → per-chapter stride sampling
      4. Return top chunks
    """
    resolved_chapters = _resolve_chapters(chapters, subject, class_grade)
    if resolved_chapters != chapters:
        logger.info(f"Chapter resolution: {chapters} -> {resolved_chapters}")

    real_topics: List[str] = []
    for t in topics or []:
        if not t:
            continue
        if t in chapters or t in resolved_chapters:
            continue
        for part in t.split(","):
            word = part.strip()
            if len(word) >= 3:
                real_topics.append(word)

    query_text = " ".join(filter(None, [subject, str(class_grade)] + resolved_chapters + real_topics))
    logger.info(f"RAG query: '{query_text[:80]}...' | real_topics={real_topics}")

    keyword_results = keyword_search(
        keywords=real_topics,
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