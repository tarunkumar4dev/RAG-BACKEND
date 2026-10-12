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
import time
from typing import List, Dict, Set, Tuple

from app.core.config import settings
from app.core.database import get_supabase
from app.services.ncert_subjects import apply_ncert_subject_filter

logger = logging.getLogger(__name__)

# In-memory chapter resolution cache to avoid repeated 1000-row Supabase queries
_CHAPTERS_CACHE: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}
_CHAPTERS_CACHE_TTL = 3600  # 1 hour


_STOPWORDS: Set[str] = {
    "and", "of", "the", "in", "a", "an", "to", "for", "with", "on",
    "at", "by", "from", "its", "it", "is", "are", "be",
}


def _ncert_select(supabase, columns: str, subject: str):
    """ncert_chunks rows of exactly this subject (family) — never a substring match."""
    return apply_ncert_subject_filter(supabase.table("ncert_chunks").select(columns), subject)


def _escape_ilike(value: str) -> str:
    """Escape SQL ILIKE wildcards to prevent pattern injection."""
    return value.replace("%", r"\%").replace("_", r"\_")


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
    cache_key = ((subject or "").strip().lower(), str(class_grade).strip())
    now = time.time()
    if cache_key in _CHAPTERS_CACHE:
        cached_time, cached_chapters = _CHAPTERS_CACHE[cache_key]
        if now - cached_time < _CHAPTERS_CACHE_TTL and cached_chapters:
            logger.info(f"Using cached chapter list for {subject} class {class_grade} ({len(cached_chapters)} chapters)")
            db_chapters = list(cached_chapters)
        else:
            db_chapters = []
    else:
        db_chapters = []

    if not db_chapters:
        supabase = get_supabase()
        try:
            response = (
                _ncert_select(supabase, "chapter", subject)
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
            if db_chapters:
                _CHAPTERS_CACHE[cache_key] = (now, list(db_chapters))
        except Exception as e:
            logger.error(f"Chapter resolution query failed: {e}")

    if len(db_chapters) < 3:
        supabase = get_supabase()
        logger.warning("Few chapters from bulk query, trying per-chapter ILIKE...")
        for req_ch in requested_chapters:
            try:
                normalized = _normalize_chapter_name(req_ch)
                search_words = [w for w in normalized.split() if len(w) > 2 and w not in _STOPWORDS][:3]
                if not search_words:
                    continue
                ilike_pattern = f"%{'%'.join(_escape_ilike(w) for w in search_words)}%"
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
        if db_chapters:
            _CHAPTERS_CACHE[cache_key] = (now, list(db_chapters))

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


# Every chosen chapter gets its own share of chunks (the per-chapter prompt uses at most
# MAX_CONTEXT_CHUNKS of them). A fixed 20-chunk total used to leave chapters 6+ with nothing,
# and their questions were then written from the first chapters' content.
PER_CHAPTER_CHUNKS = settings.MAX_CONTEXT_CHUNKS          # 8
MAX_TOTAL_CHUNKS = 120                                     # = 15 chapters x 8
MIN_PER_CHAPTER = 2


def _per_chapter_quota(num_chapters: int) -> int:
    if num_chapters <= 0:
        return PER_CHAPTER_CHUNKS
    return max(MIN_PER_CHAPTER, min(PER_CHAPTER_CHUNKS, MAX_TOTAL_CHUNKS // num_chapters))


def _sample_chapter(supabase, ch_name, subject, class_grade, quota, seen_ids):
    """Up to `quota` chunks of ONE chapter, spread over the chapter (skips the intro chunks)."""
    rows = (
        _ncert_select(supabase, "id, chapter, subject, class_grade, content", subject)
        .eq("class_grade", str(class_grade))
        .eq("chapter", ch_name)
        .limit(100)
        .execute()
    ).data or []
    rows = [r for r in rows if r["id"] not in seen_ids]
    if len(rows) > quota:
        skip_intro = min(2, len(rows) // 4)
        body = rows[skip_intro:]
        if len(body) > quota:
            stride = len(body) / quota
            rows = [body[int(i * stride)] for i in range(quota)]
        else:
            rows = body[:quota]
    for r in rows:
        r["similarity"] = 0.7
        seen_ids.add(r["id"])
    return rows


def keyword_search(keywords, subject, class_grade, chapters, limit=None):
    """
    Search NCERT chunks of the chosen chapters only.
      1. Keywords (subtopics) → chunks of the chosen chapters that contain them
      2. Every chapter is then topped up to its quota with its own chunks, so a chapter whose
         keyword isn't found (or that has no subtopic) is never left empty.
    """
    supabase = get_supabase()
    chapters = list(chapters or [])
    quota = _per_chapter_quota(len(chapters))
    total_cap = min(MAX_TOTAL_CHUNKS, quota * len(chapters)) if chapters else (limit or PER_CHAPTER_CHUNKS)

    try:
        results: List[Dict] = []
        seen_ids: Set = set()
        per_chapter: Dict[str, int] = {c: 0 for c in chapters}

        # ── 1. Keyword hits inside the chosen chapters ──
        for keyword in [k for k in (keywords or []) if k and k.strip()][:8]:
            kw = keyword.strip()
            if len(kw) < 3 or len(kw.split()) > 4:
                continue
            query = (
                _ncert_select(supabase, "id, chapter, subject, class_grade, content", subject)
                .ilike("content", f"%{_escape_ilike(kw)}%")
                .eq("class_grade", str(class_grade))
            )
            if chapters:
                query = query.in_("chapter", chapters)
            rows = query.limit(total_cap).execute().data or []
            for row in rows:
                ch = row.get("chapter")
                if row["id"] in seen_ids or (chapters and per_chapter.get(ch, 0) >= quota):
                    continue
                row["similarity"] = 0.6
                results.append(row)
                seen_ids.add(row["id"])
                if ch in per_chapter:
                    per_chapter[ch] += 1

        # ── 2. Top every chapter up to its quota with its own chunks ──
        for ch in chapters:
            need = quota - per_chapter[ch]
            if need <= 0:
                continue
            rows = _sample_chapter(supabase, ch, subject, class_grade, need, seen_ids)
            results.extend(rows)
            per_chapter[ch] += len(rows)
            logger.info(f"  '{ch}': {per_chapter[ch]} chunks (quota {quota})")

        return results[:total_cap]

    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []


def retrieve_context(chapters, topics, subject, class_grade, max_chunks=None):
    """
    Retrieval pipeline:
      1. Resolve chapter names against DB
      2. Build keywords ONLY from real topics (subtopics)
      3. Each chapter gets up to PER_CHAPTER_CHUNKS chunks (total capped at MAX_TOTAL_CHUNKS)
    `max_chunks` is kept for old callers; the per-chapter quota decides the size now.
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

    logger.info(f"RAG: {len(resolved_chapters)} chapters, real_topics={real_topics}")
    final = keyword_search(
        keywords=real_topics,
        subject=subject,
        class_grade=str(class_grade),
        chapters=resolved_chapters,
        limit=max_chunks,
    )
    chapter_set = set(r.get("chapter", "?") for r in final)
    logger.info(f"Final: {len(final)} chunks from {len(chapter_set)} chapters")
    return final
