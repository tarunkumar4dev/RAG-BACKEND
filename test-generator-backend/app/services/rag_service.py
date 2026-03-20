"""
RAG Service — Keyword Search (Vercel-compatible, no torch/sentence-transformers)

Uses existing ncert_chunks table: id, class_grade, subject, chapter, content, embedding
Vector search disabled for Vercel deployment (size limit).
Keyword search is accurate enough for NCERT structured content.
"""

import logging
import re
from typing import List, Dict, Optional

from app.core.config import settings
from app.core.database import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chapter Name Normalization
# ---------------------------------------------------------------------------
def _normalize_chapter_name(name: str) -> str:
    text = name.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" & ", " and ").replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _resolve_chapters(
    requested_chapters: List[str],
    subject: str,
    class_grade: str,
) -> List[str]:
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
                search_name = _normalize_chapter_name(req_ch)
                search_words = [w for w in search_name.split() if len(w) > 2][:3]
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
        logger.warning(f"No chapters found in DB for {subject} class {class_grade}")
        return requested_chapters

    db_lookup: Dict[str, str] = {}
    for ch in db_chapters:
        db_lookup[_normalize_chapter_name(ch).lower()] = ch

    matched: List[str] = []
    unmatched: List[str] = []

    for req_ch in requested_chapters:
        req_norm = _normalize_chapter_name(req_ch).lower()

        if req_norm in db_lookup:
            matched.append(db_lookup[req_norm])
            continue

        found = False
        for db_norm, db_actual in db_lookup.items():
            if req_norm in db_norm or db_norm in req_norm:
                matched.append(db_actual)
                found = True
                break

        if not found:
            req_words = set(req_norm.split()) - {"and", "of", "the", "in", "a"}
            for db_norm, db_actual in db_lookup.items():
                db_words = set(db_norm.split()) - {"and", "of", "the", "in", "a"}
                if len(req_words & db_words) >= 2:
                    matched.append(db_actual)
                    found = True
                    break

        if not found:
            unmatched.append(req_ch)

    if unmatched:
        logger.warning(f"Could not match chapters: {unmatched}. DB has: {db_chapters}")
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
# Keyword Search (primary search method)
# ---------------------------------------------------------------------------
def keyword_search(
    keywords: List[str],
    subject: str,
    class_grade: str,
    chapters: List[str],
    limit: int = 10,
) -> List[Dict]:
    supabase = get_supabase()

    try:
        results: List[Dict] = []
        seen_ids: set = set()

        for keyword in keywords[:8]:
            if not keyword or len(keyword.strip()) < 2:
                continue

            query = (
                supabase.table("ncert_chunks")
                .select("id, chapter, subject, class_grade, content")
                .ilike("content", f"%{keyword}%")
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
                    .ilike("content", f"%{keyword}%")
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

        return results[:limit]

    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def retrieve_context(
    chapters: List[str],
    topics: List[str],
    subject: str,
    class_grade: str,
    max_chunks: int = 10,
) -> List[Dict]:
    """
    Retrieval pipeline (keyword-only for Vercel):
      1. Resolve chapter names against DB
      2. Keyword search with chapter-filter fallback
      3. Return top chunks
    """
    resolved_chapters = _resolve_chapters(chapters, subject, class_grade)
    if resolved_chapters != chapters:
        logger.info(f"Chapter resolution: {chapters} -> {resolved_chapters}")

    # Build keywords from topics + chapters
    split_keywords: List[str] = []
    for t in topics:
        for part in t.split(","):
            word = part.strip()
            if len(word) >= 3:
                split_keywords.append(word)
    split_keywords.extend(resolved_chapters)

    query_text = " ".join(filter(None, [subject, str(class_grade)] + resolved_chapters + topics))
    logger.info(f"RAG query: '{query_text[:80]}...'")

    # Keyword search only (vector search disabled for Vercel size limit)
    keyword_results = keyword_search(
        keywords=split_keywords,
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