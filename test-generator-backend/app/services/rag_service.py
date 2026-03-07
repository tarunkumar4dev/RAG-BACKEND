"""
RAG Service — Hybrid Search (pgvector cosine + ILIKE keyword)
Uses existing ncert_chunks table: id, class_grade, subject, chapter, content, embedding

Embeddings: all-MiniLM-L6-v2 (384-dim, local) — matches ingestion model.
No Gemini API needed for search.
"""

import logging
import re
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.database import get_supabase

logger = logging.getLogger(__name__)

# Local embedding model (same as ingestion!)
_embed_model: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    """Load embedding model (cached after first call)."""
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded (384-dim)")
    return _embed_model


# ---------------------------------------------------------------------------
# Chapter Name Normalization
# ---------------------------------------------------------------------------
def _normalize_chapter_name(name: str) -> str:
    """Normalize chapter names: '&' <-> 'and', whitespace, casing."""
    text = name.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" & ", " and ")
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _resolve_chapters(
    requested_chapters: List[str],
    subject: str,
    class_grade: str,
) -> List[str]:
    """Match requested chapter names against actual DB-stored names."""
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

    # Fallback: per-chapter ILIKE search
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

    # Build normalized lookup
    db_lookup: Dict[str, str] = {}
    for ch in db_chapters:
        db_lookup[_normalize_chapter_name(ch).lower()] = ch

    matched: List[str] = []
    unmatched: List[str] = []

    for req_ch in requested_chapters:
        req_norm = _normalize_chapter_name(req_ch).lower()

        # 1. Exact normalized match
        if req_norm in db_lookup:
            matched.append(db_lookup[req_norm])
            continue

        # 2. Substring match
        found = False
        for db_norm, db_actual in db_lookup.items():
            if req_norm in db_norm or db_norm in req_norm:
                matched.append(db_actual)
                found = True
                break

        # 3. Word overlap (>=2 significant words)
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
# Embeddings — Local model (matches ingestion!)
# ---------------------------------------------------------------------------
def get_embedding(text: str) -> List[float]:
    """Get 384-dim embedding using local all-MiniLM-L6-v2 model."""
    try:
        model = _get_embed_model()
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Vector Search
# ---------------------------------------------------------------------------
def vector_search(
    query_embedding: List[float],
    subject: str,
    class_grade: str,
    chapters: List[str],
    limit: int = 8,
    threshold: float = 0.65,
) -> List[Dict]:
    """pgvector cosine similarity search via match_ncert_chunks RPC."""
    if not query_embedding:
        return []

    supabase = get_supabase()
    chapter_filter = chapters if chapters else []

    try:
        response = supabase.rpc(
            "match_ncert_chunks",
            {
                "query_embedding": query_embedding,
                "subject_filter": subject,
                "class_filter": str(class_grade),
                "chapter_filter": chapter_filter,
                "match_threshold": threshold,
                "match_count": limit,
            },
        ).execute()

        results = response.data or []

        # Retry with lowercase subject
        if not results:
            response = supabase.rpc(
                "match_ncert_chunks",
                {
                    "query_embedding": query_embedding,
                    "subject_filter": subject.lower(),
                    "class_filter": str(class_grade),
                    "chapter_filter": chapter_filter,
                    "match_threshold": threshold,
                    "match_count": limit,
                },
            ).execute()
            results = response.data or []

        # Retry with relaxed threshold
        if not results and threshold > 0.3:
            logger.info("Retrying vector search with relaxed threshold (0.3)...")
            response = supabase.rpc(
                "match_ncert_chunks",
                {
                    "query_embedding": query_embedding,
                    "subject_filter": subject,
                    "class_filter": str(class_grade),
                    "chapter_filter": chapter_filter,
                    "match_threshold": 0.3,
                    "match_count": limit,
                },
            ).execute()
            results = response.data or []

        return results

    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Keyword Search
# ---------------------------------------------------------------------------
def keyword_search(
    keywords: List[str],
    subject: str,
    class_grade: str,
    chapters: List[str],
    limit: int = 6,
) -> List[Dict]:
    """Fallback keyword search using ILIKE."""
    supabase = get_supabase()

    try:
        results: List[Dict] = []
        seen_ids: set = set()

        for keyword in keywords[:6]:
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

            # Retry WITHOUT chapter filter
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
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def rrf_fusion(
    vector_results: List[Dict],
    keyword_results: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """Merge vector + keyword results by rank."""
    scores: Dict[int, float] = {}
    all_items: Dict[int, Dict] = {}

    for rank, item in enumerate(vector_results):
        item_id = item["id"]
        scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank + 1)
        all_items[item_id] = item

    for rank, item in enumerate(keyword_results):
        item_id = item["id"]
        scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank + 1)
        all_items[item_id] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [all_items[i] for i in sorted_ids]


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
    Main retrieval pipeline:
      1. Resolve chapter names against DB
      2. Vector search with local embeddings (matches ingestion model!)
      3. Keyword search with chapter-filter fallback
      4. RRF fusion -> top chunks
    """
    resolved_chapters = _resolve_chapters(chapters, subject, class_grade)
    if resolved_chapters != chapters:
        logger.info(f"Chapter resolution: {chapters} -> {resolved_chapters}")

    query_parts = [subject, str(class_grade)] + resolved_chapters + topics
    query_text = " ".join(filter(None, query_parts))
    logger.info(f"RAG query: '{query_text[:80]}...'")

    # Vector search
    vector_results: List[Dict] = []
    try:
        embedding = get_embedding(query_text)
        if embedding:
            vector_results = vector_search(
                query_embedding=embedding,
                subject=subject,
                class_grade=str(class_grade),
                chapters=resolved_chapters,
                limit=max_chunks,
                threshold=settings.SIMILARITY_THRESHOLD,
            )
        logger.info(f"Vector search: {len(vector_results)} results")
    except Exception as e:
        logger.error(f"Vector search failed: {e}")

    # Keyword search — split comma-separated topics into individual keywords
    split_keywords: List[str] = []
    for t in topics:
        for part in t.split(","):
            word = part.strip()
            if len(word) >= 3:
                split_keywords.append(word)
    # Add chapter names as keywords too
    split_keywords.extend(resolved_chapters)

    keyword_results = keyword_search(
        keywords=split_keywords,
        subject=subject,
        class_grade=str(class_grade),
        chapters=resolved_chapters,
        limit=6,
    )
    logger.info(f"Keyword search: {len(keyword_results)} results")

    # Fuse and return
    fused = rrf_fusion(vector_results, keyword_results)
    final = fused[:max_chunks]

    chapter_set = set(r.get("chapter", "?") for r in final)
    logger.info(f"Final: {len(final)} chunks from {len(chapter_set)} chapters")

    return final