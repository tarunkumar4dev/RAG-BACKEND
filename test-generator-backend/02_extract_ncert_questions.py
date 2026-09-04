"""
a4ai — NCERT Question Extraction Script (One-Time Batch)
=========================================================
Reads raw text chunks from `ncert_chunks` table,
sends them to Gemini Pro for structured question extraction,
and inserts results into `ncert_questions` table.

USAGE:
  pip install supabase google-genai
  
  Set environment variables:
    SUPABASE_URL=https://dcmnzvjftmdbywrjkust.supabase.co
    SUPABASE_SERVICE_KEY=<your-service-role-key>
    GEMINI_API_KEY=<your-gemini-api-key>
  
  python 02_extract_ncert_questions.py

  Optional flags:
    --subject "Science"        Only process this subject
    --class_grade "10"         Only process this class
    --chapter "Light"          Only process this chapter (partial match)
    --model "gemini-3.1-pro-preview"   Gemini model to use
    --dry-run                  Print prompts, don't call Gemini
    --resume                   Skip (class,subject,chapter) combos that already have rows
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import List, Dict, Optional
from collections import defaultdict
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

import sys
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))

# ── Config ──────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://dcmnzvjftmdbywrjkust.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", os.environ.get("SUPABASE_SERVICE_KEY", ""))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_KEY", "")

# Best model for accuracy — change if needed
DEFAULT_MODEL = "gemini-3.6-flash"

# Rate limiting
REQUESTS_PER_MINUTE = 10  # conservative for Pro model
DELAY_BETWEEN_CALLS = 60 / REQUESTS_PER_MINUTE  # 6 seconds

# Chunk grouping
MAX_CHARS_PER_BATCH = 15000  # ~3-4 chunks per Gemini call (within context limits)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",   
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extraction_log.txt", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Gemini Prompt ───────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert NCERT textbook analyst for Indian school education.

Given the following NCERT textbook content from Class {class_grade} {subject}, Chapter: "{chapter}", 
extract ALL questions, examples, and activities present in this text.

CONTENT:
---
{content}
---

INSTRUCTIONS:
1. Extract EVERY question you find — exercise questions, in-text questions, examples with solutions, activities, HOTS questions.
2. For each question, identify:
   - The exact question text (copy verbatim from the content — do NOT modify or rephrase)
   - Question number as it appears (Q.1, Q 2, Example 5.3, Activity 9.2, etc.)
   - Section number if visible (like 9.1, 9.2, Exercise, In-Text Questions, etc.)
   - Question type: "exercise" | "example" | "intext" | "activity" | "hots"
   - Answer or solution if present in the text (copy verbatim)
   - If the question has MCQ options, list them
   - Estimated difficulty: "easy" | "medium" | "hard"
   - Estimated marks: 1 (MCQ/fill-blank), 2 (short answer), 3 (reasoning), 5 (long answer/numerical)
3. If the text is purely explanatory with NO questions, return an empty array.
4. Do NOT invent questions. Only extract what is ACTUALLY in the text.
5. Keep the original language (English or Hindi as it appears).

RESPOND WITH ONLY valid JSON, no markdown, no commentary:
{{
  "questions": [
    {{
      "question_number": "Q.1",
      "question_text": "Exact question text here",
      "question_type": "exercise",
      "section": "Exercise",
      "answer": "Answer if available, else null",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "difficulty": "medium",
      "marks": 2
    }}
  ]
}}

If no questions found, return: {{"questions": []}}
"""


# ── Supabase Client ────────────────────────────────────────────────

def get_supabase():
    from supabase import create_client
    if not SUPABASE_KEY:
        raise ValueError("SUPABASE_SERVICE_KEY not set!")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Gemini Client ──────────────────────────────────────────────────

def get_gemini_client():
    from google import genai
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set!")
    return genai.Client(api_key=GEMINI_API_KEY)


def call_gemini(client, model: str, prompt: str, max_retries: int = 3) -> Optional[Dict]:
    """Call Gemini with retries and JSON parsing."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=8192,
    ),
)

            if not response or not response.text:
                logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")
                time.sleep(5)
                continue

            # Clean and parse JSON
            text = response.text.strip()
            # Remove markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()

            parsed = json.loads(text)
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error (attempt {attempt + 1}): {e}")
            logger.debug(f"Raw response: {response.text[:500] if response else 'None'}")
            time.sleep(3)
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                wait_time = 30 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "500" in error_str or "503" in error_str:
                wait_time = 10 * (attempt + 1)
                logger.warning(f"Server error, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Gemini error (attempt {attempt + 1}): {e}")
                time.sleep(5)

    logger.error("All retries exhausted for this batch")
    return None


# ── Fetch Chunks ───────────────────────────────────────────────────

def fetch_all_chunks(
    supabase,
    subject_filter: Optional[str] = None,
    class_filter: Optional[str] = None,
    chapter_filter: Optional[str] = None,
) -> List[Dict]:
    """Fetch chunks from ncert_chunks with optional filters."""
    query = supabase.table("ncert_chunks").select("id, class_grade, subject, chapter, content")

    if subject_filter:
        query = query.ilike("subject", f"%{subject_filter}%")
    if class_filter:
        query = query.eq("class_grade", class_filter)
    if chapter_filter:
        query = query.ilike("chapter", f"%{chapter_filter}%")

    # Supabase has row limits — paginate
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        result = query.range(offset, offset + page_size - 1).execute()
        rows = result.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    logger.info(f"Fetched {len(all_rows)} chunks total")
    return all_rows


def group_chunks(chunks: List[Dict]) -> Dict[tuple, List[Dict]]:
    """Group chunks by (class_grade, subject, chapter)."""
    groups = defaultdict(list)
    for chunk in chunks:
        key = (chunk["class_grade"], chunk["subject"], chunk["chapter"])
        groups[key].append(chunk)
    
    logger.info(f"Grouped into {len(groups)} (class, subject, chapter) combos")
    return groups


def batch_chunks(chunks: List[Dict], max_chars: int = MAX_CHARS_PER_BATCH) -> List[List[Dict]]:
    """Split a chapter's chunks into batches that fit within char limit."""
    batches = []
    current_batch = []
    current_chars = 0

    for chunk in chunks:
        content_len = len(chunk.get("content", ""))
        if current_chars + content_len > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(chunk)
        current_chars += content_len

    if current_batch:
        batches.append(current_batch)

    return batches


# ── Insert Questions ───────────────────────────────────────────────

def insert_questions(
    supabase,
    questions: List[Dict],
    class_grade: str,
    subject: str,
    chapter: str,
    source_chunk_ids: List[int],
) -> int:
    """Insert extracted questions into ncert_questions table."""
    if not questions:
        return 0

    rows = []
    for q in questions:
        q_text = (q.get("question_text") or "").strip()
        if not q_text or len(q_text) < 5:
            continue  # Skip garbage

        # Deduplicate by checking text similarity (simple exact match)
        row = {
            "class_grade": str(class_grade),
            "subject": subject,
            "chapter": chapter,
            "section": q.get("section"),
            "question_number": q.get("question_number"),
            "question_text": q_text,
            "question_type": q.get("question_type", "exercise"),
            "answer": q.get("answer"),
            "options": json.dumps(q.get("options") or []),
            "marks": q.get("marks", 2),
            "difficulty": q.get("difficulty", "medium"),
            "source_chunk_id": source_chunk_ids[0] if source_chunk_ids else None,
        }
        rows.append(row)

    if not rows:
        return 0

    try:
        # Insert in batches of 50
        inserted = 0
        for i in range(0, len(rows), 50):
            batch = rows[i:i+50]
            supabase.table("ncert_questions").insert(batch).execute()
            inserted += len(batch)
        return inserted
    except Exception as e:
        logger.error(f"Insert failed for {subject} {class_grade} {chapter}: {e}")
        # Try one-by-one on failure
        inserted = 0
        for row in rows:
            try:
                supabase.table("ncert_questions").insert(row).execute()
                inserted += 1
            except Exception as e2:
                logger.error(f"Single insert failed: {e2} — question: {row['question_text'][:80]}")
        return inserted


# ── Check existing (for --resume) ─────────────────────────────────

def get_existing_combos(supabase) -> set:
    """Get set of (class_grade, subject, chapter) that already have questions."""
    result = supabase.table("ncert_questions") \
        .select("class_grade, subject, chapter") \
        .execute()
    
    combos = set()
    for row in (result.data or []):
        combos.add((str(row["class_grade"]), row["subject"], row["chapter"]))
    return combos


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract NCERT questions using Gemini")
    parser.add_argument("--subject", help="Filter by subject (e.g., 'Science')")
    parser.add_argument("--class_grade", help="Filter by class (e.g., '10')")
    parser.add_argument("--chapter", help="Filter by chapter (partial match)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only, don't call Gemini")
    parser.add_argument("--resume", action="store_true", help="Skip already-extracted chapters")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"a4ai NCERT Question Extraction")
    logger.info(f"Model: {args.model}")
    logger.info(f"Filters: subject={args.subject}, class={args.class_grade}, chapter={args.chapter}")
    logger.info(f"Dry run: {args.dry_run}, Resume: {args.resume}")
    logger.info("=" * 60)

    # Init clients
    supabase = get_supabase()
    if not args.dry_run:
        gemini = get_gemini_client()
    else:
        gemini = None

    # Fetch & group
    chunks = fetch_all_chunks(supabase, args.subject, args.class_grade, args.chapter)
    if not chunks:
        logger.error("No chunks found with given filters!")
        return

    groups = group_chunks(chunks)

    # Resume support
    skip_combos = set()
    if args.resume:
        skip_combos = get_existing_combos(supabase)
        logger.info(f"Resume mode: {len(skip_combos)} combos already extracted, will skip")

    # Stats
    total_questions = 0
    total_batches = 0
    total_errors = 0
    total_skipped = 0

    for (class_grade, subject, chapter), chapter_chunks in sorted(groups.items()):
        # Skip if already done (resume mode)
        if (str(class_grade), subject, chapter) in skip_combos:
            total_skipped += 1
            logger.info(f"⏭️  SKIP (already extracted): {subject} {class_grade} — {chapter}")
            continue

        logger.info(f"\n{'─' * 50}")
        logger.info(f"📖 Processing: {subject} Class {class_grade} — {chapter} ({len(chapter_chunks)} chunks)")

        # Batch chunks
        batches = batch_chunks(chapter_chunks)
        logger.info(f"   Split into {len(batches)} batch(es)")

        chapter_questions = 0

        for batch_idx, batch in enumerate(batches):
            # Build combined content
            combined_content = ""
            chunk_ids = []
            for chunk in batch:
                combined_content += chunk["content"] + "\n\n---\n\n"
                chunk_ids.append(chunk["id"])

            # Build prompt
            prompt = EXTRACTION_PROMPT.format(
                class_grade=class_grade,
                subject=subject,
                chapter=chapter,
                content=combined_content.strip(),
            )

            if args.dry_run:
                logger.info(f"   [DRY RUN] Batch {batch_idx + 1}/{len(batches)}: "
                            f"{len(combined_content)} chars, {len(chunk_ids)} chunks")
                logger.debug(f"   Prompt preview: {prompt[:200]}...")
                total_batches += 1
                continue

            # Call Gemini
            logger.info(f"   🤖 Batch {batch_idx + 1}/{len(batches)}: "
                        f"{len(combined_content)} chars → Gemini...")

            result = call_gemini(gemini, args.model, prompt)
            total_batches += 1

            if result is None:
                logger.error(f"   ❌ Gemini failed for batch {batch_idx + 1}")
                total_errors += 1
                continue

            questions = result.get("questions", [])
            logger.info(f"   ✅ Extracted {len(questions)} questions")

            if questions:
                inserted = insert_questions(
                    supabase, questions,
                    class_grade, subject, chapter,
                    chunk_ids,
                )
                chapter_questions += inserted
                logger.info(f"   💾 Inserted {inserted} questions into DB")

            # Rate limit
            time.sleep(DELAY_BETWEEN_CALLS)

        total_questions += chapter_questions
        if chapter_questions > 0:
            logger.info(f"   📊 Chapter total: {chapter_questions} questions")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ EXTRACTION COMPLETE")
    logger.info(f"   Total questions extracted: {total_questions}")
    logger.info(f"   Total batches processed:   {total_batches}")
    logger.info(f"   Errors:                    {total_errors}")
    logger.info(f"   Skipped (resume):          {total_skipped}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()