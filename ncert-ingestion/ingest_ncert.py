"""
NCERT PDF Ingestion Script — Production Grade (Local Embeddings)

Usage:
  # Existing flat-folder usage (unchanged):
  python ingest_ncert.py --folder pdfs/Class10_Science --subject Science --class_grade 10
  python ingest_ncert.py --folder pdfs/Class10_Maths --subject Mathematics --class_grade 10

  # New: book + chapter_type for English (run 3 times for Class 10):
  python ingest_ncert.py --folder pdfs/Class10_English/First_Flight_Prose \
      --subject English --class_grade 10 \
      --book first_flight --chapter_type prose

  python ingest_ncert.py --folder pdfs/Class10_English/First_Flight_Poems \
      --subject English --class_grade 10 \
      --book first_flight --chapter_type poem

  python ingest_ncert.py --folder pdfs/Class10_English/Footprints_Without_Feet \
      --subject English --class_grade 10 \
      --book footprints_without_feet --chapter_type prose

What it does:
  1. Reads all PDFs from a folder
  2. Extracts text with proper cleaning
  3. Chunks text into ~500-word overlapping segments
  4. Generates embeddings using all-MiniLM-L6-v2 (384-dim, LOCAL, FREE)
  5. Uploads to Supabase ncert_chunks table (with book / chapter_type / chapter_order
     when --book and --chapter_type are provided)
  6. Verifies upload

Prerequisites:
  pip install PyPDF2 asyncpg sentence-transformers python-dotenv
"""

import asyncio
import asyncpg
import os
import re
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Sentence Transformers (local embeddings — free, no API)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

# PDF reader
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("❌ PyPDF2 not installed. Run: pip install PyPDF2")
    sys.exit(1)

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================
CHUNK_SIZE = 500         # words per chunk
CHUNK_OVERLAP = 75       # overlap words between chunks
MIN_CHUNK_LENGTH = 50    # minimum words to keep a chunk
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, free
EMBEDDING_BATCH_SIZE = 32                    # encode this many chunks at once

# Supabase DB config
DB_HOST = "db.dcmnzvjftmdbywrjkust.supabase.co"
DB_PORT = 5432
DB_USER = "postgres"
DB_NAME = "postgres"

# Global embedding model (loaded once)
_embed_model: Optional[SentenceTransformer] = None


# ===========================================================================
# Chapter Order Mapping
# ===========================================================================
# Hardcoded syllabus order for English. Keys must match PDF filenames exactly
# (without .pdf extension). If a chapter name is not in the map, falls back to
# alphabetical index of the file in the folder.
CHAPTER_ORDER_MAP: Dict[Tuple[str, str], Dict[str, int]] = {
    ("first_flight", "prose"): {
        "A Letter to God": 1,
        "Nelson Mandela - Long Walk to Freedom": 2,
        "Stories About Flying": 3,
        "From the Diary of Anne Frank": 4,
        "Glimpses of India": 5,
        "Mijbil the Otter": 6,
        "Madam Rides the Bus": 7,
        "The Sermon at Benares": 8,
        "The Proposal (Play)": 9,
    },
    ("first_flight", "poem"): {
        "Dust of Snow": 1,
        "Fire and Ice": 2,
        "A Tiger in the Zoo": 3,
        "How to Tell Wild Animals": 4,
        "The Ball Poem": 5,
        "Amanda": 6,
        "The Trees": 7,
        "Fog": 8,
        "The Tale of Custard Dragon": 9,
        "For Anne Gregory": 10,
    },
    ("footprints_without_feet", "prose"): {
        "A Triumph of Surgery": 1,
        "The Thief's Story": 2,
        "The Midnight Visitor": 3,
        "A Question of Trust": 4,
        "Footprints Without Feet": 5,
        "The Making of a Scientist": 6,
        "The Necklace": 7,
        "Bholi": 8,
        "The Book that Saved the Earth": 9,
    },
}


def get_chapter_order(
    book: Optional[str],
    chapter_type: Optional[str],
    chapter_name: str,
    fallback_idx: int,
) -> int:
    """Resolve chapter_order from map; fall back to alphabetical index."""
    if book and chapter_type:
        order_map = CHAPTER_ORDER_MAP.get((book, chapter_type), {})
        if chapter_name in order_map:
            return order_map[chapter_name]
        if order_map:
            logger.warning(
                f"  ⚠ '{chapter_name}' not in CHAPTER_ORDER_MAP for "
                f"({book}, {chapter_type}); using fallback order={fallback_idx}"
            )
    return fallback_idx


def get_embed_model() -> SentenceTransformer:
    """Load embedding model (cached after first call)."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"✅ Model loaded (dim={_embed_model.get_sentence_embedding_dimension()})")
    return _embed_model


# ===========================================================================
# PDF Text Extraction
# ===========================================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and clean text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(text)

        full_text = "\n\n".join(pages)

        # Clean up common PDF artifacts
        full_text = re.sub(r"\x00", "", full_text)
        full_text = re.sub(r"(\n\s*){3,}", "\n\n", full_text)
        full_text = re.sub(r"[ \t]+", " ", full_text)
        full_text = re.sub(r"(\d+)\s*\n\s*Rationalised", "", full_text)
        full_text = re.sub(r"NCERT.*?not to be republished", "", full_text, flags=re.IGNORECASE)

        logger.info(f"  Extracted {len(reader.pages)} pages, {len(full_text)} chars from {Path(pdf_path).name}")
        return full_text.strip()

    except Exception as e:
        logger.error(f"  Failed to read {pdf_path}: {e}")
        return ""


# ===========================================================================
# Smart Text Chunking
# ===========================================================================
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_length: int = MIN_CHUNK_LENGTH,
) -> List[str]:
    """Split text into overlapping word-based chunks."""
    if not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: List[str] = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()

        if len(current_words) + len(para_words) > chunk_size and current_words:
            chunk_text_str = " ".join(current_words)
            if len(current_words) >= min_length:
                chunks.append(chunk_text_str)

            if overlap > 0 and len(current_words) > overlap:
                current_words = current_words[-overlap:]
            else:
                current_words = []

        current_words.extend(para_words)

        while len(current_words) > chunk_size:
            chunk_words = current_words[:chunk_size]
            chunk_text_str = " ".join(chunk_words)
            if len(chunk_words) >= min_length:
                chunks.append(chunk_text_str)
            current_words = current_words[chunk_size - overlap:]

    if current_words and len(current_words) >= min_length:
        chunks.append(" ".join(current_words))

    return chunks


# ===========================================================================
# Batch Embedding Generation (LOCAL — no API calls!)
# ===========================================================================
def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts using local model."""
    model = get_embed_model()
    embeddings = model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


# ===========================================================================
# Database Operations
# ===========================================================================
class DatabaseManager:
    def __init__(self):
        self.conn: Optional[asyncpg.Connection] = None

    async def connect(self) -> bool:
        try:
            self.conn = await asyncpg.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=os.getenv("DATABASE_PASSWORD"),
                database=DB_NAME,
                ssl="require",
                timeout=30,
            )
            logger.info("✅ Database connected")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    async def close(self):
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed")

    async def delete_all_for_class(
        self,
        subject: str,
        class_grade: str,
        book: Optional[str] = None,
        chapter_type: Optional[str] = None,
    ) -> int:
        """
        Delete chunks for a subject + class.
        If `book` AND `chapter_type` are both provided, only delete that slice
        (so re-running for First_Flight_Poems doesn't wipe First_Flight_Prose).
        Otherwise (default), delete the whole subject+class — original behavior.
        """
        if book and chapter_type:
            result = await self.conn.execute(
                """
                DELETE FROM ncert_chunks
                WHERE subject ILIKE $1 AND class_grade = $2
                  AND book = $3 AND chapter_type = $4
                """,
                f"%{subject}%",
                str(class_grade),
                book,
                chapter_type,
            )
        else:
            result = await self.conn.execute(
                """
                DELETE FROM ncert_chunks
                WHERE subject ILIKE $1 AND class_grade = $2
                """,
                f"%{subject}%",
                str(class_grade),
            )
        return int(result.split()[-1]) if result else 0

    async def insert_chunks_batch(
        self,
        class_grade: str,
        subject: str,
        chapter: str,
        chunks: List[str],
        embeddings: List[List[float]],
        book: Optional[str] = None,
        chapter_type: Optional[str] = None,
        chapter_order: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Insert multiple chunks at once. Returns (success, failed)."""
        success = 0
        failed = 0

        for content, embedding in zip(chunks, embeddings):
            try:
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                await self.conn.execute(
                    """
                    INSERT INTO ncert_chunks
                    (class_grade, subject, chapter, content, embedding, created_at,
                     book, chapter_type, chapter_order)
                    VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9)
                    """,
                    str(class_grade),
                    subject,
                    chapter,
                    content,
                    embedding_str,
                    datetime.now(),
                    book,
                    chapter_type,
                    chapter_order,
                )
                success += 1
            except Exception as e:
                logger.error(f"  Insert failed: {e}")
                failed += 1

        return success, failed

    async def get_chapter_stats(
        self,
        subject: str,
        class_grade: str,
        book: Optional[str] = None,
        chapter_type: Optional[str] = None,
    ) -> List[Dict]:
        if book and chapter_type:
            rows = await self.conn.fetch(
                """
                SELECT chapter, chapter_order, COUNT(*) as chunks,
                       LENGTH(MIN(content)) as min_len,
                       LENGTH(MAX(content)) as max_len
                FROM ncert_chunks
                WHERE subject ILIKE $1 AND class_grade = $2
                  AND book = $3 AND chapter_type = $4
                GROUP BY chapter, chapter_order
                ORDER BY chapter_order NULLS LAST, chapter
                """,
                f"%{subject}%",
                str(class_grade),
                book,
                chapter_type,
            )
        else:
            rows = await self.conn.fetch(
                """
                SELECT chapter, COUNT(*) as chunks,
                       LENGTH(MIN(content)) as min_len,
                       LENGTH(MAX(content)) as max_len
                FROM ncert_chunks
                WHERE subject ILIKE $1 AND class_grade = $2
                GROUP BY chapter
                ORDER BY chapter
                """,
                f"%{subject}%",
                str(class_grade),
            )
        return [dict(r) for r in rows]


# ===========================================================================
# Main Ingestion Pipeline
# ===========================================================================
async def ingest_folder(
    folder_path: str,
    subject: str,
    class_grade: str,
    clean_first: bool = True,
    book: Optional[str] = None,
    chapter_type: Optional[str] = None,
):
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {folder_path}")
        return

    logger.info(f"Found {len(pdf_files)} PDFs in {folder_path}")
    for f in pdf_files:
        logger.info(f"   -> {f.stem}")

    # Check DB password
    db_password = os.getenv("DATABASE_PASSWORD", "").strip()
    if not db_password:
        logger.error("DATABASE_PASSWORD not set in .env")
        return

    # Load embedding model FIRST
    get_embed_model()

    # Connect to DB
    db = DatabaseManager()
    if not await db.connect():
        return

    try:
        if clean_first:
            deleted = await db.delete_all_for_class(
                subject, class_grade, book=book, chapter_type=chapter_type
            )
            scope = (
                f"{subject} class {class_grade} ({book}/{chapter_type})"
                if (book and chapter_type)
                else f"{subject} class {class_grade}"
            )
            logger.info(f"Deleted {deleted} existing chunks for {scope}")

        total_chunks = 0
        total_uploaded = 0
        total_failed = 0
        chapter_stats: List[Dict] = []
        overall_start = time.time()

        for idx, pdf_file in enumerate(pdf_files, start=1):
            chapter_name = pdf_file.stem
            chapter_order = get_chapter_order(book, chapter_type, chapter_name, idx)

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {chapter_name} (order={chapter_order})")
            logger.info(f"{'='*60}")

            chapter_start = time.time()

            # Step 1: Extract text
            text = extract_text_from_pdf(str(pdf_file))
            if not text:
                logger.warning(f"  No text extracted, skipping")
                continue

            # Step 2: Chunk
            chunks = chunk_text(text)
            avg_words = sum(len(c.split()) for c in chunks) // max(len(chunks), 1)
            logger.info(f"  Created {len(chunks)} chunks (avg {avg_words} words)")
            total_chunks += len(chunks)

            if not chunks:
                logger.warning(f"  No valid chunks, skipping")
                continue

            # Step 3: Batch embed (LOCAL — instant!)
            logger.info(f"  Generating {len(chunks)} embeddings...")
            embeddings = generate_embeddings_batch(chunks)
            logger.info(f"  Embeddings generated")

            # Step 4: Upload to DB
            logger.info(f"  Uploading to database...")
            uploaded, failed = await db.insert_chunks_batch(
                class_grade=str(class_grade),
                subject=subject,
                chapter=chapter_name,
                chunks=chunks,
                embeddings=embeddings,
                book=book,
                chapter_type=chapter_type,
                chapter_order=chapter_order,
            )

            total_uploaded += uploaded
            total_failed += failed
            elapsed = round(time.time() - chapter_start, 1)

            chapter_stats.append({
                "chapter": chapter_name,
                "order": chapter_order,
                "chunks": len(chunks),
                "uploaded": uploaded,
                "failed": failed,
                "time": elapsed,
            })
            logger.info(f"  Done: {uploaded}/{len(chunks)} chunks ({elapsed}s)")

        # Final summary
        total_time = round(time.time() - overall_start, 1)
        logger.info(f"\n{'='*60}")
        logger.info("INGESTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Subject: {subject} | Class: {class_grade}")
        if book or chapter_type:
            logger.info(f"Book: {book} | Type: {chapter_type}")
        logger.info(f"PDFs processed: {len(pdf_files)}")
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(f"Uploaded: {total_uploaded}")
        logger.info(f"Failed: {total_failed}")
        logger.info(f"Total time: {total_time}s")
        logger.info(f"\nPer chapter:")

        for stat in chapter_stats:
            status = "OK" if stat["failed"] == 0 else "WARN"
            logger.info(
                f"  [{status}] #{stat['order']} {stat['chapter']}: "
                f"{stat['uploaded']}/{stat['chunks']} chunks ({stat['time']}s)"
            )

        # Verify from DB
        logger.info(f"\nVerifying in database...")
        db_stats = await db.get_chapter_stats(
            subject, class_grade, book=book, chapter_type=chapter_type
        )
        for row in db_stats:
            order_str = f"#{row['chapter_order']} " if row.get("chapter_order") else ""
            logger.info(
                f"  {order_str}{row['chapter']}: {row['chunks']} chunks "
                f"(content: {row['min_len']}-{row['max_len']} chars)"
            )

        logger.info(f"\nIngestion complete!")

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await db.close()


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="NCERT PDF Ingestion")
    parser.add_argument("--folder", required=True, help="Folder with chapter PDFs")
    parser.add_argument("--subject", required=True, help="Subject name")
    parser.add_argument("--class_grade", required=True, help="Class grade")
    parser.add_argument("--no-clean", action="store_true", help="Don't delete existing data")
    parser.add_argument(
        "--book",
        default=None,
        help="Book name (e.g., first_flight, footprints_without_feet). "
             "Optional — only used for English currently.",
    )
    parser.add_argument(
        "--chapter_type",
        default=None,
        help="Chapter type: prose, poem, play. Optional.",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("NCERT PDF INGESTION (Local Embeddings)")
    print(f"{'='*60}")
    print(f"Folder:       {args.folder}")
    print(f"Subject:      {args.subject}")
    print(f"Class:        {args.class_grade}")
    print(f"Book:         {args.book or '(none)'}")
    print(f"Chapter type: {args.chapter_type or '(none)'}")
    print(f"Embeddings:   {EMBEDDING_MODEL_NAME} (384-dim, local)")
    print(f"Clean:        {'No' if args.no_clean else 'Yes'}")
    print(f"{'='*60}\n")

    if not args.no_clean:
        if args.book and args.chapter_type:
            scope = f"{args.subject} class {args.class_grade} (book={args.book}, type={args.chapter_type})"
        else:
            scope = f"ALL {args.subject} class {args.class_grade}"
        confirm = input(
            f"This will DELETE existing data for {scope} and re-ingest. Continue? (y/N): "
        )
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    asyncio.run(
        ingest_folder(
            folder_path=args.folder,
            subject=args.subject,
            class_grade=args.class_grade,
            clean_first=not args.no_clean,
            book=args.book,
            chapter_type=args.chapter_type,
        )
    )


if __name__ == "__main__":
    main()