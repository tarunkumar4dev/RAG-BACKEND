"""
NCERT PDF Ingestion Script — Production Grade (Local Embeddings)

Usage:
  python ingest_ncert.py --folder pdfs/Class10_Science --subject Science --class_grade 10
  python ingest_ncert.py --folder pdfs/Class10_Maths --subject Mathematics --class_grade 10
  python ingest_ncert.py --folder pdfs/Class9_Science --subject Science --class_grade 9
  python ingest_ncert.py --folder pdfs/Class9_Maths --subject Mathematics --class_grade 9

What it does:
  1. Reads all PDFs from a folder
  2. Extracts text with proper cleaning
  3. Chunks text into ~500-word overlapping segments
  4. Generates embeddings using all-MiniLM-L6-v2 (384-dim, LOCAL, FREE)
  5. Uploads to Supabase ncert_chunks table
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

    async def delete_all_for_class(self, subject: str, class_grade: str) -> int:
        """Delete ALL chunks for a subject + class."""
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
                    (class_grade, subject, chapter, content, embedding, created_at)
                    VALUES ($1, $2, $3, $4, $5::vector, $6)
                    """,
                    str(class_grade),
                    subject,
                    chapter,
                    content,
                    embedding_str,
                    datetime.now(),
                )
                success += 1
            except Exception as e:
                logger.error(f"  Insert failed: {e}")
                failed += 1

        return success, failed

    async def get_chapter_stats(self, subject: str, class_grade: str) -> List[Dict]:
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
            deleted = await db.delete_all_for_class(subject, class_grade)
            logger.info(f"Deleted {deleted} existing chunks for {subject} class {class_grade}")

        total_chunks = 0
        total_uploaded = 0
        total_failed = 0
        chapter_stats: List[Dict] = []
        overall_start = time.time()

        for pdf_file in pdf_files:
            chapter_name = pdf_file.stem
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {chapter_name}")
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
            )

            total_uploaded += uploaded
            total_failed += failed
            elapsed = round(time.time() - chapter_start, 1)

            chapter_stats.append({
                "chapter": chapter_name,
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
        logger.info(f"PDFs processed: {len(pdf_files)}")
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(f"Uploaded: {total_uploaded}")
        logger.info(f"Failed: {total_failed}")
        logger.info(f"Total time: {total_time}s")
        logger.info(f"\nPer chapter:")

        for stat in chapter_stats:
            status = "OK" if stat["failed"] == 0 else "WARN"
            logger.info(
                f"  [{status}] {stat['chapter']}: "
                f"{stat['uploaded']}/{stat['chunks']} chunks ({stat['time']}s)"
            )

        # Verify from DB
        logger.info(f"\nVerifying in database...")
        db_stats = await db.get_chapter_stats(subject, class_grade)
        for row in db_stats:
            logger.info(
                f"  {row['chapter']}: {row['chunks']} chunks "
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

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("NCERT PDF INGESTION (Local Embeddings)")
    print(f"{'='*60}")
    print(f"Folder:     {args.folder}")
    print(f"Subject:    {args.subject}")
    print(f"Class:      {args.class_grade}")
    print(f"Embeddings: {EMBEDDING_MODEL_NAME} (384-dim, local)")
    print(f"Clean:      {'No' if args.no_clean else 'Yes'}")
    print(f"{'='*60}\n")

    if not args.no_clean:
        confirm = input(
            f"This will DELETE all existing {args.subject} class {args.class_grade} "
            f"data and re-ingest. Continue? (y/N): "
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
        )
    )


if __name__ == "__main__":
    main()