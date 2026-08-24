"""
a4ai Module Service
====================
Path: RAG-BACKEND/module_service.py

Uses same patterns as rag_system.py:
  - psycopg2 for database
  - google.generativeai for Gemini
  - Supabase client for storage only
"""

import os
import json
import uuid
import logging
import tempfile
import time

logger = logging.getLogger(__name__)

# ─── Lazy imports (same pattern as rag_system.py) ─────────
_fitz = None
_genai = None
_supabase_client = None

def get_fitz():
    global _fitz
    if _fitz is None:
        import fitz
        _fitz = fitz
    return _fitz

def get_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or "")
        _genai = genai
    return _genai

def get_supabase():
    """Supabase client — only for Storage operations (download/delete files)."""
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL") or ""
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY") or ""
        if not url or not key:
            logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
            return None
        _supabase_client = create_client(url, key)
    return _supabase_client

def get_db_connection():
    """Get psycopg2 connection — same as DatabaseManager in rag_system.py."""
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "db.dcmnzvjftmdbywrjkust.supabase.co"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            port=int(os.getenv("DB_PORT", "5432")),
            sslmode="require",
            connect_timeout=10,
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return None


# ─── Config ────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.6-flash"
CHUNK_SIZE_CHARS = 3000       # ~750 tokens per chunk
CHUNK_OVERLAP_CHARS = 400
SCANNED_THRESHOLD = 50        # avg chars/page below this = scanned


# ═══════════════════════════════════════════════════════════
# MODULE SERVICE CLASS
# ═══════════════════════════════════════════════════════════

class ModuleService:

    # ────────────────────────────────────────────────────────
    # CREATE MODULE (Step 1: just insert row, return id)
    # ────────────────────────────────────────────────────────
    @staticmethod
    def create_module(teacher_id, storage_path, original_filename, subject, class_level,
                      file_type="pdf", file_size_bytes=None, institute_id=None):
        """Insert module row with status='processing'. Returns module_id."""
        conn = get_db_connection()
        if not conn:
            return None, "Database connection failed"

        module_id = str(uuid.uuid4())
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO modules (id, teacher_id, storage_path, original_filename,
                    subject, class, file_type, file_size_bytes, institute_id, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'processing')
            """, (module_id, teacher_id, storage_path, original_filename,
                  subject, class_level, file_type, file_size_bytes, institute_id))
            conn.close()
            logger.info(f"✅ Module created: {module_id}")
            return module_id, None
        except Exception as e:
            conn.close()
            logger.error(f"❌ Create module failed: {e}")
            return None, str(e)

    # ────────────────────────────────────────────────────────
    # PROCESS MODULE (Step 2: extract → summarize → chunk)
    # ────────────────────────────────────────────────────────
    @staticmethod
    def process_module(module_id):
        """
        Main pipeline:
        1. Download PDF from Supabase Storage
        2. Extract text (PyMuPDF, fallback Gemini for scanned)
        3. Summarize with Gemini → structured JSON
        4. Chunk text → store in module_chunks
        5. Update status = 'ready'
        """
        conn = get_db_connection()
        if not conn:
            return {"success": False, "error": "Database connection failed"}

        try:
            # Fetch module row
            cur = conn.cursor()
            cur.execute("SELECT storage_path, subject, class, file_type, teacher_id FROM modules WHERE id = %s", (module_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                return {"success": False, "error": "Module not found"}

            storage_path, subject, class_level, file_type, teacher_id = row

            # ── STEP 2a: Download from Supabase Storage ──
            ModuleService._update_status(conn, module_id, "extracting")
            sb = get_supabase()
            if not sb:
                raise Exception("Supabase client not available. Check SUPABASE_URL and SUPABASE_SERVICE_KEY env vars.")

            file_bytes = sb.storage.from_("Modules").download(storage_path)

            # Save to temp file
            suffix = ".pdf" if file_type == "pdf" else ".docx"
            tmp_path = tempfile.mktemp(suffix=suffix)
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)

            try:
                # ── STEP 2b: Extract text ──
                if file_type == "pdf":
                    full_text, page_count, is_scanned = ModuleService._extract_pdf(tmp_path)
                else:
                    full_text, page_count, is_scanned = ModuleService._extract_docx(tmp_path)

                # Scanned PDF? Use Gemini to read it
                if is_scanned and file_type == "pdf":
                    logger.info(f"Scanned PDF detected for {module_id}, using Gemini OCR")
                    full_text = ModuleService._extract_with_gemini(file_bytes)

                if not full_text or len(full_text.strip()) < 50:
                    raise Exception("PDF se text extract nahi ho paya. File empty ya corrupt ho sakti hai.")

                token_count = len(full_text) // 4

                cur.execute("""
                    UPDATE modules SET full_text=%s, page_count=%s, is_scanned=%s, token_count=%s
                    WHERE id=%s
                """, (full_text, page_count, is_scanned, token_count, module_id))

                # ── STEP 2c: Summarize with Gemini ──
                ModuleService._update_status(conn, module_id, "summarizing")
                summary = ModuleService._generate_summary(full_text, subject, class_level, page_count)

                title = summary.get("title", "Untitled Module")
                summary_json = json.dumps(summary, ensure_ascii=False)

                cur.execute("""
                    UPDATE modules SET summary=%s, title=%s WHERE id=%s
                """, (summary_json, title, module_id))

                # ── STEP 2d: Chunk and store ──
                ModuleService._update_status(conn, module_id, "chunking")
                chunks = ModuleService._create_chunks(full_text, page_count)
                ModuleService._store_chunks(conn, module_id, chunks)

                # ── DONE ──
                ModuleService._update_status(conn, module_id, "ready")
                conn.close()

                return {
                    "success": True,
                    "module_id": module_id,
                    "title": title,
                    "page_count": page_count,
                    "chunks_count": len(chunks),
                    "topics_count": len(summary.get("topics", [])),
                }

            finally:
                # Cleanup temp file
                import os as _os
                if _os.path.exists(tmp_path):
                    _os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"❌ Module processing failed [{module_id}]: {e}")
            try:
                cur = conn.cursor()
                cur.execute("UPDATE modules SET status='failed', error_message=%s WHERE id=%s",
                            (str(e)[:500], module_id))
            except:
                pass
            conn.close()
            return {"success": False, "error": str(e)}

    # ────────────────────────────────────────────────────────
    # TEXT EXTRACTION
    # ────────────────────────────────────────────────────────
    @staticmethod
    def _extract_pdf(filepath):
        """Extract text from PDF using PyMuPDF. Returns (text, page_count, is_scanned)."""
        fitz = get_fitz()
        doc = fitz.open(filepath)
        page_count = len(doc)
        pages_text = []

        for i in range(page_count):
            text = doc[i].get_text("text")
            pages_text.append(f"--- Page {i+1} ---\n{text}")

        doc.close()
        full_text = "\n\n".join(pages_text)

        # Check if scanned
        clean_len = len(full_text.replace(" ", "").replace("\n", ""))
        avg_chars = clean_len / max(page_count, 1)
        is_scanned = avg_chars < SCANNED_THRESHOLD

        return full_text, page_count, is_scanned

    @staticmethod
    def _extract_docx(filepath):
        """Extract text from DOCX."""
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        page_count = max(1, len(full_text) // 3000)
        return full_text, page_count, False

    @staticmethod
    def _extract_with_gemini(file_bytes):
        """Use Gemini to OCR a scanned PDF."""
        genai = get_genai()
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Upload file to Gemini
        import io
        tmp = tempfile.mktemp(suffix=".pdf")
        with open(tmp, "wb") as f:
            f.write(file_bytes)

        uploaded = genai.upload_file(tmp, mime_type="application/pdf")

        response = model.generate_content(
            [
                uploaded,
                "Extract ALL text from this document exactly as written. "
                "Preserve headings, bullet points, numbering. "
                "If text is in Hindi/Devanagari, keep it as-is. "
                "Output only the extracted text, nothing else."
            ],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=65000,
            ),
        )

        # Cleanup
        import os as _os
        if _os.path.exists(tmp):
            _os.unlink(tmp)

        return response.text

    # ────────────────────────────────────────────────────────
    # SUMMARY GENERATION
    # ────────────────────────────────────────────────────────
    @staticmethod
    def _generate_summary(full_text, subject, class_level, page_count):
        """Generate structured module summary using Gemini."""
        genai = get_genai()
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Truncate if very large (Gemini 2.5 Flash handles 1M tokens but let's be safe)
        text_for_prompt = full_text[:800000]

        prompt = f"""You are an expert Indian education content analyzer.
Analyze this {subject} document for Class {class_level} and create a structured module summary.

DOCUMENT ({page_count} pages):
{text_for_prompt}

Create a JSON summary with this EXACT structure:
{{
    "title": "descriptive title for this module",
    "subject": "{subject}",
    "class": "{class_level}",
    "overview": "2-3 sentence summary of what this document covers",
    "topics": [
        {{
            "name": "Topic/Chapter name",
            "key_points": ["point 1", "point 2", "point 3"],
            "subtopics": ["subtopic 1", "subtopic 2"]
        }}
    ],
    "important_terms": [
        {{"term": "term name", "definition": "brief definition"}}
    ],
    "learning_objectives": ["objective 1", "objective 2"],
    "formulas_or_rules": ["formula 1", "rule 1"],
    "difficulty_level": "easy or medium or hard",
    "estimated_study_time": "X hours",
    "question_types_possible": ["MCQ", "Short Answer", "Long Answer", "Fill in the Blanks", "True/False"],
    "total_pages": {page_count}
}}

RULES:
- Identify ALL distinct topics/chapters in the document
- key_points: most important facts/concepts a student must know
- important_terms: terms a teacher would want to test
- formulas_or_rules: formulas, theorems, rules, dates worth memorizing
- If content is in Hindi, keep Hindi text as-is
- Be thorough — this summary will be used to generate test papers
- Output ONLY valid JSON, no markdown, no backticks, no explanation"""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=8000,
                response_mime_type="application/json",
            ),
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)

    # ────────────────────────────────────────────────────────
    # CHUNKING
    # ────────────────────────────────────────────────────────
    @staticmethod
    def _create_chunks(full_text, page_count):
        """Split text into overlapping chunks."""
        chunks = []
        idx = 0
        chunk_index = 0

        while idx < len(full_text):
            end = min(idx + CHUNK_SIZE_CHARS, len(full_text))

            # Break at sentence boundary
            if end < len(full_text):
                for sep in ["।", ".", "\n\n", "\n"]:
                    pos = full_text.rfind(sep, idx, end)
                    if pos > idx + CHUNK_SIZE_CHARS // 2:
                        end = pos + len(sep)
                        break

            chunk_text = full_text[idx:end].strip()
            if chunk_text and len(chunk_text) > 30:
                # Estimate page number
                ratio = idx / max(len(full_text), 1)
                est_page = max(1, min(page_count, int(ratio * page_count) + 1))

                chunks.append({
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "page_no": est_page,
                })
                chunk_index += 1

            idx = end - CHUNK_OVERLAP_CHARS if end < len(full_text) else end

        return chunks

    @staticmethod
    def _store_chunks(conn, module_id, chunks):
        """Store chunks in module_chunks table (without embeddings for now)."""
        if not chunks:
            return

        cur = conn.cursor()
        for chunk in chunks:
            cur.execute("""
                INSERT INTO module_chunks (id, module_id, chunk_index, page_no, content)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()),
                module_id,
                chunk["chunk_index"],
                chunk["page_no"],
                chunk["content"],
            ))

        logger.info(f"✅ Stored {len(chunks)} chunks for module {module_id}")

    # ────────────────────────────────────────────────────────
    # LIST / GET / DELETE
    # ────────────────────────────────────────────────────────
    @staticmethod
    def list_modules(teacher_id, subject=None, class_level=None):
        """List teacher's modules."""
        conn = get_db_connection()
        if not conn:
            return []

        try:
            query = """
                SELECT id, title, subject, class, status, page_count,
                       original_filename, is_scanned, created_at, error_message
                FROM modules
                WHERE teacher_id = %s
            """
            params = [teacher_id]

            if subject:
                query += " AND subject = %s"
                params.append(subject)
            if class_level:
                query += " AND class = %s"
                params.append(class_level)

            query += " ORDER BY created_at DESC LIMIT 50"

            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            conn.close()

            modules = []
            for r in rows:
                modules.append({
                    "id": r[0],
                    "title": r[1],
                    "subject": r[2],
                    "class": r[3],
                    "status": r[4],
                    "page_count": r[5],
                    "original_filename": r[6],
                    "is_scanned": r[7],
                    "created_at": r[8].isoformat() if r[8] else None,
                    "error_message": r[9],
                })
            return modules

        except Exception as e:
            logger.error(f"List modules failed: {e}")
            conn.close()
            return []

    @staticmethod
    def get_module(module_id, teacher_id):
        """Get module details including summary."""
        conn = get_db_connection()
        if not conn:
            return None

        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, subject, class, status, page_count, original_filename,
                       is_scanned, summary, error_message, token_count, created_at
                FROM modules
                WHERE id = %s AND teacher_id = %s
            """, (module_id, teacher_id))
            r = cur.fetchone()
            conn.close()

            if not r:
                return None

            summary = r[8]
            if isinstance(summary, str):
                try:
                    summary = json.loads(summary)
                except:
                    pass

            return {
                "id": r[0],
                "title": r[1],
                "subject": r[2],
                "class": r[3],
                "status": r[4],
                "page_count": r[5],
                "original_filename": r[6],
                "is_scanned": r[7],
                "summary": summary,
                "error_message": r[9],
                "token_count": r[10],
                "created_at": r[11].isoformat() if r[11] else None,
            }

        except Exception as e:
            logger.error(f"Get module failed: {e}")
            conn.close()
            return None

    @staticmethod
    def delete_module(module_id, teacher_id):
        """Delete module, chunks (cascade), and storage file."""
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"

        try:
            cur = conn.cursor()
            cur.execute("SELECT storage_path FROM modules WHERE id=%s AND teacher_id=%s",
                        (module_id, teacher_id))
            row = cur.fetchone()
            if not row:
                conn.close()
                return False, "Module not found"

            storage_path = row[0]

            # Delete from DB (chunks cascade automatically)
            cur.execute("DELETE FROM modules WHERE id=%s", (module_id,))
            conn.close()

            # Delete from storage
            try:
                sb = get_supabase()
                if sb:
                    sb.storage.from_("Modules").remove([storage_path])
            except Exception as e:
                logger.warning(f"Storage delete failed (non-critical): {e}")

            return True, None

        except Exception as e:
            conn.close()
            return False, str(e)

    # ────────────────────────────────────────────────────────
    # TEST GENERATION FROM MODULE
    # ────────────────────────────────────────────────────────
    @staticmethod
    def generate_test_from_module(module_id, teacher_id, num_questions=10,
                                   question_types=None, difficulty="medium", topics=None):
        """Generate test paper from module content using Gemini."""
        conn = get_db_connection()
        if not conn:
            return None, "Database connection failed"

        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT title, subject, class, full_text, summary, token_count, status
                FROM modules WHERE id=%s AND teacher_id=%s
            """, (module_id, teacher_id))
            row = cur.fetchone()

            if not row:
                conn.close()
                return None, "Module not found"

            title, subject, class_level, full_text, summary_raw, token_count, status = row

            if status != "ready":
                conn.close()
                return None, f"Module not ready. Current status: {status}"

            # Parse summary
            summary = {}
            if summary_raw:
                if isinstance(summary_raw, str):
                    try:
                        summary = json.loads(summary_raw)
                    except:
                        pass
                else:
                    summary = summary_raw

            # For small docs, use full text. For large, use chunks.
            if token_count and token_count > 100000:
                # Get chunks
                cur.execute("""
                    SELECT content FROM module_chunks
                    WHERE module_id=%s ORDER BY chunk_index LIMIT 20
                """, (module_id,))
                chunk_rows = cur.fetchall()
                context_text = "\n\n".join([c[0] for c in chunk_rows])
            else:
                context_text = full_text

            conn.close()

            if question_types is None:
                question_types = ["MCQ", "Short Answer", "Long Answer"]

            # Build summary context
            summary_context = ""
            if summary:
                summary_context = f"""
MODULE OVERVIEW: {summary.get('overview', '')}
TOPICS: {json.dumps(summary.get('topics', []), ensure_ascii=False)}
IMPORTANT TERMS: {json.dumps(summary.get('important_terms', []), ensure_ascii=False)}
FORMULAS/RULES: {json.dumps(summary.get('formulas_or_rules', []), ensure_ascii=False)}
"""

            prompt = f"""You are an expert Indian school teacher creating a test paper.

SUBJECT: {subject}
CLASS: {class_level}
MODULE: {title}
DIFFICULTY: {difficulty}
NUMBER OF QUESTIONS: {num_questions}
QUESTION TYPES: {', '.join(question_types)}
{summary_context}

CONTENT TO BASE QUESTIONS ON:
{context_text[:500000]}

Generate a test paper with EXACTLY {num_questions} questions.

RULES:
1. Questions must be DIRECTLY based on the content provided
2. Mix question types as specified
3. For MCQs: provide 4 options (a, b, c, d) with one correct answer
4. Difficulty: {difficulty}
5. Include marks for each question
6. If content has Hindi text, questions can be in Hindi too

Output as JSON:
{{
    "title": "Test: {title}",
    "subject": "{subject}",
    "class": "{class_level}",
    "total_marks": <sum>,
    "duration_minutes": <appropriate time>,
    "questions": [
        {{
            "q_no": 1,
            "type": "MCQ",
            "question": "question text",
            "options": ["a) ...", "b) ...", "c) ...", "d) ..."],
            "answer": "correct answer",
            "marks": 1,
            "difficulty": "easy",
            "topic": "topic name",
            "solution": "brief explanation"
        }}
    ]
}}
Output ONLY valid JSON."""

            genai = get_genai()
            model = genai.GenerativeModel(GEMINI_MODEL)

            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=16000,
                    response_mime_type="application/json",
                ),
            )

            test_data = json.loads(response.text)
            test_data["module_id"] = module_id
            test_data["source"] = "module"

            return test_data, None

        except Exception as e:
            logger.error(f"Test generation from module failed: {e}")
            return None, str(e)

    # ────────────────────────────────────────────────────────
    # UTILITY
    # ────────────────────────────────────────────────────────
    @staticmethod
    def _update_status(conn, module_id, status):
        cur = conn.cursor()
        cur.execute("UPDATE modules SET status=%s WHERE id=%s", (status, module_id))