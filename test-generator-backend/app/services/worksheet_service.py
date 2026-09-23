"""
a4ai Worksheet / Assignment Generator Service
================================================
Generates clean assignment PDFs from module content.
Features:
- Question generation using Gemini
- Optional school name + logo (header + watermark)
- Two downloads: with answer key / without
- Clean professional PDF format using ReportLab
"""

import os
import json
import io
import logging
import tempfile
from typing import Optional, List

logger = logging.getLogger(__name__)

# Lazy imports
def get_genai():
    from google import genai
    return genai

from app.core.db_pool import get_db_connection

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


class WorksheetService:

    # ─────────────────────────────────────────────
    # 1. GENERATE QUESTIONS FROM MODULE
    # ─────────────────────────────────────────────
    @staticmethod
    def generate_worksheet_questions(
        module_id: str,
        teacher_id: str,
        num_questions: int = 10,
        question_types: List[str] = None,
        difficulty: str = "medium",
    ) -> tuple:
        """Generate assignment questions from module content. Returns (questions_list, error)."""
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
            conn.close()

            if not row:
                return None, "Module not found"

            title, subject, class_level, full_text, summary_raw, token_count, status = row

            if status != "ready":
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

            # Use full text for small docs, chunks for large
            context_text = full_text if (token_count and token_count < 100000) else (full_text[:200000] if full_text else "")

            if question_types is None:
                question_types = ["MCQ", "Fill in the Blanks", "True/False", "Short Answer", "Long Answer"]

            # Summary context
            summary_ctx = ""
            if summary:
                summary_ctx = f"""
MODULE TOPICS: {json.dumps(summary.get('topics', []), ensure_ascii=False)}
IMPORTANT TERMS: {json.dumps(summary.get('important_terms', []), ensure_ascii=False)}
FORMULAS: {json.dumps(summary.get('formulas_or_rules', []), ensure_ascii=False)}
"""

            prompt = f"""You are an expert Indian school teacher creating an assignment worksheet.

SUBJECT: {subject}
CLASS: {class_level}
CHAPTER: {title}
DIFFICULTY: {difficulty}
TOTAL QUESTIONS: {num_questions}
QUESTION TYPES TO INCLUDE: {', '.join(question_types)}

{summary_ctx}

SOURCE CONTENT:
{context_text[:400000]}

Generate EXACTLY {num_questions} questions for a student assignment/worksheet.

RULES:
1. Questions must be DIRECTLY from the source content — no outside knowledge
2. Mix the question types as specified
3. For MCQs: 4 options (a, b, c, d) with one correct
4. For Fill in the Blanks: use _______ for the blank
5. For True/False: state clearly
6. Short Answer: 2-3 line answers expected
7. Long Answer: 5-6 line answers expected
8. Questions should be in logical order — easy to hard
9. If content has Hindi, questions can be in Hindi too
10. Use proper chemical equations, formulas, scientific notation where needed

Output as JSON:
{{
    "questions": [
        {{
            "q_no": 1,
            "type": "MCQ",
            "question": "question text here",
            "options": ["a) option1", "b) option2", "c) option3", "d) option4"],
            "answer": "correct answer with brief explanation"
        }},
        {{
            "q_no": 2,
            "type": "Fill in the Blanks",
            "question": "_______ is the process of...",
            "answer": "Oxidation"
        }},
        {{
            "q_no": 3,
            "type": "True/False",
            "question": "Rusting is a chemical change. (True/False)",
            "answer": "True. Rusting involves formation of iron oxide which is a new substance."
        }},
        {{
            "q_no": 4,
            "type": "Short Answer",
            "question": "What is a balanced chemical equation?",
            "answer": "A balanced chemical equation has equal number of atoms..."
        }}
    ]
}}
Output ONLY valid JSON."""

            genai = get_genai()
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or "")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 16000, "response_mime_type": "application/json"},
            )

            data = json.loads(response.text)
            questions = data.get("questions", [])

            return {
                "title": title,
                "subject": subject,
                "class": class_level,
                "questions": questions,
                "num_questions": len(questions),
            }, None

        except Exception as e:
            logger.error(f"Worksheet question generation failed: {e}")
            return None, str(e)

    # ─────────────────────────────────────────────
    # 2. GENERATE PDF
    # ─────────────────────────────────────────────
    @staticmethod
    def generate_pdf(
        worksheet_data: dict,
        include_answers: bool = False,
        school_name: str = None,
        logo_bytes: bytes = None,
        logo_ext: str = "png",
    ) -> bytes:
        """Generate a clean assignment PDF. Returns PDF bytes."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm, cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.colors import HexColor, Color
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable, Image
        )
        from reportlab.lib import colors

        title = worksheet_data.get("title", "Assignment")
        subject = worksheet_data.get("subject", "")
        class_level = worksheet_data.get("class", "")
        questions = worksheet_data.get("questions", [])

        # ── Setup PDF ──
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=2 * cm,
            bottomMargin=1.5 * cm,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
        )

        # Colors
        primary = HexColor("#1e3a5f")
        accent = HexColor("#4f46e5")
        light_bg = HexColor("#f8fafc")
        border_color = HexColor("#cbd5e1")
        text_color = HexColor("#1e293b")

        # Styles
        styles = getSampleStyleSheet()

        school_style = ParagraphStyle(
            'SchoolName', parent=styles['Normal'],
            fontSize=16, fontName='Helvetica-Bold',
            alignment=TA_CENTER, textColor=primary,
            spaceAfter=2 * mm,
        )

        title_style = ParagraphStyle(
            'AssignmentTitle', parent=styles['Normal'],
            fontSize=13, fontName='Helvetica-Bold',
            alignment=TA_CENTER, textColor=accent,
            spaceAfter=3 * mm,
        )

        meta_style = ParagraphStyle(
            'Meta', parent=styles['Normal'],
            fontSize=10, fontName='Helvetica',
            alignment=TA_CENTER, textColor=text_color,
            spaceAfter=6 * mm,
        )

        question_style = ParagraphStyle(
            'Question', parent=styles['Normal'],
            fontSize=11, fontName='Helvetica-Bold',
            textColor=text_color, leading=15,
            spaceBefore=4 * mm, spaceAfter=2 * mm,
            leftIndent=0,
        )

        option_style = ParagraphStyle(
            'Option', parent=styles['Normal'],
            fontSize=10, fontName='Helvetica',
            textColor=text_color, leading=14,
            leftIndent=8 * mm,
        )

        type_badge_style = ParagraphStyle(
            'TypeBadge', parent=styles['Normal'],
            fontSize=8, fontName='Helvetica-Bold',
            textColor=HexColor("#6366f1"),
        )

        answer_style = ParagraphStyle(
            'Answer', parent=styles['Normal'],
            fontSize=10, fontName='Helvetica-Oblique',
            textColor=HexColor("#16a34a"), leading=14,
            leftIndent=8 * mm, spaceBefore=2 * mm,
        )

        answer_heading_style = ParagraphStyle(
            'AnswerHeading', parent=styles['Normal'],
            fontSize=14, fontName='Helvetica-Bold',
            alignment=TA_CENTER, textColor=accent,
            spaceBefore=8 * mm, spaceAfter=6 * mm,
        )

        # ── Watermark callback ──
        logo_path_tmp = None
        if logo_bytes:
            logo_path_tmp = tempfile.mktemp(suffix=f".{logo_ext}")
            with open(logo_path_tmp, "wb") as f:
                f.write(logo_bytes)

        def add_watermark(canvas, doc):
            if logo_path_tmp:
                canvas.saveState()
                canvas.setFillAlpha(0.06)
                page_w, page_h = A4
                try:
                    canvas.drawImage(
                        logo_path_tmp,
                        page_w / 2 - 80, page_h / 2 - 80,
                        width=160, height=160,
                        mask='auto',
                        preserveAspectRatio=True,
                    )
                except:
                    pass
                canvas.restoreState()

        # ── Build content ──
        story = []

        # Logo in header
        if logo_path_tmp:
            try:
                logo_img = Image(logo_path_tmp, width=18 * mm, height=18 * mm)
                logo_img.hAlign = 'CENTER'
                story.append(logo_img)
                story.append(Spacer(1, 3 * mm))
            except:
                pass

        # School name
        if school_name:
            story.append(Paragraph(school_name, school_style))

        # Title
        story.append(Paragraph("ASSIGNMENT / WORKSHEET", title_style))

        # Separator line
        story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=3 * mm))

        # Meta info
        from datetime import datetime
        today = datetime.now().strftime("%d %B %Y")

        meta_parts = []
        if class_level:
            meta_parts.append(f"<b>Class:</b> {class_level}")
        if subject:
            meta_parts.append(f"<b>Subject:</b> {subject}")
        if title:
            meta_parts.append(f"<b>Chapter:</b> {title}")
        meta_parts.append(f"<b>Date:</b> {today}")

        if meta_parts:
            story.append(Paragraph("&nbsp;&nbsp;|&nbsp;&nbsp;".join(meta_parts), meta_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=border_color, spaceAfter=5 * mm))

        # ── Questions ──
        current_type = None
        for q in questions:
            q_type = q.get("type", "")
            q_no = q.get("q_no", "")
            question_text = q.get("question", "")
            options = q.get("options", [])

            # Type section header (group by type)
            if q_type != current_type:
                current_type = q_type
                story.append(Spacer(1, 3 * mm))
                story.append(Paragraph(
                    f'<font color="#4f46e5"><b>● {q_type}</b></font>',
                    type_badge_style
                ))
                story.append(Spacer(1, 2 * mm))

            # Question
            story.append(Paragraph(
                f"<b>{q_no}.</b>&nbsp;&nbsp;{question_text}",
                question_style
            ))

            # Options (for MCQ)
            if options:
                for opt in options:
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{opt}", option_style))

            # Answer (only if include_answers)
            if include_answers:
                answer_text = q.get("answer", "")
                if answer_text:
                    story.append(Paragraph(
                        f"<b>Ans:</b> {answer_text}",
                        answer_style
                    ))

            story.append(Spacer(1, 2 * mm))

        # ── Answer Key page (separate) ──
        if include_answers:
            story.append(PageBreak())
            story.append(Paragraph("ANSWER KEY", answer_heading_style))
            story.append(HRFlowable(width="100%", thickness=1, color=accent, spaceAfter=5 * mm))

            for q in questions:
                q_no = q.get("q_no", "")
                answer_text = q.get("answer", "")
                q_type = q.get("type", "")

                ans_text = f"<b>Q{q_no} ({q_type}):</b> {answer_text}"
                story.append(Paragraph(ans_text, ParagraphStyle(
                    'AnsKey', parent=styles['Normal'],
                    fontSize=10, fontName='Helvetica',
                    textColor=text_color, leading=14,
                    spaceBefore=2 * mm,
                )))

        # ── Build PDF ──
        doc.build(story, onFirstPage=add_watermark, onLaterPages=add_watermark)

        # Cleanup
        if logo_path_tmp and os.path.exists(logo_path_tmp):
            os.unlink(logo_path_tmp)

        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes