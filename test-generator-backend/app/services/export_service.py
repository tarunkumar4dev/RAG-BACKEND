"""
Export Service — Professional PDF and DOCX test paper generation.

Features:
  - Institute logo in header (from base64)
  - Professional school-style header layout
  - Proper LaTeX → formatted text (superscripts, subscripts, fractions, symbols)
  - Clean question numbering with marks
  - MCQ options (A, B, C, D) properly aligned
  - Separate answer key page
  - Teacher copy with inline explanations
  - Footer with preparation info
"""

import io
import re
import base64
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# LaTeX → Clean Text (for ReportLab Paragraphs)
# Uses <super> and <sub> tags (NOT Unicode chars which break in ReportLab)
# ═══════════════════════════════════════════════════════════════════════

SYMBOL_MAP = {
    r'\times': '×', r'\div': '÷', r'\pm': '±', r'\mp': '∓',
    r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
    r'\infty': '∞', r'\therefore': '∴', r'\because': '∵',
    r'\rightarrow': '→', r'\leftarrow': '←', r'\Rightarrow': '⇒',
    r'\degree': '°', r'\circ': '°',
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ', r'\mu': 'μ',
    r'\pi': 'π', r'\sigma': 'σ', r'\omega': 'ω', r'\Omega': 'Ω',
    r'\phi': 'φ', r'\psi': 'ψ', r'\rho': 'ρ', r'\tau': 'τ',
    r'\Delta': 'Δ', r'\nabla': '∇', r'\sum': 'Σ', r'\prod': 'Π',
}

TRIG_FUNCS = {
    r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
    r'\log': 'log', r'\ln': 'ln', r'\sec': 'sec',
    r'\csc': 'csc', r'\cot': 'cot',
}


def _latex_to_paragraph(text: str) -> str:
    """
    Convert LaTeX math to ReportLab Paragraph XML.
    Uses <super> and <sub> tags for proper rendering.
    """
    if not text:
        return ""

    result = text

    # 1. Replace symbols first
    for latex, symbol in SYMBOL_MAP.items():
        result = result.replace(latex, symbol)

    for latex, func in TRIG_FUNCS.items():
        result = result.replace(latex, func)

    # 2. Remove $ delimiters
    result = re.sub(r'\$([^$]+)\$', r'\1', result)

    # 3. \frac{a}{b} → (a/b)
    result = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', result)

    # 4. \sqrt{x} → √(x)
    result = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', result)

    # 5. ^{...} → <super>...</super>
    result = re.sub(r'\^\{([^}]*)\}', r'<super>\1</super>', result)
    # ^x (single char)
    result = re.sub(r'\^([a-zA-Z0-9])', r'<super>\1</super>', result)

    # 6. _{...} → <sub>...</sub>
    result = re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', result)
    # _x (single char)
    result = re.sub(r'_([a-zA-Z0-9])', r'<sub>\1</sub>', result)

    # 7. \text{...} → plain text
    result = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]*)\}', r'\1', result)

    # 8. Clean remaining backslash commands
    result = re.sub(r'\\([a-zA-Z]+)', r'\1', result)

    # 9. Clean braces
    result = result.replace('{', '').replace('}', '')

    # 10. Escape XML special chars (but preserve our <super> <sub> tags)
    # Must be careful not to escape our tags
    result = result.replace('&', '&amp;')
    # Don't escape < and > that are part of our tags
    # Escape < and > but protect our tags
    tags_placeholder = {}
    for i, tag in enumerate(re.findall(r'</?(?:super|sub|b|i)>', result)):
        placeholder = f"__TAG{i}__"
        tags_placeholder[placeholder] = tag
        result = result.replace(tag, placeholder, 1)
    result = result.replace('<', '&lt;').replace('>', '&gt;')
    for placeholder, tag in tags_placeholder.items():
        result = result.replace(placeholder, tag)

    return result


def _latex_to_plain(text: str) -> str:
    """Convert LaTeX to plain text (for DOCX and plain copy)."""
    if not text:
        return ""

    result = text
    for latex, symbol in SYMBOL_MAP.items():
        result = result.replace(latex, symbol)
    for latex, func in TRIG_FUNCS.items():
        result = result.replace(latex, func)

    result = re.sub(r'\$([^$]+)\$', r'\1', result)
    result = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', result)
    result = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', result)
    result = re.sub(r'\^\{([^}]*)\}', r'^\1', result)
    result = re.sub(r'_\{([^}]*)\}', r'_\1', result)
    result = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\([a-zA-Z]+)', r'\1', result)
    result = result.replace('{', '').replace('}', '')

    return result.strip()


# ═══════════════════════════════════════════════════════════════════════
# PDF Generation — Professional School Paper Format
# ═══════════════════════════════════════════════════════════════════════

def generate_pdf(
    questions: List[dict],
    exam_title: str = "Test Paper",
    board: str = "CBSE",
    class_grade: str = "10",
    subject: str = "Science",
    include_answers: bool = False,
    include_explanations: bool = False,
    logo_base64: Optional[str] = None,
) -> bytes:
    """Generate a professional PDF test paper with optional logo."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image as RLImage,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 4 * cm  # usable width

    # ── Custom Styles ───────────────────────────────────────────────
    styles.add(ParagraphStyle(
        name='SchoolName', parent=styles['Title'],
        fontSize=14, leading=18, spaceAfter=2,
        alignment=TA_CENTER, textColor=HexColor('#1a1a2e'),
        fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='ExamMeta', parent=styles['Normal'],
        fontSize=10, alignment=TA_CENTER,
        textColor=HexColor('#4a4a6a'), spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle', parent=styles['Heading2'],
        fontSize=11, spaceBefore=14, spaceAfter=6,
        textColor=HexColor('#1a1a2e'), fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='QText', parent=styles['Normal'],
        fontSize=10, spaceBefore=8, spaceAfter=3,
        leading=14, textColor=HexColor('#1f1f3a'),
    ))
    styles.add(ParagraphStyle(
        name='QTextBold', parent=styles['Normal'],
        fontSize=10, spaceBefore=8, spaceAfter=3,
        leading=14, textColor=HexColor('#1f1f3a'),
        fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='Option', parent=styles['Normal'],
        fontSize=9.5, leftIndent=18, spaceBefore=2, spaceAfter=2,
        leading=13, textColor=HexColor('#333355'),
    ))
    styles.add(ParagraphStyle(
        name='CorrectOption', parent=styles['Normal'],
        fontSize=9.5, leftIndent=18, spaceBefore=2, spaceAfter=2,
        leading=13, textColor=HexColor('#047857'), fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='AnswerLine', parent=styles['Normal'],
        fontSize=9, leftIndent=18, spaceBefore=2,
        textColor=HexColor('#047857'), fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='Explanation', parent=styles['Normal'],
        fontSize=8.5, leftIndent=18, spaceBefore=2, spaceAfter=6,
        textColor=HexColor('#6b7280'), leading=12,
    ))
    styles.add(ParagraphStyle(
        name='Marks', parent=styles['Normal'],
        fontSize=9, alignment=TA_RIGHT, textColor=HexColor('#9ca3af'),
    ))
    styles.add(ParagraphStyle(
        name='Instruction', parent=styles['Normal'],
        fontSize=9, leftIndent=12, spaceBefore=2, spaceAfter=2,
        textColor=HexColor('#4a4a6a'), leading=12,
    ))
    styles.add(ParagraphStyle(
        name='FooterText', parent=styles['Normal'],
        fontSize=8, textColor=HexColor('#9ca3af'), alignment=TA_CENTER,
    ))

    story = []

    # ── Header with Logo ────────────────────────────────────────────
    header_elements = []

    # Logo (if provided)
    logo_img = None
    if logo_base64:
        try:
            # Strip data URI prefix if present
            if ',' in logo_base64:
                logo_base64 = logo_base64.split(',', 1)[1]
            logo_data = base64.b64decode(logo_base64)
            logo_buf = io.BytesIO(logo_data)
            logo_img = RLImage(logo_buf, width=1.8 * cm, height=1.8 * cm)
            logo_img.hAlign = 'CENTER'
        except Exception as e:
            logger.warning(f"Failed to load logo: {e}")
            logo_img = None

    # Build header table: [Logo | Title Block | Date/Page Block]
    title_block = []
    title_block.append(Paragraph(f"<b>{exam_title}</b>", styles['SchoolName']))
    title_block.append(Paragraph(
        f"{board} Board | Class {class_grade} | {subject}",
        styles['ExamMeta'],
    ))

    today = datetime.now().strftime("%d/%m/%Y")
    total_marks = sum(q.get('marks', 1) for q in questions)

    info_block = []
    info_block.append(Paragraph(f"Date: {today}", styles['ExamMeta']))
    info_block.append(Paragraph(f"Total Marks: {total_marks}", styles['ExamMeta']))
    info_block.append(Paragraph(f"Total Questions: {len(questions)}", styles['ExamMeta']))

    if logo_img:
        header_data = [[logo_img, title_block, info_block]]
        header_table = Table(header_data, colWidths=[2.5 * cm, W - 6 * cm, 3.5 * cm])
    else:
        header_data = [[title_block, info_block]]
        header_table = Table(header_data, colWidths=[W - 4 * cm, 4 * cm])

    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)

    # Divider line
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=HexColor('#1a1a2e'), spaceAfter=8))

    # ── Instructions ────────────────────────────────────────────────
    story.append(Paragraph("<b>General Instructions:</b>", styles['SectionTitle']))
    instructions = [
        "All questions are compulsory.",
        "Read each question carefully before answering.",
        "For MCQs, select the <b>best answer</b> from the given choices.",
        f"Total marks: <b>{total_marks}</b>. Time allotted as per school schedule.",
    ]
    for inst in instructions:
        story.append(Paragraph(f"• {inst}", styles['Instruction']))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e5e7eb'), spaceAfter=6))

    # ── Questions ───────────────────────────────────────────────────
    option_labels = ["A", "B", "C", "D", "E", "F"]

    for q_idx, q in enumerate(questions, 1):
        text = _latex_to_paragraph(q.get('text', ''))
        marks = q.get('marks', 1)
        marks_label = f"[{marks} {'mark' if marks == 1 else 'marks'}]"

        # Question + marks in a table row
        q_para = Paragraph(f"<b>Q{q_idx}.</b> {text}", styles['QText'])
        m_para = Paragraph(marks_label, styles['Marks'])
        qt = Table([[q_para, m_para]], colWidths=[W * 0.88, W * 0.12])
        qt.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(qt)

        # Options
        options = q.get('options', [])
        correct_answer = q.get('correctAnswer', q.get('correct_answer', ''))

        for opt_idx, opt in enumerate(options):
            opt_text = _latex_to_paragraph(opt)
            letter = option_labels[opt_idx] if opt_idx < len(option_labels) else str(opt_idx + 1)

            # Check if this option is the correct one (for answer key)
            is_correct = False
            if include_answers and correct_answer:
                # Match by letter prefix or full text
                ca = correct_answer.strip()
                if ca.upper().startswith(letter):
                    is_correct = True
                elif opt.strip() == ca.strip():
                    is_correct = True

            style = styles['CorrectOption'] if is_correct else styles['Option']
            # Strip existing "A) " or "A." prefix if present
            opt_clean = re.sub(r'^[A-F][).\s]+\s*', '', opt_text).strip()
            prefix = f"<b>{letter})</b> " if is_correct else f"{letter}) "
            story.append(Paragraph(f"{prefix}{opt_clean}", style))

        # Answer inline (teacher copy)
        if include_answers and include_explanations:
            ans_text = _latex_to_paragraph(correct_answer)
            story.append(Paragraph(f"<b>Answer:</b> {ans_text}", styles['AnswerLine']))

        # Explanation (teacher copy)
        if include_explanations:
            explanation = _latex_to_paragraph(q.get('explanation', ''))
            if explanation:
                story.append(Paragraph(f"<b>Explanation:</b> {explanation}", styles['Explanation']))

        story.append(Spacer(1, 4))

    # ── Answer Key (separate page, for answer-key mode) ─────────────
    if include_answers and not include_explanations:
        story.append(PageBreak())
        story.append(Paragraph("<b>Answer Key</b>", styles['SchoolName']))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#1a1a2e'), spaceAfter=10))

        # Build answer grid (5 columns)
        ans_data = []
        row = []
        for q_idx, q in enumerate(questions, 1):
            correct = _latex_to_paragraph(q.get('correctAnswer', q.get('correct_answer', '')))
            cell = Paragraph(f"<b>Q{q_idx}.</b> {correct}", styles['QText'])
            row.append(cell)
            if len(row) == 5:
                ans_data.append(row)
                row = []
        if row:
            while len(row) < 5:
                row.append(Paragraph("", styles['QText']))
            ans_data.append(row)

        if ans_data:
            col_w = W / 5
            ans_table = Table(ans_data, colWidths=[col_w] * 5)
            ans_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e5e7eb')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(ans_table)

    # ── Footer ──────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e5e7eb'), spaceAfter=6))
    story.append(Paragraph(
        f"Generated by Test Engine · {board} {subject} Class {class_grade} · {today}",
        styles['FooterText'],
    ))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# DOCX Generation — Professional Format
# ═══════════════════════════════════════════════════════════════════════

def generate_docx(
    questions: List[dict],
    exam_title: str = "Test Paper",
    board: str = "CBSE",
    class_grade: str = "10",
    subject: str = "Science",
    include_answers: bool = False,
    include_explanations: bool = False,
    logo_base64: Optional[str] = None,
) -> bytes:
    """Generate a professional DOCX test paper with optional logo."""
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Cm(1.5)
        section.bottom_margin = Cm(1.5)
        section.left_margin = Cm(2)
        section.right_margin = Cm(2)

    # ── Logo ────────────────────────────────────────────────────────
    if logo_base64:
        try:
            if ',' in logo_base64:
                logo_base64 = logo_base64.split(',', 1)[1]
            logo_data = base64.b64decode(logo_base64)
            logo_buf = io.BytesIO(logo_data)
            logo_para = doc.add_paragraph()
            logo_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            logo_para.add_run().add_picture(logo_buf, width=Cm(2))
        except Exception as e:
            logger.warning(f"Failed to add logo to DOCX: {e}")

    # ── Title ───────────────────────────────────────────────────────
    title = doc.add_heading(exam_title or "Test Paper", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(f"{board} Board | Class {class_grade} | {subject}")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(74, 74, 106)

    total_marks = sum(q.get('marks', 1) for q in questions)
    today = datetime.now().strftime("%d/%m/%Y")

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run(f"Total Questions: {len(questions)} | Total Marks: {total_marks} | Date: {today}")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(107, 114, 128)

    doc.add_paragraph("━" * 50)

    # ── Instructions ────────────────────────────────────────────────
    doc.add_heading("General Instructions", level=2)
    instructions = [
        "All questions are compulsory.",
        "Read each question carefully before answering.",
        "For MCQs, select the best answer from the given choices.",
        f"Total marks: {total_marks}.",
    ]
    for inst in instructions:
        p = doc.add_paragraph(inst, style='List Bullet')
        p.paragraph_format.space_after = Pt(2)

    doc.add_paragraph("━" * 50)

    # ── Questions ───────────────────────────────────────────────────
    option_labels = ["A", "B", "C", "D", "E", "F"]

    for q_idx, q in enumerate(questions, 1):
        text = _latex_to_plain(q.get('text', ''))
        marks = q.get('marks', 1)

        # Question
        p = doc.add_paragraph()
        run_q = p.add_run(f"Q{q_idx}. ")
        run_q.bold = True
        run_q.font.size = Pt(11)
        run_t = p.add_run(text)
        run_t.font.size = Pt(11)
        run_m = p.add_run(f"  [{marks} {'mark' if marks == 1 else 'marks'}]")
        run_m.font.size = Pt(8)
        run_m.font.color.rgb = RGBColor(156, 163, 175)

        # Options
        options = q.get('options', [])
        correct_answer = q.get('correctAnswer', q.get('correct_answer', ''))

        for opt_idx, opt in enumerate(options):
            opt_clean = _latex_to_plain(opt)
            letter = option_labels[opt_idx] if opt_idx < len(option_labels) else str(opt_idx + 1)

            is_correct = False
            if include_answers and correct_answer:
                ca = correct_answer.strip()
                if ca.upper().startswith(letter):
                    is_correct = True

            op = doc.add_paragraph()
            op.paragraph_format.left_indent = Pt(24)
            op.paragraph_format.space_after = Pt(2)
            opt_stripped = re.sub(r'^[A-F][).\s]+\s*', '', opt_clean).strip()
            run = op.add_run(f"{letter}) {opt_stripped}")
            run.font.size = Pt(10)
            if is_correct:
                run.bold = True
                run.font.color.rgb = RGBColor(4, 120, 87)

        # Answer (teacher copy inline)
        if include_answers and include_explanations:
            correct = _latex_to_plain(correct_answer)
            ap = doc.add_paragraph()
            ap.paragraph_format.left_indent = Pt(24)
            run_a = ap.add_run("Answer: ")
            run_a.bold = True
            run_a.font.size = Pt(10)
            run_a.font.color.rgb = RGBColor(4, 120, 87)
            run_av = ap.add_run(correct)
            run_av.font.size = Pt(10)
            run_av.font.color.rgb = RGBColor(4, 120, 87)

        # Explanation
        if include_explanations:
            explanation = _latex_to_plain(q.get('explanation', ''))
            if explanation:
                ep = doc.add_paragraph()
                ep.paragraph_format.left_indent = Pt(24)
                run_e = ep.add_run("Explanation: ")
                run_e.bold = True
                run_e.font.size = Pt(8)
                run_e.font.color.rgb = RGBColor(107, 114, 128)
                run_ev = ep.add_run(explanation)
                run_ev.font.size = Pt(8)
                run_ev.font.color.rgb = RGBColor(107, 114, 128)

        doc.add_paragraph()

    # ── Answer Key (separate page) ──────────────────────────────────
    if include_answers and not include_explanations:
        doc.add_page_break()
        h = doc.add_heading("Answer Key", level=0)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for q_idx, q in enumerate(questions, 1):
            correct = _latex_to_plain(q.get('correctAnswer', q.get('correct_answer', '')))
            p = doc.add_paragraph()
            run = p.add_run(f"Q{q_idx}. ")
            run.bold = True
            p.add_run(correct)

    # ── Footer ──────────────────────────────────────────────────────
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run(f"Generated by Test Engine · {board} {subject} Class {class_grade} · {today}")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(156, 163, 175)

    # Save
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()