"""
Export Service v4 — Professional PDF and DOCX generation.

v4 fixes:
  - Unicode subscript ₂₃₆ → <sub> tags (fixes ■ squares in PDF)
  - Chemical formulas without $ delimiters handled (CO2 → CO<sub>2</sub>)
  - All v3 LaTeX fixes retained
"""

import io
import re
import base64
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Unicode → ReportLab tag conversion (fixes ■ in PDF)
# ═══════════════════════════════════════════════════════════════════════

UNICODE_SUBSCRIPTS = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '₊': '+', '₋': '-', '₌': '=',
    'ₐ': 'a', 'ₑ': 'e', 'ₒ': 'o', 'ₓ': 'x', 'ₙ': 'n',
}

UNICODE_SUPERSCRIPTS = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁼': '=',
    'ⁿ': 'n', 'ⁱ': 'i',
}

# Common chemical formulas — detect and fix subscripts even without $
CHEMICAL_PATTERN = re.compile(
    r'([A-Z][a-z]?)(\d+)',  # e.g., H2, O2, CO2, H2O, C6H12O6, Al2O3
)


def _fix_unicode_scripts(text: str, use_tags: bool = True) -> str:
    """Convert Unicode sub/superscript chars to ReportLab tags or plain text."""
    if not text:
        return text

    result = text

    # Convert consecutive Unicode subscripts to single <sub> tag
    # e.g., "H₁₂" → "H<sub>12</sub>"
    if use_tags:
        # Group consecutive subscripts
        sub_pattern = '([' + ''.join(re.escape(k) for k in UNICODE_SUBSCRIPTS.keys()) + ']+)'
        def sub_replacer(m):
            chars = m.group(1)
            converted = ''.join(UNICODE_SUBSCRIPTS.get(c, c) for c in chars)
            return f'<sub>{converted}</sub>'
        result = re.sub(sub_pattern, sub_replacer, result)

        # Group consecutive superscripts
        sup_pattern = '([' + ''.join(re.escape(k) for k in UNICODE_SUPERSCRIPTS.keys()) + ']+)'
        def sup_replacer(m):
            chars = m.group(1)
            converted = ''.join(UNICODE_SUPERSCRIPTS.get(c, c) for c in chars)
            return f'<super>{converted}</super>'
        result = re.sub(sup_pattern, sup_replacer, result)
    else:
        # Plain text — just convert to normal digits
        for uni, plain in UNICODE_SUBSCRIPTS.items():
            result = result.replace(uni, plain)
        for uni, plain in UNICODE_SUPERSCRIPTS.items():
            result = result.replace(uni, plain)

    return result


def _fix_chemical_formulas(text: str, use_tags: bool = True) -> str:
    """Convert plain chemical formulas like CO2, H2O to proper sub/superscript.
    Only applies to text NOT inside $ delimiters (LaTeX handles those).
    """
    if not text:
        return text

    # Don't touch text inside $ delimiters
    parts = re.split(r'(\$[^$]+\$)', text)
    result_parts = []

    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            # LaTeX — leave alone
            result_parts.append(part)
        else:
            # Plain text — fix chemical formulas
            if use_tags:
                fixed = CHEMICAL_PATTERN.sub(
                    lambda m: f'{m.group(1)}<sub>{m.group(2)}</sub>', part
                )
            else:
                fixed = part  # Plain text keeps numbers as-is
            result_parts.append(fixed)

    return ''.join(result_parts)


# ═══════════════════════════════════════════════════════════════════════
# LaTeX → Clean Text
# ═══════════════════════════════════════════════════════════════════════

SYMBOL_MAP = {
    r'\times': '×', r'\div': '÷', r'\pm': '±', r'\mp': '∓', r'\cdot': '·',
    r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
    r'\equiv': '≡', r'\sim': '~', r'\propto': '∝',
    r'\infty': '∞', r'\therefore': '∴', r'\because': '∵',
    r'\cup': '∪', r'\cap': '∩', r'\subset': '⊂', r'\supset': '⊃',
    r'\subseteq': '⊆', r'\supseteq': '⊇', r'\in': '∈', r'\notin': '∉',
    r'\emptyset': '∅', r'\forall': '∀', r'\exists': '∃',
    r'\rightarrow': '→', r'\leftarrow': '←', r'\Rightarrow': '⇒',
    r'\Leftarrow': '⇐', r'\leftrightarrow': '↔', r'\to': '→',
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
    r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
    r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
    r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
    r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Phi': 'Φ',
    r'\Psi': 'Ψ', r'\Omega': 'Ω',
    r'\degree': '°', r'\circ': '°', r'\nabla': '∇',
    r'\partial': '∂', r'\ell': 'ℓ', r'\hbar': 'ℏ',
    r'\sum': 'Σ', r'\prod': 'Π', r'\int': '∫',
    r'\left': '', r'\right': '',
    r'\bigl': '', r'\bigr': '',
    r'\Bigl': '', r'\Bigr': '',
    r'\langle': '⟨', r'\rangle': '⟩',
    r'\lfloor': '⌊', r'\rfloor': '⌋',
    r'\lceil': '⌈', r'\rceil': '⌉',
}

TRIG_FUNCS = {
    r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
    r'\cot': 'cot', r'\sec': 'sec', r'\csc': 'csc',
    r'\log': 'log', r'\ln': 'ln', r'\exp': 'exp',
    r'\lim': 'lim', r'\max': 'max', r'\min': 'min',
    r'\det': 'det', r'\gcd': 'gcd',
}


def _process_latex(text: str, use_tags: bool = False) -> str:
    """Convert LaTeX to formatted text."""
    if not text:
        return ""

    result = text

    # ── Step 0: Fix chemical formulas in plain text ─────────────────
    result = _fix_chemical_formulas(result, use_tags)

    # ── Step 1: Remove $ delimiters ─────────────────────────────────
    result = re.sub(r'\$([^$]+)\$', r'\1', result)

    # ── Step 2: Handle \mathbb, \text etc ───────────────────────────
    result = re.sub(r'\\mathbb\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\(?:text|mathrm|mathbf|textbf|textit|mathit)\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\(?:overline|underline|bar|hat|tilde|vec)\{([^}]*)\}', r'\1', result)

    # ── Step 3: Nested fractions ────────────────────────────────────
    for _ in range(3):
        result = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1/\2)', result)

    # ── Step 4: \sqrt ───────────────────────────────────────────────
    result = re.sub(r'\\sqrt\[([^]]*)\]\{([^}]*)\}', r'\1√(\2)', result)
    result = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', result)

    # ── Step 5: \binom ──────────────────────────────────────────────
    result = re.sub(r'\\binom\{([^}]*)\}\{([^}]*)\}', r'C(\1,\2)', result)

    # ── Step 6: Symbols ─────────────────────────────────────────────
    for latex, symbol in sorted(SYMBOL_MAP.items(), key=lambda x: -len(x[0])):
        result = result.replace(latex, symbol)
    for latex, func in sorted(TRIG_FUNCS.items(), key=lambda x: -len(x[0])):
        result = result.replace(latex, func)

    # ── Step 7: Superscripts and subscripts ─────────────────────────
    if use_tags:
        result = re.sub(r'\^\{([^}]*)\}', r'<super>\1</super>', result)
        result = re.sub(r'\^([a-zA-Z0-9°])', r'<super>\1</super>', result)
        result = re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', result)
        result = re.sub(r'_([a-zA-Z0-9])', r'<sub>\1</sub>', result)
    else:
        result = re.sub(r'\^\{([^}]*)\}', r'^\1', result)
        result = re.sub(r'\^([a-zA-Z0-9°])', r'^\1', result)
        result = re.sub(r'_\{([^}]*)\}', r'_\1', result)

    # ── Step 8: Remove remaining \commands ──────────────────────────
    result = re.sub(r'\\([a-zA-Z]+)', r'\1', result)

    # ── Step 9: Clean braces ────────────────────────────────────────
    result = result.replace('{', '').replace('}', '')

    # ── Step 10: Fix Unicode subscripts/superscripts ────────────────
    result = _fix_unicode_scripts(result, use_tags)

    # ── Step 11: Clean whitespace ───────────────────────────────────
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def _latex_to_paragraph(text: str) -> str:
    """Convert LaTeX to ReportLab Paragraph XML."""
    result = _process_latex(text, use_tags=True)

    # Escape XML but preserve our tags
    tags = {}
    for i, tag in enumerate(re.findall(r'</?(?:super|sub|b|i)>', result)):
        ph = f"__TAG{i}__"
        tags[ph] = tag
        result = result.replace(tag, ph, 1)

    result = result.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    for ph, tag in tags.items():
        result = result.replace(ph, tag)

    return result


def _latex_to_plain(text: str) -> str:
    """Convert LaTeX to plain text for DOCX."""
    return _process_latex(text, use_tags=False)


# ═══════════════════════════════════════════════════════════════════════
# PDF Generation
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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image as RLImage,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 4 * cm

    custom_styles = {
        'SchoolName': dict(parent=styles['Title'], fontSize=14, leading=18, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor('#1a1a2e'), fontName='Helvetica-Bold'),
        'ExamMeta': dict(parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=HexColor('#4a4a6a'), spaceAfter=4),
        'SectionTitle': dict(parent=styles['Heading2'], fontSize=11, spaceBefore=14, spaceAfter=6, textColor=HexColor('#1a1a2e'), fontName='Helvetica-Bold'),
        'QText': dict(parent=styles['Normal'], fontSize=10, spaceBefore=8, spaceAfter=3, leading=14, textColor=HexColor('#1f1f3a')),
        'Option': dict(parent=styles['Normal'], fontSize=9.5, leftIndent=18, spaceBefore=2, spaceAfter=2, leading=13, textColor=HexColor('#333355')),
        'CorrectOption': dict(parent=styles['Normal'], fontSize=9.5, leftIndent=18, spaceBefore=2, spaceAfter=2, leading=13, textColor=HexColor('#047857'), fontName='Helvetica-Bold'),
        'AnswerLine': dict(parent=styles['Normal'], fontSize=9, leftIndent=18, spaceBefore=2, textColor=HexColor('#047857'), fontName='Helvetica-Bold'),
        'Explanation': dict(parent=styles['Normal'], fontSize=8.5, leftIndent=18, spaceBefore=2, spaceAfter=6, textColor=HexColor('#6b7280'), leading=12),
        'Marks': dict(parent=styles['Normal'], fontSize=9, alignment=TA_RIGHT, textColor=HexColor('#9ca3af')),
        'Instruction': dict(parent=styles['Normal'], fontSize=9, leftIndent=12, spaceBefore=2, spaceAfter=2, textColor=HexColor('#4a4a6a'), leading=12),
        'FooterText': dict(parent=styles['Normal'], fontSize=8, textColor=HexColor('#9ca3af'), alignment=TA_CENTER),
    }
    for name, props in custom_styles.items():
        try:
            styles.add(ParagraphStyle(name=name, **props))
        except KeyError:
            pass

    story = []

    # Header
    logo_img = None
    if logo_base64:
        try:
            if ',' in logo_base64:
                logo_base64 = logo_base64.split(',', 1)[1]
            logo_img = RLImage(io.BytesIO(base64.b64decode(logo_base64)), width=1.8 * cm, height=1.8 * cm)
            logo_img.hAlign = 'CENTER'
        except Exception as e:
            logger.warning(f"Logo failed: {e}")

    title_block = [
        Paragraph(f"<b>{exam_title}</b>", styles['SchoolName']),
        Paragraph(f"{board} Board | Class {class_grade} | {subject}", styles['ExamMeta']),
    ]
    today = datetime.now().strftime("%d/%m/%Y")
    total_marks = sum(q.get('marks', 1) for q in questions)
    info_block = [
        Paragraph(f"Date: {today}", styles['ExamMeta']),
        Paragraph(f"Total Marks: {total_marks}", styles['ExamMeta']),
        Paragraph(f"Total Questions: {len(questions)}", styles['ExamMeta']),
    ]

    if logo_img:
        ht = Table([[logo_img, title_block, info_block]], colWidths=[2.5 * cm, W - 6 * cm, 3.5 * cm])
    else:
        ht = Table([[title_block, info_block]], colWidths=[W - 4 * cm, 4 * cm])
    ht.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    story.append(ht)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=HexColor('#1a1a2e'), spaceAfter=8))

    # Instructions
    story.append(Paragraph("<b>General Instructions:</b>", styles['SectionTitle']))
    for inst in [
        "All questions are compulsory.",
        "Read each question carefully before answering.",
        "For MCQs, select the <b>best answer</b> from the given choices.",
        f"Total marks: <b>{total_marks}</b>. Time allotted as per school schedule.",
    ]:
        story.append(Paragraph(f"• {inst}", styles['Instruction']))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e5e7eb'), spaceAfter=6))

    # Questions
    labels = ["A", "B", "C", "D", "E", "F"]

    for q_idx, q in enumerate(questions, 1):
        text = _latex_to_paragraph(q.get('text', ''))
        marks = q.get('marks', 1)
        marks_label = f"[{marks} {'mark' if marks == 1 else 'marks'}]"

        qt = Table(
            [[Paragraph(f"<b>Q{q_idx}.</b> {text}", styles['QText']), Paragraph(marks_label, styles['Marks'])]],
            colWidths=[W * 0.88, W * 0.12],
        )
        qt.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0)]))
        story.append(qt)

        options = q.get('options', [])
        correct_answer = q.get('correctAnswer', q.get('correct_answer', ''))

        for opt_idx, opt in enumerate(options):
            opt_text = _latex_to_paragraph(opt)
            letter = labels[opt_idx] if opt_idx < len(labels) else str(opt_idx + 1)

            is_correct = False
            if include_answers and correct_answer:
                ca = correct_answer.strip()
                if ca.upper().startswith(letter):
                    is_correct = True
                elif opt.strip() == ca.strip():
                    is_correct = True

            style = styles['CorrectOption'] if is_correct else styles['Option']
            opt_clean = re.sub(r'^[A-F][).\s]+\s*', '', opt_text).strip()
            prefix = f"<b>{letter})</b> " if is_correct else f"{letter}) "
            story.append(Paragraph(f"{prefix}{opt_clean}", style))

        if include_answers and include_explanations:
            ans = _latex_to_paragraph(correct_answer)
            story.append(Paragraph(f"<b>Answer:</b> {ans}", styles['AnswerLine']))

        if include_explanations:
            exp = _latex_to_paragraph(q.get('explanation', ''))
            if exp:
                story.append(Paragraph(f"<b>Explanation:</b> {exp}", styles['Explanation']))

        story.append(Spacer(1, 4))

    # Answer Key
    if include_answers and not include_explanations:
        story.append(PageBreak())
        story.append(Paragraph("<b>Answer Key</b>", styles['SchoolName']))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#1a1a2e'), spaceAfter=10))

        ans_data, row = [], []
        for q_idx, q in enumerate(questions, 1):
            correct = _latex_to_paragraph(q.get('correctAnswer', q.get('correct_answer', '')))
            row.append(Paragraph(f"<b>Q{q_idx}.</b> {correct}", styles['QText']))
            if len(row) == 5:
                ans_data.append(row)
                row = []
        if row:
            row.extend([Paragraph("", styles['QText'])] * (5 - len(row)))
            ans_data.append(row)

        if ans_data:
            at = Table(ans_data, colWidths=[W / 5] * 5)
            at.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e5e7eb')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(at)

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e5e7eb'), spaceAfter=6))
    story.append(Paragraph(f"Generated by Test Engine · {board} {subject} Class {class_grade} · {today}", styles['FooterText']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# DOCX Generation
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
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(1.5)
        section.bottom_margin = Cm(1.5)
        section.left_margin = Cm(2)
        section.right_margin = Cm(2)

    if logo_base64:
        try:
            if ',' in logo_base64:
                logo_base64 = logo_base64.split(',', 1)[1]
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(io.BytesIO(base64.b64decode(logo_base64)), width=Cm(2))
        except Exception as e:
            logger.warning(f"DOCX logo failed: {e}")

    title = doc.add_heading(exam_title or "Test Paper", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run(f"{board} Board | Class {class_grade} | {subject}")
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(74, 74, 106)

    total_marks = sum(q.get('marks', 1) for q in questions)
    today = datetime.now().strftime("%d/%m/%Y")

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = meta.add_run(f"Total Questions: {len(questions)} | Total Marks: {total_marks} | Date: {today}")
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(107, 114, 128)

    doc.add_paragraph("━" * 50)
    doc.add_heading("General Instructions", level=2)
    for inst in ["All questions are compulsory.", "Read each question carefully.", "For MCQs, select the best answer.", f"Total marks: {total_marks}."]:
        p = doc.add_paragraph(inst, style='List Bullet')
        p.paragraph_format.space_after = Pt(2)
    doc.add_paragraph("━" * 50)

    labels = ["A", "B", "C", "D", "E", "F"]

    for q_idx, q in enumerate(questions, 1):
        text = _latex_to_plain(q.get('text', ''))
        marks = q.get('marks', 1)

        p = doc.add_paragraph()
        rq = p.add_run(f"Q{q_idx}. ")
        rq.bold = True
        rq.font.size = Pt(11)
        rt = p.add_run(text)
        rt.font.size = Pt(11)
        rm = p.add_run(f"  [{marks} {'mark' if marks == 1 else 'marks'}]")
        rm.font.size = Pt(8)
        rm.font.color.rgb = RGBColor(156, 163, 175)

        options = q.get('options', [])
        correct_answer = q.get('correctAnswer', q.get('correct_answer', ''))

        for opt_idx, opt in enumerate(options):
            opt_clean = _latex_to_plain(opt)
            letter = labels[opt_idx] if opt_idx < len(labels) else str(opt_idx + 1)
            is_correct = include_answers and correct_answer and correct_answer.strip().upper().startswith(letter)

            op = doc.add_paragraph()
            op.paragraph_format.left_indent = Pt(24)
            op.paragraph_format.space_after = Pt(2)
            opt_stripped = re.sub(r'^[A-F][).\s]+\s*', '', opt_clean).strip()
            run = op.add_run(f"{letter}) {opt_stripped}")
            run.font.size = Pt(10)
            if is_correct:
                run.bold = True
                run.font.color.rgb = RGBColor(4, 120, 87)

        if include_answers and include_explanations:
            correct = _latex_to_plain(correct_answer)
            ap = doc.add_paragraph()
            ap.paragraph_format.left_indent = Pt(24)
            ra = ap.add_run("Answer: ")
            ra.bold = True
            ra.font.size = Pt(10)
            ra.font.color.rgb = RGBColor(4, 120, 87)
            rv = ap.add_run(correct)
            rv.font.size = Pt(10)
            rv.font.color.rgb = RGBColor(4, 120, 87)

        if include_explanations:
            exp = _latex_to_plain(q.get('explanation', ''))
            if exp:
                ep = doc.add_paragraph()
                ep.paragraph_format.left_indent = Pt(24)
                re2 = ep.add_run("Explanation: ")
                re2.bold = True
                re2.font.size = Pt(8)
                re2.font.color.rgb = RGBColor(107, 114, 128)
                rv2 = ep.add_run(exp)
                rv2.font.size = Pt(8)
                rv2.font.color.rgb = RGBColor(107, 114, 128)

        doc.add_paragraph()

    if include_answers and not include_explanations:
        doc.add_page_break()
        h = doc.add_heading("Answer Key", level=0)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for q_idx, q in enumerate(questions, 1):
            correct = _latex_to_plain(q.get('correctAnswer', q.get('correct_answer', '')))
            p = doc.add_paragraph()
            p.add_run(f"Q{q_idx}. ").bold = True
            p.add_run(correct)

    doc.add_paragraph()
    ft = doc.add_paragraph()
    ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = ft.add_run(f"Generated by Test Engine · {board} {subject} Class {class_grade} · {today}")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(156, 163, 175)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()