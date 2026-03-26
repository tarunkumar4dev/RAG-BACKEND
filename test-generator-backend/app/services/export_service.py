"""
Export Service v6 — Professional PDF and DOCX with CBSE Sections + Accountancy Tables.

v6 changes:
  - Accountancy table rendering (Journal Entry, Ledger, Trial Balance) in PDF
  - Accountancy table rendering in DOCX
  - answer_table field support in both PDF and DOCX
  - All v5 features retained (CBSE sections, Unicode, LaTeX cleanup)
"""

import io
import re
import base64
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CBSE Section Definitions (must match generation_service)
# ═══════════════════════════════════════════════════════════════════════

CBSE_SECTIONS_META = {
    "A": {"title": "Section A", "subtitle": "(1 mark each — MCQ / Assertion-Reason)", "marks": 1, "instruction": "All questions are compulsory. Each carries 1 mark."},
    "B": {"title": "Section B", "subtitle": "(2 marks each — Very Short Answer)", "marks": 2, "instruction": "All questions are compulsory. Each carries 2 marks."},
    "C": {"title": "Section C", "subtitle": "(3 marks each — Short Answer)", "marks": 3, "instruction": "All questions are compulsory. Each carries 3 marks."},
    "D": {"title": "Section D", "subtitle": "(5 marks each — Long Answer)", "marks": 5, "instruction": "All questions are compulsory. Each carries 5 marks."},
    "E": {"title": "Section E", "subtitle": "(4 marks each — Case Study Based)", "marks": 4, "instruction": "All questions are compulsory. Each carries 4 marks. Answer all sub-parts."},
}

SECTION_ORDER = ["A", "B", "C", "D", "E"]


# ═══════════════════════════════════════════════════════════════════════
# Unicode sub/super scripts (v4 compat)
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

CHEMICAL_PATTERN = re.compile(r'([A-Z][a-z]?)(\d+)')


def _fix_unicode_scripts(text: str, use_tags: bool = True) -> str:
    if not text:
        return text
    result = text
    if use_tags:
        sub_pattern = '([' + ''.join(re.escape(k) for k in UNICODE_SUBSCRIPTS.keys()) + ']+)'
        def sub_replacer(m):
            chars = m.group(1)
            converted = ''.join(UNICODE_SUBSCRIPTS.get(c, c) for c in chars)
            return f'<sub>{converted}</sub>'
        result = re.sub(sub_pattern, sub_replacer, result)

        sup_pattern = '([' + ''.join(re.escape(k) for k in UNICODE_SUPERSCRIPTS.keys()) + ']+)'
        def sup_replacer(m):
            chars = m.group(1)
            converted = ''.join(UNICODE_SUPERSCRIPTS.get(c, c) for c in chars)
            return f'<super>{converted}</super>'
        result = re.sub(sup_pattern, sup_replacer, result)
    else:
        for uni, plain in UNICODE_SUBSCRIPTS.items():
            result = result.replace(uni, plain)
        for uni, plain in UNICODE_SUPERSCRIPTS.items():
            result = result.replace(uni, plain)
    return result


def _fix_chemical_formulas(text: str, use_tags: bool = True) -> str:
    if not text:
        return text
    parts = re.split(r'(\$[^$]+\$)', text)
    result_parts = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            result_parts.append(part)
        else:
            if use_tags:
                fixed = CHEMICAL_PATTERN.sub(lambda m: f'{m.group(1)}<sub>{m.group(2)}</sub>', part)
            else:
                fixed = part
            result_parts.append(fixed)
    return ''.join(result_parts)


# ═══════════════════════════════════════════════════════════════════════
# LaTeX → Clean Text (improved v5 — handles Unicode output from v11 gen)
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
    r'\sigma': 'σ', r'\tau': 'τ', r'\phi': 'φ', r'\chi': 'χ',
    r'\psi': 'ψ', r'\omega': 'ω',
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Sigma': 'Σ', r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
    r'\degree': '°', r'\circ': '°', r'\nabla': '∇',
    r'\partial': '∂', r'\ell': 'ℓ',
    r'\sum': 'Σ', r'\prod': 'Π', r'\int': '∫',
    r'\left': '', r'\right': '',
    r'\bigl': '', r'\bigr': '',
    r'\langle': '⟨', r'\rangle': '⟩',
    r'\lfloor': '⌊', r'\rfloor': '⌋', r'\lceil': '⌈', r'\rceil': '⌉',
    r'\setminus': ' \\ ',
}

TRIG_FUNCS = {
    r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
    r'\cot': 'cot', r'\sec': 'sec', r'\csc': 'csc',
    r'\log': 'log', r'\ln': 'ln', r'\exp': 'exp',
    r'\lim': 'lim', r'\max': 'max', r'\min': 'min',
}

BARE_COMMANDS = {
    'setminus': ' \\ ', 'mathbb': '', 'mathrm': '', 'mathbf': '',
    'textbf': '', 'textit': '', 'overline': '', 'underline': '',
}


def _process_latex(text: str, use_tags: bool = False) -> str:
    if not text:
        return ""

    result = text
    result = result.replace('₹', 'Rs.')   # ← ADDED: Fix 1 - Replace Rupee symbol for ReportLab compatibility

    result = _fix_chemical_formulas(result, use_tags)
    result = re.sub(r'\$([^$]+)\$', r'\1', result)

    mathbb_map = {'R': 'ℝ', 'Z': 'ℤ', 'N': 'ℕ', 'Q': 'ℚ', 'C': 'ℂ'}
    for letter, symbol in mathbb_map.items():
        result = result.replace(f'\\mathbb{{{letter}}}', symbol)
        result = result.replace(f'mathbb{{{letter}}}', symbol)
        result = re.sub(rf'(?<![a-zA-Z])mathbb\s*{letter}(?![a-zA-Z])', symbol, result)

    result = re.sub(r'\\(?:text|mathrm|mathbf|textbf|textit|mathit)\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\(?:overline|underline|bar|hat|tilde|vec)\{([^}]*)\}', r'\1', result)

    for _ in range(3):
        result = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1/\2)', result)
    for _ in range(3):
        result = re.sub(r'(?<![a-zA-Z])frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1/\2)', result)

    result = re.sub(r'\\sqrt\[([^]]*)\]\{([^}]*)\}', r'\1√(\2)', result)
    result = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', result)

    result = re.sub(r'\\binom\{([^}]*)\}\{([^}]*)\}', r'C(\1,\2)', result)

    for latex, symbol in sorted(SYMBOL_MAP.items(), key=lambda x: -len(x[0])):
        result = result.replace(latex, symbol)
    for latex, func in sorted(TRIG_FUNCS.items(), key=lambda x: -len(x[0])):
        result = result.replace(latex, func)

    for cmd, replacement in BARE_COMMANDS.items():
        result = re.sub(rf'(?<![a-zA-Z\\]){cmd}\{{([^}}]*)\}}', r'\1', result)
        result = re.sub(rf'(?<![a-zA-Z\\]){cmd}(?![a-zA-Z])', replacement, result)

    if use_tags:
        result = re.sub(r'\^\{([^}]*)\}', r'<super>\1</super>', result)
        result = re.sub(r'\^([a-zA-Z0-9°])', r'<super>\1</super>', result)
        result = re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', result)
        result = re.sub(r'_([a-zA-Z0-9])', r'<sub>\1</sub>', result)
    else:
        result = re.sub(r'\^\{([^}]*)\}', r'^\1', result)
        result = re.sub(r'\^([a-zA-Z0-9°])', r'^\1', result)
        result = re.sub(r'_\{([^}]*)\}', r'_\1', result)

    result = re.sub(r'\\([a-zA-Z]+)\{([^}]*)\}', r'\2', result)
    result = re.sub(r'\\([a-zA-Z]+)', '', result)

    result = result.replace('{', '').replace('}', '')
    result = _fix_unicode_scripts(result, use_tags)
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def _latex_to_paragraph(text: str) -> str:
    result = _process_latex(text, use_tags=True)
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
    return _process_latex(text, use_tags=False)


# ═══════════════════════════════════════════════════════════════════════
# Markdown Table Parser (v7 — inline tables in question text)
# ═══════════════════════════════════════════════════════════════════════

def _parse_table_row(row_str: str) -> List[str]:
    """Parse a pipe-separated row string into clean cells."""
    cells = [c.strip() for c in row_str.split('|')]
    cells = [c for c in cells if c]
    cells = [re.sub(r'\*\*(.+?)\*\*', r'\1', c) for c in cells]
    cells = [re.sub(r'\*(.+?)\*', r'\1', c) for c in cells]
    return cells


def _parse_pipe_table_lines(lines: List[str]) -> tuple:
    """Parse pipe-table lines into (headers, data_rows)."""
    headers: List[str] = []
    rows: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip separator rows like |---|---|
        if re.match(r'^\|?[\s\-:]+\|[\s\-:|]*$', stripped):
            continue
        row = _parse_table_row(stripped)
        if not headers:
            headers = row
        elif row:
            rows.append(row)
    return headers, rows


def _split_text_and_tables(text: str) -> List[dict]:
    """
    Split question text into plain-text and markdown-table segments.
    Handles both newline-separated and collapsed inline formats.
    Returns list of {'type': 'text'|'table', 'content': str|(headers,rows)}
    """
    if not text:
        return [{'type': 'text', 'content': ''}]

    segments: List[dict] = []

    if '\n' in text:
        # ── Newline format (preferred — after generation service fix) ──
        lines = text.split('\n')
        i = 0
        current_text: List[str] = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            has_pipes = stripped.startswith('|') and stripped.count('|') >= 2
            next_is_sep = (
                i + 1 < len(lines)
                and re.match(r'^\s*\|?[\s\-:]+\|[\s\-:|]*$', lines[i + 1].strip())
            )

            if has_pipes and next_is_sep:
                if current_text:
                    segments.append({'type': 'text', 'content': ' '.join(current_text).strip()})
                    current_text = []
                # Collect all contiguous table lines
                table_lines = []
                while i < len(lines):
                    tl = lines[i].strip()
                    if tl.startswith('|') or re.match(r'^\|?[\s\-:]+\|', tl):
                        table_lines.append(tl)
                        i += 1
                    else:
                        break
                headers, rows = _parse_pipe_table_lines(table_lines)
                if headers:
                    segments.append({'type': 'table', 'content': (headers, rows)})
            else:
                if stripped:
                    current_text.append(stripped)
                i += 1

        if current_text:
            segments.append({'type': 'text', 'content': ' '.join(current_text).strip()})

    else:
        # ── Collapsed format: detect ---|--- separator ──
        sep_re = re.compile(r'\s*\|?\s*-{3,}(?:[\|\-\s:]*-{3,})+\s*\|?\s*')
        sep_m = sep_re.search(text)

        if not sep_m:
            return [{'type': 'text', 'content': text}]

        pre_sep = text[:sep_m.start()]
        post_sep = text[sep_m.end():]

        # Header row = last |..| block in pre_sep
        hdr_m = re.search(r'((?:\|[^|]+)+\|)\s*$', pre_sep)
        if not hdr_m:
            return [{'type': 'text', 'content': text}]

        pre_table = pre_sep[:hdr_m.start()].strip()
        headers = _parse_table_row(hdr_m.group(1))

        # Data rows from post_sep
        rows: List[List[str]] = []
        last_end = 0
        for m in re.finditer(r'((?:\|[^|]+)+\|)', post_sep):
            row = _parse_table_row(m.group(1))
            if row:
                rows.append(row)
            last_end = m.end()

        post_table = post_sep[last_end:].strip()

        if pre_table:
            segments.append({'type': 'text', 'content': pre_table})
        if headers and rows:
            segments.append({'type': 'table', 'content': (headers, rows)})
        if post_table:
            segments.append({'type': 'text', 'content': post_table})

    return segments if segments else [{'type': 'text', 'content': text}]


def _render_inline_table_pdf(headers: List[str], rows: List[List[str]], styles, W: float) -> list:
    """Render a markdown-parsed inline table as a ReportLab Table."""
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

    if not headers or not rows:
        return []

    num_cols = len(headers)
    cell_style = styles.get('Option', styles['Normal'])

    # Header row
    table_data = [[Paragraph(f"<b>{h}</b>", cell_style) for h in headers]]

    for row in rows:
        padded = (row + [''] * num_cols)[:num_cols]
        table_data.append([
            Paragraph(_latex_to_paragraph(str(c)), cell_style)
            for c in padded
        ])

    # Column widths
    if num_cols == 2:
        col_widths = [W * 0.60, W * 0.40]
    elif num_cols == 3:
        col_widths = [W * 0.50, W * 0.25, W * 0.25]
    elif num_cols == 4:
        col_widths = [W * 0.40, W * 0.20, W * 0.20, W * 0.20]
    elif num_cols == 5:
        col_widths = [W * 0.10, W * 0.42, W * 0.08, W * 0.20, W * 0.20]
    else:
        col_widths = [W / num_cols] * num_cols

    col_widths = col_widths[:num_cols]

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('GRID',          (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
        ('BACKGROUND',    (0, 0), (-1, 0),  HexColor('#f3f4f6')),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  HexColor('#1f2937')),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
        ('ALIGN',         (1, 0), (-1, -1), 'RIGHT'),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 6)]


# ═══════════════════════════════════════════════════════════════════════
# Helper: Group questions by section
# ═══════════════════════════════════════════════════════════════════════

def _group_by_section(questions: List[dict]) -> dict:
    groups = {}
    for q in questions:
        sec = q.get('section') or q.get('_section') or 'NONE'
        if sec not in groups:
            groups[sec] = []
        groups[sec].append(q)
    return groups


def _has_sections(questions: List[dict]) -> bool:
    for q in questions:
        sec = q.get('section') or q.get('_section')
        if sec and sec in CBSE_SECTIONS_META:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# NEW v6: Accountancy Table Rendering — PDF
# ═══════════════════════════════════════════════════════════════════════

def _render_answer_table_pdf(answer_table, styles, W):
    """
    Render an Accountancy answer table (Journal Entry / Ledger / Trial Balance)
    as a ReportLab Table with proper formatting.

    Returns a list of story elements.
    """
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

    elements = []
    if not answer_table or not isinstance(answer_table, dict):
        return elements

    table_type = answer_table.get("type", "")
    headers = answer_table.get("headers", [])
    rows = answer_table.get("rows", [])
    total_row = answer_table.get("total_row")

    if not headers or not rows:
        return elements

    num_cols = len(headers)

    # Table title
    title_map = {
        "journal_entry": "Journal Entry",
        "ledger": "Ledger Account",
        "trial_balance": "Trial Balance",
    }
    title = title_map.get(table_type, "Answer Table")
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"<b>{title}:</b>",
        styles.get('AnswerLine', styles['Normal'])
    ))
    elements.append(Spacer(1, 4))

    # Build table data — header row
    header_style = styles.get('Option', styles['Normal'])
    table_data = [[Paragraph(f"<b>{h}</b>", header_style) for h in headers]]

    # Data rows
    for row in rows:
        if not isinstance(row, list):
            continue
        padded = (row + [""] * num_cols)[:num_cols]
        table_data.append([
            Paragraph(_latex_to_paragraph(str(cell)), styles.get('Option', styles['Normal']))
            for cell in padded
        ])

    # Total row
    if total_row and isinstance(total_row, list):
        padded_total = (total_row + [""] * num_cols)[:num_cols]
        table_data.append([
            Paragraph(f"<b>{_latex_to_paragraph(str(cell))}</b>", styles.get('Option', styles['Normal']))
            for cell in padded_total
        ])

    # Column widths based on table type
    if table_type == "journal_entry":
        col_widths = [W * 0.13, W * 0.40, W * 0.07, W * 0.20, W * 0.20]
    elif table_type == "ledger":
        col_w = W / 8
        col_widths = [col_w] * num_cols
    elif table_type == "trial_balance":
        col_widths = [W * 0.08, W * 0.42, W * 0.10, W * 0.20, W * 0.20]
    else:
        col_widths = [W / num_cols] * num_cols

    col_widths = col_widths[:num_cols]

    # Create table
    t = Table(table_data, colWidths=col_widths, repeatRows=1)

    # Styling
    style_cmds = [
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1f2937')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]

    # Right-align amount columns
    if table_type == "journal_entry":
        style_cmds.append(('ALIGN', (3, 0), (4, -1), 'RIGHT'))
    elif table_type == "trial_balance":
        style_cmds.append(('ALIGN', (3, 0), (4, -1), 'RIGHT'))
    elif table_type == "ledger":
        if num_cols >= 8:
            style_cmds.append(('ALIGN', (3, 0), (3, -1), 'RIGHT'))
            style_cmds.append(('ALIGN', (7, 0), (7, -1), 'RIGHT'))
            style_cmds.append(('LINEAFTER', (3, 0), (3, -1), 1.5, HexColor('#374151')))

    # Total row styling
    if total_row:
        last_row = len(table_data) - 1
        style_cmds.extend([
            ('BACKGROUND', (0, last_row), (-1, last_row), HexColor('#e5e7eb')),
            ('LINEABOVE', (0, last_row), (-1, last_row), 1.5, HexColor('#374151')),
        ])

    t.setStyle(TableStyle(style_cmds))
    elements.append(t)
    elements.append(Spacer(1, 6))

    return elements


# ═══════════════════════════════════════════════════════════════════════
# NEW v6: Accountancy Table Rendering — DOCX
# ═══════════════════════════════════════════════════════════════════════

def _render_answer_table_docx(doc, answer_table):
    """
    Render an Accountancy answer table in a DOCX document.
    Uses python-docx Table with proper formatting.
    """
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    if not answer_table or not isinstance(answer_table, dict):
        return

    table_type = answer_table.get("type", "")
    headers = answer_table.get("headers", [])
    rows = answer_table.get("rows", [])
    total_row = answer_table.get("total_row")

    if not headers or not rows:
        return

    num_cols = len(headers)

    # Title
    title_map = {
        "journal_entry": "Journal Entry",
        "ledger": "Ledger Account",
        "trial_balance": "Trial Balance",
    }
    title = title_map.get(table_type, "Answer Table")
    tp = doc.add_paragraph()
    tr = tp.add_run(f"{title}:")
    tr.bold = True
    tr.font.size = Pt(10)
    tr.font.color.rgb = RGBColor(4, 120, 87)

    # Calculate total rows
    total_data_rows = 1 + len(rows) + (1 if total_row else 0)

    # Create table
    table = doc.add_table(rows=total_data_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'

    # Header row
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(header)
        r.bold = True
        r.font.size = Pt(9)
        # Header background - light gray
        shading = cell._element.get_or_add_tcPr()
        shd = shading.makeelement(qn('w:shd'), {
            qn('w:fill'): 'F3F4F6',
            qn('w:val'): 'clear',
        })
        shading.append(shd)

    # Data rows
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        padded = (row + [""] * num_cols)[:num_cols]
        for j, cell_text in enumerate(padded):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(cell_text))
            r.font.size = Pt(9)

            # Right-align amount columns
            if table_type in ("journal_entry", "trial_balance") and j >= num_cols - 2:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif table_type == "ledger" and num_cols >= 8 and j in (3, 7):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # Total row
    if total_row and isinstance(total_row, list):
        row_idx = 1 + len(rows)
        padded_total = (total_row + [""] * num_cols)[:num_cols]
        for j, cell_text in enumerate(padded_total):
            cell = table.rows[row_idx].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(cell_text))
            r.bold = True
            r.font.size = Pt(9)

            # Right-align amounts
            if table_type in ("journal_entry", "trial_balance") and j >= num_cols - 2:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

            # Gray background for total row
            shading = cell._element.get_or_add_tcPr()
            shd = shading.makeelement(qn('w:shd'), {
                qn('w:fill'): 'E5E7EB',
                qn('w:val'): 'clear',
            })
            shading.append(shd)

    doc.add_paragraph()  # spacing after table


# ═══════════════════════════════════════════════════════════════════════
# Markdown Table Rendering — DOCX
# ═══════════════════════════════════════════════════════════════════════

def _render_inline_table_docx(doc, headers: List[str], rows: List[List[str]]):
    """Render a markdown-parsed inline table inside a DOCX document."""
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT

    if not headers or not rows:
        return

    num_cols = len(headers)
    total_rows = 1 + len(rows)

    table = doc.add_table(rows=total_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'

    # Header
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)

    # Data rows
    for i, row in enumerate(rows):
        padded = (row + [''] * num_cols)[:num_cols]
        for j, val in enumerate(padded):
            cell = table.rows[i + 1].cells[j]
            cell.text = ''
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(9)
            if j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    doc.add_paragraph()


# ═══════════════════════════════════════════════════════════════════════
# PDF Generation (v7 — with inline markdown table support)
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
        PageBreak, HRFlowable, Image as RLImage, KeepTogether,
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
        'SectionHeader': dict(parent=styles['Heading1'], fontSize=12, spaceBefore=18, spaceAfter=4, textColor=HexColor('#1a1a2e'), fontName='Helvetica-Bold', alignment=TA_CENTER),
        'SectionSub': dict(parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=HexColor('#6b7280'), spaceAfter=2),
        'SectionInstruction': dict(parent=styles['Normal'], fontSize=8.5, alignment=TA_CENTER, textColor=HexColor('#9ca3af'), spaceAfter=8, leading=11),
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

    # ── Header ──
    logo_img = None
    if logo_base64:
        try:
            if ',' in logo_base64:
                logo_base64 = logo_base64.split(',', 1)[1]
            logo_img = RLImage(io.BytesIO(base64.b64decode(logo_base64)), width=1.8 * cm, height=1.8 * cm)
            logo_img.hAlign = 'CENTER'
        except Exception as e:
            logger.warning(f"Logo failed: {e}")

    today = datetime.now().strftime("%d/%m/%Y")
    total_marks = sum(q.get('marks', 1) for q in questions)

    title_block = [
        Paragraph(f"<b>{exam_title}</b>", styles['SchoolName']),
        Paragraph(f"{board} Board | Class {class_grade} | {subject}", styles['ExamMeta']),
    ]
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

    # ── General Instructions ──
    has_sec = _has_sections(questions)

    story.append(Paragraph("<b>General Instructions:</b>", styles['SectionTitle']))

    base_instructions = [
        "All questions are compulsory.",
        "Read each question carefully before answering.",
    ]

    if has_sec:
        base_instructions.extend([
            f"This question paper has <b>5 Sections</b> — A, B, C, D, and E.",
            f"<b>Section A</b> has 20 MCQs / Assertion-Reason (1 mark each).",
            f"<b>Section B</b> has 5 Very Short Answer questions (2 marks each).",
            f"<b>Section C</b> has 6 Short Answer questions (3 marks each).",
            f"<b>Section D</b> has 4 Long Answer questions (5 marks each).",
            f"<b>Section E</b> has 3 Case Study questions (4 marks each).",
        ])
    else:
        base_instructions.extend([
            "For MCQs, select the <b>best answer</b> from the given choices.",
        ])

    base_instructions.append(f"Total marks: <b>{total_marks}</b>. Time allotted as per school schedule.")

    for inst in base_instructions:
        story.append(Paragraph(f"• {inst}", styles['Instruction']))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e5e7eb'), spaceAfter=6))

    # ── Questions ──
    labels = ["A", "B", "C", "D", "E", "F"]
    q_num = 0

    def _render_question(q, q_num):
        """Render a single question — handles inline markdown tables in question text."""
        from reportlab.platypus import Table as RLTable, TableStyle as RLTableStyle
        elements = []
        raw_text = q.get('text', '')
        marks = q.get('marks', 1)
        marks_label = f"[{marks} {'mark' if marks == 1 else 'marks'}]"

        # Split question text into text/table segments
        segments = _split_text_and_tables(raw_text)

        # First text segment → Q number header
        first_text = ''
        for seg in segments:
            if seg['type'] == 'text' and seg['content']:
                first_text = _latex_to_paragraph(seg['content'])
                break
        if not first_text:
            first_text = _latex_to_paragraph(raw_text)

        qt = RLTable(
            [[Paragraph(f"<b>Q{q_num}.</b> {first_text}", styles['QText']),
              Paragraph(marks_label, styles['Marks'])]],
            colWidths=[W * 0.88, W * 0.12],
        )
        qt.setStyle(RLTableStyle([
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(qt)

        # Remaining segments (inline tables + continuation text)
        first_text_skipped = False
        for seg in segments:
            if seg['type'] == 'text':
                if not first_text_skipped:
                    first_text_skipped = True
                    continue  # already rendered above
                content = _latex_to_paragraph(seg['content'])
                if content:
                    elements.append(Paragraph(content, styles['QText']))
            elif seg['type'] == 'table':
                hdrs, rws = seg['content']
                elements.extend(_render_inline_table_pdf(hdrs, rws, styles, W))

        # Options (MCQ)
        options = q.get('options', [])
        correct_answer = q.get('correctAnswer', q.get('correct_answer', ''))

        if options:
            for opt_idx, opt in enumerate(options):
                opt_text = _latex_to_paragraph(opt)
                letter = labels[opt_idx] if opt_idx < len(labels) else str(opt_idx + 1)
                is_correct = False
                if include_answers and correct_answer:
                    ca = correct_answer.strip()
                    if ca.upper().startswith(letter) or opt.strip() == ca.strip():
                        is_correct = True
                style = styles['CorrectOption'] if is_correct else styles['Option']
                opt_clean = re.sub(r'^[A-F][).\s]+\s*', '', opt_text).strip()
                prefix = f"<b>{letter})</b> " if is_correct else f"{letter}) "
                elements.append(Paragraph(f"{prefix}{opt_clean}", style))
        else:
            fmt = q.get('format', 'mcq')
            if not include_answers:
                if fmt == 'short_answer':
                    elements.append(Spacer(1, 24))
                elif fmt == 'long_answer':
                    elements.append(Spacer(1, 60))
                elif fmt in ('journal_entry', 'ledger', 'trial_balance'):
                    elements.append(Spacer(1, 80))

        # Answer + explanation
        if include_answers and include_explanations:
            raw_table = q.get('answer_table') or q.get('answerTable')
            if raw_table and isinstance(raw_table, dict):
                elements.extend(_render_answer_table_pdf(raw_table, styles, W))
            else:
                ans = _latex_to_paragraph(correct_answer)
                elements.append(Paragraph(f"<b>Answer:</b> {ans}", styles['AnswerLine']))

        if include_explanations:
            exp = _latex_to_paragraph(q.get('explanation', ''))
            if exp:
                elements.append(Paragraph(f"<b>Explanation:</b> {exp}", styles['Explanation']))

        elements.append(Spacer(1, 4))
        return elements

    if has_sec:
        grouped = _group_by_section(questions)

        for sec_key in SECTION_ORDER:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue

            meta = CBSE_SECTIONS_META.get(sec_key, {})

            story.append(HRFlowable(width="60%", thickness=1, color=HexColor('#1a1a2e'), spaceBefore=12, spaceAfter=4))
            story.append(Paragraph(f"<b>{meta.get('title', sec_key)}</b>", styles['SectionHeader']))
            story.append(Paragraph(meta.get('subtitle', ''), styles['SectionSub']))
            story.append(Paragraph(meta.get('instruction', ''), styles['SectionInstruction']))
            story.append(HRFlowable(width="40%", thickness=0.5, color=HexColor('#e5e7eb'), spaceAfter=6))

            for q in sec_qs:
                q_num += 1
                elements = _render_question(q, q_num)
                story.extend(elements)
    else:
        for q in questions:
            q_num += 1
            elements = _render_question(q, q_num)
            story.extend(elements)

    # ── Answer Key ──
    if include_answers and not include_explanations:
        story.append(PageBreak())
        story.append(Paragraph("<b>Answer Key</b>", styles['SchoolName']))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#1a1a2e'), spaceAfter=10))

        q_num_ak = 0
        all_qs_ordered = []
        if has_sec:
            grouped = _group_by_section(questions)
            for sec_key in SECTION_ORDER:
                all_qs_ordered.extend(grouped.get(sec_key, []))
        else:
            all_qs_ordered = questions

        # v6: For table-based questions, render tables in answer key too
        for q in all_qs_ordered:
            q_num_ak += 1
            raw_table = q.get('answer_table') or q.get('answerTable')
            if raw_table and isinstance(raw_table, dict):
                # Table answer — render full table in answer key
                story.append(Paragraph(f"<b>Q{q_num_ak}.</b>", styles['QText']))
                table_elements = _render_answer_table_pdf(raw_table, styles, W)
                story.extend(table_elements)
                story.append(Spacer(1, 4))
            else:
                # Text answer — compact grid (existing logic)
                correct = _latex_to_paragraph(q.get('correctAnswer', q.get('correct_answer', '')))
                story.append(Paragraph(f"<b>Q{q_num_ak}.</b> {correct}", styles['QText']))
                story.append(Spacer(1, 2))

    # ── Footer ──
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e5e7eb'), spaceAfter=6))
    story.append(Paragraph(f"Generated by A4AI Test Engine · {board} {subject} Class {class_grade} · {today}", styles['FooterText']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# DOCX Generation (v7 — with inline markdown table support)
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

    # Instructions
    has_sec = _has_sections(questions)
    doc.add_heading("General Instructions", level=2)

    instructions = ["All questions are compulsory.", "Read each question carefully."]
    if has_sec:
        instructions.extend([
            "This paper has 5 Sections — A, B, C, D, and E.",
            "Section A: 20 questions × 1 mark (MCQ / Assertion-Reason)",
            "Section B: 5 questions × 2 marks (Very Short Answer)",
            "Section C: 6 questions × 3 marks (Short Answer)",
            "Section D: 4 questions × 5 marks (Long Answer)",
            "Section E: 3 questions × 4 marks (Case Study Based)",
        ])
    else:
        instructions.append("For MCQs, select the best answer.")
    instructions.append(f"Total marks: {total_marks}.")

    for inst in instructions:
        p = doc.add_paragraph(inst, style='List Bullet')
        p.paragraph_format.space_after = Pt(2)
    doc.add_paragraph("━" * 50)

    labels = ["A", "B", "C", "D", "E", "F"]
    q_num = 0

    def _render_q_docx(q, q_num):
        raw_text = q.get('text', '')
        marks = q.get('marks', 1)

        # Q number + first text segment
        segments = _split_text_and_tables(raw_text)
        first_text = ''
        for seg in segments:
            if seg['type'] == 'text' and seg['content']:
                first_text = _latex_to_plain(seg['content'])
                break
        if not first_text:
            first_text = _latex_to_plain(raw_text)

        p = doc.add_paragraph()
        rq = p.add_run(f"Q{q_num}. ")
        rq.bold = True
        rq.font.size = Pt(11)
        rt = p.add_run(first_text)
        rt.font.size = Pt(11)
        rm = p.add_run(f"  [{marks} {'mark' if marks == 1 else 'marks'}]")
        rm.font.size = Pt(8)
        rm.font.color.rgb = RGBColor(156, 163, 175)

        # Remaining segments
        first_text_skipped = False
        for seg in segments:
            if seg['type'] == 'text':
                if not first_text_skipped:
                    first_text_skipped = True
                    continue
                content = _latex_to_plain(seg['content'])
                if content:
                    cp = doc.add_paragraph()
                    cp.add_run(content).font.size = Pt(11)
            elif seg['type'] == 'table':
                hdrs, rws = seg['content']
                _render_inline_table_docx(doc, hdrs, rws)

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

        # v6: Answer with table support
        if include_answers and include_explanations:
            raw_table = q.get('answer_table') or q.get('answerTable')
            if raw_table and isinstance(raw_table, dict):
                _render_answer_table_docx(doc, raw_table)
            else:
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

    if has_sec:
        grouped = _group_by_section(questions)
        for sec_key in SECTION_ORDER:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue

            sec_meta = CBSE_SECTIONS_META.get(sec_key, {})

            doc.add_paragraph("━" * 50)
            h = doc.add_heading(f"{sec_meta.get('title', sec_key)} {sec_meta.get('subtitle', '')}", level=1)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER

            inst_p = doc.add_paragraph()
            inst_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            inst_r = inst_p.add_run(sec_meta.get('instruction', ''))
            inst_r.font.size = Pt(9)
            inst_r.font.color.rgb = RGBColor(107, 114, 128)
            inst_r.italic = True

            for q in sec_qs:
                q_num += 1
                _render_q_docx(q, q_num)
    else:
        for q in questions:
            q_num += 1
            _render_q_docx(q, q_num)

    # Answer Key
    if include_answers and not include_explanations:
        doc.add_page_break()
        h = doc.add_heading("Answer Key", level=0)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER

        q_num_ak = 0
        all_qs_ordered = []
        if has_sec:
            grouped = _group_by_section(questions)
            for sec_key in SECTION_ORDER:
                all_qs_ordered.extend(grouped.get(sec_key, []))
        else:
            all_qs_ordered = questions

        for q in all_qs_ordered:
            q_num_ak += 1
            # v6: Table answers in answer key
            raw_table = q.get('answer_table') or q.get('answerTable')
            if raw_table and isinstance(raw_table, dict):
                p = doc.add_paragraph()
                p.add_run(f"Q{q_num_ak}. ").bold = True
                _render_answer_table_docx(doc, raw_table)
            else:
                correct = _latex_to_plain(q.get('correctAnswer', q.get('correct_answer', '')))
                p = doc.add_paragraph()
                p.add_run(f"Q{q_num_ak}. ").bold = True
                p.add_run(correct)

    doc.add_paragraph()
    ft = doc.add_paragraph()
    ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = ft.add_run(f"Generated by A4AI Test Engine · {board} {subject} Class {class_grade} · {today}")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(156, 163, 175)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()