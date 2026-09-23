"""
Export Service v14 — Multi-Template Support (Classic / Modern / Compact / Colorful)

v14 changes:
  - "colorful" template now uses a distinct INSTITUTE-PAPER layout
    (layout_style="institute_paper") matching a reference institute exam
    format: Institute name header, "CLASS X — SUBJECT" line, optional
    Topic line, Teacher / Subject + Max Marks / Date / Time meta row, a
    signature multi-color section rule, inline "[marks]" per question (no
    separate column), MCQ options rendered as (a)/(b) two-column pairs,
    plain "SECTION A (desc)" headings, and a "— All the Best —" footer.
  - generate_pdf() / generate_docx() gain new optional params:
    teacher_name, institute_name, duration, topic — only rendered by the
    institute_paper layout; other templates ignore them (still accepted,
    so callers never break).
  - modern / classic / compact keep the original card-based layout
    (layout_style="card_based") — completely unaffected.

v13 features retained:
  - TEMPLATE_PRESETS: 4 selectable visual templates for PDF + DOCX export
  - get_available_templates() metadata endpoint
  - Semantic colors (Accountancy emerald, Statistics blue) are NOT
    template-driven — they carry meaning, not brand styling.

v12 / v11 features retained:
  - Matrix bracket rendering, question tables (Statistics), Accountancy
    answer tables, CBSE section grouping, manual question images, LaTeX/
    Unicode cleanup, paper date support, card-style question boxes.
"""

import io
import re
import base64
import logging
from typing import List, Optional
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CBSE Section Definitions
# ═══════════════════════════════════════════════════════════════════════

CBSE_SECTIONS_META = {
    "A": {"title": "Section A", "subtitle": "(1 mark each — MCQ / Assertion-Reason)", "marks": 1, "instruction": "All questions are compulsory. Each carries 1 mark."},
    "B": {"title": "Section B", "subtitle": "(2 marks each — Very Short Answer)", "marks": 2, "instruction": "All questions are compulsory. Each carries 2 marks."},
    "C": {"title": "Section C", "subtitle": "(3 marks each — Short Answer)", "marks": 3, "instruction": "All questions are compulsory. Each carries 3 marks."},
    "D": {"title": "Section D", "subtitle": "(5 marks each — Long Answer)", "marks": 5, "instruction": "All questions are compulsory. Each carries 5 marks."},
    "E": {"title": "Section E", "subtitle": "(4 marks each — Case Study Based)", "marks": 4, "instruction": "All questions are compulsory. Each carries 4 marks. Answer all sub-parts."},
}

ACCOUNTANCY_SECTIONS_META = {
    "A_1m":  {"title": "Part A", "subtitle": "(1 mark each — MCQ / Assertion-Reason)", "marks": 1, "instruction": "Questions carry 1 mark each."},
    "A_3m":  {"title": "Part A", "subtitle": "(3 marks each)", "marks": 3, "instruction": "Questions carry 3 marks each."},
    "A_4m":  {"title": "Part A", "subtitle": "(4 marks each)", "marks": 4, "instruction": "Questions carry 4 marks each."},
    "A_6m":  {"title": "Part A", "subtitle": "(6 marks each)", "marks": 6, "instruction": "Questions carry 6 marks each."},
    "B1_1m": {"title": "Part B (Option I)", "subtitle": "Analysis of Financial Statements — (1 mark each)", "marks": 1, "instruction": "Questions carry 1 mark each."},
    "B1_3m": {"title": "Part B (Option I)", "subtitle": "Analysis of Financial Statements — (3 marks each)", "marks": 3, "instruction": "Questions carry 3 marks each."},
    "B1_4m": {"title": "Part B (Option I)", "subtitle": "Analysis of Financial Statements — (4 marks each)", "marks": 4, "instruction": "Questions carry 4 marks each."},
    "B1_6m": {"title": "Part B (Option I)", "subtitle": "Analysis of Financial Statements — (6 marks each)", "marks": 6, "instruction": "Questions carry 6 marks each."},
}

SECTION_ORDER = ["A", "B", "C", "D", "E"]
ACCOUNTANCY_SECTION_ORDER = ["A_1m", "A_3m", "A_4m", "A_6m", "B1_1m", "B1_3m", "B1_4m", "B1_6m"]


# ═══════════════════════════════════════════════════════════════════════
# Template Presets  (v14: each preset now carries a "layout_style")
# ═══════════════════════════════════════════════════════════════════════
#
# layout_style:
#   "card_based"       → modern / classic / compact — existing renderer
#   "institute_paper"  → colorful — institute exam-paper renderer (v14)
#
# card_style:
#   "card"   → rounded/bordered box around each question
#   "flat"   → no box, just spacing
#   "stripe" → bordered box with a colored left accent stripe per section

TEMPLATE_PRESETS = {
    "modern": {
        "label": "Modern",
        "description": "CBSE exam paper with signature coral-orange section bars and clean layout.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#FF7043",
        "accent_light": "#FFF3E0",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#E5E7EB",
        "card_bg": "#FFFFFF",
        "card_border": "#E5E7EB",
        "correct": "#047857",
        "table_header_bg": "#F3F4F6",
        "table_header_text": "#1F2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#FF7043", "B": "#FF7043", "C": "#FF7043",
            "D": "#FF7043", "E": "#FF7043", "F": "#FF7043",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
    "classic": {
        "label": "Classic",
        "description": "Traditional serif exam-paper look — formal, no boxes.",
        "layout_style": "card_based",
        "font_body": "Times-Roman",
        "font_bold": "Times-Bold",
        "docx_font": "Times New Roman",
        "primary": "#1a1a2e",
        "secondary": "#3f3f3f",
        "muted": "#595959",
        "light_muted": "#7a7a7a",
        "border": "#cfcfcf",
        "card_bg": "#FFFFFF",
        "card_border": "#cfcfcf",
        "correct": "#1e5631",
        "table_header_bg": "#f0f0f0",
        "table_header_text": "#1a1a1a",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": None,
        "spacing_scale": 1.0,
        "margins_cm": (1.2, 1.2, 1.8, 1.8),
    },
    "compact": {
        "label": "Compact",
        "description": "Dense layout, smaller fonts — fits more on fewer pages.",
        "layout_style": "card_based",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Arial",
        "primary": "#1a1a2e",
        "secondary": "#4a4a6a",
        "muted": "#6b7280",
        "light_muted": "#9ca3af",
        "border": "#e5e7eb",
        "card_bg": "#FFFFFF",
        "card_border": "#e5e7eb",
        "correct": "#047857",
        "table_header_bg": "#f3f4f6",
        "table_header_text": "#1f2937",
        "card_style": "flat",
        "header_style": "minimal",
        "section_colors": None,
        "spacing_scale": 0.55,
        "margins_cm": (0.7, 0.7, 1.1, 1.1),
    },
    "colorful": {
        "label": "Colorful",
        "description": "Institute-style paper — inline marks, 2-column MCQ options, colorful section accents.",
        "layout_style": "institute_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Arial",
        "primary": "#1a1a2e",
        "secondary": "#4a4a6a",
        "muted": "#6b7280",
        "light_muted": "#9ca3af",
        "border": "#e5e7eb",
        "card_bg": "#FFFFFF",
        "card_border": "#e5e7eb",
        "correct": "#047857",
        "table_header_bg": "#f3f4f6",
        "table_header_text": "#1f2937",
        "card_style": "stripe",
        "header_style": "banner",
        "banner_bg": "#EEF2FF",
        "section_colors": {
            "A": "#2563eb",
            "B": "#7c3aed",
            "C": "#ea580c",
            "D": "#db2777",
            "E": "#0d9488",
            "F": "#65a30d",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.5, 1.5),
    },

    "teal": {
        "label": "Teal",
        "description": "CBSE exam paper with teal section bars and accents.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#0f766e",
        "accent_light": "#ccfbf1",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#d1d5db",
        "card_bg": "#FFFFFF",
        "card_border": "#e5e7eb",
        "correct": "#047857",
        "table_header_bg": "#f3f4f6",
        "table_header_text": "#1f2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#0f766e", "B": "#0f766e", "C": "#0f766e",
            "D": "#0f766e", "E": "#0f766e", "F": "#0f766e",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
    "navy": {
        "label": "Navy Blue",
        "description": "CBSE exam paper with navy blue section bars and accents.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#1e3a8a",
        "accent_light": "#dbeafe",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#d1d5db",
        "card_bg": "#FFFFFF",
        "card_border": "#e5e7eb",
        "correct": "#047857",
        "table_header_bg": "#f3f4f6",
        "table_header_text": "#1f2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#1e3a8a", "B": "#1e3a8a", "C": "#1e3a8a",
            "D": "#1e3a8a", "E": "#1e3a8a", "F": "#1e3a8a",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
    "dark_green": {
        "label": "Dark Green",
        "description": "CBSE exam paper with dark green section bars and accents.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#166534",
        "accent_light": "#dcfce7",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#d1d5db",
        "card_bg": "#FFFFFF",
        "card_border": "#e5e7eb",
        "correct": "#047857",
        "table_header_bg": "#f3f4f6",
        "table_header_text": "#1f2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#166534", "B": "#166534", "C": "#166534",
            "D": "#166534", "E": "#166534", "F": "#166534",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
    "orange": {
        "label": "Orange",
        "description": "CBSE exam paper with warm coral-orange section bars and accents.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#FF7043",
        "accent_light": "#FFF3E0",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#E5E7EB",
        "card_bg": "#FFFFFF",
        "card_border": "#E5E7EB",
        "correct": "#047857",
        "table_header_bg": "#F3F4F6",
        "table_header_text": "#1F2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#FF7043", "B": "#FF7043", "C": "#FF7043",
            "D": "#FF7043", "E": "#FF7043", "F": "#FF7043",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
    "teal_premium": {
        "label": "Teal (Premium)",
        "description": "Clean CBSE exam paper with teal section bars and accents.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#0f766e",
        "accent_light": "#ccfbf1",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#d1d5db",
        "card_bg": "#FFFFFF",
        "card_border": "#e5e7eb",
        "correct": "#047857",
        "table_header_bg": "#f3f4f6",
        "table_header_text": "#1f2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#0f766e", "B": "#0f766e", "C": "#0f766e",
            "D": "#0f766e", "E": "#0f766e", "F": "#0f766e",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
    "navy_premium": {
        "label": "Navy Blue (Premium)",
        "description": "Clean CBSE exam paper with navy section bars and accents.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#1e3a8a",
        "accent_light": "#dbeafe",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#d1d5db",
        "card_bg": "#FFFFFF",
        "card_border": "#e5e7eb",
        "correct": "#047857",
        "table_header_bg": "#f3f4f6",
        "table_header_text": "#1f2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#1e3a8a", "B": "#1e3a8a", "C": "#1e3a8a",
            "D": "#1e3a8a", "E": "#1e3a8a", "F": "#1e3a8a",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
    "dark_green_premium": {
        "label": "Dark Green (Premium)",
        "description": "Clean CBSE exam paper with green section bars and accents.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#166534",
        "accent_light": "#dcfce7",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#d1d5db",
        "card_bg": "#FFFFFF",
        "card_border": "#e5e7eb",
        "correct": "#047857",
        "table_header_bg": "#f3f4f6",
        "table_header_text": "#1f2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#166534", "B": "#166534", "C": "#166534",
            "D": "#166534", "E": "#166534", "F": "#166534",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
    "orange_premium": {
        "label": "Orange (Premium)",
        "description": "Clean CBSE exam paper with orange section bars and accents.",
        "layout_style": "cbse_exam_paper",
        "font_body": "Helvetica",
        "font_bold": "Helvetica-Bold",
        "docx_font": "Calibri",
        "primary": "#1F2937",
        "accent": "#FF7043",
        "accent_light": "#FFF3E0",
        "accent_text": "#FFFFFF",
        "secondary": "#374151",
        "muted": "#6B7280",
        "light_muted": "#9CA3AF",
        "border": "#E5E7EB",
        "card_bg": "#FFFFFF",
        "card_border": "#E5E7EB",
        "correct": "#047857",
        "table_header_bg": "#F3F4F6",
        "table_header_text": "#1F2937",
        "card_style": "flat",
        "header_style": "formal",
        "section_colors": {
            "A": "#FF7043", "B": "#FF7043", "C": "#FF7043",
            "D": "#FF7043", "E": "#FF7043", "F": "#FF7043",
        },
        "spacing_scale": 1.0,
        "margins_cm": (1.0, 1.0, 1.58, 1.58),
    },
}

DEFAULT_TEMPLATE = "modern"


def get_available_templates() -> list:
    """Metadata list for a frontend template-picker dropdown."""
    return [
        {"id": key, "label": val["label"], "description": val.get("description", "")}
        for key, val in TEMPLATE_PRESETS.items()
    ]


def _get_template(name: Optional[str]) -> dict:
    if not name:
        return TEMPLATE_PRESETS[DEFAULT_TEMPLATE]
    key = str(name).strip().lower()
    if key not in TEMPLATE_PRESETS:
        logger.warning(f"Unknown template '{key}', falling back to '{DEFAULT_TEMPLATE}'")
    return TEMPLATE_PRESETS.get(key, TEMPLATE_PRESETS[DEFAULT_TEMPLATE])


def _section_color(tpl: dict, section_label: Optional[str]) -> str:
    """Per-section accent color for 'colorful' template; falls back to primary."""
    sc = tpl.get("section_colors")
    if sc and section_label and section_label in sc:
        return sc[section_label]
    return tpl["primary"]


def _sc(val: float, tpl: dict) -> float:
    """Scale a PDF spacing value (points) by the template's spacing_scale."""
    return max(1, val * tpl.get("spacing_scale", 1.0))


def _hexnc(hexstr: str) -> str:
    """Hex color without '#', uppercased — for docx OXML shading/border fills."""
    return hexstr.lstrip('#').upper()


def _rgb(hexstr: str):
    """Hex string -> docx RGBColor."""
    from docx.shared import RGBColor
    h = hexstr.lstrip('#')
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ═══════════════════════════════════════════════════════════════════════
# Unicode sub/super scripts
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
# Matrix Bracket Converter
# ═══════════════════════════════════════════════════════════════════════

def _convert_matrix_brackets(text: str, use_tags: bool = False) -> str:
    """Convert [[a,b],[c,d]] notation to proper matrix box format."""
    if not text:
        return text

    if '[[' in text:
        logger.warning(f"MATRIX_IN: {text[:150]!r}")

    bracket_pattern = re.compile(r'\[\[(.*?)\]\]', re.DOTALL)

    def build_matrix(inner_content: str) -> str:
        rows_raw = re.split(r'\],\s*\[', inner_content)
        if len(rows_raw) <= 1:
            return f"[[{inner_content}]]"

        matrix_rows = []
        max_cols = 0
        for row in rows_raw:
            values = [v.strip() for v in row.split(',') if v.strip()]
            matrix_rows.append(values)
            max_cols = max(max_cols, len(values))

        if not matrix_rows:
            return f"[[{inner_content}]]"

        if use_tags:
            lines = ['┌' + ' ' * (max_cols * 6) + '┐']
            for row in matrix_rows:
                padded = row + [''] * (max_cols - len(row))
                line = '│ ' + '  '.join(f'{v:>4}' for v in padded) + ' │'
                lines.append(line)
            lines.append('└' + ' ' * (max_cols * 6) + '┘')
            return '<br/>' + '<br/>'.join(lines) + '<br/>'
        else:
            lines = ['┌' + ' ' * (max_cols * 5) + '┐']
            for row in matrix_rows:
                padded = row + [''] * (max_cols - len(row))
                line = '│ ' + '  '.join(f'{v:>3}' for v in padded) + ' │'
                lines.append(line)
            lines.append('└' + ' ' * (max_cols * 5) + '┘')
            return '\n'.join(lines)

    def replace_matrix(match):
        inner = match.group(1)
        if not inner or ',' not in inner:
            return match.group(0)
        return build_matrix(inner)

    result = bracket_pattern.sub(replace_matrix, text)

    if '[[' in text:
        logger.warning(f"MATRIX_OUT: {result[:150]!r}")

    return result


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
    result = result.replace('₹', 'Rs.')

    MODIFIER_LETTERS = {
        '\u1D57': 't', '\u02B0': 'h', '\u02E2': 's', '\u1D48': 'd',
        '\u02B3': 'r', '\u02E1': 'l', '\u1D43': 'a', '\u1D49': 'e',
        '\u1D52': 'o',
    }
    for mod, plain in MODIFIER_LETTERS.items():
        result = result.replace(mod, plain)

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
        result = re.sub(r'_([a-zA-Z0-9])', r'_\1', result)

    result = result.replace('{', '').replace('}', '')
    result = _fix_unicode_scripts(result, use_tags)
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r' *\n *', '\n', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()

    result = _convert_matrix_brackets(result, use_tags)

    return result


def _latex_to_paragraph(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\s*\n', '__PARABREAK__', text)
    text = text.replace('\n', '__LINEBREAK__')

    result = _process_latex(text, use_tags=True)

    result = result.replace('__PARABREAK__', '<br/><br/>')
    result = result.replace('__LINEBREAK__', '<br/>')

    tags = {}
    for i, tag in enumerate(re.findall(r'</?(?:super|sub|b|i|font[^>]*)>|<br\s*/?>', result)):
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
# Date Formatting
# ═══════════════════════════════════════════════════════════════════════

def _format_date_for_display(paper_date: Optional[str] = None) -> str:
    if paper_date:
        try:
            dt = datetime.strptime(paper_date, "%Y-%m-%d")
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            try:
                dt = datetime.fromisoformat(paper_date.replace('Z', '+00:00'))
                return dt.strftime("%d/%m/%Y")
            except Exception:
                logger.warning(f"Invalid paperDate format: {paper_date}, using today's date")
                return datetime.now().strftime("%d/%m/%Y")
    else:
        return datetime.now().strftime("%d/%m/%Y")


# ═══════════════════════════════════════════════════════════════════════
# Rich Field Helpers (Marking Scheme, Sub-Parts, Model Answer)
# ═══════════════════════════════════════════════════════════════════════

def _format_marking_scheme_text(marking_scheme) -> str:
    """Format structured marking scheme into a clean readable string."""
    if not marking_scheme:
        return ""
    if isinstance(marking_scheme, str):
        return marking_scheme.strip()
    if not isinstance(marking_scheme, list):
        return ""
    parts = []
    for item in marking_scheme:
        if isinstance(item, dict):
            step = item.get("step") or item.get("description") or ""
            marks = item.get("marks")
            if step and marks is not None:
                parts.append(f"• {step} [{marks}M]")
            elif step:
                parts.append(f"• {step}")
        elif isinstance(item, str):
            parts.append(f"• {item}")
    return "  ".join(parts)


def _get_marking_scheme(q: dict):
    return q.get("markingScheme") or q.get("marking_scheme") or None


def _get_sub_parts(q: dict) -> list:
    return q.get("subParts") or q.get("sub_parts") or []


def _get_model_answer(q: dict) -> Optional[str]:
    return q.get("modelAnswer") or q.get("model_answer") or None


# ═══════════════════════════════════════════════════════════════════════
# Manual Question + Image Helpers
# ═══════════════════════════════════════════════════════════════════════

def _is_manual(q: dict) -> bool:
    return bool(
        q.get("isManual")
        or q.get("is_manual")
        or q.get("validationStatus") == "manual"
        or q.get("validation_status") == "manual"
    )


def _get_image_url(q: dict) -> Optional[str]:
    return q.get("imageUrl") or q.get("image_url") or None


def _get_question_table(q: dict) -> Optional[dict]:
    qt = q.get("questionTable") or q.get("question_table")
    if not qt or not isinstance(qt, dict):
        return None
    if not qt.get("headers") or not qt.get("rows"):
        return None
    return qt


def _fetch_image_bytes(url: str, timeout: int = 8) -> Optional[bytes]:
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        req = Request(url, headers={"User-Agent": "A4AI-ExportService/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning(f"Image fetch returned {resp.status} for {url}")
                return None
            return resp.read()
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning(f"Failed to fetch image {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching image {url}: {e}")
        return None


def _render_manual_question_image_pdf(image_url: str, W: float):
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage, Spacer

    img_bytes = _fetch_image_bytes(image_url)
    if not img_bytes:
        return []

    try:
        img_stream = io.BytesIO(img_bytes)
        max_w = W * 0.80
        max_h = 8 * cm
        img = RLImage(img_stream, width=max_w, height=max_h, kind='proportional')
        img.hAlign = 'CENTER'
        return [Spacer(1, 4), img, Spacer(1, 6)]
    except Exception as e:
        logger.warning(f"Failed to render image in PDF: {e}")
        return []


def _render_manual_question_image_docx(container, image_url: str):
    from docx.shared import Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    img_bytes = _fetch_image_bytes(image_url)
    if not img_bytes:
        return

    try:
        img_stream = io.BytesIO(img_bytes)
        p = container.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(img_stream, width=Cm(10))
    except Exception as e:
        logger.warning(f"Failed to embed image in DOCX: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Markdown Table Parser
# ═══════════════════════════════════════════════════════════════════════

def _parse_table_row(row_str: str) -> List[str]:
    cells = [c.strip() for c in row_str.split('|')]
    cells = [c for c in cells if c]
    cells = [re.sub(r'\*\*(.+?)\*\*', r'\1', c) for c in cells]
    cells = [re.sub(r'\*(.+?)\*', r'\1', c) for c in cells]
    return cells


def _parse_pipe_table_lines(lines: List[str]) -> tuple:
    headers: List[str] = []
    rows: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^\|?[\s\-:]+\|[\s\-:|]*$', stripped):
            continue
        row = _parse_table_row(stripped)
        if not headers:
            headers = row
        elif row:
            rows.append(row)
    return headers, rows


def _split_text_and_tables(text: str) -> List[dict]:
    if not text:
        return [{'type': 'text', 'content': ''}]

    segments: List[dict] = []

    if '\n' in text:
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
        sep_re = re.compile(r'\s*\|?\s*-{3,}(?:[\|\-\s:]*-{3,})+\s*\|?\s*')
        sep_m = sep_re.search(text)

        if not sep_m:
            return [{'type': 'text', 'content': text}]

        pre_sep = text[:sep_m.start()]
        post_sep = text[sep_m.end():]

        hdr_m = re.search(r'((?:\|[^|]+)+\|)\s*$', pre_sep)
        if not hdr_m:
            return [{'type': 'text', 'content': text}]

        pre_table = pre_sep[:hdr_m.start()].strip()
        headers = _parse_table_row(hdr_m.group(1))

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


def _strip_markdown_table_from_text(text: str) -> str:
    if not text or '\n' not in text:
        return text

    lines = text.split('\n')
    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        has_pipes = stripped.startswith('|') and stripped.count('|') >= 2
        next_is_sep = (
            i + 1 < len(lines)
            and re.match(r'^\s*\|?[\s\-:]+\|[\s\-:|]*$', lines[i + 1].strip())
        )

        if has_pipes and next_is_sep:
            while i < len(lines):
                tl = lines[i].strip()
                if tl.startswith('|') or re.match(r'^\|?[\s\-:]+\|', tl):
                    i += 1
                else:
                    break
        else:
            output_lines.append(line)
            i += 1

    result = '\n'.join(output_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def _render_inline_table_pdf(headers: List[str], rows: List[List[str]], styles, W: float, tpl: dict) -> list:
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

    if not headers or not rows:
        return []

    num_cols = len(headers)
    cell_style = styles.get('Option', styles['Normal'])

    table_data = [[Paragraph(f"<b>{h}</b>", cell_style) for h in headers]]

    for row in rows:
        padded = (row + [''] * num_cols)[:num_cols]
        table_data.append([
            Paragraph(_latex_to_paragraph(str(c)), cell_style)
            for c in padded
        ])

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
        ('GRID',          (0, 0), (-1, -1), 0.5, HexColor(tpl['border'])),
        ('BACKGROUND',    (0, 0), (-1, 0),  HexColor(tpl['table_header_bg'])),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  HexColor(tpl['table_header_text'])),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
        ('ALIGN',         (1, 0), (-1, -1), 'RIGHT'),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 6)]


# ═══════════════════════════════════════════════════════════════════════
# Group questions by section
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


def _has_accountancy_sections(questions: List[dict]) -> bool:
    for q in questions:
        sec = q.get('section') or q.get('_section') or ''
        if sec in ACCOUNTANCY_SECTIONS_META or sec.startswith(('A_', 'B1_')):
            return True
    return False


def _get_section_order(questions: List[dict]) -> tuple:
    if _has_accountancy_sections(questions):
        return ACCOUNTANCY_SECTION_ORDER, ACCOUNTANCY_SECTIONS_META
    elif _has_sections(questions):
        return SECTION_ORDER, CBSE_SECTIONS_META
    return None, None


# ═══════════════════════════════════════════════════════════════════════
# Question Table Rendering — PDF  (Statistics — stays semantic blue)
# ═══════════════════════════════════════════════════════════════════════

def _render_question_table_pdf(question_table, styles, W):
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer

    elements = []
    if not question_table or not isinstance(question_table, dict):
        return elements

    headers = question_table.get("headers", [])
    rows = question_table.get("rows", [])
    caption = question_table.get("caption")

    if not headers or not rows:
        return elements

    num_cols = len(headers)

    if caption:
        elements.append(Spacer(1, 2))
        cap_style = styles.get('QText', styles['Normal'])
        elements.append(Paragraph(
            f"<i>{_latex_to_paragraph(str(caption))}</i>",
            cap_style
        ))
        elements.append(Spacer(1, 3))
    else:
        elements.append(Spacer(1, 4))

    cell_style = styles.get('Option', styles['Normal'])

    table_data = [[Paragraph(f"<b>{_latex_to_paragraph(str(h))}</b>", cell_style) for h in headers]]

    for row in rows:
        if not isinstance(row, list):
            continue
        padded = (row + [""] * num_cols)[:num_cols]
        table_data.append([
            Paragraph(_latex_to_paragraph(str(cell)), cell_style)
            for cell in padded
        ])

    if num_cols == 2:
        col_widths = [W * 0.55, W * 0.45]
    elif num_cols == 3:
        col_widths = [W * 0.40, W * 0.30, W * 0.30]
    elif num_cols == 4:
        col_widths = [W * 0.34, W * 0.22, W * 0.22, W * 0.22]
    elif num_cols == 5:
        col_widths = [W * 0.28, W * 0.18, W * 0.18, W * 0.18, W * 0.18]
    else:
        col_widths = [W / num_cols] * num_cols

    col_widths = col_widths[:num_cols]

    t = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign='CENTER')

    style_cmds = [
        ('GRID',          (0, 0), (-1, -1), 0.5, HexColor('#bfdbfe')),
        ('BACKGROUND',    (0, 0), (-1, 0),  HexColor('#dbeafe')),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  HexColor('#1e3a8a')),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN',         (0, 0), (-1, 0),  'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
        ('ALIGN',         (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN',         (0, 1), (0, -1),  'LEFT'),
    ]

    t.setStyle(TableStyle(style_cmds))
    elements.append(t)
    elements.append(Spacer(1, 8))

    return elements


# ═══════════════════════════════════════════════════════════════════════
# Question Table Rendering — DOCX (Statistics — stays semantic blue)
# ═══════════════════════════════════════════════════════════════════════

def _render_question_table_docx(container, question_table):
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    if not question_table or not isinstance(question_table, dict):
        return

    headers = question_table.get("headers", [])
    rows = question_table.get("rows", [])
    caption = question_table.get("caption")

    if not headers or not rows:
        return

    num_cols = len(headers)

    if caption:
        cp = container.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cp.add_run(_latex_to_plain(str(caption)))
        cr.italic = True
        cr.font.size = Pt(9)
        cr.font.color.rgb = RGBColor(75, 85, 99)

    total_data_rows = 1 + len(rows)

    table = container.add_table(rows=total_data_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'

    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(_latex_to_plain(str(header)))
        r.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(30, 58, 138)

        shading = cell._element.get_or_add_tcPr()
        shd = shading.makeelement(qn('w:shd'), {
            qn('w:fill'): 'DBEAFE',
            qn('w:val'): 'clear',
        })
        shading.append(shd)

    for i, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        padded = (row + [""] * num_cols)[:num_cols]
        for j, cell_text in enumerate(padded):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(_latex_to_plain(str(cell_text)))
            r.font.size = Pt(10)

            if j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    container.add_paragraph()


# ═══════════════════════════════════════════════════════════════════════
# Accountancy Answer Table Rendering — PDF (stays semantic emerald)
# ═══════════════════════════════════════════════════════════════════════

def _render_answer_table_pdf(answer_table, styles, W):
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

    header_style = styles.get('Option', styles['Normal'])
    table_data = [[Paragraph(f"<b>{h}</b>", header_style) for h in headers]]

    for row in rows:
        if not isinstance(row, list):
            continue
        padded = (row + [""] * num_cols)[:num_cols]
        table_data.append([
            Paragraph(_latex_to_paragraph(str(cell)), styles.get('Option', styles['Normal']))
            for cell in padded
        ])

    if total_row and isinstance(total_row, list):
        padded_total = (total_row + [""] * num_cols)[:num_cols]
        table_data.append([
            Paragraph(f"<b>{_latex_to_paragraph(str(cell))}</b>", styles.get('Option', styles['Normal']))
            for cell in padded_total
        ])

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

    t = Table(table_data, colWidths=col_widths, repeatRows=1)

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

    if table_type == "journal_entry":
        style_cmds.append(('ALIGN', (3, 0), (4, -1), 'RIGHT'))
    elif table_type == "trial_balance":
        style_cmds.append(('ALIGN', (3, 0), (4, -1), 'RIGHT'))
    elif table_type == "ledger":
        if num_cols >= 8:
            style_cmds.append(('ALIGN', (3, 0), (3, -1), 'RIGHT'))
            style_cmds.append(('ALIGN', (7, 0), (7, -1), 'RIGHT'))
            style_cmds.append(('LINEAFTER', (3, 0), (3, -1), 1.5, HexColor('#374151')))

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
# Accountancy Answer Table Rendering — DOCX (stays semantic emerald)
# ═══════════════════════════════════════════════════════════════════════

def _render_answer_table_docx(container, answer_table):
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

    title_map = {
        "journal_entry": "Journal Entry",
        "ledger": "Ledger Account",
        "trial_balance": "Trial Balance",
    }
    title = title_map.get(table_type, "Answer Table")
    tp = container.add_paragraph()
    tr = tp.add_run(f"{title}:")
    tr.bold = True
    tr.font.size = Pt(10)
    tr.font.color.rgb = RGBColor(4, 120, 87)

    total_data_rows = 1 + len(rows) + (1 if total_row else 0)

    table = container.add_table(rows=total_data_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'

    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(header)
        r.bold = True
        r.font.size = Pt(9)
        shading = cell._element.get_or_add_tcPr()
        shd = shading.makeelement(qn('w:shd'), {
            qn('w:fill'): 'F3F4F6',
            qn('w:val'): 'clear',
        })
        shading.append(shd)

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

            if table_type in ("journal_entry", "trial_balance") and j >= num_cols - 2:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif table_type == "ledger" and num_cols >= 8 and j in (3, 7):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

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

            if table_type in ("journal_entry", "trial_balance") and j >= num_cols - 2:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

            shading = cell._element.get_or_add_tcPr()
            shd = shading.makeelement(qn('w:shd'), {
                qn('w:fill'): 'E5E7EB',
                qn('w:val'): 'clear',
            })
            shading.append(shd)

    container.add_paragraph()


def _render_inline_table_docx(container, headers: List[str], rows: List[List[str]]):
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT

    if not headers or not rows:
        return

    num_cols = len(headers)
    total_rows = 1 + len(rows)

    table = container.add_table(rows=total_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'

    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)

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

    container.add_paragraph()


# ═══════════════════════════════════════════════════════════════════════
# Question Card Wrapper (PDF) — card_based layout, template-aware
# ═══════════════════════════════════════════════════════════════════════

def _wrap_question_card(elements, W, tpl: dict, section_label: Optional[str] = None):
    from reportlab.platypus import Table, TableStyle, Spacer
    from reportlab.lib.colors import HexColor

    clean = list(elements)
    while clean and isinstance(clean[-1], Spacer):
        clean.pop()

    if not clean:
        return [Spacer(1, _sc(4, tpl))]

    style = tpl.get("card_style", "card")

    if style == "flat":
        return clean + [Spacer(1, _sc(8, tpl))]

    t = Table([[clean]], colWidths=[W])

    style_cmds = [
        ('BACKGROUND',    (0, 0), (-1, -1), HexColor(tpl['card_bg'])),
        ('BOX',           (0, 0), (-1, -1), 0.5, HexColor(tpl['card_border'])),
        ('TOPPADDING',    (0, 0), (-1, -1), _sc(5, tpl)),
        ('BOTTOMPADDING', (0, 0), (-1, -1), _sc(5, tpl)),
        ('LEFTPADDING',   (0, 0), (-1, -1), _sc(6, tpl)),
        ('RIGHTPADDING',  (0, 0), (-1, -1), _sc(6, tpl)),
    ]

    if style == "stripe":
        accent = _section_color(tpl, section_label)
        style_cmds.append(('LINEBEFORE', (0, 0), (0, -1), 3, HexColor(accent)))
        style_cmds.append(('LEFTPADDING', (0, 0), (-1, -1), _sc(10, tpl)))

    try:
        style_cmds.append(('ROUNDEDCORNERS', [6, 6, 6, 6]))
    except Exception:
        pass

    t.setStyle(TableStyle(style_cmds))

    return [t, Spacer(1, _sc(4, tpl))]


def _render_or_separator(styles, W, tpl: dict):
    from reportlab.platypus import Paragraph, Spacer, HRFlowable, Table
    from reportlab.lib.colors import HexColor

    or_elements = [
        Spacer(1, 2),
        Table(
            [[
                HRFlowable(width="30%", thickness=0.5, color=HexColor(tpl['border'])),
                Paragraph("<b>OR</b>", styles.get('SectionHeader', styles['Normal'])),
                HRFlowable(width="30%", thickness=0.5, color=HexColor(tpl['border'])),
            ]],
            colWidths=[W * 0.35, W * 0.30, W * 0.35],
        ),
        Spacer(1, 2),
    ]
    return or_elements


# ═══════════════════════════════════════════════════════════════════════
# PDF Generation  (entry point — routes to card_based or institute_paper)
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
    paper_date: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
    teacher_name: Optional[str] = None,
    institute_name: Optional[str] = None,
    duration: Optional[str] = None,
    topic: Optional[str] = None,
) -> bytes:
    tpl = _get_template(template)

    # Route Accountancy to authentic CBSE Accountancy Tabular layout
    sub_lower = (subject or "").lower()
    if sub_lower in ("accountancy", "accounts", "accounting") or _has_accountancy_sections(questions):
        return _generate_pdf_accountancy_exam(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject,
            include_answers=include_answers, include_explanations=include_explanations,
            logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
            teacher_name=teacher_name, institute_name=institute_name,
            duration=duration, topic=topic,
        )

    if tpl.get("layout_style") in ("cbse_exam_paper", "case_study_paper"):
        return _generate_pdf_cbse_exam(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject,
            include_answers=include_answers, include_explanations=include_explanations,
            logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
            teacher_name=teacher_name, institute_name=institute_name,
            duration=duration, topic=topic,
        )

    # v14: colorful -> institute-paper layout
    if tpl.get("layout_style") == "institute_paper":
        return _generate_pdf_institute(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject,
            include_answers=include_answers, include_explanations=include_explanations,
            logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
            teacher_name=teacher_name, institute_name=institute_name,
            duration=duration, topic=topic,
        )

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

    top_m, bottom_m, left_m, right_m = tpl['margins_cm']
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=top_m * cm, bottomMargin=bottom_m * cm,
        leftMargin=left_m * cm, rightMargin=right_m * cm,
    )

    styles = getSampleStyleSheet()

    W = A4[0] - (left_m + right_m) * cm

    fb = tpl['font_body']
    fbd = tpl['font_bold']

    custom_styles = {
        'SchoolName': dict(parent=styles['Title'], fontSize=14, leading=18, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor(tpl['primary']), fontName=fbd),
        'ExamMeta': dict(parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=HexColor(tpl['secondary']), spaceAfter=4, fontName=fb),
        'SectionHeader': dict(parent=styles['Heading1'], fontSize=11, spaceBefore=_sc(10, tpl), spaceAfter=2, textColor=HexColor(tpl['primary']), fontName=fbd, alignment=TA_CENTER),
        'SectionSub': dict(parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=HexColor(tpl['muted']), spaceAfter=2, fontName=fb),
        'SectionInstruction': dict(parent=styles['Normal'], fontSize=8.5, alignment=TA_CENTER, textColor=HexColor(tpl['light_muted']), spaceAfter=4, leading=10, fontName=fb),
        'SectionTitle': dict(parent=styles['Heading2'], fontSize=11, spaceBefore=_sc(14, tpl), spaceAfter=6, textColor=HexColor(tpl['primary']), fontName=fbd),
        'QText': dict(parent=styles['Normal'], fontSize=10, spaceBefore=2, spaceAfter=1, leading=12, textColor=HexColor('#1f1f3a'), fontName=fb),
        'Option': dict(parent=styles['Normal'], fontSize=9.5, leftIndent=14, spaceBefore=1, spaceAfter=1, leading=11.5, textColor=HexColor('#333355'), fontName=fb),
        'CorrectOption': dict(parent=styles['Normal'], fontSize=9.5, leftIndent=14, spaceBefore=1, spaceAfter=1, leading=11.5, textColor=HexColor(tpl['correct']), fontName=fbd),
        'AnswerLine': dict(parent=styles['Normal'], fontSize=9, leftIndent=14, spaceBefore=1, textColor=HexColor(tpl['correct']), fontName=fbd),
        'Explanation': dict(parent=styles['Normal'], fontSize=8.5, leftIndent=14, spaceBefore=1, spaceAfter=3, textColor=HexColor(tpl['muted']), leading=11, fontName=fb),
        'Marks': dict(parent=styles['Normal'], fontSize=9, alignment=TA_RIGHT, textColor=HexColor(tpl['light_muted']), fontName=fb),
        'Instruction': dict(parent=styles['Normal'], fontSize=9, leftIndent=12, spaceBefore=2, spaceAfter=2, textColor=HexColor(tpl['secondary']), leading=12, fontName=fb),
        'FooterText': dict(parent=styles['Normal'], fontSize=8, textColor=HexColor(tpl['light_muted']), alignment=TA_CENTER, fontName=fb),
        'ORText': dict(parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=HexColor(tpl['muted']), fontName=fbd, spaceBefore=2, spaceAfter=2),
        'ManualBadge': dict(parent=styles['Normal'], fontSize=7.5, textColor=HexColor('#4f46e5'), fontName=fbd),
    }
    for name, props in custom_styles.items():
        try:
            styles.add(ParagraphStyle(name=name, **props))
        except KeyError:
            pass

    story = []

    display_date = _format_date_for_display(paper_date)

    logo_img = None
    if logo_base64:
        try:
            if ',' in logo_base64:
                logo_base64 = logo_base64.split(',', 1)[1]
            logo_img = RLImage(io.BytesIO(base64.b64decode(logo_base64)), width=1.8 * cm, height=1.8 * cm)
            logo_img.hAlign = 'CENTER'
        except Exception as e:
            logger.warning(f"Logo failed: {e}")

    total_marks = sum(q.get('marks', 1) for q in questions)

    title_block = [
        Paragraph(f"<b>{exam_title}</b>", styles['SchoolName']),
        Paragraph(f"{board} Board | Class {class_grade} | {subject}", styles['ExamMeta']),
    ]
    info_block = [
        Paragraph(f"Date: {display_date}", styles['ExamMeta']),
        Paragraph(f"Total Marks: {total_marks}", styles['ExamMeta']),
        Paragraph(f"Total Questions: {len(questions)}", styles['ExamMeta']),
    ]

    if logo_img:
        ht = Table([[logo_img, title_block, info_block]], colWidths=[2.5 * cm, W - 6 * cm, 3.5 * cm])
    else:
        ht = Table([[title_block, info_block]], colWidths=[W - 4 * cm, 4 * cm])
    ht.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))

    if tpl['header_style'] == 'banner':
        banner = Table([[ht]], colWidths=[W])
        banner.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(tpl.get('banner_bg', '#F5F3FF'))),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(banner)
        rule_thickness = 2.5
    elif tpl['header_style'] == 'minimal':
        story.append(ht)
        rule_thickness = 0.75
    else:
        story.append(ht)
        rule_thickness = 1.5

    story.append(Spacer(1, _sc(4, tpl)))
    story.append(HRFlowable(width="100%", thickness=rule_thickness, color=HexColor(tpl['primary']), spaceAfter=_sc(8, tpl)))

    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None

    story.append(Paragraph("<b>General Instructions:</b>", styles['SectionTitle']))

    base_instructions = [
        "All questions are compulsory.",
        "Read each question carefully before answering.",
    ]

    if sec_meta_dict is ACCOUNTANCY_SECTIONS_META:
        base_instructions.extend([
            "This question paper is divided into <b>Part A</b> and <b>Part B</b>.",
            "<b>Part A</b> is compulsory for all candidates.",
            "<b>Part B</b> has two options — attempt only one.",
            "Internal choice has been provided in some questions.",
        ])
    elif has_sec:
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
    story.append(Spacer(1, _sc(6, tpl)))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor(tpl['border']), spaceAfter=_sc(6, tpl)))

    labels = ["A", "B", "C", "D", "E", "F"]

    QW = W - 20

    def _render_question(q, q_num):
        from reportlab.platypus import Table as RLTable, TableStyle as RLTableStyle
        elements = []
        raw_text = q.get('text', '')
        marks = q.get('marks', 1)
        marks_label = f"[{marks} {'mark' if marks == 1 else 'marks'}]"

        question_table = _get_question_table(q)

        if question_table:
            raw_text = _strip_markdown_table_from_text(raw_text)

        segments = _split_text_and_tables(raw_text)

        first_text = ''
        for seg in segments:
            if seg['type'] == 'text' and seg['content']:
                first_text = _latex_to_paragraph(seg['content'])
                break
        if not first_text:
            first_text = _latex_to_paragraph(raw_text)

        q_text_html = f"<b>Q{q_num}.</b> {first_text}"

        qt = RLTable(
            [[Paragraph(q_text_html, styles['QText']),
              Paragraph(marks_label, styles['Marks'])]],
            colWidths=[QW * 0.84, QW * 0.16],
        )
        qt.setStyle(RLTableStyle([
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(qt)

        if question_table:
            qt_elements = _render_question_table_pdf(question_table, styles, QW)
            elements.extend(qt_elements)

        image_url = _get_image_url(q)
        if image_url:
            img_elements = _render_manual_question_image_pdf(image_url, QW)
            elements.extend(img_elements)

        first_text_skipped = False
        for seg in segments:
            if seg['type'] == 'text':
                if not first_text_skipped:
                    first_text_skipped = True
                    continue
                content = _latex_to_paragraph(seg['content'])
                if content:
                    elements.append(Paragraph(content, styles['QText']))
            elif seg['type'] == 'table':
                if question_table:
                    continue
                hdrs, rws = seg['content']
                elements.extend(_render_inline_table_pdf(hdrs, rws, styles, QW, tpl))

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
                    elements.append(Spacer(1, _sc(18, tpl)))
                elif fmt == 'long_answer':
                    elements.append(Spacer(1, _sc(40, tpl)))
                elif fmt in ('journal_entry', 'ledger', 'trial_balance'):
                    elements.append(Spacer(1, _sc(50, tpl)))
                elif fmt == 'image':
                    elements.append(Spacer(1, _sc(20, tpl)))

        if include_answers and include_explanations:
            raw_table = q.get('answer_table') or q.get('answerTable')
            if raw_table and isinstance(raw_table, dict):
                elements.extend(_render_answer_table_pdf(raw_table, styles, QW))
            else:
                ans = _latex_to_paragraph(correct_answer)
                elements.append(Paragraph(f"<b>Answer:</b> {ans}", styles['AnswerLine']))

        if include_explanations:
            exp = _latex_to_paragraph(q.get('explanation', ''))
            if exp:
                elements.append(Paragraph(f"<b>Explanation:</b> {exp}", styles['Explanation']))

        return elements

    q_num = 0

    if has_sec:
        grouped = _group_by_section(questions)
        last_section_title = None

        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue

            meta = sec_meta_dict.get(sec_key, {})
            current_title = meta.get('title', sec_key)
            sec_letter = sec_key[:1]
            sec_color = _section_color(tpl, sec_letter)

            if current_title != last_section_title:
                story.append(HRFlowable(width="60%", thickness=1, color=HexColor(sec_color), spaceBefore=_sc(14, tpl), spaceAfter=_sc(4, tpl)))
                story.append(Paragraph(f'<font color="{sec_color}"><b>{current_title}</b></font>', styles['SectionHeader']))
                last_section_title = current_title

            story.append(Paragraph(meta.get('subtitle', ''), styles['SectionSub']))
            story.append(Paragraph(meta.get('instruction', ''), styles['SectionInstruction']))
            story.append(HRFlowable(width="40%", thickness=0.5, color=HexColor(tpl['border']), spaceAfter=_sc(6, tpl)))

            main_qs = [q for q in sec_qs if not q.get('_is_or', False)]
            or_qs = [q for q in sec_qs if q.get('_is_or', False)]
            or_queue = list(or_qs)

            for q in main_qs:
                q_num += 1
                elements = _render_question(q, q_num)
                card = _wrap_question_card(elements, W, tpl, sec_letter)
                story.extend(card)

                if or_queue:
                    or_q = or_queue.pop(0)
                    story.extend(_render_or_separator(styles, W, tpl))
                    or_elements = _render_question(or_q, q_num)
                    or_card = _wrap_question_card(or_elements, W, tpl, sec_letter)
                    story.extend(or_card)

            for or_q in or_queue:
                q_num += 1
                story.extend(_render_or_separator(styles, W, tpl))
                or_elements = _render_question(or_q, q_num)
                or_card = _wrap_question_card(or_elements, W, tpl, sec_letter)
                story.extend(or_card)

        unsectioned = grouped.get('NONE', [])
        if unsectioned:
            story.append(HRFlowable(width="60%", thickness=1, color=HexColor('#4f46e5'), spaceBefore=_sc(14, tpl), spaceAfter=_sc(4, tpl)))
            story.append(Paragraph(f"<b>Additional Questions</b>", styles['SectionHeader']))
            story.append(Paragraph("(Added by teacher)", styles['SectionSub']))
            story.append(HRFlowable(width="40%", thickness=0.5, color=HexColor(tpl['border']), spaceAfter=_sc(6, tpl)))

            for q in unsectioned:
                q_num += 1
                elements = _render_question(q, q_num)
                card = _wrap_question_card(elements, W, tpl, None)
                story.extend(card)

    else:
        for q in questions:
            q_num += 1
            elements = _render_question(q, q_num)
            card = _wrap_question_card(elements, W, tpl, None)
            story.extend(card)

    if include_answers and not include_explanations:
        story.append(PageBreak())
        story.append(Paragraph("<b>Answer Key</b>", styles['SchoolName']))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor(tpl['primary']), spaceAfter=_sc(10, tpl)))

        q_num_ak = 0
        all_qs_ordered = []
        if has_sec:
            grouped = _group_by_section(questions)
            for sec_key in sec_order:
                all_qs_ordered.extend(grouped.get(sec_key, []))
            all_qs_ordered.extend(grouped.get('NONE', []))
        else:
            all_qs_ordered = questions

        for q in all_qs_ordered:
            q_num_ak += 1
            raw_table = q.get('answer_table') or q.get('answerTable')
            if raw_table and isinstance(raw_table, dict):
                story.append(Paragraph(f"<b>Q{q_num_ak}.</b>", styles['QText']))
                table_elements = _render_answer_table_pdf(raw_table, styles, W)
                story.extend(table_elements)
                story.append(Spacer(1, 4))
            else:
                correct = _latex_to_paragraph(q.get('correctAnswer', q.get('correct_answer', '')))
                story.append(Paragraph(f"<b>Q{q_num_ak}.</b> {correct}", styles['QText']))
                story.append(Spacer(1, 2))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor(tpl['border']), spaceAfter=6))
    story.append(Paragraph(f"Generated by a4ai · {board} {subject} Class {class_grade} · {display_date}", styles['FooterText']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# DOCX Generation  (entry point — routes to card_based or institute_paper)
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
    paper_date: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
    teacher_name: Optional[str] = None,
    institute_name: Optional[str] = None,
    duration: Optional[str] = None,
    topic: Optional[str] = None,
) -> bytes:
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn

    tpl = _get_template(template)

    # Route Accountancy to authentic CBSE Accountancy Tabular layout
    sub_lower = (subject or "").lower()
    if sub_lower in ("accountancy", "accounts", "accounting") or _has_accountancy_sections(questions):
        return _generate_docx_accountancy_exam(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject,
            include_answers=include_answers, include_explanations=include_explanations,
            logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
            teacher_name=teacher_name, institute_name=institute_name,
            duration=duration, topic=topic,
        )

    if tpl.get("layout_style") in ("cbse_exam_paper", "case_study_paper"):
        return _generate_docx_cbse_exam(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject,
            include_answers=include_answers, include_explanations=include_explanations,
            logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
            teacher_name=teacher_name, institute_name=institute_name,
            duration=duration, topic=topic,
        )

    # v14: colorful -> institute-paper layout
    if tpl.get("layout_style") == "institute_paper":
        return _generate_docx_institute(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject,
            include_answers=include_answers, include_explanations=include_explanations,
            logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
            teacher_name=teacher_name, institute_name=institute_name,
            duration=duration, topic=topic,
        )

    doc = Document()

    try:
        normal_style = doc.styles['Normal']
        normal_style.font.name = tpl['docx_font']
        rpr = normal_style.element.get_or_add_rPr()
        rFonts = rpr.find(qn('w:rFonts'))
        if rFonts is None:
            rFonts = rpr.makeelement(qn('w:rFonts'), {})
            rpr.append(rFonts)
        rFonts.set(qn('w:eastAsia'), tpl['docx_font'])
    except Exception as e:
        logger.warning(f"Could not set default docx font: {e}")

    top_m, bottom_m, left_m, right_m = tpl['margins_cm']
    for section in doc.sections:
        section.top_margin = Cm(top_m)
        section.bottom_margin = Cm(bottom_m)
        section.left_margin = Cm(left_m)
        section.right_margin = Cm(right_m)

    def _spt(val: float) -> "Pt":
        return Pt(max(1, val * tpl.get('spacing_scale', 1.0)))

    display_date = _format_date_for_display(paper_date)

    header_container = doc
    if tpl['header_style'] == 'banner':
        banner_table = doc.add_table(rows=1, cols=1)
        banner_cell = banner_table.rows[0].cells[0]
        tcPr = banner_cell._element.get_or_add_tcPr()
        shd = tcPr.makeelement(qn('w:shd'), {qn('w:fill'): _hexnc(tpl.get('banner_bg', '#EEF2FF')), qn('w:val'): 'clear'})
        tcPr.append(shd)
        header_container = banner_cell

    if logo_base64:
        try:
            if ',' in logo_base64:
                logo_base64 = logo_base64.split(',', 1)[1]
            p = header_container.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(io.BytesIO(base64.b64decode(logo_base64)), width=Cm(2))
        except Exception as e:
            logger.warning(f"DOCX logo failed: {e}")

    title_p = header_container.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run(exam_title or "Test Paper")
    title_run.bold = True
    title_run.font.size = Pt(20)
    title_run.font.color.rgb = _rgb(tpl['primary'])
    title_run.font.name = tpl['docx_font']

    sub = header_container.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run(f"{board} Board | Class {class_grade} | {subject}")
    r.font.size = Pt(11)
    r.font.color.rgb = _rgb(tpl['secondary'])
    r.font.name = tpl['docx_font']

    total_marks = sum(q.get('marks', 1) for q in questions)

    meta = header_container.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = meta.add_run(f"Total Questions: {len(questions)} | Total Marks: {total_marks} | Date: {display_date}")
    r.font.size = Pt(9)
    r.font.color.rgb = _rgb(tpl['muted'])
    r.font.name = tpl['docx_font']

    rule_char = "─" if tpl['header_style'] == 'minimal' else "━"
    rule_len = 36 if tpl['spacing_scale'] < 1.0 else 50
    doc.add_paragraph(rule_char * rule_len)

    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None

    doc.add_heading("General Instructions", level=2)

    instructions = ["All questions are compulsory.", "Read each question carefully."]
    if sec_meta_dict is ACCOUNTANCY_SECTIONS_META:
        instructions.extend([
            "This paper is divided into Part A and Part B.",
            "Part A is compulsory for all candidates.",
            "Part B has two options — attempt only one.",
            "Internal choice has been provided in some questions.",
        ])
    elif has_sec:
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
        p.paragraph_format.space_after = _spt(2)
    doc.add_paragraph(rule_char * rule_len)

    labels = ["A", "B", "C", "D", "E", "F"]
    q_num = 0

    def _add_docx_separator(container):
        sep_p = container.add_paragraph()
        sep_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sep_p.paragraph_format.space_before = _spt(2)
        sep_p.paragraph_format.space_after = _spt(2)
        r = sep_p.add_run("─" * 60)
        r.font.size = Pt(6)
        r.font.color.rgb = _rgb(tpl['border'])

    def _add_docx_or_separator(container):
        sep_p = container.add_paragraph()
        sep_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sep_p.paragraph_format.space_before = _spt(4)
        sep_p.paragraph_format.space_after = _spt(4)
        r = sep_p.add_run("─── OR ───")
        r.font.size = Pt(10)
        r.font.color.rgb = _rgb(tpl['muted'])
        r.bold = True

    def _start_question_container(section_letter=None):
        style = tpl.get('card_style', 'card')
        if style == 'flat':
            return doc, None

        table = doc.add_table(rows=1, cols=1)
        cell = table.rows[0].cells[0]
        tcPr = cell._element.get_or_add_tcPr()

        shd = tcPr.makeelement(qn('w:shd'), {qn('w:fill'): _hexnc(tpl['card_bg']), qn('w:val'): 'clear'})
        tcPr.append(shd)

        tcBorders = tcPr.makeelement(qn('w:tcBorders'), {})
        if style == 'stripe':
            accent = _hexnc(_section_color(tpl, section_letter))
            left = tcBorders.makeelement(qn('w:left'), {qn('w:val'): 'single', qn('w:sz'): '24', qn('w:color'): accent})
            tcBorders.append(left)
        else:
            border_hex = _hexnc(tpl['card_border'])
            for side in ('top', 'left', 'bottom', 'right'):
                b = tcBorders.makeelement(qn(f'w:{side}'), {qn('w:val'): 'single', qn('w:sz'): '6', qn('w:color'): border_hex})
                tcBorders.append(b)
        tcPr.append(tcBorders)

        return cell, table

    def _render_q_docx(q, q_num, section_letter=None):
        container, wrapper_table = _start_question_container(section_letter)

        raw_text = q.get('text', '')
        marks = q.get('marks', 1)

        question_table = _get_question_table(q)

        if question_table:
            raw_text = _strip_markdown_table_from_text(raw_text)

        segments = _split_text_and_tables(raw_text)
        first_text = ''
        for seg in segments:
            if seg['type'] == 'text' and seg['content']:
                first_text = _latex_to_plain(seg['content'])
                break
        if not first_text:
            first_text = _latex_to_plain(raw_text)

        p = container.add_paragraph()
        rq = p.add_run(f"Q{q_num}. ")
        rq.bold = True
        rq.font.size = Pt(11)
        rq.font.name = tpl['docx_font']
        rt = p.add_run(first_text)
        rt.font.size = Pt(11)
        rt.font.name = tpl['docx_font']

        rm = p.add_run(f"  [{marks} {'mark' if marks == 1 else 'marks'}]")
        rm.font.size = Pt(8)
        rm.font.color.rgb = _rgb(tpl['light_muted'])
        rm.font.name = tpl['docx_font']

        if question_table:
            _render_question_table_docx(container, question_table)

        image_url = _get_image_url(q)
        if image_url:
            _render_manual_question_image_docx(container, image_url)

        first_text_skipped = False
        for seg in segments:
            if seg['type'] == 'text':
                if not first_text_skipped:
                    first_text_skipped = True
                    continue
                content = _latex_to_plain(seg['content'])
                if content:
                    cp = container.add_paragraph()
                    crun = cp.add_run(content)
                    crun.font.size = Pt(11)
                    crun.font.name = tpl['docx_font']
            elif seg['type'] == 'table':
                if question_table:
                    continue
                hdrs, rws = seg['content']
                _render_inline_table_docx(container, hdrs, rws)

        options = q.get('options', [])
        correct_answer = q.get('correctAnswer', q.get('correct_answer', ''))

        for opt_idx, opt in enumerate(options):
            opt_clean = _latex_to_plain(opt)
            letter = labels[opt_idx] if opt_idx < len(labels) else str(opt_idx + 1)
            is_correct = include_answers and correct_answer and correct_answer.strip().upper().startswith(letter)

            op = container.add_paragraph()
            op.paragraph_format.left_indent = Pt(24)
            op.paragraph_format.space_after = _spt(2)
            opt_stripped = re.sub(r'^[A-F][).\s]+\s*', '', opt_clean).strip()
            run = op.add_run(f"{letter}) {opt_stripped}")
            run.font.size = Pt(10)
            run.font.name = tpl['docx_font']
            if is_correct:
                run.bold = True
                run.font.color.rgb = _rgb(tpl['correct'])

        if include_answers and include_explanations:
            raw_table = q.get('answer_table') or q.get('answerTable')
            if raw_table and isinstance(raw_table, dict):
                _render_answer_table_docx(container, raw_table)
            else:
                correct = _latex_to_plain(correct_answer)
                ap = container.add_paragraph()
                ap.paragraph_format.left_indent = Pt(24)
                ra = ap.add_run("Answer: ")
                ra.bold = True
                ra.font.size = Pt(10)
                ra.font.color.rgb = _rgb(tpl['correct'])
                rv = ap.add_run(correct)
                rv.font.size = Pt(10)
                rv.font.color.rgb = _rgb(tpl['correct'])

        if include_explanations:
            exp = _latex_to_plain(q.get('explanation', ''))
            if exp:
                ep = container.add_paragraph()
                ep.paragraph_format.left_indent = Pt(24)
                re2 = ep.add_run("Explanation: ")
                re2.bold = True
                re2.font.size = Pt(8)
                re2.font.color.rgb = _rgb(tpl['muted'])
                rv2 = ep.add_run(exp)
                rv2.font.size = Pt(8)
                rv2.font.color.rgb = _rgb(tpl['muted'])

    if has_sec:
        grouped = _group_by_section(questions)
        last_section_title = None

        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue

            sec_meta = sec_meta_dict.get(sec_key, {})
            current_title = sec_meta.get('title', sec_key)
            sec_letter = sec_key[:1]
            sec_color = _section_color(tpl, sec_letter)

            if current_title != last_section_title:
                doc.add_paragraph(rule_char * rule_len)
                h = doc.add_heading(current_title, level=1)
                h.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for hr in h.runs:
                    hr.font.color.rgb = _rgb(sec_color)
                    hr.font.name = tpl['docx_font']
                last_section_title = current_title

            sub_h = doc.add_paragraph()
            sub_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            sub_r = sub_h.add_run(sec_meta.get('subtitle', ''))
            sub_r.font.size = Pt(9)
            sub_r.font.color.rgb = _rgb(tpl['muted'])
            sub_r.font.name = tpl['docx_font']

            inst_p = doc.add_paragraph()
            inst_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            inst_r = inst_p.add_run(sec_meta.get('instruction', ''))
            inst_r.font.size = Pt(9)
            inst_r.font.color.rgb = _rgb(tpl['muted'])
            inst_r.font.name = tpl['docx_font']
            inst_r.italic = True

            main_qs = [q for q in sec_qs if not q.get('_is_or', False)]
            or_qs = [q for q in sec_qs if q.get('_is_or', False)]
            or_queue = list(or_qs)

            for i, q in enumerate(main_qs):
                q_num += 1
                _render_q_docx(q, q_num, sec_letter)

                if or_queue:
                    or_q = or_queue.pop(0)
                    _add_docx_or_separator(doc)
                    _render_q_docx(or_q, q_num, sec_letter)

                if tpl.get('card_style') == 'flat' and (i < len(main_qs) - 1 or or_queue):
                    _add_docx_separator(doc)

        unsectioned = grouped.get('NONE', [])
        if unsectioned:
            doc.add_paragraph(rule_char * rule_len)
            h = doc.add_heading("Additional Questions", level=1)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for hr in h.runs:
                hr.font.name = tpl['docx_font']

            sub_h = doc.add_paragraph()
            sub_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            sub_r = sub_h.add_run("(Added by teacher)")
            sub_r.font.size = Pt(9)
            sub_r.font.color.rgb = _rgb(tpl['muted'])
            sub_r.font.name = tpl['docx_font']
            sub_r.italic = True

            for i, q in enumerate(unsectioned):
                q_num += 1
                _render_q_docx(q, q_num, None)
                if tpl.get('card_style') == 'flat' and i < len(unsectioned) - 1:
                    _add_docx_separator(doc)

    else:
        for i, q in enumerate(questions):
            q_num += 1
            _render_q_docx(q, q_num, None)
            if tpl.get('card_style') == 'flat' and i < len(questions) - 1:
                _add_docx_separator(doc)

    if include_answers and not include_explanations:
        doc.add_page_break()
        h = doc.add_heading("Answer Key", level=0)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for hr in h.runs:
            hr.font.color.rgb = _rgb(tpl['primary'])
            hr.font.name = tpl['docx_font']

        q_num_ak = 0
        all_qs_ordered = []
        if has_sec:
            grouped = _group_by_section(questions)
            for sec_key in sec_order:
                all_qs_ordered.extend(grouped.get(sec_key, []))
            all_qs_ordered.extend(grouped.get('NONE', []))
        else:
            all_qs_ordered = questions

        for q in all_qs_ordered:
            q_num_ak += 1
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
    r = ft.add_run(f"Generated by a4ai · {board} {subject} Class {class_grade} · {display_date}")
    r.font.size = Pt(8)
    r.font.color.rgb = _rgb(tpl['light_muted'])
    r.font.name = tpl['docx_font']

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# v14 — INSTITUTE-PAPER LAYOUT  (used by "colorful" template)
# ═══════════════════════════════════════════════════════════════════════
#
# Reference institute exam-paper format:
#   • Institute name header  →  "Class X — Subject"  →  optional Topic line
#   • Teacher / Max-Marks / Time / Date meta row
#   • Signature multi-color section rule
#   • Inline "[marks]" at end of each question (NOT a separate column)
#   • MCQ options rendered as (a)/(b) two-column pairs
#   • Plain "SECTION A (description)" headings
#   • "— All the Best —" footer
#
# Shared LaTeX/table/image helpers are reused; only the layout differs.


def _institute_section_heading(sec_key: str, meta_dict: dict) -> tuple:
    """Return (title, description) for an institute-style section heading."""
    meta = meta_dict.get(sec_key, {}) if meta_dict else {}
    title = meta.get("title", f"Section {sec_key}")
    subtitle = (meta.get("subtitle", "") or "").strip()
    # normalise "(1 mark each — MCQ / Assertion-Reason)" -> clean parenthetical
    desc = subtitle.strip("() ")
    return title.upper(), desc


def _institute_multicolor_rule_pdf(W: float, tpl: dict, thickness: float = 3.5):
    """A thin horizontal rule split into the template's section accent colors."""
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Table, TableStyle

    sc = tpl.get("section_colors") or {}
    order = ["A", "B", "C", "D", "E", "F"]
    colors = [sc.get(k, tpl["primary"]) for k in order if sc.get(k)]
    if not colors:
        colors = [tpl["primary"]]

    n = len(colors)
    seg_w = W / n
    t = Table([[""] * n], colWidths=[seg_w] * n, rowHeights=[thickness])
    cmds = [
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]
    for i, c in enumerate(colors):
        cmds.append(("BACKGROUND", (i, 0), (i, 0), HexColor(c)))
    t.setStyle(TableStyle(cmds))
    return t


def _generate_pdf_institute(
    questions, exam_title, board, class_grade, subject,
    include_answers, include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image as RLImage, KeepTogether,
    )

    buffer = io.BytesIO()
    top_m, bottom_m, left_m, right_m = tpl["margins_cm"]
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=top_m * cm, bottomMargin=bottom_m * cm,
        leftMargin=left_m * cm, rightMargin=right_m * cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - (left_m + right_m) * cm
    fb, fbd = tpl["font_body"], tpl["font_bold"]

    ist_styles = {
        "InstName":   dict(parent=styles["Title"], fontSize=18, leading=21, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor(tpl["primary"]), fontName=fbd),
        "InstExam":   dict(parent=styles["Normal"], fontSize=12.5, leading=15, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor(tpl["primary"]), fontName=fbd),
        "InstClass":  dict(parent=styles["Normal"], fontSize=11, leading=13, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor(tpl["secondary"]), fontName=fb),
        "InstTopic":  dict(parent=styles["Normal"], fontSize=10, leading=12, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor(tpl["muted"]), fontName=fb),
        "MetaL":      dict(parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_LEFT, textColor=HexColor(tpl["secondary"]), fontName=fb),
        "MetaR":      dict(parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_RIGHT, textColor=HexColor(tpl["secondary"]), fontName=fb),
        "GenInst":    dict(parent=styles["Normal"], fontSize=9, leading=12, alignment=TA_LEFT, textColor=HexColor(tpl["muted"]), fontName=fb),
        "SecHead":    dict(parent=styles["Heading2"], fontSize=11.5, leading=14, spaceBefore=12, spaceAfter=1, alignment=TA_LEFT, fontName=fbd),
        "QText":      dict(parent=styles["Normal"], fontSize=10.5, leading=13.5, spaceBefore=5, spaceAfter=2, alignment=TA_LEFT, textColor=HexColor("#1f1f3a"), fontName=fb),
        "Opt":        dict(parent=styles["Normal"], fontSize=10, leading=13, alignment=TA_LEFT, textColor=HexColor("#2b2b45"), fontName=fb),
        "OptCorrect": dict(parent=styles["Normal"], fontSize=10, leading=13, alignment=TA_LEFT, textColor=HexColor(tpl["correct"]), fontName=fbd),
        "Ans":        dict(parent=styles["Normal"], fontSize=9.5, leading=12, alignment=TA_LEFT, textColor=HexColor(tpl["correct"]), fontName=fbd),
        "Expl":       dict(parent=styles["Normal"], fontSize=9, leading=11.5, alignment=TA_LEFT, textColor=HexColor(tpl["muted"]), fontName=fb),
        "OrText":     dict(parent=styles["Normal"], fontSize=10, leading=13, alignment=TA_CENTER, textColor=HexColor(tpl["secondary"]), fontName=fbd, spaceBefore=3, spaceAfter=3),
        "Footer":     dict(parent=styles["Normal"], fontSize=8, alignment=TA_CENTER, textColor=HexColor(tpl["light_muted"]), fontName=fb),
        "AllBest":    dict(parent=styles["Normal"], fontSize=11, leading=14, alignment=TA_CENTER, textColor=HexColor(tpl["primary"]), fontName=fbd, spaceBefore=10),
    }
    # reportlab needs 'Option'/'AnswerLine' keys for shared table helpers
    for name, props in ist_styles.items():
        try:
            styles.add(ParagraphStyle(name=name, **props))
        except KeyError:
            pass
    # alias styles that shared table renderers look up by name
    for shared_name, src in (("Option", "Opt"), ("AnswerLine", "Ans"), ("QText", "QText")):
        if shared_name not in styles:
            try:
                styles.add(ParagraphStyle(name=shared_name, parent=styles[src]))
            except Exception:
                pass

    story = []
    display_date = _format_date_for_display(paper_date)
    total_marks = sum(q.get("marks", 1) for q in questions)

    # ── Header ──────────────────────────────────────────────────────
    if logo_base64:
        try:
            lb = logo_base64.split(",", 1)[1] if "," in logo_base64 else logo_base64
            logo_img = RLImage(io.BytesIO(base64.b64decode(lb)), width=1.6 * cm, height=1.6 * cm)
            logo_img.hAlign = "CENTER"
            story.append(logo_img)
            story.append(Spacer(1, 2))
        except Exception as e:
            logger.warning(f"Institute logo failed: {e}")

    header_name = (institute_name or "").strip() or (exam_title or "Test Paper")
    story.append(Paragraph(header_name, styles["InstName"]))

    # If both institute + exam title given and they differ, show exam title too
    if institute_name and exam_title and exam_title.strip() and exam_title.strip().lower() != header_name.strip().lower():
        story.append(Paragraph(exam_title.strip(), styles["InstExam"]))

    story.append(Paragraph(f"Class {class_grade} &nbsp;•&nbsp; {subject} &nbsp;•&nbsp; {board}", styles["InstClass"]))
    if topic and str(topic).strip():
        story.append(Paragraph(f"<i>Topic: {str(topic).strip()}</i>", styles["InstTopic"]))

    story.append(Spacer(1, 5))
    story.append(_institute_multicolor_rule_pdf(W, tpl))
    story.append(Spacer(1, 5))

    # ── Meta row (Teacher | Max Marks / Time / Date) ────────────────
    teacher_disp = (teacher_name or "").strip() or "______________"
    left_lines = [f"<b>Teacher:</b> {teacher_disp}"]
    right_lines = [f"<b>Max Marks:</b> {total_marks}"]
    if duration and str(duration).strip():
        right_lines.append(f"<b>Time:</b> {str(duration).strip()}")
    right_lines.append(f"<b>Date:</b> {display_date}")

    meta_left = Paragraph("<br/>".join(left_lines), styles["MetaL"])
    meta_right = Paragraph("<br/>".join(right_lines), styles["MetaR"])
    meta_tbl = Table([[meta_left, meta_right]], colWidths=[W * 0.55, W * 0.45])
    meta_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.75, color=HexColor(tpl["border"]), spaceAfter=5))

    # ── Compact general instructions (single line, institute style) ─
    story.append(Paragraph(
        "<b>General Instructions:</b> All questions are compulsory. "
        "Marks for each question are indicated against it. "
        "Write answers neatly in the space provided.",
        styles["GenInst"],
    ))
    story.append(Spacer(1, 4))

    labels_lower = ["a", "b", "c", "d", "e", "f"]

    def _mcq_two_column(options, correct_answer):
        """Render MCQ options as (a)/(b) two-column pairs."""
        cells = []
        for idx, opt in enumerate(options):
            letter = labels_lower[idx] if idx < len(labels_lower) else str(idx + 1)
            opt_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_paragraph(opt)).strip()
            is_correct = False
            if include_answers and correct_answer:
                ca = correct_answer.strip()
                if ca.upper().startswith(letter.upper()) or opt.strip() == ca.strip():
                    is_correct = True
            style = styles["OptCorrect"] if is_correct else styles["Opt"]
            cells.append(Paragraph(f"({letter}) {opt_clean}", style))

        # pack into 2-column rows
        rows = []
        for i in range(0, len(cells), 2):
            left = cells[i]
            right = cells[i + 1] if i + 1 < len(cells) else ""
            rows.append([left, right])
        if not rows:
            return []
        t = Table(rows, colWidths=[W * 0.5, W * 0.5])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return [Spacer(1, 1), t]

    def _render_question_institute(q, q_num):
        elements = []
        raw_text = q.get("text", "")
        marks = q.get("marks", 1)
        marks_tag = f'<font color="{tpl["muted"]}"><b>[{marks}]</b></font>'

        question_table = _get_question_table(q)
        if question_table:
            raw_text = _strip_markdown_table_from_text(raw_text)

        segments = _split_text_and_tables(raw_text)
        first_text = ""
        for seg in segments:
            if seg["type"] == "text" and seg["content"]:
                first_text = _latex_to_paragraph(seg["content"])
                break
        if not first_text:
            first_text = _latex_to_paragraph(raw_text)

        # inline marks at end of question line (no separate column)
        elements.append(Paragraph(f"<b>{q_num}.</b> {first_text} &nbsp;{marks_tag}", styles["QText"]))

        if question_table:
            elements.extend(_render_question_table_pdf(question_table, styles, W))

        image_url = _get_image_url(q)
        if image_url:
            elements.extend(_render_manual_question_image_pdf(image_url, W))

        first_skipped = False
        for seg in segments:
            if seg["type"] == "text":
                if not first_skipped:
                    first_skipped = True
                    continue
                content = _latex_to_paragraph(seg["content"])
                if content:
                    elements.append(Paragraph(content, styles["QText"]))
            elif seg["type"] == "table":
                if question_table:
                    continue
                hdrs, rws = seg["content"]
                elements.extend(_render_inline_table_pdf(hdrs, rws, styles, W, tpl))

        options = q.get("options", [])
        correct_answer = q.get("correctAnswer", q.get("correct_answer", ""))

        if options:
            elements.extend(_mcq_two_column(options, correct_answer))
        else:
            fmt = q.get("format", "mcq")
            if not include_answers:
                gap = {"short_answer": 20, "long_answer": 44,
                       "journal_entry": 52, "ledger": 52, "trial_balance": 52,
                       "image": 22}.get(fmt, 0)
                if gap:
                    elements.append(Spacer(1, gap))

        if include_answers and include_explanations:
            raw_table = q.get("answer_table") or q.get("answerTable")
            if raw_table and isinstance(raw_table, dict):
                elements.extend(_render_answer_table_pdf(raw_table, styles, W))
            elif not options:
                ans = _latex_to_paragraph(correct_answer)
                elements.append(Paragraph(f"<b>Ans:</b> {ans}", styles["Ans"]))
        elif include_answers and not options:
            ans = _latex_to_paragraph(correct_answer)
            elements.append(Paragraph(f"<b>Ans:</b> {ans}", styles["Ans"]))

        if include_explanations:
            exp = _latex_to_paragraph(q.get("explanation", ""))
            if exp:
                elements.append(Paragraph(f"<b>Explanation:</b> {exp}", styles["Expl"]))

        return elements

    def _section_heading_flowables(sec_key, meta_dict):
        title, desc = _institute_section_heading(sec_key, meta_dict)
        accent = _section_color(tpl, sec_key[:1])
        head = f'<font color="{accent}">{title}</font>'
        if desc:
            head += f'  <font color="{tpl["muted"]}" size="9">({desc})</font>'
        return [
            Spacer(1, 6),
            Paragraph(head, styles["SecHead"]),
            HRFlowable(width="100%", thickness=1.2, color=HexColor(accent), spaceAfter=4),
        ]

    # ── Body ────────────────────────────────────────────────────────
    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None
    q_num = 0

    if has_sec:
        grouped = _group_by_section(questions)
        last_title = None
        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue
            meta = sec_meta_dict.get(sec_key, {})
            current_title = meta.get("title", sec_key)
            if current_title != last_title:
                story.extend(_section_heading_flowables(sec_key, sec_meta_dict))
                last_title = current_title

            main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
            or_qs = [q for q in sec_qs if q.get("_is_or", False)]
            or_queue = list(or_qs)

            for q in main_qs:
                q_num += 1
                story.append(KeepTogether(_render_question_institute(q, q_num)))
                if or_queue:
                    or_q = or_queue.pop(0)
                    story.append(Paragraph("OR", styles["OrText"]))
                    story.append(KeepTogether(_render_question_institute(or_q, q_num)))

            for or_q in or_queue:
                q_num += 1
                story.append(Paragraph("OR", styles["OrText"]))
                story.append(KeepTogether(_render_question_institute(or_q, q_num)))

        unsectioned = grouped.get("NONE", [])
        if unsectioned:
            accent = _section_color(tpl, "F")
            story.append(Spacer(1, 6))
            story.append(Paragraph(f'<font color="{accent}">ADDITIONAL QUESTIONS</font>', styles["SecHead"]))
            story.append(HRFlowable(width="100%", thickness=1.2, color=HexColor(accent), spaceAfter=4))
            for q in unsectioned:
                q_num += 1
                story.append(KeepTogether(_render_question_institute(q, q_num)))
    else:
        for q in questions:
            q_num += 1
            story.append(KeepTogether(_render_question_institute(q, q_num)))

    # ── Answer key (answers-only mode) ──────────────────────────────
    if include_answers and not include_explanations:
        story.append(PageBreak())
        story.append(Paragraph("Answer Key", styles["InstName"]))
        story.append(Spacer(1, 4))
        story.append(_institute_multicolor_rule_pdf(W, tpl))
        story.append(Spacer(1, 6))

        all_qs = []
        if has_sec:
            grouped = _group_by_section(questions)
            for sec_key in sec_order:
                all_qs.extend(grouped.get(sec_key, []))
            all_qs.extend(grouped.get("NONE", []))
        else:
            all_qs = questions

        for i, q in enumerate(all_qs, 1):
            raw_table = q.get("answer_table") or q.get("answerTable")
            if raw_table and isinstance(raw_table, dict):
                story.append(Paragraph(f"<b>{i}.</b>", styles["QText"]))
                story.extend(_render_answer_table_pdf(raw_table, styles, W))
            else:
                correct = _latex_to_paragraph(q.get("correctAnswer", q.get("correct_answer", "")))
                story.append(Paragraph(f"<b>{i}.</b> {correct}", styles["QText"]))

    # ── Footer ──────────────────────────────────────────────────────
    story.append(Paragraph("— All the Best —", styles["AllBest"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor(tpl["border"]), spaceAfter=4))
    story.append(Paragraph(f"Generated by a4ai · {board} {subject} Class {class_grade} · {display_date}", styles["Footer"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _institute_multicolor_rule_docx(doc, tpl: dict):
    """A thin horizontal rule split into the template's section accent colors (DOCX)."""
    from docx.shared import Pt
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ROW_HEIGHT_RULE
    from docx.oxml.ns import qn

    sc = tpl.get("section_colors") or {}
    order = ["A", "B", "C", "D", "E", "F"]
    colors = [sc.get(k) for k in order if sc.get(k)]
    if not colors:
        colors = [tpl["primary"]]

    n = len(colors)
    table = doc.add_table(rows=1, cols=n)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    row = table.rows[0]
    row.height = Pt(4)
    row.height_rule = WD_ROW_HEIGHT_RULE.EXACTLY

    for i, c in enumerate(colors):
        cell = row.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(" ")
        run.font.size = Pt(1)
        tcPr = cell._element.get_or_add_tcPr()
        shd = tcPr.makeelement(qn('w:shd'), {qn('w:fill'): _hexnc(c), qn('w:val'): 'clear'})
        tcPr.append(shd)
        # zero cell margins
        tcMar = tcPr.makeelement(qn('w:tcMar'), {})
        for side in ('top', 'bottom', 'start', 'end'):
            m = tcMar.makeelement(qn(f'w:{side}'), {qn('w:w'): '0', qn('w:type'): 'dxa'})
            tcMar.append(m)
        tcPr.append(tcMar)


def _generate_docx_institute(
    questions, exam_title, board, class_grade, subject,
    include_answers, include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    doc = Document()
    font_name = tpl["docx_font"]

    try:
        normal_style = doc.styles["Normal"]
        normal_style.font.name = font_name
        rpr = normal_style.element.get_or_add_rPr()
        rFonts = rpr.find(qn('w:rFonts'))
        if rFonts is None:
            rFonts = rpr.makeelement(qn('w:rFonts'), {})
            rpr.append(rFonts)
        rFonts.set(qn('w:eastAsia'), font_name)
    except Exception as e:
        logger.warning(f"Institute docx font set failed: {e}")

    top_m, bottom_m, left_m, right_m = tpl["margins_cm"]
    for section in doc.sections:
        section.top_margin = Cm(top_m)
        section.bottom_margin = Cm(bottom_m)
        section.left_margin = Cm(left_m)
        section.right_margin = Cm(right_m)

    display_date = _format_date_for_display(paper_date)
    total_marks = sum(q.get("marks", 1) for q in questions)
    labels_lower = ["a", "b", "c", "d", "e", "f"]

    def _center_run(text, size, color_hex, bold=True, italic=False):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
        r.font.size = Pt(size)
        r.font.color.rgb = _rgb(color_hex)
        r.font.name = font_name
        return p

    # ── Header ──────────────────────────────────────────────────────
    if logo_base64:
        try:
            lb = logo_base64.split(",", 1)[1] if "," in logo_base64 else logo_base64
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(io.BytesIO(base64.b64decode(lb)), width=Cm(1.8))
        except Exception as e:
            logger.warning(f"Institute docx logo failed: {e}")

    header_name = (institute_name or "").strip() or (exam_title or "Test Paper")
    _center_run(header_name, 18, tpl["primary"], bold=True)

    if institute_name and exam_title and exam_title.strip() and exam_title.strip().lower() != header_name.strip().lower():
        _center_run(exam_title.strip(), 12.5, tpl["primary"], bold=True)

    _center_run(f"Class {class_grade}  •  {subject}  •  {board}", 11, tpl["secondary"], bold=False)
    if topic and str(topic).strip():
        _center_run(f"Topic: {str(topic).strip()}", 10, tpl["muted"], bold=False, italic=True)

    _institute_multicolor_rule_docx(doc, tpl)

    # ── Meta row ────────────────────────────────────────────────────
    teacher_disp = (teacher_name or "").strip() or "______________"
    meta_tbl = doc.add_table(rows=1, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    lc = meta_tbl.rows[0].cells[0]
    lc.text = ""
    lp = lc.paragraphs[0]
    lr = lp.add_run("Teacher: ")
    lr.bold = True; lr.font.size = Pt(10); lr.font.name = font_name
    lr2 = lp.add_run(teacher_disp)
    lr2.font.size = Pt(10); lr2.font.name = font_name

    rc = meta_tbl.rows[0].cells[1]
    rc.text = ""
    rp = rc.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    right_bits = [("Max Marks: ", str(total_marks))]
    if duration and str(duration).strip():
        right_bits.append(("Time: ", str(duration).strip()))
    right_bits.append(("Date: ", display_date))
    for k, (lbl, val) in enumerate(right_bits):
        rb = rp.add_run(lbl); rb.bold = True; rb.font.size = Pt(10); rb.font.name = font_name
        rv = rp.add_run(val); rv.font.size = Pt(10); rv.font.name = font_name
        if k < len(right_bits) - 1:
            sep = rp.add_run("    "); sep.font.size = Pt(10)

    # thin separator line
    sep_p = doc.add_paragraph()
    sep_r = sep_p.add_run("─" * 60)
    sep_r.font.size = Pt(7)
    sep_r.font.color.rgb = _rgb(tpl["border"])

    # ── General instructions (single line) ──────────────────────────
    gi = doc.add_paragraph()
    gir = gi.add_run("General Instructions: ")
    gir.bold = True; gir.font.size = Pt(9); gir.font.name = font_name
    gir.font.color.rgb = _rgb(tpl["muted"])
    giv = gi.add_run("All questions are compulsory. Marks for each question are indicated against it. "
                     "Write answers neatly in the space provided.")
    giv.font.size = Pt(9); giv.font.name = font_name
    giv.font.color.rgb = _rgb(tpl["muted"])

    def _mcq_two_column_docx(options, correct_answer):
        n = len(options)
        if n == 0:
            return
        nrows = (n + 1) // 2
        table = doc.add_table(rows=nrows, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.LEFT
        for idx, opt in enumerate(options):
            letter = labels_lower[idx] if idx < len(labels_lower) else str(idx + 1)
            opt_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_plain(opt)).strip()
            is_correct = bool(include_answers and correct_answer
                              and correct_answer.strip().upper().startswith(letter.upper()))
            row_i, col_i = idx // 2, idx % 2
            cell = table.rows[row_i].cells[col_i]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.left_indent = Pt(14)
            r = p.add_run(f"({letter}) {opt_clean}")
            r.font.size = Pt(10)
            r.font.name = font_name
            if is_correct:
                r.bold = True
                r.font.color.rgb = _rgb(tpl["correct"])

    def _render_q_docx_institute(q, q_num):
        raw_text = q.get("text", "")
        marks = q.get("marks", 1)

        question_table = _get_question_table(q)
        if question_table:
            raw_text = _strip_markdown_table_from_text(raw_text)

        segments = _split_text_and_tables(raw_text)
        first_text = ""
        for seg in segments:
            if seg["type"] == "text" and seg["content"]:
                first_text = _latex_to_plain(seg["content"])
                break
        if not first_text:
            first_text = _latex_to_plain(raw_text)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(2)
        rq = p.add_run(f"{q_num}. ")
        rq.bold = True; rq.font.size = Pt(10.5); rq.font.name = font_name
        rt = p.add_run(first_text)
        rt.font.size = Pt(10.5); rt.font.name = font_name
        rm = p.add_run(f"   [{marks}]")
        rm.bold = True; rm.font.size = Pt(9); rm.font.name = font_name
        rm.font.color.rgb = _rgb(tpl["muted"])

        if question_table:
            _render_question_table_docx(doc, question_table)

        image_url = _get_image_url(q)
        if image_url:
            _render_manual_question_image_docx(doc, image_url)

        first_skipped = False
        for seg in segments:
            if seg["type"] == "text":
                if not first_skipped:
                    first_skipped = True
                    continue
                content = _latex_to_plain(seg["content"])
                if content:
                    cp = doc.add_paragraph()
                    cr = cp.add_run(content)
                    cr.font.size = Pt(10.5); cr.font.name = font_name
            elif seg["type"] == "table":
                if question_table:
                    continue
                hdrs, rws = seg["content"]
                _render_inline_table_docx(doc, hdrs, rws)

        options = q.get("options", [])
        correct_answer = q.get("correctAnswer", q.get("correct_answer", ""))

        if options:
            _mcq_two_column_docx(options, correct_answer)
        else:
            fmt = q.get("format", "mcq")
            if not include_answers:
                blanks = {"short_answer": 2, "long_answer": 4,
                          "journal_entry": 5, "ledger": 5, "trial_balance": 5,
                          "image": 2}.get(fmt, 0)
                for _ in range(blanks):
                    doc.add_paragraph()

        if include_answers and include_explanations:
            raw_table = q.get("answer_table") or q.get("answerTable")
            if raw_table and isinstance(raw_table, dict):
                _render_answer_table_docx(doc, raw_table)
            elif not options:
                ap = doc.add_paragraph()
                ar = ap.add_run("Ans: "); ar.bold = True
                ar.font.size = Pt(10); ar.font.color.rgb = _rgb(tpl["correct"]); ar.font.name = font_name
                av = ap.add_run(_latex_to_plain(correct_answer))
                av.font.size = Pt(10); av.font.color.rgb = _rgb(tpl["correct"]); av.font.name = font_name
        elif include_answers and not options:
            ap = doc.add_paragraph()
            ar = ap.add_run("Ans: "); ar.bold = True
            ar.font.size = Pt(10); ar.font.color.rgb = _rgb(tpl["correct"]); ar.font.name = font_name
            av = ap.add_run(_latex_to_plain(correct_answer))
            av.font.size = Pt(10); av.font.color.rgb = _rgb(tpl["correct"]); av.font.name = font_name

        if include_explanations:
            exp = _latex_to_plain(q.get("explanation", ""))
            if exp:
                ep = doc.add_paragraph()
                er = ep.add_run("Explanation: "); er.bold = True
                er.font.size = Pt(8.5); er.font.color.rgb = _rgb(tpl["muted"]); er.font.name = font_name
                ev = ep.add_run(exp)
                ev.font.size = Pt(8.5); ev.font.color.rgb = _rgb(tpl["muted"]); ev.font.name = font_name

    def _section_heading_docx(sec_key, meta_dict):
        title, desc = _institute_section_heading(sec_key, meta_dict)
        accent = _section_color(tpl, sec_key[:1])
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(title)
        r.bold = True; r.font.size = Pt(11.5); r.font.name = font_name
        r.font.color.rgb = _rgb(accent)
        if desc:
            rd = p.add_run(f"  ({desc})")
            rd.font.size = Pt(9); rd.font.name = font_name
            rd.font.color.rgb = _rgb(tpl["muted"])
        line = doc.add_paragraph()
        lr = line.add_run("─" * 60)
        lr.font.size = Pt(7); lr.font.color.rgb = _rgb(accent)

    def _or_docx():
        op = doc.add_paragraph()
        op.alignment = WD_ALIGN_PARAGRAPH.CENTER
        orr = op.add_run("OR")
        orr.bold = True; orr.font.size = Pt(10); orr.font.name = font_name
        orr.font.color.rgb = _rgb(tpl["secondary"])

    # ── Body ────────────────────────────────────────────────────────
    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None
    q_num = 0

    if has_sec:
        grouped = _group_by_section(questions)
        last_title = None
        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue
            meta = sec_meta_dict.get(sec_key, {})
            current_title = meta.get("title", sec_key)
            if current_title != last_title:
                _section_heading_docx(sec_key, sec_meta_dict)
                last_title = current_title

            main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
            or_qs = [q for q in sec_qs if q.get("_is_or", False)]
            or_queue = list(or_qs)

            for q in main_qs:
                q_num += 1
                _render_q_docx_institute(q, q_num)
                if or_queue:
                    or_q = or_queue.pop(0)
                    _or_docx()
                    _render_q_docx_institute(or_q, q_num)

            for or_q in or_queue:
                q_num += 1
                _or_docx()
                _render_q_docx_institute(or_q, q_num)

        unsectioned = grouped.get("NONE", [])
        if unsectioned:
            accent = _section_color(tpl, "F")
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            r = p.add_run("ADDITIONAL QUESTIONS")
            r.bold = True; r.font.size = Pt(11.5); r.font.name = font_name
            r.font.color.rgb = _rgb(accent)
            line = doc.add_paragraph()
            lr = line.add_run("─" * 60); lr.font.size = Pt(7); lr.font.color.rgb = _rgb(accent)
            for q in unsectioned:
                q_num += 1
                _render_q_docx_institute(q, q_num)
    else:
        for q in questions:
            q_num += 1
            _render_q_docx_institute(q, q_num)

    # ── Answer key (answers-only mode) ──────────────────────────────
    if include_answers and not include_explanations:
        doc.add_page_break()
        _center_run("Answer Key", 18, tpl["primary"], bold=True)
        _institute_multicolor_rule_docx(doc, tpl)

        all_qs = []
        if has_sec:
            grouped = _group_by_section(questions)
            for sec_key in sec_order:
                all_qs.extend(grouped.get(sec_key, []))
            all_qs.extend(grouped.get("NONE", []))
        else:
            all_qs = questions

        for i, q in enumerate(all_qs, 1):
            raw_table = q.get("answer_table") or q.get("answerTable")
            if raw_table and isinstance(raw_table, dict):
                p = doc.add_paragraph()
                p.add_run(f"{i}. ").bold = True
                _render_answer_table_docx(doc, raw_table)
            else:
                correct = _latex_to_plain(q.get("correctAnswer", q.get("correct_answer", "")))
                p = doc.add_paragraph()
                p.add_run(f"{i}. ").bold = True
                p.add_run(correct)

    # ── Footer ──────────────────────────────────────────────────────
    _center_run("— All the Best —", 11, tpl["primary"], bold=True)
    ft = doc.add_paragraph()
    ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = ft.add_run(f"Generated by a4ai · {board} {subject} Class {class_grade} · {display_date}")
    r.font.size = Pt(8); r.font.color.rgb = _rgb(tpl["light_muted"]); r.font.name = font_name

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

    # ═══════════════════════════════════════════════════════════════════════
# PATCH 2 — Add standalone Answer Key generators
# ═══════════════════════════════════════════════════════════════════════
#
# LOCATION: app/services/export_service.py
# INSERT: At the END of the file (after everything else)
#
# These are STANDALONE answer key generators — they produce a separate
# PDF/DOCX file containing ONLY the answer key, matching the sample PDF's
# clean institute-paper style.
#
# ═══════════════════════════════════════════════════════════════════════


def generate_answer_key_pdf(
    questions: List[dict],
    exam_title: str = "Test Paper",
    board: str = "CBSE",
    class_grade: str = "10",
    subject: str = "Science",
    include_explanations: bool = False,
    logo_base64: Optional[str] = None,
    paper_date: Optional[str] = None,
    template: str = "teal",
    teacher_name: Optional[str] = None,
    institute_name: Optional[str] = None,
    duration: Optional[str] = None,
    topic: Optional[str] = None,
) -> bytes:
    """
    Standalone Answer Key PDF (separate file — not appended to question paper).
    Uses the same institute-paper header as the question paper for consistency.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, KeepTogether,
    )

    tpl = _get_template(template)
    if tpl.get("layout_style") in ("cbse_exam_paper", "case_study_paper"):
        return _generate_pdf_cbse_answer_key(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject,
            include_explanations=include_explanations,
            logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
            teacher_name=teacher_name, institute_name=institute_name,
            duration=duration, topic=topic,
        )

    # Force institute layout for answer key (uniform look regardless of picked template)
    if tpl.get("layout_style") != "institute_paper":
        # Fall back to teal for card-based templates
        tpl = _get_template("teal")

    buffer = io.BytesIO()
    top_m, bottom_m, left_m, right_m = tpl["margins_cm"]
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=top_m * cm, bottomMargin=bottom_m * cm,
        leftMargin=left_m * cm, rightMargin=right_m * cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - (left_m + right_m) * cm
    fb, fbd = tpl["font_body"], tpl["font_bold"]

    ak_styles = {
        "AkInstName":   dict(parent=styles["Title"], fontSize=18, leading=21, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor(tpl["primary"]), fontName=fbd),
        "AkTitle":      dict(parent=styles["Title"], fontSize=16, leading=20, spaceBefore=6, spaceAfter=3, alignment=TA_CENTER, textColor=HexColor(tpl["primary"]), fontName=fbd),
        "AkExam":       dict(parent=styles["Normal"], fontSize=12, leading=15, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor(tpl["primary"]), fontName=fbd),
        "AkClass":      dict(parent=styles["Normal"], fontSize=11, leading=13, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor(tpl["secondary"]), fontName=fb),
        "AkMetaL":      dict(parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_LEFT, textColor=HexColor(tpl["secondary"]), fontName=fb),
        "AkMetaR":      dict(parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_RIGHT, textColor=HexColor(tpl["secondary"]), fontName=fb),
        "AkSecHead":    dict(parent=styles["Heading2"], fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=2, alignment=TA_LEFT, textColor=HexColor(tpl["primary"]), fontName=fbd),
        "AkQNum":       dict(parent=styles["Normal"], fontSize=11, leading=15, spaceBefore=3, spaceAfter=1, alignment=TA_LEFT, textColor=HexColor("#1f1f3a"), fontName=fbd),
        "AkAns":        dict(parent=styles["Normal"], fontSize=10.5, leading=14, alignment=TA_LEFT, textColor=HexColor(tpl["correct"]), fontName=fbd),
        "AkExpl":       dict(parent=styles["Normal"], fontSize=9.5, leading=12.5, spaceAfter=4, alignment=TA_LEFT, textColor=HexColor(tpl["muted"]), fontName=fb),
        "AkFooter":     dict(parent=styles["Normal"], fontSize=8, alignment=TA_CENTER, textColor=HexColor(tpl["light_muted"]), fontName=fb),
    }
    for name, props in ak_styles.items():
        try:
            styles.add(ParagraphStyle(name=name, **props))
        except KeyError:
            pass

    story = []
    display_date = _format_date_for_display(paper_date)
    total_marks = sum(q.get("marks", 1) for q in questions)

    # ── Header ──────────────────────────────────────────────────────
    if logo_base64:
        try:
            lb = logo_base64.split(",", 1)[1] if "," in logo_base64 else logo_base64
            logo_img = RLImage(io.BytesIO(base64.b64decode(lb)), width=1.6 * cm, height=1.6 * cm)
            logo_img.hAlign = "CENTER"
            story.append(logo_img)
            story.append(Spacer(1, 2))
        except Exception as e:
            logger.warning(f"AK logo failed: {e}")

    header_name = (institute_name or "").strip() or (exam_title or "Test Paper")
    story.append(Paragraph(header_name, styles["AkInstName"]))
    if institute_name and exam_title and exam_title.strip() and exam_title.strip().lower() != header_name.strip().lower():
        story.append(Paragraph(exam_title.strip(), styles["AkExam"]))

    story.append(Paragraph(f"Class {class_grade} &nbsp;•&nbsp; {subject} &nbsp;•&nbsp; {board}", styles["AkClass"]))

    story.append(Spacer(1, 5))
    story.append(_institute_multicolor_rule_pdf(W, tpl))
    story.append(Spacer(1, 6))

    # Big "ANSWER KEY" title
    story.append(Paragraph("ANSWER KEY", styles["AkTitle"]))
    story.append(Spacer(1, 2))

    # ── Meta row ────────────────────────────────────────────────────
    teacher_disp = (teacher_name or "").strip() or "______________"
    left_lines = [f"<b>Teacher:</b> {teacher_disp}"]
    right_lines = [f"<b>Max Marks:</b> {total_marks}"]
    if duration and str(duration).strip():
        right_lines.append(f"<b>Time:</b> {str(duration).strip()}")
    right_lines.append(f"<b>Date:</b> {display_date}")

    meta_left = Paragraph("<br/>".join(left_lines), styles["AkMetaL"])
    meta_right = Paragraph("<br/>".join(right_lines), styles["AkMetaR"])
    meta_tbl = Table([[meta_left, meta_right]], colWidths=[W * 0.55, W * 0.45])
    meta_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.75, color=HexColor(tpl["border"]), spaceAfter=8))

    # ── Build answer list ───────────────────────────────────────────
    labels_lower = ["a", "b", "c", "d", "e", "f"]

    def _find_correct_letter(options, correct_answer):
        """Given options list + correct answer, return the letter (a/b/c/d) or None."""
        if not options or not correct_answer:
            return None
        ca = str(correct_answer).strip()
        # If correct_answer already starts with a letter marker
        m = re.match(r'^([A-Fa-f])[).\s]', ca)
        if m:
            return m.group(1).lower()
        # Match against option text
        ca_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', ca).strip().lower()
        for idx, opt in enumerate(options):
            opt_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', str(opt)).strip().lower()
            if opt_clean == ca_clean:
                return labels_lower[idx] if idx < len(labels_lower) else None
        # First-letter match on single-char answer
        if len(ca) == 1 and ca.lower() in labels_lower:
            return ca.lower()
        return None

    def _render_answer_row(q, q_num):
        """Return list of flowables for one answer entry."""
        elements = []
        options = q.get("options", [])
        correct = q.get("correctAnswer", q.get("correct_answer", ""))
        explanation = q.get("explanation", "") if include_explanations else ""
        sub_parts = _get_sub_parts(q)
        marking_scheme = _get_marking_scheme(q)
        model_answer = _get_model_answer(q)

        # Check for answer table (Accountancy etc.)
        raw_table = q.get("answer_table") or q.get("answerTable")
        if raw_table and isinstance(raw_table, dict):
            elements.append(Paragraph(f"<b>{q_num}.</b>", styles["AkQNum"]))
            elements.extend(_render_answer_table_pdf(raw_table, styles, W))
            if model_answer and include_explanations:
                ma = _latex_to_paragraph(model_answer)
                elements.append(Paragraph(f"<b>Model Answer:</b> {ma}", styles["AkExpl"]))
            if explanation:
                exp = _latex_to_paragraph(explanation)
                elements.append(Paragraph(f"<i>Explanation:</i> {exp}", styles["AkExpl"]))
            if marking_scheme and include_explanations:
                ms_text = _latex_to_paragraph(_format_marking_scheme_text(marking_scheme))
                if ms_text:
                    elements.append(Paragraph(f"<b>Marking Scheme:</b> {ms_text}", styles["AkExpl"]))
            return elements

        # Handle sub-parts (case-based / multi-part)
        if sub_parts:
            elements.append(Paragraph(f"<b>{q_num}.</b>", styles["AkAns"]))
            labels_roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
            for sp_idx, sp in enumerate(sub_parts):
                sp_label = sp.get("label") or (labels_roman[sp_idx] if sp_idx < len(labels_roman) else str(sp_idx + 1))
                sp_ans = sp.get("correctAnswer", sp.get("correct_answer", "")) or sp.get("answer", "") or "—"
                sp_marks = sp.get("marks")
                marks_str = f" [{sp_marks}M]" if sp_marks else ""
                elements.append(Paragraph(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<b>({sp_label})</b> {_latex_to_paragraph(str(sp_ans))}{marks_str}",
                    styles["AkAns"]
                ))
            if model_answer and include_explanations:
                ma = _latex_to_paragraph(model_answer)
                elements.append(Paragraph(f"<b>Model Answer:</b> {ma}", styles["AkExpl"]))
            if explanation:
                exp = _latex_to_paragraph(explanation)
                elements.append(Paragraph(f"<i>Explanation:</i> {exp}", styles["AkExpl"]))
            if marking_scheme and include_explanations:
                ms_text = _latex_to_paragraph(_format_marking_scheme_text(marking_scheme))
                if ms_text:
                    elements.append(Paragraph(f"<b>Marking Scheme:</b> {ms_text}", styles["AkExpl"]))
            return elements

        if options:
            letter = _find_correct_letter(options, correct)
            if letter:
                # Show both letter AND option text — clearest for teacher
                letter_idx = labels_lower.index(letter)
                opt_text = ""
                if letter_idx < len(options):
                    opt_text = re.sub(r'^[A-Fa-f][).\s]+\s*', '', str(options[letter_idx])).strip()
                if opt_text:
                    ans_str = f"({letter}) {_latex_to_paragraph(opt_text)}"
                else:
                    ans_str = f"({letter})"
            else:
                # Fallback: just show whatever correct answer we have
                ans_str = _latex_to_paragraph(str(correct)) if correct else "—"
        else:
            ans_str = _latex_to_paragraph(str(correct)) if correct else "—"

        elements.append(Paragraph(f"<b>{q_num}.</b> &nbsp;{ans_str}", styles["AkAns"]))

        if model_answer and include_explanations:
            ma = _latex_to_paragraph(model_answer)
            elements.append(Paragraph(f"<b>Model Answer:</b> {ma}", styles["AkExpl"]))

        if explanation:
            exp = _latex_to_paragraph(explanation)
            elements.append(Paragraph(f"<i>Explanation:</i> {exp}", styles["AkExpl"]))

        if marking_scheme and include_explanations:
            ms_text = _latex_to_paragraph(_format_marking_scheme_text(marking_scheme))
            if ms_text:
                elements.append(Paragraph(f"<b>Marking Scheme:</b> {ms_text}", styles["AkExpl"]))

        return elements

    # ── Group by section if applicable ──────────────────────────────
    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None
    q_num = 0

    if has_sec:
        grouped = _group_by_section(questions)
        last_title = None
        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue
            meta = sec_meta_dict.get(sec_key, {})
            current_title = meta.get("title", sec_key)
            if current_title != last_title:
                subtitle = meta.get("subtitle", "")
                head = current_title
                if subtitle:
                    head += f'  <font color="{tpl["muted"]}" size="9">{subtitle}</font>'
                story.append(Paragraph(head, styles["AkSecHead"]))
                story.append(HRFlowable(width="100%", thickness=0.8, color=HexColor(tpl["primary"]), spaceAfter=4))
                last_title = current_title

            main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
            or_qs = [q for q in sec_qs if q.get("_is_or", False)]
            or_queue = list(or_qs)

            for q in main_qs:
                q_num += 1
                story.append(KeepTogether(_render_answer_row(q, q_num)))
                if or_queue:
                    or_q = or_queue.pop(0)
                    story.append(Paragraph("<i>OR</i>", styles["AkExpl"]))
                    story.append(KeepTogether(_render_answer_row(or_q, q_num)))
                    story.append(KeepTogether(_render_answer_row(or_q, f"{q_num} (OR)")))

            for or_q in or_queue:
                q_num += 1
                story.append(Paragraph("<i>OR</i>", styles["AkExpl"]))
                story.append(KeepTogether(_render_answer_row(or_q, q_num)))
                story.append(KeepTogether(_render_answer_row(or_q, f"{q_num} (OR)")))

        unsectioned = grouped.get("NONE", [])
        if unsectioned:
            story.append(Paragraph("Additional Questions", styles["AkSecHead"]))
            story.append(HRFlowable(width="100%", thickness=0.8, color=HexColor(tpl["primary"]), spaceAfter=4))
            for q in unsectioned:
                q_num += 1
                story.append(KeepTogether(_render_answer_row(q, q_num)))
    else:
        for q in questions:
            q_num += 1
            story.append(KeepTogether(_render_answer_row(q, q_num)))

    # ── Footer ──────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor(tpl["border"]), spaceAfter=4))
    story.append(Paragraph(f"Answer Key • Generated by a4ai · {board} {subject} Class {class_grade} · {display_date}", styles["AkFooter"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_answer_key_docx(
    questions: List[dict],
    exam_title: str = "Test Paper",
    board: str = "CBSE",
    class_grade: str = "10",
    subject: str = "Science",
    include_explanations: bool = False,
    logo_base64: Optional[str] = None,
    paper_date: Optional[str] = None,
    template: str = "teal",
    teacher_name: Optional[str] = None,
    institute_name: Optional[str] = None,
    duration: Optional[str] = None,
    topic: Optional[str] = None,
) -> bytes:
    """
    Standalone Answer Key DOCX (separate file).
    """
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    tpl = _get_template(template)
    if tpl.get("layout_style") in ("cbse_exam_paper", "case_study_paper"):
        return _generate_docx_cbse_answer_key(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject,
            include_explanations=include_explanations,
            logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
            teacher_name=teacher_name, institute_name=institute_name,
            duration=duration, topic=topic,
        )

    if tpl.get("layout_style") != "institute_paper":
        tpl = _get_template("teal")

    doc = Document()

    # Page margins
    top_m, bottom_m, left_m, right_m = tpl["margins_cm"]
    for section in doc.sections:
        section.top_margin = Cm(top_m)
        section.bottom_margin = Cm(bottom_m)
        section.left_margin = Cm(left_m)
        section.right_margin = Cm(right_m)

    # Normal style default font
    normal_style = doc.styles["Normal"]
    normal_style.font.name = tpl.get("docx_font", "Calibri")
    normal_style.font.size = Pt(10.5)

    display_date = _format_date_for_display(paper_date)
    total_marks = sum(q.get("marks", 1) for q in questions)
    header_name = (institute_name or "").strip() or (exam_title or "Test Paper")

    # ── Institute name (title) ──
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(header_name)
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = _rgb(tpl["primary"])

    if institute_name and exam_title and exam_title.strip() and exam_title.strip().lower() != header_name.strip().lower():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(exam_title.strip())
        run.bold = True
        run.font.size = Pt(12)
        run.font.color.rgb = _rgb(tpl["primary"])

    # Class • Subject • Board
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"Class {class_grade}  •  {subject}  •  {board}")
    run.font.size = Pt(11)
    run.font.color.rgb = _rgb(tpl["secondary"])

    # Multi-color rule (or single primary bar for uniform templates)
    _institute_multicolor_rule_docx(doc, tpl)

    # "ANSWER KEY" big title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run("ANSWER KEY")
    run.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = _rgb(tpl["primary"])

    # ── Meta row (Teacher | Max Marks / Time / Date) using a 2-col table ──
    teacher_disp = (teacher_name or "").strip() or "______________"
    meta_tbl = doc.add_table(rows=1, cols=2)
    meta_tbl.autofit = True
    cell_l, cell_r = meta_tbl.rows[0].cells

    # Left cell: Teacher
    pl = cell_l.paragraphs[0]
    pl.paragraph_format.space_before = Pt(0)
    pl.paragraph_format.space_after = Pt(0)
    r = pl.add_run("Teacher: ")
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = _rgb(tpl["secondary"])
    r2 = pl.add_run(teacher_disp)
    r2.font.size = Pt(10)
    r2.font.color.rgb = _rgb(tpl["secondary"])

    # Right cell: Max Marks / Time / Date
    pr = cell_r.paragraphs[0]
    pr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pr.paragraph_format.space_before = Pt(0)
    pr.paragraph_format.space_after = Pt(0)

    def _add_meta_line(para, label, value, first=False):
        if not first:
            para.add_run("\n")
        r = para.add_run(f"{label}: ")
        r.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = _rgb(tpl["secondary"])
        r2 = para.add_run(str(value))
        r2.font.size = Pt(10)
        r2.font.color.rgb = _rgb(tpl["secondary"])

    _add_meta_line(pr, "Max Marks", total_marks, first=True)
    if duration and str(duration).strip():
        _add_meta_line(pr, "Time", str(duration).strip())
    _add_meta_line(pr, "Date", display_date)

    # Small spacer
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # ── Answers ──
    labels_lower = ["a", "b", "c", "d", "e", "f"]

    def _find_correct_letter(options, correct_answer):
        if not options or not correct_answer:
            return None
        ca = str(correct_answer).strip()
        m = re.match(r'^([A-Fa-f])[).\s]', ca)
        if m:
            return m.group(1).lower()
        ca_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', ca).strip().lower()
        for idx, opt in enumerate(options):
            opt_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', str(opt)).strip().lower()
            if opt_clean == ca_clean:
                return labels_lower[idx] if idx < len(labels_lower) else None
        if len(ca) == 1 and ca.lower() in labels_lower:
            return ca.lower()
        return None

    def _write_answer_row(q, q_num):
        options = q.get("options", [])
        correct = q.get("correctAnswer", q.get("correct_answer", ""))
        explanation = q.get("explanation", "") if include_explanations else ""
        sub_parts = _get_sub_parts(q)
        marking_scheme = _get_marking_scheme(q)
        model_answer = _get_model_answer(q)

        # Check for answer table (Accountancy etc.)
        raw_table = q.get("answer_table") or q.get("answerTable")
        if raw_table and isinstance(raw_table, dict):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(3)
            r = p.add_run(f"{q_num}. ")
            r.bold = True; r.font.size = Pt(11)
            _render_answer_table_docx(doc, raw_table)
            if model_answer and include_explanations:
                pma = doc.add_paragraph()
                r = pma.add_run("Model Answer: "); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = _rgb(tpl["primary"])
                r2 = pma.add_run(_latex_to_plain(model_answer)); r2.font.size = Pt(9.5); r2.font.color.rgb = _rgb(tpl["muted"])
            if explanation:
                pe = doc.add_paragraph()
                pe.paragraph_format.space_after = Pt(4)
                r = pe.add_run("Explanation: "); r.italic = True; r.font.size = Pt(9.5); r.font.color.rgb = _rgb(tpl["muted"])
                r2 = pe.add_run(_latex_to_plain(str(explanation))); r2.font.size = Pt(9.5); r2.font.color.rgb = _rgb(tpl["muted"])
            if marking_scheme and include_explanations:
                ms_text = _format_marking_scheme_text(marking_scheme)
                if ms_text:
                    pms = doc.add_paragraph()
                    r = pms.add_run("Marking Scheme: "); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = _rgb(tpl["primary"])
                    r2 = pms.add_run(ms_text); r2.font.size = Pt(9.5); r2.font.color.rgb = _rgb(tpl["muted"])
            return

        # Handle sub-parts (Case-study / multi-part)
        if sub_parts:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(3)
            r = p.add_run(f"{q_num}. ")
            r.bold = True; r.font.size = Pt(11)
            labels_roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
            for sp_idx, sp in enumerate(sub_parts):
                sp_label = sp.get("label") or (labels_roman[sp_idx] if sp_idx < len(labels_roman) else str(sp_idx + 1))
                sp_ans = sp.get("correctAnswer", sp.get("correct_answer", "")) or sp.get("answer", "") or "—"
                sp_marks = sp.get("marks")
                marks_str = f" [{sp_marks}M]" if sp_marks else ""
                ap = doc.add_paragraph()
                ap.paragraph_format.left_indent = Pt(14)
                r = ap.add_run(f"({sp_label}) {_latex_to_plain(str(sp_ans))}{marks_str}")
                r.font.size = Pt(10)
                r.font.color.rgb = _rgb(tpl["correct"])
            if model_answer and include_explanations:
                pma = doc.add_paragraph()
                r = pma.add_run("Model Answer: "); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = _rgb(tpl["primary"])
                r2 = pma.add_run(_latex_to_plain(model_answer)); r2.font.size = Pt(9.5); r2.font.color.rgb = _rgb(tpl["muted"])
            if explanation:
                pe = doc.add_paragraph()
                pe.paragraph_format.space_after = Pt(4)
                r = pe.add_run("Explanation: "); r.italic = True; r.font.size = Pt(9.5); r.font.color.rgb = _rgb(tpl["muted"])
                r2 = pe.add_run(_latex_to_plain(str(explanation))); r2.font.size = Pt(9.5); r2.font.color.rgb = _rgb(tpl["muted"])
            if marking_scheme and include_explanations:
                ms_text = _format_marking_scheme_text(marking_scheme)
                if ms_text:
                    pms = doc.add_paragraph()
                    r = pms.add_run("Marking Scheme: "); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = _rgb(tpl["primary"])
                    r2 = pms.add_run(ms_text); r2.font.size = Pt(9.5); r2.font.color.rgb = _rgb(tpl["muted"])
            return

        if options:
            letter = _find_correct_letter(options, correct)
            if letter:
                letter_idx = labels_lower.index(letter)
                opt_text = ""
                if letter_idx < len(options):
                    opt_text = re.sub(r'^[A-Fa-f][).\s]+\s*', '', str(options[letter_idx])).strip()
                ans_str = f"({letter}) {opt_text}" if opt_text else f"({letter})"
                ans_str = f"({letter}) {_latex_to_plain(opt_text)}" if opt_text else f"({letter})"
            else:
                ans_str = str(correct) if correct else "—"
        else:
            ans_str = str(correct) if correct else "—"

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(f"{q_num}. ")
        r.bold = True
        r.font.size = Pt(11)
        r2 = p.add_run(f" {ans_str}")
        r2.bold = True
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = _rgb(tpl["correct"])

        if model_answer and include_explanations:
            pma = doc.add_paragraph()
            r = pma.add_run("Model Answer: "); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = _rgb(tpl["primary"])
            r2 = pma.add_run(_latex_to_plain(model_answer)); r2.font.size = Pt(9.5); r2.font.color.rgb = _rgb(tpl["muted"])

        if explanation:
            pe = doc.add_paragraph()
            pe.paragraph_format.space_after = Pt(4)
            r = pe.add_run("Explanation: ")
            r.italic = True
            r.font.size = Pt(9.5)
            r.font.color.rgb = _rgb(tpl["muted"])
            r2 = pe.add_run(str(explanation))
            r2.font.size = Pt(9.5)
            r2.font.color.rgb = _rgb(tpl["muted"])

        if marking_scheme and include_explanations:
            ms_text = _format_marking_scheme_text(marking_scheme)
            if ms_text:
                pms = doc.add_paragraph()
                r = pms.add_run("Marking Scheme: "); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = _rgb(tpl["primary"])
                r2 = pms.add_run(ms_text); r2.font.size = Pt(9.5); r2.font.color.rgb = _rgb(tpl["muted"])

    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None
    q_num = 0

    if has_sec:
        grouped = _group_by_section(questions)
        last_title = None
        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue
            meta = sec_meta_dict.get(sec_key, {})
            current_title = meta.get("title", sec_key)
            if current_title != last_title:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(2)
                r = p.add_run(current_title)
                r.bold = True
                r.font.size = Pt(11.5)
                r.font.color.rgb = _rgb(tpl["primary"])
                subtitle = meta.get("subtitle", "")
                if subtitle:
                    r2 = p.add_run(f"  {subtitle}")
                    r2.font.size = Pt(9)
                    r2.font.color.rgb = _rgb(tpl["muted"])
                last_title = current_title

            main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
            or_qs = [q for q in sec_qs if q.get("_is_or", False)]
            or_queue = list(or_qs)

            for q in main_qs:
                q_num += 1
                _write_answer_row(q, q_num)
                if or_queue:
                    or_q = or_queue.pop(0)
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = p.add_run("OR")
                    r.italic = True
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = _rgb(tpl["muted"])
                    _write_answer_row(or_q, q_num)
                    _write_answer_row(or_q, f"{q_num} (OR)")

            for or_q in or_queue:
                q_num += 1
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run("OR")
                r.italic = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = _rgb(tpl["muted"])
                _write_answer_row(or_q, q_num)
                _write_answer_row(or_q, f"{q_num} (OR)")

        unsectioned = grouped.get("NONE", [])
        if unsectioned:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            r = p.add_run("Additional Questions")
            r.bold = True
            r.font.size = Pt(11.5)
            r.font.color.rgb = _rgb(tpl["primary"])
            for q in unsectioned:
                q_num += 1
                _write_answer_row(q, q_num)
    else:
        for q in questions:
            q_num += 1
            _write_answer_row(q, q_num)

    # Footer
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Answer Key • Generated by a4ai · {board} {subject} Class {class_grade} · {display_date}")
    r.font.size = Pt(8)
    r.font.color.rgb = _rgb(tpl["light_muted"])

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════
# CBSE EXAM PAPER & ANSWER KEY LAYOUT — Exact Class 12 Physics format
# ═══════════════════════════════════════════════════════════════════════

def _build_cbse_answer_key_elements(
    questions, exam_title, board, class_grade, subject, tpl, styles, W,
    include_explanations=False,
):
    """
    Constructs ReportLab flowables for the CBSE Exam Answer Key / Marking Scheme.
    Matches Page 3 of the reference format:
      - Centered coral ANSWER KEY / MARKING SCHEME title
      - Subtitle: Class X Subject — Test Paper: Title
      - Solid accent section bars: Section A, Section B, etc.
      - Bold coral question numbers: Q1., Q2., ...
      - Sub-part answers (i), (ii), etc. for case studies
      - Sub-footer: Generated with a4ai • a4ai.in
    """
    from reportlab.lib.colors import HexColor, white
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, HRFlowable

    accent = tpl.get("accent", "#FF7043")
    elements = []

    # Header
    elements.append(Paragraph(
        f'<font color="{accent}"><b>ANSWER KEY / MARKING SCHEME</b></font>',
        styles["AkTitle"],
    ))
    class_num = re.sub(r'[^0-9]', '', str(class_grade)) or class_grade
    elements.append(Paragraph(
        f"Class {class_num} {subject.title()} — Test Paper: {exam_title}",
        styles["AkSubTitle"],
    ))
    elements.append(Spacer(1, 3))
    elements.append(HRFlowable(width="100%", thickness=1.0, color=HexColor(accent), spaceAfter=4))

    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None

    def _ak_bar(title):
        bar_tbl = Table(
            [[Paragraph(f"<b>{title}</b>", styles["AkSecBarText"])]],
            colWidths=[W],
            rowHeights=[14],
        )
        bar_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor(accent)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        return bar_tbl

    labels_lower = ["a", "b", "c", "d", "e", "f"]
    labels_roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]

    def _render_ak_item(q, q_label):
        sub_parts = _get_sub_parts(q)
        correct_answer = q.get("correctAnswer", q.get("correct_answer", ""))
        explanation = q.get("explanation", "")
        model_ans = _get_model_answer(q)
        options = q.get("options", [])

        if sub_parts:
            sp_texts = []
            for sp_idx, sp in enumerate(sub_parts):
                sp_label = sp.get("label") or (labels_roman[sp_idx] if sp_idx < len(labels_roman) else str(sp_idx + 1))
                sp_ans = sp.get("correctAnswer", sp.get("correct_answer", "")) or sp.get("answer", "") or "—"
                sp_clean = _latex_to_paragraph(str(sp_ans).strip())
                sp_texts.append(f"<b>({sp_label})</b> {sp_clean}")
            ans_full = " &nbsp;&nbsp; ".join(sp_texts)
            return Paragraph(f'<b><font color="{accent}">Q{q_label}.</font></b> &nbsp; {ans_full}', styles["AkItem"])
        elif options:
            ca = str(correct_answer).strip()
            letter = ""
            opt_text = ca
            for o_idx, opt in enumerate(options):
                lbl = labels_lower[o_idx] if o_idx < len(labels_lower) else str(o_idx + 1)
                opt_c = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_paragraph(opt)).strip()
                if ca.upper().startswith(lbl.upper()) or opt.strip() == ca:
                    letter = lbl
                    opt_text = opt_c
                    break
            if not letter and ca and len(ca) <= 2 and ca.lower() in labels_lower:
                letter = ca.lower()
                idx_found = labels_lower.index(letter)
                if idx_found < len(options):
                    opt_text = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_paragraph(options[idx_found])).strip()

            prefix = f"({letter}) {opt_text}" if letter else opt_text
            clean_exp = _latex_to_paragraph(explanation).strip() if explanation else ""
            if clean_exp and not clean_exp.startswith("(") and len(clean_exp) < 130:
                ans_full = f"{prefix} &nbsp;—&nbsp; {clean_exp}"
            else:
                ans_full = prefix
            return Paragraph(f'<b><font color="{accent}">Q{q_label}.</font></b> &nbsp; {ans_full}', styles["AkItem"])
        else:
            main_ans = _latex_to_paragraph(model_ans or correct_answer or "")
            clean_exp = _latex_to_paragraph(explanation).strip() if (include_explanations and explanation) else ""
            if clean_exp and clean_exp != main_ans and len(clean_exp) < 140:
                ans_full = f"{main_ans} &nbsp;—&nbsp; {clean_exp}"
            else:
                ans_full = main_ans
            return Paragraph(f'<b><font color="{accent}">Q{q_label}.</font></b> &nbsp; {ans_full}', styles["AkItem"])

    q_num = 0
    if has_sec:
        grouped = _group_by_section(questions)
        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue
            meta = sec_meta_dict.get(sec_key, {})
            title = meta.get("title", f"Section {sec_key}")
            if elements and len(elements) > 4:
                elements.append(Spacer(1, 3))
            elements.append(_ak_bar(title))
            elements.append(Spacer(1, 2))

            main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
            or_qs = [q for q in sec_qs if q.get("_is_or", False)]
            or_queue = list(or_qs)

            for q in main_qs:
                q_num += 1
                elements.append(_render_ak_item(q, str(q_num)))
                if or_queue:
                    or_q = or_queue.pop(0)
                    elements.append(_render_ak_item(or_q, f"{q_num} (OR)"))

            for or_q in or_queue:
                q_num += 1
                elements.append(_render_ak_item(or_q, f"{q_num} (OR)"))

        unsectioned = grouped.get("NONE", [])
        if unsectioned:
            elements.append(Spacer(1, 3))
            elements.append(_ak_bar("Additional Questions"))
            elements.append(Spacer(1, 2))
            for q in unsectioned:
                q_num += 1
                elements.append(_render_ak_item(q, str(q_num)))
    else:
        for q in questions:
            q_num += 1
            elements.append(_render_ak_item(q, str(q_num)))

    elements.append(Spacer(1, 6))
    elements.append(Paragraph("<i>Generated with a4ai &nbsp;•&nbsp; a4ai.in</i>", styles["Footer"]))
    return elements


def _generate_pdf_cbse_answer_key(
    questions, exam_title, board, class_grade, subject,
    include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    """Standalone CBSE Exam Answer Key PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.colors import HexColor, white
    from reportlab.platypus import SimpleDocTemplate

    top_m, bottom_m, left_m, right_m = tpl.get("margins_cm", (1.0, 1.0, 1.58, 1.58))
    W = A4[0] - (left_m + right_m) * cm
    accent = tpl.get("accent", "#FF7043")
    primary = tpl.get("primary", "#1F2937")
    muted = tpl.get("muted", "#6B7280")
    fb = tpl["font_body"]
    fbd = tpl["font_bold"]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=top_m * cm, bottomMargin=bottom_m * cm,
        leftMargin=left_m * cm, rightMargin=right_m * cm,
    )
    styles = getSampleStyleSheet()
    ak_styles = {
        "AkTitle":      dict(parent=styles["Title"], fontSize=13.0, leading=16, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor(accent), fontName=fbd),
        "AkSubTitle":   dict(parent=styles["Normal"], fontSize=8.5, leading=11, spaceAfter=3, alignment=TA_CENTER, textColor=HexColor(muted), fontName=fb),
        "AkSecBarText": dict(parent=styles["Normal"], fontSize=9.0, leading=11, alignment=TA_LEFT, textColor=white, fontName=fbd),
        "AkItem":       dict(parent=styles["Normal"], fontSize=8.0, leading=10.5, spaceBefore=1.5, spaceAfter=1.5, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fb),
        "Footer":       dict(parent=styles["Normal"], fontSize=6.5, alignment=TA_CENTER, textColor=HexColor(muted), fontName="Helvetica-Oblique"),
    }
    for name, props in ak_styles.items():
        try:
            styles.add(ParagraphStyle(name=name, **props))
        except KeyError:
            pass

    elements = _build_cbse_answer_key_elements(
        questions=questions, exam_title=exam_title, board=board,
        class_grade=class_grade, subject=subject, tpl=tpl,
        styles=styles, W=W, include_explanations=include_explanations,
    )
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def _generate_docx_cbse_answer_key(
    questions, exam_title, board, class_grade, subject,
    include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    """Standalone CBSE Exam Answer Key DOCX."""
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    doc = Document()
    font_name = tpl.get("docx_font", "Calibri")
    accent = tpl.get("accent", "#FF7043")
    primary = tpl.get("primary", "#1F2937")
    muted = tpl.get("muted", "#6B7280")

    top_m, bottom_m, left_m, right_m = tpl.get("margins_cm", (1.0, 1.0, 1.58, 1.58))
    for section in doc.sections:
        section.top_margin = Cm(top_m)
        section.bottom_margin = Cm(bottom_m)
        section.left_margin = Cm(left_m)
        section.right_margin = Cm(right_m)

    class_num = re.sub(r'[^0-9]', '', str(class_grade)) or class_grade

    # Header
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run("ANSWER KEY / MARKING SCHEME")
    r.bold = True; r.font.size = Pt(13); r.font.name = font_name; r.font.color.rgb = _rgb(accent)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(2)
    r2 = p2.add_run(f"Class {class_num} {subject.title()} — Test Paper: {exam_title}")
    r2.font.size = Pt(8.5); r2.font.name = font_name; r2.font.color.rgb = _rgb(muted)

    p_hr = doc.add_paragraph()
    p_hr.paragraph_format.space_after = Pt(3)
    r_hr = p_hr.add_run("━" * 55)
    r_hr.font.size = Pt(6); r_hr.font.color.rgb = _rgb(accent)

    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None
    labels_lower = ["a", "b", "c", "d", "e", "f"]
    labels_roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]

    def _add_ak_bar(title):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
        cell = tbl.rows[0].cells[0]
        cell.text = ""
        cp = cell.paragraphs[0]
        cp.paragraph_format.space_before = Pt(1.5)
        cp.paragraph_format.space_after = Pt(1.5)
        cp.paragraph_format.left_indent = Pt(6)
        cr = cp.add_run(title)
        cr.bold = True; cr.font.size = Pt(9.0); cr.font.name = font_name
        cr.font.color.rgb = _rgb("#FFFFFF")
        tcPr = cell._element.get_or_add_tcPr()
        shd = tcPr.makeelement(qn('w:shd'), {
            qn('w:fill'): _hexnc(accent),
            qn('w:val'): 'clear',
        })
        tcPr.append(shd)

    def _add_ak_row(q, q_lbl):
        sub_parts = _get_sub_parts(q)
        correct_answer = q.get("correctAnswer", q.get("correct_answer", ""))
        explanation = q.get("explanation", "")
        model_ans = _get_model_answer(q)
        options = q.get("options", [])

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        r_num = p.add_run(f"Q{q_lbl}.  ")
        r_num.bold = True; r_num.font.size = Pt(8.0); r_num.font.name = font_name
        r_num.font.color.rgb = _rgb(accent)

        if sub_parts:
            sp_texts = []
            for sp_idx, sp in enumerate(sub_parts):
                sp_label = sp.get("label") or (labels_roman[sp_idx] if sp_idx < len(labels_roman) else str(sp_idx + 1))
                sp_ans = sp.get("correctAnswer", sp.get("correct_answer", "")) or sp.get("answer", "") or "—"
                sp_texts.append(f"({sp_label}) {_latex_to_plain(str(sp_ans).strip())}")
            r_ans = p.add_run("   ".join(sp_texts))
            r_ans.font.size = Pt(8.0); r_ans.font.name = font_name; r_ans.font.color.rgb = _rgb(primary)
        elif options:
            ca = str(correct_answer).strip()
            letter = ""
            opt_text = ca
            for o_idx, opt in enumerate(options):
                lbl = labels_lower[o_idx] if o_idx < len(labels_lower) else str(o_idx + 1)
                opt_c = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_plain(opt)).strip()
                if ca.upper().startswith(lbl.upper()) or opt.strip() == ca:
                    letter = lbl
                    opt_text = opt_c
                    break
            prefix = f"({letter}) {opt_text}" if letter else opt_text
            clean_exp = _latex_to_plain(explanation).strip() if explanation else ""
            full_txt = f"{prefix}  —  {clean_exp}" if (clean_exp and len(clean_exp) < 130) else prefix
            r_ans = p.add_run(full_txt)
            r_ans.font.size = Pt(8.0); r_ans.font.name = font_name; r_ans.font.color.rgb = _rgb(primary)
        else:
            main_ans = _latex_to_plain(model_ans or correct_answer or "")
            clean_exp = _latex_to_plain(explanation).strip() if (include_explanations and explanation) else ""
            full_txt = f"{main_ans}  —  {clean_exp}" if (clean_exp and clean_exp != main_ans and len(clean_exp) < 140) else main_ans
            r_ans = p.add_run(full_txt)
            r_ans.font.size = Pt(8.0); r_ans.font.name = font_name; r_ans.font.color.rgb = _rgb(primary)

    q_num = 0
    if has_sec:
        grouped = _group_by_section(questions)
        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue
            meta = sec_meta_dict.get(sec_key, {})
            title = meta.get("title", f"Section {sec_key}")
            _add_ak_bar(title)
            main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
            or_qs = [q for q in sec_qs if q.get("_is_or", False)]
            or_queue = list(or_qs)
            for q in main_qs:
                q_num += 1
                _add_ak_row(q, str(q_num))
                if or_queue:
                    or_q = or_queue.pop(0)
                    _add_ak_row(or_q, f"{q_num} (OR)")
            for or_q in or_queue:
                q_num += 1
                _add_ak_row(or_q, f"{q_num} (OR)")
    else:
        for q in questions:
            q_num += 1
            _add_ak_row(q, str(q_num))

    pf = doc.add_paragraph()
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.paragraph_format.space_before = Pt(8)
    rf = pf.add_run("Generated with a4ai • a4ai.in")
    rf.italic = True; rf.font.size = Pt(6.5); rf.font.name = font_name; rf.font.color.rgb = _rgb(muted)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# CBSE EXAM PAPER LAYOUT — Matches reference Class 12 Physics format
# ═══════════════════════════════════════════════════════════════════════

def _generate_pdf_cbse_exam(
    questions, exam_title, board, class_grade, subject,
    include_answers, include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    """
    CBSE Exam Paper layout — matches Class 12 Physics reference format.
    Features:
      - a4ai branding (clean, bold accent color, no extra slogans)
      - CLASS X — SUBJECT (large bold)
      - Test Paper: Title (accent color)
      - Topic/chapter subtitle & CBSE pattern
      - Name / Class / Roll No blanks
      - Time / Maximum Marks row (auto-formatted duration)
      - Bulleted General Instructions
      - Solid colored SECTION bars with bold white text
      - Inline [marks M] in accent color
      - 2-column MCQ options with accent-colored (a)/(b) labels
      - Case-study passage in shaded rounded box with (i)-(iv) sub-parts
      - "— End of Question Paper —" footer
      - Appended Answer Key matching Page 3 of reference format
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.colors import HexColor, white
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, KeepTogether, Image as RLImage,
    )

    buffer = io.BytesIO()
    top_m, bottom_m, left_m, right_m = tpl.get("margins_cm", (1.0, 1.0, 1.58, 1.58))
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=top_m * cm, bottomMargin=bottom_m * cm,
        leftMargin=left_m * cm, rightMargin=right_m * cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - (left_m + right_m) * cm
    fb = tpl["font_body"]
    fbd = tpl["font_bold"]
    accent = tpl.get("accent", "#FF7043")
    primary = tpl.get("primary", "#1F2937")
    muted = tpl.get("muted", "#6B7280")
    secondary = tpl.get("secondary", "#374151")

    # ── Styles ──────────────────────────────────────────────────────
    cst = {
        "Tagline":        dict(parent=styles["Normal"], fontSize=12, leading=15, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor(accent), fontName=fbd),
        "BigClass":       dict(parent=styles["Title"], fontSize=14, leading=17, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor(primary), fontName=fbd),
        "TestTitle":      dict(parent=styles["Normal"], fontSize=11, leading=14, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor(accent), fontName=fbd),
        "TopicLine":      dict(parent=styles["Normal"], fontSize=8.5, leading=11, spaceAfter=3, alignment=TA_CENTER, textColor=HexColor(muted), fontName=fb),
        "MetaLabel":      dict(parent=styles["Normal"], fontSize=8, leading=11, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fbd),
        "MetaValue":      dict(parent=styles["Normal"], fontSize=8, leading=11, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fb),
        "MetaRight":      dict(parent=styles["Normal"], fontSize=8, leading=11, alignment=TA_RIGHT, textColor=HexColor(primary), fontName=fbd),
        "InstHead":       dict(parent=styles["Normal"], fontSize=8, leading=11, spaceBefore=3, spaceAfter=1.5, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fbd),
        "InstBullet":     dict(parent=styles["Normal"], fontSize=7.5, leading=9.5, leftIndent=10, spaceBefore=0.5, spaceAfter=0.5, textColor=HexColor(primary), fontName=fb),
        "InstName":       dict(parent=styles["Title"], fontSize=15, leading=18, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor(primary), fontName=fbd),
        "SecBarText":     dict(parent=styles["Normal"], fontSize=9.0, leading=11, alignment=TA_LEFT, textColor=white, fontName=fbd),
        "SecSub":         dict(parent=styles["Normal"], fontSize=7.0, leading=9, spaceBefore=1, spaceAfter=2, alignment=TA_LEFT, textColor=HexColor(muted), fontName="Helvetica-Oblique"),
        "QText":          dict(parent=styles["Normal"], fontSize=8.0, leading=10.5, spaceBefore=2.5, spaceAfter=1, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fb),
        "PassagePrompt":  dict(parent=styles["Normal"], fontSize=8.0, leading=11, spaceBefore=2, spaceAfter=2, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fbd),
        "PassageText":    dict(parent=styles["Normal"], fontSize=8.0, leading=10.5, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fb),
        "SubQText":       dict(parent=styles["Normal"], fontSize=8.0, leading=10.5, leftIndent=10, spaceBefore=2, spaceAfter=1, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fb),
        "QBold":          dict(parent=styles["Normal"], fontSize=8.5, leading=12, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fbd),
        "Opt":            dict(parent=styles["Normal"], fontSize=7.5, leading=9.5, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fb),
        "OptCorrect":     dict(parent=styles["Normal"], fontSize=7.5, leading=9.5, alignment=TA_LEFT, textColor=HexColor(tpl["correct"]), fontName=fbd),
        "Ans":            dict(parent=styles["Normal"], fontSize=8.0, leading=11, textColor=HexColor(tpl["correct"]), fontName=fbd),
        "Expl":           dict(parent=styles["Normal"], fontSize=7.5, leading=10, textColor=HexColor(muted), fontName=fb),
        "OrText":         dict(parent=styles["Normal"], fontSize=9.0, leading=11, alignment=TA_CENTER, textColor=HexColor(secondary), fontName=fbd, spaceBefore=2, spaceAfter=2),
        "EndPaper":       dict(parent=styles["Normal"], fontSize=8.0, leading=11, alignment=TA_CENTER, textColor=HexColor(muted), fontName=fbd, spaceBefore=8),
        "Footer":         dict(parent=styles["Normal"], fontSize=6.5, alignment=TA_CENTER, textColor=HexColor(muted), fontName="Helvetica-Oblique"),
        "AkTitle":        dict(parent=styles["Title"], fontSize=13.0, leading=16, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor(accent), fontName=fbd),
        "AkSubTitle":     dict(parent=styles["Normal"], fontSize=8.5, leading=11, spaceAfter=3, alignment=TA_CENTER, textColor=HexColor(muted), fontName=fb),
        "AkSecBarText":   dict(parent=styles["Normal"], fontSize=9.0, leading=11, alignment=TA_LEFT, textColor=white, fontName=fbd),
        "AkItem":         dict(parent=styles["Normal"], fontSize=8.0, leading=10.5, spaceBefore=1.5, spaceAfter=1.5, alignment=TA_LEFT, textColor=HexColor(primary), fontName=fb),
    }
    for name, props in cst.items():
        try:
            styles.add(ParagraphStyle(name=name, **props))
        except KeyError:
            pass
    for alias, src in (("Option", "Opt"), ("AnswerLine", "Ans")):
        if alias not in styles:
            try:
                styles.add(ParagraphStyle(name=alias, parent=styles[src]))
            except Exception:
                pass

    story = []
    display_date = _format_date_for_display(paper_date)
    total_marks = sum(q.get("marks", 1) for q in questions)
    class_num = re.sub(r'[^0-9]', '', str(class_grade)) or class_grade

    # ── Top Left Logo & Header ──────────────────────────────────────
    logo_flowable = None
    if logo_base64:
        try:
            lb = logo_base64.split(",", 1)[1] if "," in logo_base64 else logo_base64
            img_data = io.BytesIO(base64.b64decode(lb))
            logo_flowable = RLImage(img_data, width=2.2 * cm, height=2.2 * cm)
            logo_flowable.hAlign = "LEFT"
        except Exception as e:
            logger.warning(f"Could not load logo in cbse exam pdf: {e}")
            logo_flowable = None

    topic_parts = []
    if topic and str(topic).strip():
        topic_parts.append(str(topic).strip())
    topic_parts.append(f"{board} Pattern")

    center_items = []
    if institute_name and institute_name.strip():
        center_items.append(Paragraph(f"<b>{institute_name.strip().upper()}</b>", styles["InstName"]))
        center_items.append(Spacer(1, 2))

    # a4ai brand mark (clean, to the point, no extra slogans)
    center_items.append(Paragraph(f'<b><font color="{accent}">a4ai</font></b>', styles["Tagline"]))
    center_items.append(Paragraph(f"CLASS {class_num} — {subject.upper()}", styles["BigClass"]))
    center_items.append(Paragraph(f"Test Paper: {exam_title}", styles["TestTitle"]))
    if topic_parts:
        center_items.append(Paragraph(" — ".join(topic_parts), styles["TopicLine"]))

    if logo_flowable:
        logo_col_w = 2.4 * cm
        center_col_w = W - (2 * logo_col_w)
        header_tbl = Table(
            [[logo_flowable, center_items, ""]],
            colWidths=[logo_col_w, center_col_w, logo_col_w],
        )
        header_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(header_tbl)
    else:
        for it in center_items:
            story.append(it)

    story.append(HRFlowable(width="100%", thickness=1.0, color=HexColor(accent), spaceAfter=3))

    # ── Name / Class / Roll No / Teacher blanks ─────────────────────
    if teacher_name and teacher_name.strip():
        name_row = Table(
            [[
                Paragraph("<b>Name:</b> ____________________", styles["MetaLabel"]),
                Paragraph("<b>Class/Sec:</b> __________", styles["MetaLabel"]),
                Paragraph("<b>Roll No:</b> _______", styles["MetaLabel"]),
                Paragraph(f"<b>Teacher:</b> {teacher_name.strip()}", styles["MetaLabel"]),
            ]],
            colWidths=[W * 0.35, W * 0.22, W * 0.18, W * 0.25],
        )
    else:
        name_row = Table(
            [[
                Paragraph("<b>Name:</b> ____________________", styles["MetaLabel"]),
                Paragraph("<b>Class/Sec:</b> __________", styles["MetaLabel"]),
                Paragraph("<b>Roll No:</b> _______", styles["MetaLabel"]),
            ]],
            colWidths=[W * 0.44, W * 0.31, W * 0.25],
        )
    name_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(name_row)

    # ── Time / Maximum Marks (auto-formatted duration) ──────────────
    if not duration or str(duration).strip().lower() in ("none", "as per schedule", ""):
        if total_marks <= 25:
            dur_str = "1 Hour"
        elif total_marks <= 40:
            dur_str = "1½ Hours"
        elif total_marks <= 60:
            dur_str = "2 Hours"
        else:
            dur_str = "3 Hours"
    else:
        dur_str = str(duration).strip()

    time_row = Table(
        [[
            Paragraph(f"<b>Time Allowed:</b> {dur_str}", styles["MetaLabel"]),
            Paragraph(f"<b>Maximum Marks:</b> {total_marks}", styles["MetaRight"]),
        ]],
        colWidths=[W * 0.55, W * 0.45],
    )
    time_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(time_row)
    story.append(Spacer(1, 2))

    # ── General Instructions ────────────────────────────────────────
    story.append(Paragraph("<b>General Instructions:</b>", styles["InstHead"]))
    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None

    instructions = [
        f"All questions are compulsory. There are {len(questions)} questions in total"
        + (", divided into 5 Sections." if has_sec else "."),
    ]
    if has_sec and sec_meta_dict is CBSE_SECTIONS_META:
        grouped_temp = _group_by_section(questions)
        sec_items = list(sec_meta_dict.keys())
        curr_q = 1
        for sk in sec_items:
            sq = grouped_temp.get(sk, [])
            if not sq:
                continue
            cnt = len([q for q in sq if not q.get("_is_or", False)])
            meta = sec_meta_dict.get(sk, {})
            q_range = f"Q{curr_q}–Q{curr_q + cnt - 1}" if cnt > 1 else f"Q{curr_q}"
            curr_q += cnt
            marks_val = meta.get("marks", 1)
            if sk == "A":
                instructions.append(f"Section A: {q_range} are MCQs / Assertion-Reason of 1 mark each.")
            elif sk == "B":
                instructions.append(f"Section B: {q_range} are Very Short Answer questions of 2 marks each.")
            elif sk == "C":
                instructions.append(f"Section C: {q_range} are Short Answer questions of 3 marks each.")
            elif sk == "D":
                instructions.append(f"Section D: {q_range} are Long Answer questions of 5 marks each.")
            elif sk == "E":
                instructions.append(f"Section E: {q_range} is a Case-Study based question of {marks_val} marks (1 mark each sub-part).")
            else:
                subtitle_clean = meta.get("subtitle", "").strip("() ")
                instructions.append(f"Section {sk}: {q_range} are {subtitle_clean} ({marks_val} marks each).")
    elif has_sec:
        instructions.append("This paper has multiple sections as indicated.")

    instructions.append("Use of calculator is not permitted.")
    for inst in instructions:
        story.append(Paragraph(f"•  {inst}", styles["InstBullet"]))
    story.append(Spacer(1, 3))

    # ── Section bar helper ──────────────────────────────────────────
    def _section_bar(title_text):
        """Solid colored bar with white bold text — exact reference look."""
        bar_tbl = Table(
            [[Paragraph(f"<b>{title_text}</b>", styles["SecBarText"])]],
            colWidths=[W],
            rowHeights=[14],
        )
        bar_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor(accent)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        return bar_tbl

    labels_lower = ["a", "b", "c", "d", "e", "f"]

    def _render_q(q, q_num):
        elements = []
        raw_text = q.get("text", "")
        marks = q.get("marks", 1)
        marks_tag = f'<font color="{accent}"><b>[{marks} M]</b></font>'
        sub_parts = q.get("subParts") or q.get("sub_parts") or []

        question_table = _get_question_table(q)
        if question_table:
            raw_text = _strip_markdown_table_from_text(raw_text)

        segments = _split_text_and_tables(raw_text)
        first_text = ""
        for seg in segments:
            if seg["type"] == "text" and seg["content"]:
                first_text = _latex_to_paragraph(seg["content"])
                break
        if not first_text:
            first_text = _latex_to_paragraph(raw_text)

        # Handle sub-parts (Case-Study Questions with shaded passage box)
        if sub_parts:
            elements.append(Paragraph("<b><i>Read the passage below and answer the questions that follow:</i></b>", styles["PassagePrompt"]))
            passage_tbl = Table([[Paragraph(first_text, styles["PassageText"])]], colWidths=[W])
            passage_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F3F4F6")),
                ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(passage_tbl)
            elements.append(Spacer(1, 3))

            labels_roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
            for sp_idx, sp in enumerate(sub_parts):
                sp_text = sp.get("text", "")
                sp_marks = sp.get("marks", 1)
                sp_label = sp.get("label") or (labels_roman[sp_idx] if sp_idx < len(labels_roman) else str(sp_idx + 1))
                sp_marks_tag = f'<font color="{accent}"><b>[{sp_marks} M]</b></font>'

                elements.append(Paragraph(
                    f"<b>({sp_label})</b>  {_latex_to_paragraph(sp_text)}  {sp_marks_tag}",
                    styles["SubQText"],
                ))

                sp_options = sp.get("options", [])
                sp_correct = sp.get("correctAnswer", sp.get("correct_answer", ""))
                if sp_options:
                    cells = []
                    for idx, opt in enumerate(sp_options):
                        letter = labels_lower[idx] if idx < len(labels_lower) else str(idx + 1)
                        opt_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_paragraph(opt)).strip()
                        is_correct = bool(include_answers and sp_correct and (sp_correct.strip().upper().startswith(letter.upper()) or opt.strip() == sp_correct.strip()))
                        style = styles["OptCorrect"] if is_correct else styles["Opt"]
                        cells.append(Paragraph(f'<b><font color="{accent}">({letter})</font></b> {opt_clean}', style))
                    rows = []
                    for i in range(0, len(cells), 2):
                        left = cells[i]
                        right = cells[i + 1] if i + 1 < len(cells) else ""
                        rows.append([left, right])
                    if rows:
                        t = Table(rows, colWidths=[W * 0.5, W * 0.5])
                        t.setStyle(TableStyle([
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 0.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
                            ("LEFTPADDING", (0, 0), (-1, -1), 12),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]))
                        elements.append(t)

                if include_answers and include_explanations and sp_correct and not sp_options:
                    elements.append(Paragraph(
                        f"<b>Ans:</b> {_latex_to_paragraph(sp_correct)}",
                        styles["Ans"],
                    ))
        else:
            # Standard Question text with inline marks
            if first_text.strip():
                elements.append(Paragraph(
                    f"<b>Q{q_num}.</b>  {first_text}  {marks_tag}",
                    styles["QText"],
                ))
            else:
                elements.append(Paragraph(
                    f"<b>Q{q_num}.</b>  {marks_tag}",
                    styles["QText"],
                ))

            if question_table:
                elements.extend(_render_question_table_pdf(question_table, styles, W))

            image_url = _get_image_url(q)
            if image_url:
                elements.extend(_render_manual_question_image_pdf(image_url, W))

            first_skipped = False
            for seg in segments:
                if seg["type"] == "text":
                    if not first_skipped:
                        first_skipped = True
                        continue
                    content = _latex_to_paragraph(seg["content"])
                    if content:
                        elements.append(Paragraph(content, styles["QText"]))
                elif seg["type"] == "table":
                    if question_table:
                        continue
                    hdrs, rws = seg["content"]
                    elements.extend(_render_inline_table_pdf(hdrs, rws, styles, W, tpl))

            # Standard MCQ options — 2-column layout with colored (a)/(b) labels
            options = q.get("options", [])
            correct_answer = q.get("correctAnswer", q.get("correct_answer", ""))

            if options:
                cells = []
                for idx, opt in enumerate(options):
                    letter = labels_lower[idx] if idx < len(labels_lower) else str(idx + 1)
                    opt_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_paragraph(opt)).strip()
                    is_correct = False
                    if include_answers and correct_answer:
                        ca = correct_answer.strip()
                        if ca.upper().startswith(letter.upper()) or opt.strip() == ca.strip():
                            is_correct = True
                    style = styles["OptCorrect"] if is_correct else styles["Opt"]
                    cells.append(Paragraph(f'<b><font color="{accent}">({letter})</font></b> {opt_clean}', style))

                rows = []
                for i in range(0, len(cells), 2):
                    left = cells[i]
                    right = cells[i + 1] if i + 1 < len(cells) else ""
                    rows.append([left, right])
                if rows:
                    t = Table(rows, colWidths=[W * 0.5, W * 0.5])
                    t.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 0.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]))
                    elements.append(t)
            else:
                fmt = q.get("format", "mcq")
                if not include_answers:
                    gap = {"short_answer": 18, "long_answer": 40,
                           "journal_entry": 44, "ledger": 44, "trial_balance": 44,
                           "image": 18}.get(fmt, 0)
                    if gap:
                        elements.append(Spacer(1, gap))

            if include_answers and include_explanations:
                raw_table = q.get("answer_table") or q.get("answerTable")
                if raw_table and isinstance(raw_table, dict):
                    elements.extend(_render_answer_table_pdf(raw_table, styles, W))
                elif not options:
                    ans = _latex_to_paragraph(correct_answer)
                    elements.append(Paragraph(f"<b>Ans:</b> {ans}", styles["Ans"]))

            model_ans = _get_model_answer(q)
            if include_answers and include_explanations and model_ans:
                elements.append(Paragraph(f"<b>Model Answer:</b> {_latex_to_paragraph(model_ans)}", styles["Ans"]))

            if include_explanations:
                exp = _latex_to_paragraph(q.get("explanation", ""))
                if exp:
                    elements.append(Paragraph(f"<b>Explanation:</b> {exp}", styles["Expl"]))

            marking_sch = _get_marking_scheme(q)
            if include_explanations and marking_sch:
                ms_text = _format_marking_scheme_text(marking_sch)
                if ms_text:
                    elements.append(Paragraph(f"<b>Marking Scheme:</b> {_latex_to_paragraph(ms_text)}", styles["Expl"]))

        return elements

    # ── Body — render sections ──────────────────────────────────────
    q_num = 0

    if has_sec:
        grouped = _group_by_section(questions)
        last_section_title = None

        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue

            meta = sec_meta_dict.get(sec_key, {})
            current_title = meta.get("title", f"Section {sec_key}")
            subtitle = (meta.get("subtitle", "") or "").strip("() ")

            if current_title != last_section_title:
                bar_text = f"{current_title.upper()} — {subtitle}" if subtitle else current_title.upper()
                if story and len(story) > 8:
                    story.append(Spacer(1, 3))
                story.append(_section_bar(bar_text))
                last_section_title = current_title

            # Section subtitle with dynamic question range (e.g. 1 Mark Each (Q1–5))
            main_count = len([q for q in sec_qs if not q.get("_is_or", False)])
            start_q = q_num + 1
            end_q = q_num + main_count
            q_range = f"(Q{start_q}–{end_q})" if end_q > start_q else f"(Q{start_q})"

            marks_desc = meta.get("marks_per_q") or meta.get("marks")
            if sec_key == "E" or "case" in current_title.lower():
                tot_m = meta.get("marks", sum(q.get("marks", 4) for q in sec_qs))
                sub_label = f"{tot_m} Marks {q_range}"
            elif marks_desc:
                sub_label = f"{marks_desc} Mark{'s' if str(marks_desc) != '1' else ''} Each {q_range}"
            else:
                sub_label = q_range
            story.append(Paragraph(sub_label, styles["SecSub"]))

            main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
            or_qs = [q for q in sec_qs if q.get("_is_or", False)]
            or_queue = list(or_qs)

            for q in main_qs:
                q_num += 1
                story.append(KeepTogether(_render_q(q, q_num)))
                if or_queue:
                    or_q = or_queue.pop(0)
                    story.append(Paragraph("OR", styles["OrText"]))
                    story.append(KeepTogether(_render_q(or_q, q_num)))

            for or_q in or_queue:
                q_num += 1
                story.append(Paragraph("OR", styles["OrText"]))
                story.append(KeepTogether(_render_q(or_q, q_num)))

        unsectioned = grouped.get("NONE", [])
        if unsectioned:
            story.append(Spacer(1, 3))
            story.append(_section_bar("ADDITIONAL QUESTIONS"))
            for q in unsectioned:
                q_num += 1
                story.append(KeepTogether(_render_q(q, q_num)))
    else:
        for q in questions:
            q_num += 1
            story.append(KeepTogether(_render_q(q, q_num)))

    # ── End of Question Paper marker ───────────────────────────────
    story.append(Spacer(1, 4))
    story.append(Paragraph("———  End of Question Paper  ———", styles["EndPaper"]))
    story.append(Paragraph("<i>Generated with a4ai &nbsp;•&nbsp; a4ai.in</i>", styles["Footer"]))

    # ── Answer key page (appended to end of question paper) ─────────
    if include_answers and not include_explanations:
        story.append(PageBreak())
        ak_elements = _build_cbse_answer_key_elements(
            questions=questions, exam_title=exam_title, board=board,
            class_grade=class_grade, subject=subject, tpl=tpl,
            styles=styles, W=W, include_explanations=False,
        )
        story.extend(ak_elements)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _generate_docx_cbse_exam(
    questions, exam_title, board, class_grade, subject,
    include_answers, include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    """
    CBSE Exam Paper DOCX — matches the PDF layout.
    """
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    doc = Document()
    font_name = tpl.get("docx_font", "Calibri")
    accent = tpl.get("accent", "#FF7043")
    primary = tpl.get("primary", "#1F2937")
    muted = tpl.get("muted", "#6B7280")
    secondary = tpl.get("secondary", "#374151")

    try:
        ns = doc.styles["Normal"]
        ns.font.name = font_name
        ns.font.size = Pt(10)
    except Exception:
        pass

    top_m, bottom_m, left_m, right_m = tpl.get("margins_cm", (1.0, 1.0, 1.58, 1.58))
    for section in doc.sections:
        section.top_margin = Cm(top_m)
        section.bottom_margin = Cm(bottom_m)
        section.left_margin = Cm(left_m)
        section.right_margin = Cm(right_m)

    display_date = _format_date_for_display(paper_date)
    total_marks = sum(q.get("marks", 1) for q in questions)
    class_num = re.sub(r'[^0-9]', '', str(class_grade)) or class_grade
    labels_lower = ["a", "b", "c", "d", "e", "f"]

    def _c(text, size, color_hex, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.CENTER):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
        r.font.size = Pt(size)
        r.font.color.rgb = _rgb(color_hex)
        r.font.name = font_name
        return p

    # ── Header ──────────────────────────────────────────────────────
    topic_parts = []
    if topic and str(topic).strip():
        topic_parts.append(str(topic).strip())
    topic_parts.append(f"{board} Pattern")

    has_logo_docx = False
    if logo_base64:
        try:
            lb = logo_base64.split(",", 1)[1] if "," in logo_base64 else logo_base64
            img_bytes = io.BytesIO(base64.b64decode(lb))
            htbl = doc.add_table(rows=1, cols=2)
            htbl.autofit = False
            htbl.columns[0].width = Cm(2.4)
            htbl.columns[1].width = Cm(15.5)

            p_logo = htbl.rows[0].cells[0].paragraphs[0]
            p_logo.add_run().add_picture(img_bytes, width=Cm(2.2))

            p_text = htbl.rows[0].cells[1].paragraphs[0]
            p_text.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if institute_name and institute_name.strip():
                r = p_text.add_run(institute_name.strip().upper() + "\n")
                r.bold = True; r.font.size = Pt(15); r.font.name = font_name; r.font.color.rgb = _rgb(primary)
            r = p_text.add_run("a4ai\n")
            r.bold = True; r.font.size = Pt(11); r.font.name = font_name; r.font.color.rgb = _rgb(accent)
            r = p_text.add_run(f"CLASS {class_num} — {subject.upper()}\n")
            r.bold = True; r.font.size = Pt(14); r.font.name = font_name; r.font.color.rgb = _rgb(primary)
            r = p_text.add_run(f"Test Paper: {exam_title}\n")
            r.bold = True; r.font.size = Pt(11); r.font.name = font_name; r.font.color.rgb = _rgb(accent)
            if topic_parts:
                r = p_text.add_run(" — ".join(topic_parts))
                r.italic = True; r.font.size = Pt(8.5); r.font.name = font_name; r.font.color.rgb = _rgb(muted)
            has_logo_docx = True
        except Exception as e:
            logger.warning(f"Could not add logo in cbse exam docx: {e}")
            has_logo_docx = False

    if not has_logo_docx:
        if institute_name and institute_name.strip():
            _c(institute_name.strip().upper(), 15, primary, bold=True)
        _c("a4ai", 11, accent, bold=True)
        _c(f"CLASS {class_num} — {subject.upper()}", 14, primary, bold=True)
        _c(f"Test Paper: {exam_title}", 11, accent, bold=True)
        if topic_parts:
            _c(" — ".join(topic_parts), 8.5, muted, italic=True)

    # HR
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("━" * 55)
    r.font.size = Pt(6); r.font.color.rgb = _rgb(accent)

    # Name / Class / Roll No / Teacher
    if teacher_name and teacher_name.strip():
        meta_tbl = doc.add_table(rows=1, cols=4)
        for i, lbl in enumerate(["Name: ____________________", "Class/Sec: __________", "Roll No: _______", f"Teacher: {teacher_name.strip()}"]):
            cell = meta_tbl.rows[0].cells[i]
            cell.text = ""
            cp = cell.paragraphs[0]
            cr = cp.add_run(lbl)
            cr.font.size = Pt(8); cr.font.name = font_name
            if ":" in lbl:
                cr.bold = True
    else:
        meta_tbl = doc.add_table(rows=1, cols=3)
        for i, lbl in enumerate(["Name: ____________________", "Class/Sec: __________", "Roll No: _______"]):
            cell = meta_tbl.rows[0].cells[i]
            cell.text = ""
            cp = cell.paragraphs[0]
            cr = cp.add_run(lbl)
            cr.font.size = Pt(8); cr.font.name = font_name
            if ":" in lbl:
                cr.bold = True
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Time / Maximum Marks
    if not duration or str(duration).strip().lower() in ("none", "as per schedule", ""):
        if total_marks <= 25:
            dur_str = "1 Hour"
        elif total_marks <= 40:
            dur_str = "1½ Hours"
        elif total_marks <= 60:
            dur_str = "2 Hours"
        else:
            dur_str = "3 Hours"
    else:
        dur_str = str(duration).strip()

    time_tbl = doc.add_table(rows=1, cols=2)
    time_tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    p_time = time_tbl.rows[0].cells[0].paragraphs[0]
    r = p_time.add_run(f"Time Allowed: {dur_str}")
    r.bold = True; r.font.size = Pt(8); r.font.name = font_name

    p_marks = time_tbl.rows[0].cells[1].paragraphs[0]
    p_marks.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p_marks.add_run(f"Maximum Marks: {total_marks}")
    r.bold = True; r.font.size = Pt(8); r.font.name = font_name

    # General Instructions
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run("General Instructions:")
    r.bold = True; r.font.size = Pt(8); r.font.name = font_name

    sec_order, sec_meta_dict = _get_section_order(questions)
    has_sec = sec_order is not None

    instructions = [
        f"All questions are compulsory. There are {len(questions)} questions in total"
        + (", divided into 5 Sections." if has_sec else "."),
    ]
    if has_sec and sec_meta_dict is CBSE_SECTIONS_META:
        grouped_temp = _group_by_section(questions)
        sec_items = list(sec_meta_dict.keys())
        curr_q = 1
        for sk in sec_items:
            sq = grouped_temp.get(sk, [])
            if not sq:
                continue
            cnt = len([q for q in sq if not q.get("_is_or", False)])
            meta = sec_meta_dict.get(sk, {})
            q_range = f"Q{curr_q}–Q{curr_q + cnt - 1}" if cnt > 1 else f"Q{curr_q}"
            curr_q += cnt
            marks_val = meta.get("marks", 1)
            if sk == "A":
                instructions.append(f"Section A: {q_range} are MCQs / Assertion-Reason of 1 mark each.")
            elif sk == "B":
                instructions.append(f"Section B: {q_range} are Very Short Answer questions of 2 marks each.")
            elif sk == "C":
                instructions.append(f"Section C: {q_range} are Short Answer questions of 3 marks each.")
            elif sk == "D":
                instructions.append(f"Section D: {q_range} are Long Answer questions of 5 marks each.")
            elif sk == "E":
                instructions.append(f"Section E: {q_range} is a Case-Study based question of {marks_val} marks (1 mark each sub-part).")
            else:
                subtitle_clean = meta.get("subtitle", "").strip("() ")
                instructions.append(f"Section {sk}: {q_range} are {subtitle_clean} ({marks_val} marks each).")
    elif has_sec:
        instructions.append("This paper has multiple sections as indicated.")

    instructions.append("Use of calculator is not permitted.")
    for inst in instructions:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(10)
        p.paragraph_format.space_after = Pt(0.5)
        r = p.add_run(f"•  {inst}")
        r.font.size = Pt(7.5); r.font.name = font_name; r.font.color.rgb = _rgb(primary)

    # Section bar helper
    def _add_section_bar(text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
        cell = tbl.rows[0].cells[0]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(1.5)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.left_indent = Pt(6)
        r = p.add_run(text)
        r.bold = True; r.font.size = Pt(9.0); r.font.name = font_name
        r.font.color.rgb = _rgb("#FFFFFF")
        tcPr = cell._element.get_or_add_tcPr()
        shd = tcPr.makeelement(qn('w:shd'), {
            qn('w:fill'): _hexnc(accent),
            qn('w:val'): 'clear',
        })
        tcPr.append(shd)

    def _add_q(q, q_num):
        raw_text = q.get("text", "")
        marks = q.get("marks", 1)
        sub_parts = q.get("subParts") or q.get("sub_parts") or []

        question_table = _get_question_table(q)
        if question_table:
            raw_text = _strip_markdown_table_from_text(raw_text)

        segments = _split_text_and_tables(raw_text)
        first_text = ""
        for seg in segments:
            if seg["type"] == "text" and seg["content"]:
                first_text = _latex_to_plain(seg["content"])
                break
        if not first_text:
            first_text = _latex_to_plain(raw_text)

        if sub_parts:
            # Case Study
            pp = doc.add_paragraph()
            pp.paragraph_format.space_before = Pt(2)
            pp.paragraph_format.space_after = Pt(1)
            pr = pp.add_run("Read the passage below and answer the questions that follow:")
            pr.bold = True; pr.italic = True; pr.font.size = Pt(8.0); pr.font.name = font_name

            ptbl = doc.add_table(rows=1, cols=1)
            ptbl.alignment = WD_TABLE_ALIGNMENT.LEFT
            pcell = ptbl.rows[0].cells[0]
            pcell.text = ""
            cp = pcell.paragraphs[0]
            cp.paragraph_format.space_before = Pt(3)
            cp.paragraph_format.space_after = Pt(3)
            cp.paragraph_format.left_indent = Pt(6)
            cr = cp.add_run(first_text)
            cr.font.size = Pt(8.0); cr.font.name = font_name
            tcPr = pcell._element.get_or_add_tcPr()
            shd = tcPr.makeelement(qn('w:shd'), {
                qn('w:fill'): "F3F4F6",
                qn('w:val'): 'clear',
            })
            tcPr.append(shd)

            labels_roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
            for sp_idx, sp in enumerate(sub_parts):
                sp_text = sp.get("text", "")
                sp_marks = sp.get("marks", 1)
                sp_label = sp.get("label") or (labels_roman[sp_idx] if sp_idx < len(labels_roman) else str(sp_idx + 1))

                sp_p = doc.add_paragraph()
                sp_p.paragraph_format.left_indent = Pt(10)
                sp_p.paragraph_format.space_before = Pt(1.5)
                sp_p.paragraph_format.space_after = Pt(0.5)
                sr = sp_p.add_run(f"({sp_label})  ")
                sr.bold = True; sr.font.size = Pt(8.0); sr.font.name = font_name
                st = sp_p.add_run(_latex_to_plain(sp_text) + "  ")
                st.font.size = Pt(8.0); st.font.name = font_name
                sm = sp_p.add_run(f"[{sp_marks} M]")
                sm.bold = True; sm.font.size = Pt(7.5); sm.font.name = font_name
                sm.font.color.rgb = _rgb(accent)
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(1)
            rq = p.add_run(f"Q{q_num}.  ")
            rq.bold = True; rq.font.size = Pt(8.0); rq.font.name = font_name
            rt = p.add_run(first_text + "  ")
            rt.font.size = Pt(8.0); rt.font.name = font_name
            rm = p.add_run(f"[{marks} M]")
            rm.bold = True; rm.font.size = Pt(7.5); rm.font.name = font_name
            rm.font.color.rgb = _rgb(accent)

            options = q.get("options", [])
            correct_answer = q.get("correctAnswer", q.get("correct_answer", ""))
            if options:
                n = len(options)
                nrows = (n + 1) // 2
                otbl = doc.add_table(rows=nrows, cols=2)
                otbl.alignment = WD_TABLE_ALIGNMENT.LEFT
                for idx, opt in enumerate(options):
                    letter = labels_lower[idx] if idx < len(labels_lower) else str(idx + 1)
                    opt_clean = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_plain(opt)).strip()
                    row_i, col_i = idx // 2, idx % 2
                    cell = otbl.rows[row_i].cells[col_i]
                    cell.text = ""
                    cp = cell.paragraphs[0]
                    cp.paragraph_format.left_indent = Pt(10)
                    r_lbl = cp.add_run(f"({letter})  ")
                    r_lbl.bold = True; r_lbl.font.size = Pt(7.5); r_lbl.font.name = font_name
                    r_lbl.font.color.rgb = _rgb(accent)
                    r_txt = cp.add_run(opt_clean)
                    r_txt.font.size = Pt(7.5); r_txt.font.name = font_name

    # ── Body ────────────────────────────────────────────────────────
    q_num = 0

    if has_sec:
        grouped = _group_by_section(questions)
        last_title = None
        for sec_key in sec_order:
            sec_qs = grouped.get(sec_key, [])
            if not sec_qs:
                continue
            meta = sec_meta_dict.get(sec_key, {})
            current_title = meta.get("title", f"Section {sec_key}")
            subtitle = (meta.get("subtitle", "") or "").strip("() ")

            if current_title != last_title:
                bar_text = f"{current_title.upper()} — {subtitle}" if subtitle else current_title.upper()
                _add_section_bar(bar_text)
                last_title = current_title

            marks_each = meta.get("marks", "?")
            start_q = q_num + 1
            main_count = len([q for q in sec_qs if not q.get("_is_or", False)])
            end_q = q_num + main_count
            q_range = f"(Q{start_q}–{end_q})" if end_q > start_q else f"(Q{start_q})"

            marks_desc = meta.get("marks_per_q") or meta.get("marks")
            if sec_key == "E" or "case" in current_title.lower():
                tot_m = meta.get("marks", sum(q.get("marks", 4) for q in sec_qs))
                sub_label = f"{tot_m} Marks {q_range}"
            elif marks_desc:
                sub_label = f"{marks_desc} Mark{'s' if str(marks_desc) != '1' else ''} Each {q_range}"
            else:
                sub_label = q_range

            p = doc.add_paragraph()
            r = p.add_run(sub_label)
            r.italic = True; r.font.size = Pt(7.0); r.font.color.rgb = _rgb(muted); r.font.name = font_name

            main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
            or_qs = [q for q in sec_qs if q.get("_is_or", False)]
            or_queue = list(or_qs)

            for q in main_qs:
                q_num += 1
                _add_q(q, q_num)
                if or_queue:
                    or_q = or_queue.pop(0)
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = p.add_run("OR")
                    r.bold = True; r.font.size = Pt(9); r.font.color.rgb = _rgb(secondary)
                    _add_q(or_q, q_num)

        unsectioned = grouped.get("NONE", [])
        if unsectioned:
            _add_section_bar("ADDITIONAL QUESTIONS")
            for q in unsectioned:
                q_num += 1
                _add_q(q, q_num)
    else:
        for q in questions:
            q_num += 1
            _add_q(q, q_num)

    # ── End of Question Paper marker ───────────────────────────────
    _c("———  End of Question Paper  ———", 8, muted, bold=True)
    _c("Generated with a4ai • a4ai.in", 6.5, muted, italic=True)

    # ── Appended Answer Key (if answers only, no explanations) ──────
    if include_answers and not include_explanations:
        doc.add_page_break()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run("ANSWER KEY / MARKING SCHEME")
        r.bold = True; r.font.size = Pt(13); r.font.name = font_name; r.font.color.rgb = _rgb(accent)

        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_after = Pt(2)
        r2 = p2.add_run(f"Class {class_num} {subject.title()} — Test Paper: {exam_title}")
        r2.font.size = Pt(8.5); r2.font.name = font_name; r2.font.color.rgb = _rgb(muted)

        p_hr = doc.add_paragraph()
        p_hr.paragraph_format.space_after = Pt(3)
        r_hr = p_hr.add_run("━" * 55)
        r_hr.font.size = Pt(6); r_hr.font.color.rgb = _rgb(accent)

        labels_roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]

        def _add_appended_ak_row(q, q_lbl):
            sub_parts = _get_sub_parts(q)
            correct_answer = q.get("correctAnswer", q.get("correct_answer", ""))
            explanation = q.get("explanation", "")
            model_ans = _get_model_answer(q)
            options = q.get("options", [])

            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r_num = p.add_run(f"Q{q_lbl}.  ")
            r_num.bold = True; r_num.font.size = Pt(8.0); r_num.font.name = font_name
            r_num.font.color.rgb = _rgb(accent)

            if sub_parts:
                sp_texts = []
                for sp_idx, sp in enumerate(sub_parts):
                    sp_label = sp.get("label") or (labels_roman[sp_idx] if sp_idx < len(labels_roman) else str(sp_idx + 1))
                    sp_ans = sp.get("correctAnswer", sp.get("correct_answer", "")) or sp.get("answer", "") or "—"
                    sp_texts.append(f"({sp_label}) {_latex_to_plain(str(sp_ans).strip())}")
                r_ans = p.add_run("   ".join(sp_texts))
                r_ans.font.size = Pt(8.0); r_ans.font.name = font_name; r_ans.font.color.rgb = _rgb(primary)
            elif options:
                ca = str(correct_answer).strip()
                letter = ""
                opt_text = ca
                for o_idx, opt in enumerate(options):
                    lbl = labels_lower[o_idx] if o_idx < len(labels_lower) else str(o_idx + 1)
                    opt_c = re.sub(r'^[A-Fa-f][).\s]+\s*', '', _latex_to_plain(opt)).strip()
                    if ca.upper().startswith(lbl.upper()) or opt.strip() == ca:
                        letter = lbl
                        opt_text = opt_c
                        break
                prefix = f"({letter}) {opt_text}" if letter else opt_text
                clean_exp = _latex_to_plain(explanation).strip() if explanation else ""
                full_txt = f"{prefix}  —  {clean_exp}" if (clean_exp and len(clean_exp) < 130) else prefix
                r_ans = p.add_run(full_txt)
                r_ans.font.size = Pt(8.0); r_ans.font.name = font_name; r_ans.font.color.rgb = _rgb(primary)
            else:
                main_ans = _latex_to_plain(model_ans or correct_answer or "")
                clean_exp = _latex_to_plain(explanation).strip() if (include_explanations and explanation) else ""
                full_txt = f"{main_ans}  —  {clean_exp}" if (clean_exp and clean_exp != main_ans and len(clean_exp) < 140) else main_ans
                r_ans = p.add_run(full_txt)
                r_ans.font.size = Pt(8.0); r_ans.font.name = font_name; r_ans.font.color.rgb = _rgb(primary)

        ak_num = 0
        if has_sec:
            grouped = _group_by_section(questions)
            for sec_key in sec_order:
                sec_qs = grouped.get(sec_key, [])
                if not sec_qs:
                    continue
                meta = sec_meta_dict.get(sec_key, {})
                title = meta.get("title", f"Section {sec_key}")
                _add_section_bar(title)
                main_qs = [q for q in sec_qs if not q.get("_is_or", False)]
                or_qs = [q for q in sec_qs if q.get("_is_or", False)]
                or_queue = list(or_qs)
                for q in main_qs:
                    ak_num += 1
                    _add_appended_ak_row(q, str(ak_num))
                    if or_queue:
                        or_q = or_queue.pop(0)
                        _add_appended_ak_row(or_q, f"{ak_num} (OR)")
                for or_q in or_queue:
                    ak_num += 1
                    _add_appended_ak_row(or_q, f"{ak_num} (OR)")
        else:
            for q in questions:
                ak_num += 1
                _add_appended_ak_row(q, str(ak_num))

        pf = doc.add_paragraph()
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.paragraph_format.space_before = Pt(8)
        rf = pf.add_run("Generated with a4ai • a4ai.in")
        rf.italic = True; rf.font.size = Pt(6.5); rf.font.name = font_name; rf.font.color.rgb = _rgb(muted)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# Premium Case-Study Paper Layout — PDF + DOCX
# (Delegates to CBSE Exam Paper layout to maintain Class 12 Physics format)
# ═══════════════════════════════════════════════════════════════════════

def _generate_pdf_case_study(
    questions, exam_title, board, class_grade, subject,
    include_answers, include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    """Delegates to CBSE Exam Paper layout matching Class 12 Physics format."""
    return _generate_pdf_cbse_exam(
        questions=questions, exam_title=exam_title, board=board,
        class_grade=class_grade, subject=subject,
        include_answers=include_answers, include_explanations=include_explanations,
        logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
        teacher_name=teacher_name, institute_name=institute_name,
        duration=duration, topic=topic,
    )


def _generate_docx_case_study(
    questions, exam_title, board, class_grade, subject,
    include_answers, include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    """Delegates to CBSE Exam Paper layout matching Class 12 Physics format."""
    return _generate_docx_cbse_exam(
        questions=questions, exam_title=exam_title, board=board,
        class_grade=class_grade, subject=subject,
        include_answers=include_answers, include_explanations=include_explanations,
        logo_base64=logo_base64, paper_date=paper_date, tpl=tpl,
        teacher_name=teacher_name, institute_name=institute_name,
        duration=duration, topic=topic,
    )


# ═══════════════════════════════════════════════════════════════════════
# Authentic CBSE Class 11 & 12 Accountancy Tabular Layout Engine
# ═══════════════════════════════════════════════════════════════════════

_ACCOUNTANCY_FONTS_REGISTERED = False
_ACCOUNTANCY_FONT_BODY = "Helvetica"
_ACCOUNTANCY_FONT_BOLD = "Helvetica-Bold"
_ACCOUNTANCY_HAS_TT = False


def _register_accountancy_fonts():
    global _ACCOUNTANCY_FONTS_REGISTERED, _ACCOUNTANCY_FONT_BODY, _ACCOUNTANCY_FONT_BOLD, _ACCOUNTANCY_HAS_TT
    if _ACCOUNTANCY_FONTS_REGISTERED:
        return _ACCOUNTANCY_FONT_BODY, _ACCOUNTANCY_FONT_BOLD, _ACCOUNTANCY_HAS_TT

    import os
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Try Windows Arial for native ₹ glyph support
    win_arial = "C:/Windows/Fonts/arial.ttf"
    win_arial_bd = "C:/Windows/Fonts/arialbd.ttf"
    if os.path.exists(win_arial) and os.path.exists(win_arial_bd):
        try:
            pdfmetrics.registerFont(TTFont("Arial", win_arial))
            pdfmetrics.registerFont(TTFont("Arial-Bold", win_arial_bd))
            _ACCOUNTANCY_FONT_BODY = "Arial"
            _ACCOUNTANCY_FONT_BOLD = "Arial-Bold"
            _ACCOUNTANCY_HAS_TT = True
            _ACCOUNTANCY_FONTS_REGISTERED = True
            return _ACCOUNTANCY_FONT_BODY, _ACCOUNTANCY_FONT_BOLD, _ACCOUNTANCY_HAS_TT
        except Exception as e:
            logger.warning(f"Could not register Windows Arial: {e}")

    # Try Linux DejaVuSans
    linux_dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    linux_dejavu_bd = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(linux_dejavu) and os.path.exists(linux_dejavu_bd):
        try:
            pdfmetrics.registerFont(TTFont("DejaVuSans", linux_dejavu))
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", linux_dejavu_bd))
            _ACCOUNTANCY_FONT_BODY = "DejaVuSans"
            _ACCOUNTANCY_FONT_BOLD = "DejaVuSans-Bold"
            _ACCOUNTANCY_HAS_TT = True
            _ACCOUNTANCY_FONTS_REGISTERED = True
            return _ACCOUNTANCY_FONT_BODY, _ACCOUNTANCY_FONT_BOLD, _ACCOUNTANCY_HAS_TT
        except Exception as e:
            logger.warning(f"Could not register Linux DejaVuSans: {e}")

    _ACCOUNTANCY_FONTS_REGISTERED = True
    return _ACCOUNTANCY_FONT_BODY, _ACCOUNTANCY_FONT_BOLD, _ACCOUNTANCY_HAS_TT


from reportlab.pdfgen import canvas


class AccountancyNumberedCanvas(canvas.Canvas):
    """
    Two-pass ReportLab canvas that draws:
    - 'Page X of Y' centered at 0.9 cm
    - Official CBSE Assessment Scheme Note centered at 0.5 cm
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor

        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(HexColor("#374151"))
            page_text = f"Page {self._pageNumber} of {num_pages}"
            self.drawCentredString(A4[0] / 2.0, 0.9 * cm, page_text)
            self.setFont("Helvetica", 6.5)
            self.setFillColor(HexColor("#6B7280"))
            note_text = "Please note that the assessment scheme of the academic session 2024-25 will continue in the current session i.e. 2025-26"
            self.drawCentredString(A4[0] / 2.0, 0.5 * cm, note_text)
            self.restoreState()
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)


def _format_acc_text(text: str, has_tt_font: bool = True) -> str:
    if not text:
        return ""
    cleaned = _latex_to_paragraph(str(text))
    if not has_tt_font:
        cleaned = cleaned.replace("₹", "Rs. ")
    return cleaned


def _build_accountancy_mcq_table(options, correct_answer, include_answers, col_w, styles, has_tt_font=True):
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.colors import HexColor

    if not options or len(options) < 2:
        return []

    labels = ["A", "B", "C", "D", "E", "F"]
    cleaned_options = []
    for i, opt in enumerate(options):
        cleaned = re.sub(r'^[A-Fa-f0-9][).\s\-]+\s*', '', str(opt)).strip()
        cleaned_options.append((labels[i] if i < len(labels) else str(i+1), cleaned, opt))

    is_long = any(len(c[1]) > 38 for c in cleaned_options)
    opt_style = styles["AccOpt"]
    opt_correct_style = styles["AccOptCorrect"]

    cells = []
    for lbl, text, raw_opt in cleaned_options:
        is_correct = False
        if include_answers and correct_answer:
            ca = str(correct_answer).strip().upper()
            if ca.startswith(lbl) or text.strip().lower() == str(correct_answer).strip().lower():
                is_correct = True
        st = opt_correct_style if is_correct else opt_style
        fmt_text = _format_acc_text(text, has_tt_font)
        cells.append(Paragraph(f"<b>{lbl}.</b>  {fmt_text}", st))

    if is_long:
        rows = [[c] for c in cells]
        t = Table(rows, colWidths=[col_w])
    else:
        rows = []
        for i in range(0, len(cells), 2):
            left = cells[i]
            right = cells[i+1] if i+1 < len(cells) else ""
            rows.append([left, right])
        t = Table(rows, colWidths=[col_w * 0.5, col_w * 0.5])

    t.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, HexColor('#374151')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor('#9CA3AF')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    return [Spacer(1, 3), t, Spacer(1, 4)]


def _render_accountancy_financial_table_pdf(headers, rows, col_w, styles, has_tt_font=True, caption=None):
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.colors import HexColor

    if not headers or not rows:
        return []

    num_cols = len(headers)
    cell_style = styles["AccTableBody"]
    cell_style_right = styles["AccTableBodyRight"]
    cell_style_bold_right = styles["AccTableBodyBoldRight"]
    header_style = styles["AccTableHead"]

    h_str = " ".join([str(h).lower() for h in headers])
    is_balance_sheet = ("liabilit" in h_str and "asset" in h_str) and num_cols == 4
    is_comparative = ("absolute change" in h_str or "percentage change" in h_str or "comparative" in h_str or "common size" in h_str)
    is_journal = ("particular" in h_str and ("debit" in h_str or "credit" in h_str or "l.f" in h_str))
    is_ledger = ("dr" in h_str or "cr" in h_str or num_cols == 8)

    if is_balance_sheet:
        col_widths = [col_w * 0.35, col_w * 0.15, col_w * 0.35, col_w * 0.15]
    elif is_comparative and num_cols == 5:
        col_widths = [col_w * 0.32, col_w * 0.17, col_w * 0.17, col_w * 0.17, col_w * 0.17]
    elif is_journal and num_cols == 5:
        col_widths = [col_w * 0.12, col_w * 0.48, col_w * 0.08, col_w * 0.16, col_w * 0.16]
    elif is_ledger:
        col_widths = [col_w*0.09, col_w*0.26, col_w*0.05, col_w*0.10, col_w*0.09, col_w*0.26, col_w*0.05, col_w*0.10]
    elif num_cols == 2:
        col_widths = [col_w * 0.60, col_w * 0.40]
    elif num_cols == 3:
        col_widths = [col_w * 0.50, col_w * 0.25, col_w * 0.25]
    elif num_cols == 4:
        col_widths = [col_w * 0.40, col_w * 0.20, col_w * 0.20, col_w * 0.20]
    else:
        col_widths = [col_w / num_cols] * num_cols

    col_widths = col_widths[:num_cols]

    # For wide tables (5+ columns or ledger), use compact font and padding
    is_wide = num_cols >= 5
    if is_wide:
        h_style = styles.get("AccTableHeadWide", header_style)
        c_style = styles.get("AccTableBodyWide", cell_style)
        c_style_r = styles.get("AccTableBodyWideRight", cell_style_right)
        c_style_br = styles.get("AccTableBodyWideBoldRight", cell_style_bold_right)
    else:
        h_style = header_style
        c_style = cell_style
        c_style_r = cell_style_right
        c_style_br = cell_style_bold_right

    table_data = [[Paragraph(f"<b>{_format_acc_text(str(h), has_tt_font)}</b>", h_style) for h in headers]]

    for r_idx, row in enumerate(rows):
        padded = (list(row) + [""] * num_cols)[:num_cols]
        is_last_row = (r_idx == len(rows) - 1)
        row_cells = []
        for c_idx, cell in enumerate(padded):
            val = str(cell).strip()
            fmt_val = _format_acc_text(val, has_tt_font)
            if is_balance_sheet:
                if c_idx in (1, 3):
                    st = c_style_br if is_last_row else c_style_r
                else:
                    st = c_style
            elif is_journal:
                if c_idx in (3, 4):
                    st = c_style_r
                elif c_idx == 1 and val.startswith("To "):
                    st = styles["AccTableIndented"]
                elif c_idx == 1 and (val.startswith("(") or "being" in val.lower()):
                    st = styles["AccTableNarration"]
                else:
                    st = c_style
            elif is_ledger:
                if c_idx in (3, 7):
                    st = c_style_r
                else:
                    st = c_style
            elif is_comparative:
                if c_idx >= 1:
                    st = c_style_r
                else:
                    st = c_style
            else:
                st = c_style
            row_cells.append(Paragraph(fmt_val, st))
        table_data.append(row_cells)

    pad_h = 2 if is_wide else 4
    pad_v = 1.5 if is_wide else 2.5

    t = Table(table_data, colWidths=col_widths, repeatRows=0)
    style_cmds = [
        ('BOX', (0,0), (-1,-1), 0.5, HexColor('#374151')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor('#9CA3AF')),
        ('BACKGROUND', (0,0), (-1,0), HexColor('#F9FAFB')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), pad_v),
        ('BOTTOMPADDING', (0,0), (-1,-1), pad_v),
        ('LEFTPADDING', (0,0), (-1,-1), pad_h),
        ('RIGHTPADDING', (0,0), (-1,-1), pad_h),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
    ]

    if is_balance_sheet:
        style_cmds.extend([
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
            ('LINEAFTER', (1, 0), (1, -1), 1.0, HexColor('#374151')),
        ])
        if len(rows) > 0:
            last = len(table_data) - 1
            style_cmds.append(('LINEABOVE', (0, last), (-1, last), 1.0, HexColor('#111827')))
            style_cmds.append(('LINEBELOW', (0, last), (-1, last), 1.5, HexColor('#111827')))
    elif is_journal:
        style_cmds.extend([
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),
            ('ALIGN', (3, 1), (4, -1), 'RIGHT'),
        ])
    elif is_ledger and num_cols >= 8:
        style_cmds.extend([
            ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
            ('ALIGN', (7, 1), (7, -1), 'RIGHT'),
            ('LINEAFTER', (3, 0), (3, -1), 1.5, HexColor('#111827')),
        ])

    t.setStyle(TableStyle(style_cmds))

    res = [Spacer(1, 4)]
    if caption:
        res.append(Paragraph(f"<b><i>{_format_acc_text(caption, has_tt_font)}</i></b>", styles["AccTableCaption"]))
        res.append(Spacer(1, 2))
    res.append(t)
    res.append(Spacer(1, 5))
    return res


def _generate_pdf_accountancy_exam(
    questions, exam_title, board, class_grade, subject,
    include_answers, include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    """
    Authentic CBSE Class 11 & 12 Accountancy (055) Tabular Question Paper PDF Generator.
    Matches the official 10-page CBSE Accountancy Sample Paper:
      - 3-column master table: S.No. (1.2 cm) | Question Content & Tables | Marks (1.2 cm)
      - Boxed MCQ options in a 2x2 grid (or 1x4 for long Assertion/Reason)
      - Internal choice OR questions rendered inside the question cell with centered bold OR
      - T-Shape Balance Sheets with double-line totals
      - Journal and Ledger tables with authentic financial rulings
      - Clear Part A and Part B Section Dividers
      - Page footer: Page X of Y with CBSE assessment scheme note
      - Appended Solution Key and Marking Scheme when include_answers=True
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.0 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.2 * cm, rightMargin=1.2 * cm,
    )

    W = A4[0] - 2.4 * cm
    col0_w = 1.2 * cm
    col2_w = 1.2 * cm
    col1_w = W - (col0_w + col2_w)

    fb, fbd, has_tt = _register_accountancy_fonts()

    styles = getSampleStyleSheet()

    cst = {
        "AccInstName":     dict(parent=styles["Title"], fontName=fbd, fontSize=14, leading=17, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor("#111827")),
        "AccMainTitle":    dict(parent=styles["Title"], fontName=fbd, fontSize=13, leading=16, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor("#111827")),
        "AccSubTitle":     dict(parent=styles["Normal"], fontName=fbd, fontSize=10.5, leading=14, spaceAfter=1, alignment=TA_CENTER, textColor=HexColor("#1F2937")),
        "AccClassLine":    dict(parent=styles["Normal"], fontName=fbd, fontSize=9.5, leading=13, spaceAfter=4, alignment=TA_CENTER, textColor=HexColor("#374151")),
        "AccTimeMeta":     dict(parent=styles["Normal"], fontName=fbd, fontSize=8.5, leading=11, alignment=TA_LEFT, textColor=HexColor("#111827")),
        "AccMarksMeta":    dict(parent=styles["Normal"], fontName=fbd, fontSize=8.5, leading=11, alignment=TA_RIGHT, textColor=HexColor("#111827")),
        "AccInstHeading":  dict(parent=styles["Normal"], fontName=fbd, fontSize=8.5, leading=11, alignment=TA_LEFT, spaceBefore=4, spaceAfter=2, textColor=HexColor("#111827")),
        "AccInstItem":     dict(parent=styles["Normal"], fontName=fb, fontSize=7.5, leading=9.5, leftIndent=14, firstLineIndent=-14, spaceBefore=0.5, spaceAfter=0.5, textColor=HexColor("#1F2937")),
        "AccPartHeader":   dict(parent=styles["Normal"], fontName=fbd, fontSize=9.5, leading=12, alignment=TA_CENTER, textColor=HexColor("#111827")),
        "AccSNo":          dict(parent=styles["Normal"], fontName=fbd, fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=HexColor("#111827")),
        "AccMarks":        dict(parent=styles["Normal"], fontName=fbd, fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=HexColor("#111827")),
        "AccQText":        dict(parent=styles["Normal"], fontName=fb, fontSize=8.0, leading=10.5, spaceBefore=1, spaceAfter=2, alignment=TA_LEFT, textColor=HexColor("#111827")),
        "AccOpt":          dict(parent=styles["Normal"], fontName=fb, fontSize=7.5, leading=9.5, alignment=TA_LEFT, textColor=HexColor("#1F2937")),
        "AccOptCorrect":   dict(parent=styles["Normal"], fontName=fbd, fontSize=7.5, leading=9.5, alignment=TA_LEFT, textColor=HexColor("#15803D")),
        "AccOr":           dict(parent=styles["Normal"], fontName=fbd, fontSize=9.0, leading=12, alignment=TA_CENTER, spaceBefore=4, spaceAfter=4, textColor=HexColor("#374151")),
        "AccTableHead":    dict(parent=styles["Normal"], fontName=fbd, fontSize=7.5, leading=9.5, alignment=TA_CENTER, textColor=HexColor("#111827")),
        "AccTableBody":    dict(parent=styles["Normal"], fontName=fb, fontSize=7.5, leading=9.5, alignment=TA_LEFT, textColor=HexColor("#111827")),
        "AccTableBodyRight": dict(parent=styles["Normal"], fontName=fb, fontSize=7.5, leading=9.5, alignment=TA_RIGHT, textColor=HexColor("#111827")),
        "AccTableBodyBoldRight": dict(parent=styles["Normal"], fontName=fbd, fontSize=7.5, leading=9.5, alignment=TA_RIGHT, textColor=HexColor("#111827")),
        "AccTableIndented": dict(parent=styles["Normal"], fontName=fb, fontSize=7.5, leading=9.5, leftIndent=12, alignment=TA_LEFT, textColor=HexColor("#111827")),
        "AccTableNarration": dict(parent=styles["Normal"], fontName=fb, fontSize=7.0, leading=8.5, leftIndent=8, alignment=TA_LEFT, textColor=HexColor("#4B5563")),
        "AccTableCaption": dict(parent=styles["Normal"], fontName=fbd, fontSize=7.5, leading=9.5, alignment=TA_CENTER, textColor=HexColor("#374151")),
        "AccTableHeadWide": dict(parent=styles["Normal"], fontName=fbd, fontSize=6.5, leading=8.0, alignment=TA_CENTER, textColor=HexColor("#111827")),
        "AccTableBodyWide": dict(parent=styles["Normal"], fontName=fb, fontSize=6.5, leading=8.0, alignment=TA_LEFT, textColor=HexColor("#111827")),
        "AccTableBodyWideRight": dict(parent=styles["Normal"], fontName=fb, fontSize=6.5, leading=8.0, alignment=TA_RIGHT, textColor=HexColor("#111827")),
        "AccTableBodyWideBoldRight": dict(parent=styles["Normal"], fontName=fbd, fontSize=6.5, leading=8.0, alignment=TA_RIGHT, textColor=HexColor("#111827")),
        "AccEndPaper":     dict(parent=styles["Normal"], fontName=fbd, fontSize=8.0, leading=11, alignment=TA_CENTER, textColor=HexColor("#6B7280"), spaceBefore=8),
        "AccFooter":       dict(parent=styles["Normal"], fontName=fb, fontSize=6.5, alignment=TA_CENTER, textColor=HexColor("#6B7280")),
        "AccSolTitle":     dict(parent=styles["Title"], fontName=fbd, fontSize=13, leading=16, spaceAfter=2, alignment=TA_CENTER, textColor=HexColor("#1E3A8A")),
        "AccSolSubtitle":  dict(parent=styles["Normal"], fontName=fb, fontSize=8.5, leading=11, spaceAfter=4, alignment=TA_CENTER, textColor=HexColor("#4B5563")),
        "AccSolQHead":     dict(parent=styles["Normal"], fontName=fbd, fontSize=8.5, leading=11, alignment=TA_LEFT, textColor=HexColor("#1E3A8A")),
        "AccSolText":      dict(parent=styles["Normal"], fontName=fb, fontSize=8.0, leading=10.5, alignment=TA_LEFT, textColor=HexColor("#111827")),
    }
    for name, props in cst.items():
        try:
            styles.add(ParagraphStyle(name=name, **props))
        except KeyError:
            pass

    story = []
    class_num = re.sub(r'[^0-9]', '', str(class_grade)) or class_grade
    total_marks = sum(q.get("marks", 1) for q in questions if not q.get("_is_or", False))
    main_questions_count = len([q for q in questions if not q.get("_is_or", False)])

    # 1. Header block
    if institute_name and institute_name.strip():
        story.append(Paragraph(f"<b>{institute_name.strip().upper()}</b>", styles["AccInstName"]))

    story.append(Paragraph("ACCOUNTANCY (055)", styles["AccMainTitle"]))
    title_text = exam_title.strip() if exam_title else "SAMPLE QUESTION PAPER"
    story.append(Paragraph(title_text.upper(), styles["AccSubTitle"]))
    story.append(Paragraph(f"Class {class_num} (2025-26)", styles["AccClassLine"]))

    # Time / Maximum Marks row
    if not duration or str(duration).strip().lower() in ("none", "as per schedule", ""):
        dur_str = "3 HOURS" if total_marks >= 70 else ("2 HOURS" if total_marks >= 40 else "1 HOUR")
    else:
        dur_str = str(duration).strip().upper()

    meta_tbl = Table(
        [[Paragraph(f"<b>TIME {dur_str}</b>", styles["AccTimeMeta"]), Paragraph(f"<b>MAX. MARKS {total_marks}</b>", styles["AccMarksMeta"])]],
        colWidths=[W * 0.5, W * 0.5]
    )
    meta_tbl.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 4))

    # General Instructions
    story.append(Paragraph("<b>GENERAL INSTRUCTIONS:</b>", styles["AccInstHeading"]))
    if main_questions_count == 34:
        instructions = [
            "This question paper contains 34 questions. All questions are compulsory.",
            "This question paper is divided into two parts, Part A and B.",
            "Part - A is compulsory for all candidates (Accounting for Partnership Firms and Companies).",
            "Part - B has two options i.e. (i) Analysis of Financial Statements and (ii) Computerised Accounting. Students must attempt only one of the given options.",
            "Question 1 to 16 and 27 to 30 carry 1 mark each.",
            "Questions 17 to 20, 31 and 32 carry 3 marks each.",
            "Questions from 21, 22 and 33 carry 4 marks each.",
            "Questions from 23 to 26 and 34 carry 6 marks each.",
            "There is no overall choice. However, an internal choice has been provided in questions where indicated."
        ]
    else:
        instructions = [
            f"This question paper contains {main_questions_count} questions. All questions are compulsory.",
            "This question paper is prepared as per the latest CBSE pattern and guidelines.",
            "Marks for each question are indicated against it in the right column.",
            "All parts of a question should be attempted together.",
            "All workings must form part of the answer.",
            "There is no overall choice. Internal choice has been provided in questions where indicated."
        ]
    for idx, inst in enumerate(instructions, 1):
        story.append(Paragraph(f"{idx}. {inst}", styles["AccInstItem"]))
    story.append(Spacer(1, 6))

    # Master Table setup
    table_rows = []
    # Row 0: Table Header
    table_rows.append([
        Paragraph("<b>S.No.</b>", styles["AccSNo"]),
        Paragraph("<b>Part A :- Accounting for Partnership Firms and Companies</b>", styles["AccPartHeader"]),
        Paragraph("<b>Marks</b>", styles["AccMarks"]),
    ])

    part_b_started = False
    divider_row_indices = []

    main_qs = [q for q in questions if not q.get("_is_or", False)]
    or_qs = [q for q in questions if q.get("_is_or", False)]
    or_queue = list(or_qs)

    q_num = 0
    for q in main_qs:
        q_num += 1
        sec = str(q.get("section") or q.get("_section") or "").lower()
        part = str(q.get("part") or "").lower()

        # Check if transitioning to Part B
        is_part_b = (
            "b1" in sec or "part b" in sec or "analysis" in sec or
            part == "b" or (main_questions_count == 34 and q_num == 27)
        )

        if is_part_b and not part_b_started:
            part_b_started = True
            part_b_row_idx = len(table_rows)
            divider_row_indices.append(part_b_row_idx)
            table_rows.append([
                Paragraph("", styles["AccSNo"]),
                Paragraph("<b>Part B :- Analysis of Financial Statements (Option – I)</b>", styles["AccPartHeader"]),
                Paragraph("", styles["AccMarks"]),
            ])

        def _render_single_q_content(target_q):
            q_flowables = []
            raw_text = target_q.get("text") or target_q.get("question") or ""

            # Check question_table or markdown table in text
            qt = _get_question_table(target_q)
            if qt:
                raw_text = _strip_markdown_table_from_text(raw_text)

            segments = _split_text_and_tables(raw_text)
            for seg in segments:
                if seg['type'] == 'text':
                    if seg['content'].strip():
                        q_flowables.append(Paragraph(_format_acc_text(seg['content'], has_tt), styles["AccQText"]))
                elif seg['type'] == 'table':
                    hdrs, rws = seg['content']
                    q_flowables.extend(_render_accountancy_financial_table_pdf(hdrs, rws, col1_w, styles, has_tt))

            # Render question_table if separately attached
            if qt:
                q_flowables.extend(_render_accountancy_financial_table_pdf(
                    qt.get("headers", []), qt.get("rows", []), col1_w, styles, has_tt, caption=qt.get("caption")
                ))

            # Sub-parts
            sub_parts = target_q.get("sub_parts") or target_q.get("sub_questions")
            if sub_parts and isinstance(sub_parts, list):
                for sp in sub_parts:
                    lbl = sp.get("label") or sp.get("id") or "•"
                    sp_text = sp.get("text") or sp.get("question") or ""
                    sp_marks = sp.get("marks")
                    marks_tag = f" <b>[{sp_marks} Mark{'s' if sp_marks != 1 else ''}]</b>" if sp_marks else ""
                    q_flowables.append(Paragraph(
                        f"<b>({lbl})</b>  {_format_acc_text(sp_text, has_tt)}{marks_tag}",
                        styles["AccQText"]
                    ))
                    if sp.get("options"):
                        q_flowables.extend(_build_accountancy_mcq_table(
                            sp["options"], sp.get("correct_answer"), include_answers, col1_w, styles, has_tt
                        ))

            # MCQ Options
            opts = target_q.get("options")
            if opts and isinstance(opts, list) and len(opts) >= 2:
                q_flowables.extend(_build_accountancy_mcq_table(
                    opts, target_q.get("correct_answer"), include_answers, col1_w, styles, has_tt
                ))

            return q_flowables

        flowables = _render_single_q_content(q)

        # Internal Choice OR
        or_target = None
        if q.get("or_question"):
            or_target = q["or_question"]
        elif or_queue:
            or_target = or_queue.pop(0)

        if or_target:
            flowables.append(Paragraph("<b>OR</b>", styles["AccOr"]))
            flowables.extend(_render_single_q_content(or_target))

        q_marks = q.get("marks", 1)
        table_rows.append([
            Paragraph(f"<b>{q_num}.</b>", styles["AccSNo"]),
            flowables,
            Paragraph(f"<b>{q_marks}</b>", styles["AccMarks"]),
        ])

    for or_target in or_queue:
        q_num += 1
        flowables = _render_single_q_content(or_target)
        table_rows.append([
            Paragraph(f"<b>{q_num} (OR).</b>", styles["AccSNo"]),
            flowables,
            Paragraph(f"<b>{or_target.get('marks', 1)}</b>", styles["AccMarks"]),
        ])

    # Build Master Table (repeatRows=0 matches CBSE sample paper flow)
    master_table = Table(table_rows, colWidths=[col0_w, col1_w, col2_w], repeatRows=0)
    table_style_cmds = [
        ('BOX', (0,0), (-1,-1), 0.75, HexColor('#111827')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor('#374151')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('ALIGN', (2,0), (2,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,0), (-1,0), HexColor('#F3F4F6')),
    ]
    for d_idx in divider_row_indices:
        table_style_cmds.append(('BACKGROUND', (0, d_idx), (-1, d_idx), HexColor('#F3F4F6')))

    master_table.setStyle(TableStyle(table_style_cmds))
    story.append(master_table)

    story.append(Spacer(1, 4))
    story.append(Paragraph("———  End of Question Paper  ———", styles["AccEndPaper"]))
    story.append(Paragraph("<i>Generated with a4ai &nbsp;•&nbsp; a4ai.in</i>", styles["AccFooter"]))

    # ── Appended Solution Key / Marking Scheme (if include_answers) ─────────
    if include_answers:
        story.append(PageBreak())
        story.append(Paragraph("ACCOUNTANCY (055) — MARKING SCHEME & SOLUTION KEY", styles["AccSolTitle"]))
        story.append(Paragraph("Step-by-step model solutions and marking scheme for evaluators", styles["AccSolSubtitle"]))
        story.append(Spacer(1, 4))

        ak_num = 0
        for q in main_qs:
            ak_num += 1
            ca = q.get("correct_answer") or q.get("answer") or ""
            expl = q.get("explanation") or q.get("model_answer") or ""
            ans_tbl = q.get("answer_table") or q.get("answerTable")

            story.append(Paragraph(f"<b>Question {ak_num} [{q.get('marks', 1)} Marks]</b>", styles["AccSolQHead"]))
            if ca:
                story.append(Paragraph(f"<b>Answer:</b>  {_format_acc_text(ca, has_tt)}", styles["AccSolText"]))
            if expl:
                story.append(Paragraph(f"<b>Working / Explanation:</b>  {_format_acc_text(expl, has_tt)}", styles["AccSolText"]))
            if ans_tbl and isinstance(ans_tbl, dict):
                story.extend(_render_accountancy_financial_table_pdf(
                    ans_tbl.get("headers", []), ans_tbl.get("rows", []), W * 0.95, styles, has_tt
                ))
            story.append(Spacer(1, 4))

    doc.build(story, canvasmaker=AccountancyNumberedCanvas)
    buf_val = buffer.getvalue()
    buffer.close()
    return buf_val


def _add_docx_mcq_grid(container, options, correct_answer, include_answers, font_name):
    from docx.shared import Pt
    from docx.enum.table import WD_TABLE_ALIGNMENT

    labels = ["A", "B", "C", "D", "E", "F"]
    cleaned_options = []
    for i, opt in enumerate(options):
        cleaned = re.sub(r'^[A-Fa-f0-9][).\s\-]+\s*', '', str(opt)).strip()
        cleaned_options.append((labels[i] if i < len(labels) else str(i+1), cleaned))

    is_long = any(len(c[1]) > 38 for c in cleaned_options)
    if is_long:
        tbl = container.add_table(rows=len(cleaned_options), cols=1)
        tbl.style = 'Table Grid'
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for idx, (lbl, text) in enumerate(cleaned_options):
            cell = tbl.rows[idx].cells[0]
            p = cell.paragraphs[0]
            r = p.add_run(f"{lbl}.  {text}")
            r.font.name = font_name; r.font.size = Pt(8.0)
    else:
        num_rows = (len(cleaned_options) + 1) // 2
        tbl = container.add_table(rows=num_rows, cols=2)
        tbl.style = 'Table Grid'
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for idx in range(0, len(cleaned_options), 2):
            r_idx = idx // 2
            # Left
            lbl_l, text_l = cleaned_options[idx]
            p_l = tbl.rows[r_idx].cells[0].paragraphs[0]
            r_l = p_l.add_run(f"{lbl_l}.  {text_l}")
            r_l.font.name = font_name; r_l.font.size = Pt(8.0)
            # Right
            if idx + 1 < len(cleaned_options):
                lbl_r, text_r = cleaned_options[idx + 1]
                p_r = tbl.rows[r_idx].cells[1].paragraphs[0]
                r_r = p_r.add_run(f"{lbl_r}.  {text_r}")
                r_r.font.name = font_name; r_r.font.size = Pt(8.0)


def _add_docx_financial_table(container, headers, rows, font_name):
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    if not headers or not rows:
        return
    num_cols = len(headers)
    tbl = container.add_table(rows=1 + len(rows), cols=num_cols)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    for j, h in enumerate(headers):
        cell = tbl.rows[0].cells[j]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(_latex_to_plain(str(h)))
        r.bold = True; r.font.name = font_name; r.font.size = Pt(8.0)
        shading = cell._element.get_or_add_tcPr()
        shd = shading.makeelement(qn('w:shd'), {qn('w:fill'): 'F9FAFB', qn('w:val'): 'clear'})
        shading.append(shd)

    # Data rows
    for i, row in enumerate(rows):
        padded = (list(row) + [""] * num_cols)[:num_cols]
        for j, val in enumerate(padded):
            cell = tbl.rows[i + 1].cells[j]
            p = cell.paragraphs[0]
            r = p.add_run(_latex_to_plain(str(val)))
            r.font.name = font_name; r.font.size = Pt(8.0)
            if any(term in str(headers[j]).lower() for term in ("amount", "debit", "credit", "rs", "₹", "change")):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def _generate_docx_accountancy_exam(
    questions, exam_title, board, class_grade, subject,
    include_answers, include_explanations, logo_base64, paper_date, tpl,
    teacher_name=None, institute_name=None, duration=None, topic=None,
) -> bytes:
    """
    Authentic CBSE Class 11 & 12 Accountancy (055) Tabular Question Paper DOCX Generator.
    """
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    doc = Document()
    font_name = tpl.get("docx_font", "Calibri")
    primary = "#111827"
    secondary = "#374151"

    try:
        ns = doc.styles["Normal"]
        ns.font.name = font_name
        ns.font.size = Pt(9.5)
    except Exception:
        pass

    for section in doc.sections:
        section.top_margin = Cm(1.2)
        section.bottom_margin = Cm(1.2)
        section.left_margin = Cm(1.2)
        section.right_margin = Cm(1.2)

    class_num = re.sub(r'[^0-9]', '', str(class_grade)) or class_grade
    total_marks = sum(q.get("marks", 1) for q in questions if not q.get("_is_or", False))
    main_questions_count = len([q for q in questions if not q.get("_is_or", False)])

    def _add_p(text, size=9.5, bold=False, italic=False, color="#111827", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        r = p.add_run(text)
        r.font.name = font_name
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        r.font.color.rgb = _rgb(color)
        return p

    # Header
    if institute_name and institute_name.strip():
        _add_p(institute_name.strip().upper(), size=14, bold=True, color=primary)

    _add_p("ACCOUNTANCY (055)", size=13, bold=True, color=primary)
    title_text = exam_title.strip() if exam_title else "SAMPLE QUESTION PAPER"
    _add_p(title_text.upper(), size=10.5, bold=True, color=secondary)
    _add_p(f"Class {class_num} (2025-26)", size=9.5, bold=True, color=secondary, space_after=3)

    if not duration or str(duration).strip().lower() in ("none", "as per schedule", ""):
        dur_str = "3 HOURS" if total_marks >= 70 else ("2 HOURS" if total_marks >= 40 else "1 HOUR")
    else:
        dur_str = str(duration).strip().upper()

    meta_tbl = doc.add_table(rows=1, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_tbl.autofit = False
    meta_tbl.columns[0].width = Cm(9.0)
    meta_tbl.columns[1].width = Cm(9.0)

    p0 = meta_tbl.rows[0].cells[0].paragraphs[0]
    r0 = p0.add_run(f"TIME {dur_str}")
    r0.bold = True; r0.font.name = font_name; r0.font.size = Pt(8.5)

    p1 = meta_tbl.rows[0].cells[1].paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r1 = p1.add_run(f"MAX. MARKS {total_marks}")
    r1.bold = True; r1.font.name = font_name; r1.font.size = Pt(8.5)

    # General Instructions
    p_inst_h = doc.add_paragraph()
    p_inst_h.paragraph_format.space_before = Pt(4)
    p_inst_h.paragraph_format.space_after = Pt(2)
    r_ih = p_inst_h.add_run("GENERAL INSTRUCTIONS:")
    r_ih.bold = True; r_ih.font.name = font_name; r_ih.font.size = Pt(8.5)

    if main_questions_count == 34:
        instructions = [
            "This question paper contains 34 questions. All questions are compulsory.",
            "This question paper is divided into two parts, Part A and B.",
            "Part - A is compulsory for all candidates (Accounting for Partnership Firms and Companies).",
            "Part - B has two options i.e. (i) Analysis of Financial Statements and (ii) Computerised Accounting. Students must attempt only one of the given options.",
            "Question 1 to 16 and 27 to 30 carry 1 mark each.",
            "Questions 17 to 20, 31 and 32 carry 3 marks each.",
            "Questions from 21, 22 and 33 carry 4 marks each.",
            "Questions from 23 to 26 and 34 carry 6 marks each.",
            "There is no overall choice. However, an internal choice has been provided in questions where indicated."
        ]
    else:
        instructions = [
            f"This question paper contains {main_questions_count} questions. All questions are compulsory.",
            "This question paper is prepared as per the latest CBSE pattern and guidelines.",
            "Marks for each question are indicated against it in the right column.",
            "All parts of a question should be attempted together.",
            "All workings must form part of the answer.",
            "There is no overall choice. Internal choice has been provided in questions where indicated."
        ]

    for idx, inst in enumerate(instructions, 1):
        p_inst = doc.add_paragraph()
        p_inst.paragraph_format.left_indent = Cm(0.5)
        p_inst.paragraph_format.space_after = Pt(1)
        r = p_inst.add_run(f"{idx}. {inst}")
        r.font.name = font_name; r.font.size = Pt(7.5); r.font.color.rgb = _rgb("#1F2937")

    # Master 3-Column Table
    main_qs = [q for q in questions if not q.get("_is_or", False)]
    or_qs = [q for q in questions if q.get("_is_or", False)]
    or_queue = list(or_qs)

    master_table = doc.add_table(rows=0, cols=3)
    master_table.style = 'Table Grid'
    master_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    master_table.autofit = False

    def _add_master_row(sno_text, marks_text, is_header=False, is_divider=False):
        row = master_table.add_row()
        row.cells[0].width = Cm(1.2)
        row.cells[1].width = Cm(15.2)
        row.cells[2].width = Cm(1.2)

        if is_header or is_divider:
            for cell in row.cells:
                shading = cell._element.get_or_add_tcPr()
                shd = shading.makeelement(qn('w:shd'), {qn('w:fill'): 'F3F4F6', qn('w:val'): 'clear'})
                shading.append(shd)

        p0 = row.cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r0 = p0.add_run(sno_text)
        r0.bold = True; r0.font.name = font_name; r0.font.size = Pt(8.5)

        p2 = row.cells[2].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(marks_text)
        r2.bold = True; r2.font.name = font_name; r2.font.size = Pt(8.5)

        return row.cells[1]

    # Header Row
    cell_content = _add_master_row("S.No.", "Marks", is_header=True)
    p_h = cell_content.paragraphs[0]
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rh = p_h.add_run("Part A :- Accounting for Partnership Firms and Companies")
    rh.bold = True; rh.font.name = font_name; rh.font.size = Pt(9.5)

    part_b_started = False
    q_num = 0

    for q in main_qs:
        q_num += 1
        sec = str(q.get("section") or q.get("_section") or "").lower()
        part = str(q.get("part") or "").lower()

        is_part_b = (
            "b1" in sec or "part b" in sec or "analysis" in sec or
            part == "b" or (main_questions_count == 34 and q_num == 27)
        )

        if is_part_b and not part_b_started:
            part_b_started = True
            cell_div = _add_master_row("", "", is_divider=True)
            p_div = cell_div.paragraphs[0]
            p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rd = p_div.add_run("Part B :- Analysis of Financial Statements (Option – I)")
            rd.bold = True; rd.font.name = font_name; rd.font.size = Pt(9.5)

        cell_q = _add_master_row(f"{q_num}.", str(q.get("marks", 1)))

        def _render_docx_q_content(container, target_q, is_first=True):
            raw_text = target_q.get("text") or target_q.get("question") or ""
            qt = _get_question_table(target_q)
            if qt:
                raw_text = _strip_markdown_table_from_text(raw_text)

            segments = _split_text_and_tables(raw_text)
            for s_idx, seg in enumerate(segments):
                if seg['type'] == 'text':
                    if seg['content'].strip():
                        p = container.paragraphs[0] if (is_first and s_idx == 0) else container.add_paragraph()
                        p.paragraph_format.space_after = Pt(2)
                        r = p.add_run(_latex_to_plain(seg['content']))
                        r.font.name = font_name; r.font.size = Pt(8.5)
                elif seg['type'] == 'table':
                    hdrs, rws = seg['content']
                    _add_docx_financial_table(container, hdrs, rws, font_name)

            if qt:
                _add_docx_financial_table(container, qt.get("headers", []), qt.get("rows", []), font_name)

            sub_parts = target_q.get("sub_parts") or target_q.get("sub_questions")
            if sub_parts and isinstance(sub_parts, list):
                for sp in sub_parts:
                    p_sp = container.add_paragraph()
                    p_sp.paragraph_format.left_indent = Cm(0.4)
                    p_sp.paragraph_format.space_after = Pt(1)
                    lbl = sp.get("label") or sp.get("id") or "•"
                    r_lbl = p_sp.add_run(f"({lbl})  ")
                    r_lbl.bold = True; r_lbl.font.name = font_name; r_lbl.font.size = Pt(8.5)
                    r_spt = p_sp.add_run(_latex_to_plain(sp.get("text") or sp.get("question") or ""))
                    r_spt.font.name = font_name; r_spt.font.size = Pt(8.5)

            opts = target_q.get("options")
            if opts and isinstance(opts, list) and len(opts) >= 2:
                _add_docx_mcq_grid(container, opts, target_q.get("correct_answer"), include_answers, font_name)

        _render_docx_q_content(cell_q, q, is_first=True)

        or_target = None
        if q.get("or_question"):
            or_target = q["or_question"]
        elif or_queue:
            or_target = or_queue.pop(0)

        if or_target:
            p_or = cell_q.add_paragraph()
            p_or.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_or.paragraph_format.space_before = Pt(3)
            p_or.paragraph_format.space_after = Pt(3)
            r_or = p_or.add_run("OR")
            r_or.bold = True; r_or.font.name = font_name; r_or.font.size = Pt(9.0)

            _render_docx_q_content(cell_q, or_target, is_first=False)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

