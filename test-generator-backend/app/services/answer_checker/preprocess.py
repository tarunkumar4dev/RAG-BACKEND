"""
Stage A — preprocess (pure code, no LLM).

Sources (PDF pages / phone photos, in upload order) → normalised page JPEGs (~150 dpi, long edge ≤ 2000 px),
blank-page detection, then — once the grader has reported each page's written page number and rotation —
rotate the pages upright, put them in the student's order, and raise sheet flags.

Every coordinate the renderer uses is on the FINAL page (rotated + reordered), so the marked copy and the
answer positions always agree.
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

logger = logging.getLogger(__name__)

RENDER_DPI = 150
MAX_LONG_EDGE = int(os.getenv("CHECKER_MAX_LONG_EDGE", "2000"))
JPEG_QUALITY = 80
MAX_PAGES = int(os.getenv("COPY_CHECKER_MAX_PAGES", "16"))
# Blank page: less than this share of "ink" pixels after ruled lines / margins are removed. Kept very low on
# purpose: a blank page is not sent to the grader, so a page with only "(c)" on it must never count as blank.
# Fixture: written pages 0.0049–0.022, empty ruled page with shadow + stain 0.0000.
BLANK_INK_RATIO = 0.0002

_IMAGE_TYPES = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}


class BadFile(ValueError):
    """The upload can't be read. The message is safe to show the teacher; no Gemini call is made."""


@dataclass
class NormPage:
    source_index: int                 # position in upload order (all sources, all pages)
    jpeg: bytes
    w: int
    h: int
    blank: bool
    rotation: int = 0                 # clockwise degrees applied to make it upright
    student_page_no: Optional[str] = None
    flags: list = field(default_factory=list)


def sha256_files(paths: list) -> str:
    """One hash over all source files in order (dedupe / result cache key)."""
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _open(path: str) -> pymupdf.Document:
    ext = os.path.splitext(path)[1].lower()
    if os.path.getsize(path) == 0:
        raise BadFile("One of the files is empty (0 bytes). Please upload it again.")
    try:
        if ext == ".pdf":
            doc = pymupdf.open(path, filetype="pdf")
        elif ext in _IMAGE_TYPES:
            doc = pymupdf.open(path, filetype=_IMAGE_TYPES[ext])
        else:
            raise BadFile(f"Unsupported file type '{ext}'. Upload a PDF or photos (JPG/PNG/WEBP).")
    except BadFile:
        raise
    except Exception:
        raise BadFile("A file could not be opened (it may be damaged). Please upload it again.")
    if doc.needs_pass:
        doc.close()
        raise BadFile("The PDF is password-protected. Upload a PDF without a password.")
    if doc.page_count == 0:
        doc.close()
        raise BadFile("The PDF has no pages.")
    return doc


_INK_WIDTH = 300      # px, grayscale render used only for blank detection
_TILE = 20            # px, local paper brightness per tile (shadows / stains don't count as ink)
_INK_DROP = 45        # a pixel is ink when it is this much darker than its tile's paper


def _ink_ratio(page: pymupdf.Page) -> float:
    """Share of ink pixels on a small grayscale render. Ink = clearly darker than the local paper (90th
    percentile of its tile), so shadows and stains don't count; rows/columns that are mostly dark (printed
    ruled or margin lines, scan borders) are ignored — they are on every sheet, written or not."""
    zoom = _INK_WIDTH / max(page.rect.width, 1)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False)
    w, h, s = pix.width, pix.height, pix.samples
    if w * h == 0:
        return 0.0
    paper = {}
    for ty in range(0, h, _TILE):
        for tx in range(0, w, _TILE):
            vals = sorted(s[y * w + x] for y in range(ty, min(ty + _TILE, h)) for x in range(tx, min(tx + _TILE, w)))
            paper[(ty // _TILE, tx // _TILE)] = vals[int(len(vals) * 0.9)]
    dark = bytearray(w * h)
    for y in range(h):
        row_paper = y // _TILE
        for x in range(w):
            if s[y * w + x] < paper[(row_paper, x // _TILE)] - _INK_DROP:
                dark[y * w + x] = 1
    line_rows = {y for y in range(h) if sum(dark[y * w:(y + 1) * w]) > 0.5 * w}
    line_cols = {x for x in range(w) if sum(dark[x::w]) > 0.5 * h}
    m = 4      # scanner edges
    ink = sum(1 for y in range(m, h - m) if y not in line_rows
              for x in range(m, w - m) if x not in line_cols and dark[y * w + x])
    return ink / (w * h)


def _render(page: pymupdf.Page, native_long_px: float) -> tuple:
    long_pts = max(page.rect.width, page.rect.height)
    target = min(MAX_LONG_EDGE, native_long_px)
    zoom = target / long_pts
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
    return pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY), pix.width, pix.height


def normalize_sources(paths: list, max_pages: int = MAX_PAGES) -> list:
    """PDF pages and photos (upload order) → NormPage list. Raises BadFile for unreadable input."""
    pages: list = []
    for path in paths:
        doc = _open(path)
        try:
            is_pdf = doc.is_pdf
            photo_px = None
            if not is_pdf:
                # photos: never upscale. page.rect of an image document depends on its DPI tag, not its pixels
                pix = pymupdf.Pixmap(path)
                photo_px = max(pix.width, pix.height)
                pix = None
            for page in doc:
                if len(pages) >= max_pages:
                    raise BadFile(f"Too many pages (max {max_pages}).")
                if is_pdf:
                    native = max(page.rect.width, page.rect.height) * RENDER_DPI / 72
                else:
                    native = photo_px
                jpeg, w, h = _render(page, native)
                pages.append(NormPage(source_index=len(pages), jpeg=jpeg, w=w, h=h,
                                      blank=_ink_ratio(page) < BLANK_INK_RATIO))
        finally:
            doc.close()
    if not pages:
        raise BadFile("No pages found in the upload.")
    return pages


def rotate_jpeg(jpeg: bytes, w: int, h: int, clockwise: int) -> tuple:
    """Turn a page image by 90/180/270 degrees clockwise → (jpeg, w, h)."""
    clockwise %= 360
    if clockwise == 0:
        return jpeg, w, h
    nw, nh = (h, w) if clockwise in (90, 270) else (w, h)
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=nw, height=nh)
        # insert_image rotates counter-clockwise
        page.insert_image(page.rect, stream=jpeg, rotate=(360 - clockwise) % 360)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), colorspace=pymupdf.csRGB, alpha=False)
        return pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY), pix.width, pix.height
    finally:
        doc.close()


def _page_number(s: Optional[str]) -> Optional[int]:
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else None


def finalize_pages(pages: list, graded_index: list, page_meta: list) -> tuple:
    """
    Apply the grader's page report.
      pages        — NormPage list in upload order
      graded_index — for each image sent to the grader, its index in `pages` (blank pages are not sent)
      page_meta    — grader's report per sent image: {student_page_number, rotation_needed, blank}
    Returns (ordered_pages, old_to_new, sheet_flags):
      ordered_pages — NormPage list in the student's order, rotated upright
      old_to_new    — graded image index → index in ordered_pages (to remap answer positions)
    """
    flags: list = []
    for gi, meta in enumerate(page_meta):
        p = pages[graded_index[gi]]
        p.student_page_no = meta.get("student_page_number")
        rot = meta.get("rotation_needed") or 0
        if rot:
            p.jpeg, p.w, p.h = rotate_jpeg(p.jpeg, p.w, p.h, rot)
            p.rotation = rot
        if meta.get("blank"):
            p.blank = True

    # Reorder by the written page numbers — only when every written page has a distinct number
    written = [p for p in pages if not p.blank]
    numbers = [_page_number(p.student_page_no) for p in written]
    order = list(range(len(pages)))
    if written and None not in numbers and len(set(numbers)) == len(numbers):
        key, last = {}, 0.0
        for i, p in enumerate(pages):          # a blank page stays right after the page before it
            if p.blank:
                key[i] = last + 0.5
            else:
                last = float(_page_number(p.student_page_no))
                key[i] = last
        order = sorted(range(len(pages)), key=lambda i: (key[i], i))

    ordered = [pages[i] for i in order]
    new_pos = {old: new for new, old in enumerate(order)}
    old_to_new = [new_pos[gi] for gi in graded_index]

    if order != list(range(len(pages))):
        moved = [str(p.student_page_no) for k, p in enumerate(ordered) if order[k] != k and not p.blank]
        flags.append({"code": "PAGES_REORDERED", "page": None,
                      "message": f"Pages were uploaded out of order (student pages {', '.join(moved)}); "
                                 f"put back in order by the page numbers written on them."})
    for k, p in enumerate(ordered):
        label = p.student_page_no or str(k + 1)
        if p.rotation:
            flags.append({"code": "PAGE_ROTATED", "page": k,
                          "message": f"Page {label} was scanned {'upside down' if p.rotation == 180 else 'sideways'} "
                                     f"and has been turned upright."})
        if p.blank:
            flags.append({"code": "BLANK_PAGE", "page": k, "message": f"Page {k + 1} is blank."})
    return ordered, old_to_new, flags


def remap_positions(result: dict, old_to_new: list) -> None:
    """Answer positions from the grader (graded image index) → index on the final, ordered pages."""
    def fix(item: dict) -> None:
        pi = item.get("page_index")
        if isinstance(pi, int) and 0 <= pi < len(old_to_new):
            item["page_index"] = old_to_new[pi]
        elif pi is not None:
            item["page_index"] = None
            item["y_start"] = None

    for gq in result.get("graded_questions") or []:
        fix(gq)
        subs = gq.get("subpart_grades") or []
        for s in subs:
            fix(s)
        located = [s for s in subs if s.get("page_index") is not None]
        if located:      # the answer starts at its earliest sub-part ON THE FINAL PAGE ORDER
            first = min(located, key=lambda s: (s["page_index"], s["y_start"] if s.get("y_start") is not None else 1000))
            gq["page_index"], gq["y_start"] = first["page_index"], first.get("y_start")
