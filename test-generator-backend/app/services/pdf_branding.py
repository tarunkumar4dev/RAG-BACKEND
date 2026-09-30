"""
app/services/pdf_branding.py — Watermark any exported PDF after it is built.

Applied as a post-processing step so every paper layout (card, institute, CBSE, Accountancy,
answer keys) gets the same watermark without touching each generator:
  * a faded institute logo centred on every page, and/or
  * diagonal watermark text ("Apex Academy", "CONFIDENTIAL", ...).
Both are drawn semi-transparent over the content so table backgrounds can't hide them.
"""
import base64
import io
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def _decode_logo(logo_base64: Optional[str]) -> Optional[bytes]:
    if not logo_base64:
        return None
    try:
        data = logo_base64.split(",", 1)[1] if "," in logo_base64 else logo_base64
        return base64.b64decode(data)
    except Exception:
        return None


def _faded_png(img_bytes: bytes, opacity: float) -> Optional[bytes]:
    """Returns the logo as a PNG whose alpha is scaled to `opacity` (a true watermark tint)."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        img.thumbnail((900, 900))
        alpha = img.getchannel("A").point(lambda a: int(a * opacity))
        img.putalpha(alpha)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        logger.warning(f"Could not prepare logo watermark: {e}")
        return None


def apply_pdf_watermark(
    pdf_bytes: bytes,
    text: Optional[str] = None,
    logo_base64: Optional[str] = None,
    text_opacity: float = 0.07,
    logo_opacity: float = 0.07,
) -> bytes:
    """Stamps the watermark on every page. Returns the original bytes if anything goes wrong."""
    text = (text or "").strip()[:60]
    logo_png = None
    if logo_base64:
        raw = _decode_logo(logo_base64)
        logo_png = _faded_png(raw, logo_opacity) if raw else None
    if not text and not logo_png:
        return pdf_bytes

    try:
        import pymupdf
    except ImportError:  # older PyMuPDF
        import fitz as pymupdf  # type: ignore

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            r = page.rect
            cx, cy = r.width / 2, r.height / 2
            if logo_png:
                side = min(r.width, r.height) * 0.55
                box = pymupdf.Rect(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
                page.insert_image(box, stream=logo_png, keep_proportion=True, overlay=True)
            if text:
                # Fit the text along ~75% of the page diagonal, capped at a readable size.
                diag = math.hypot(r.width, r.height)
                width_at_1 = pymupdf.get_text_length(text, fontname="helv", fontsize=1) or 1
                size = max(18, min(84, diag * 0.75 / width_at_1))
                tw = pymupdf.TextWriter(r, opacity=text_opacity, color=(0.25, 0.25, 0.3))
                font = pymupdf.Font("helv")
                tw.append((cx - width_at_1 * size / 2, cy + size / 3), text, font=font, fontsize=size)
                angle = math.degrees(math.atan2(r.height, r.width))
                tw.write_text(page, morph=(pymupdf.Point(cx, cy), pymupdf.Matrix(angle)), overlay=True)
        out = doc.tobytes(garbage=3, deflate=True)
        doc.close()
        return out
    except Exception as e:
        logger.warning(f"Watermark failed, returning unwatermarked PDF: {e}")
        return pdf_bytes
