"""
Server-side checks for files a student uploaded through an assignment link, before any Gemini call.

The upload went straight from the browser to Storage, so nothing about it is trusted: the mime, the extension and
the sha256 in submissions.files were all written by the client. Here the real bytes decide:
  * magic bytes  → what the file really is (a PDF renamed to .jpg is rejected)
  * PyMuPDF      → PDF opens, has no password, page count (cap = the link's max_pages, hard max 30)
  * Pillow       → photo opens; EXIF (GPS, phone model, time) is removed and the photo turned upright
  * sha256       → of the real bytes, for duplicate flags

Every problem is reported per file in Hinglish (students and teachers read it), all problems at once, so a
student fixes everything in one re-upload.
"""

import hashlib
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

from app.services.answer_checker import preprocess

logger = logging.getLogger(__name__)

HARD_MAX_PAGES = 30

# kind → (mime, allowed extensions)
KINDS = {
    "pdf": ("application/pdf", {"pdf"}),
    "jpeg": ("image/jpeg", {"jpg", "jpeg"}),
    "png": ("image/png", {"png"}),
    "webp": ("image/webp", {"webp"}),
    "heic": ("image/heic", {"heic", "heif"}),
}
_KIND_LABEL = {"pdf": "PDF", "jpeg": "JPG photo", "png": "PNG image", "webp": "WEBP image", "heic": "HEIC photo"}
_HEIC_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1", b"heif"}


def sniff(data: bytes) -> Optional[str]:
    """File kind from its first bytes, None when it is none of the accepted kinds."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp" and data[8:12] in _HEIC_BRANDS:
        return "heic"
    # a PDF may start after some junk bytes (allowed by the spec, within the first 1 KB)
    if b"%PDF-" in data[:1024]:
        return "pdf"
    return None


@dataclass
class FileProblem:
    page_order: int
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"code": self.code, "page_order": self.page_order, "message": self.message}


@dataclass
class CheckedFile:
    page_order: int
    storage_path: str
    kind: str
    data: bytes                     # bytes to grade (EXIF removed for photos)
    pages: int
    sha256: str                     # of the bytes the student uploaded
    declared_sha256: Optional[str]
    exif_removed: bool = False

    @property
    def ext(self) -> str:
        return "jpg" if self.kind == "jpeg" else self.kind


@dataclass
class Validation:
    files: list = field(default_factory=list)       # CheckedFile, in page order
    problems: list = field(default_factory=list)    # FileProblem
    total_pages: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        """One line for check_jobs.error (≤ 300 chars there)."""
        return " ".join(p.message for p in self.problems)[:300]


def _strip_photo(data: bytes, kind: str) -> tuple:
    """(bytes without EXIF/GPS, removed?) — only rewritten when the photo carries metadata. The photo is turned
    upright first, because dropping the EXIF orientation tag would otherwise show it sideways."""
    from PIL import Image, ImageOps

    if kind == "heic":
        import pillow_heif
        pillow_heif.register_heif_opener()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        exif = im.getexif()
        has_meta = bool(exif) or bool(im.info.get("exif")) or bool(im.info.get("xmp"))
        if not has_meta:
            return data, False
        fmt = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "heic": "HEIF"}[kind]
        out = ImageOps.exif_transpose(im)
        if fmt in ("JPEG", "HEIF") and out.mode not in ("RGB", "L"):
            out = out.convert("RGB")
        buf = io.BytesIO()
        kwargs = {"quality": 92} if fmt in ("JPEG", "WEBP", "HEIF") else {}
        out.save(buf, format=fmt, **kwargs)        # no exif= → nothing carried over
        return buf.getvalue(), True


def _pdf_pages(data: bytes, label: str) -> tuple:
    """(page count, problem message or None)"""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        return 0, f"{label} ki PDF khul nahi rahi (file kharab lag rahi hai). Dobara save karke upload karo."
    try:
        # needs_pass only: a PDF with just an owner password (print/copy restrictions) opens fine and is graded
        if doc.needs_pass:
            return 0, f"{label} ki PDF password-protected hai. Bina password wali PDF upload karo."
        if doc.page_count == 0:
            return 0, f"{label} ki PDF mein koi page nahi hai."
        return doc.page_count, None
    finally:
        doc.close()


def validate(entries: list, blobs: dict, allowed_mimes: list, max_pages: int) -> Validation:
    """
    entries — submissions.files rows [{storage_path, mime, size, sha256, page_order}]
    blobs   — storage_path → downloaded bytes (missing key = the object is gone)
    Returns every file checked + every problem found (empty problems = all fine).
    """
    cap = max(1, min(int(max_pages or HARD_MAX_PAGES), HARD_MAX_PAGES))
    allowed = {m.lower() for m in (allowed_mimes or [])}
    v = Validation()
    for e in sorted(entries, key=lambda x: int(x.get("page_order") or 0)):
        order = int(e.get("page_order") or 0)
        path = str(e.get("storage_path") or "")
        label = f"Page {order}"
        data = blobs.get(path)
        if data is None:
            v.problems.append(FileProblem(order, "not_uploaded", f"{label} ki file server pe nahi mili. Dobara upload karo."))
            continue
        if len(data) == 0:
            v.problems.append(FileProblem(order, "empty_file", f"{label} ki file khaali hai (0 bytes). Dobara upload karo."))
            continue
        kind = sniff(data)
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if kind is None:
            v.problems.append(FileProblem(order, "bad_format",
                                          f"{label} ki file PDF ya photo (JPG/PNG/WEBP/HEIC) nahi hai. Sahi file upload karo."))
            continue
        mime, exts = KINDS[kind]
        if ext not in exts or str(e.get("mime") or "").lower() != mime:
            v.problems.append(FileProblem(order, "bad_format",
                                          f"{label} ki file ka naam .{ext or '?'} hai par andar {_KIND_LABEL[kind]} hai. "
                                          f"Asli file seedha upload karo (naam badal kar nahi)."))
            continue
        if allowed and mime not in allowed:
            v.problems.append(FileProblem(order, "bad_format",
                                          f"{label}: is assignment mein {_KIND_LABEL[kind]} allowed nahi hai."))
            continue

        sha = hashlib.sha256(data).hexdigest()
        removed = False
        if kind == "pdf":
            pages, problem = _pdf_pages(data, label)
            if problem:
                v.problems.append(FileProblem(order, "bad_pdf", problem))
                continue
        else:
            pages = 1
            try:
                data, removed = _strip_photo(data, kind)
            except Exception:
                v.problems.append(FileProblem(order, "bad_image",
                                              f"{label} ki photo khul nahi rahi (file kharab lag rahi hai). Dobara photo lo."))
                continue
        v.files.append(CheckedFile(order, path, kind, data, pages, sha, e.get("sha256"), removed))
        v.total_pages += pages

    if v.total_pages > cap:
        v.problems.append(FileProblem(0, "too_many_pages",
                                      f"Copy mein {v.total_pages} pages hain, is assignment mein max {cap} pages allowed hain. "
                                      f"Extra pages hata kar dobara upload karo."))
    return v


# Stage A / grading messages (English, from preprocess / copy_check) → what the student reads
def hinglish_bad_file(e: preprocess.BadFile) -> str:
    msg = str(e)
    if "blank" in msg.lower():
        return "Copy ke saare pages khaali (blank) lag rahe hain. Likhe hue pages ki saaf photo upload karo."
    if "Too many pages" in msg:
        return ("Copy ki photos bahut badi hain, ek saath check nahi ho sakti. Photos thodi chhoti karke "
                "ya kam pages mein dobara upload karo.")
    if "password" in msg.lower():
        return "PDF password-protected hai. Bina password wali PDF upload karo."
    return "Copy ki file padhi nahi ja saki. Saaf photo / PDF dobara upload karo."
