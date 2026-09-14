"""
add_images.py — NCERT Diagram Extraction (v3 — Smart Region Detection)
========================================================================
v3 changes vs v2:
  - Smart region selection: caption ke around SABSE dense vector-drawing
    region dhundhta hai, fixed padding nahi.
  - Table detection: agar upar wala region table hai (grid of text),
    to skip karke neeche/side check karta hai.
  - Multi-directional: caption ke upar + neeche + left + right
    candidate regions evaluate karta hai.
  - Best-candidate scoring: drawings density / area ratio se decide.

Strategy:
  1. Har page pe "Figure X.Y" / "Fig. X.Y" captions dhundho (text search).
  2. Har caption ke liye, page ke drawings + text blocks ka layout dekho.
  3. Caption ke around 4 candidate regions banao.
  4. Har candidate ka "diagram score" nikalo:
       score = (vector drawing ops in region) / (text chars in region + 1)
     Jahan score highest, wahan diagram hoga.
  5. Us region ko render karo → PNG.
  6. Supabase Storage pe upload.
  7. figure_ref match karke ncert_questions.image_url update karo.

Usage:
  python add_images.py --dry-run --pdf "Electricity.pdf"
  python add_images.py --pdf "Electricity.pdf"
  python add_images.py                     # all PDFs
  python add_images.py --class 10 --subject Science
  python add_images.py --debug --pdf "Electricity.pdf"  # verbose region scores

Requires .env:
  SUPABASE_URL=...
  SUPABASE_SERVICE_KEY=...
"""

import os
import re
import sys
import io
import argparse
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
import pymupdf
from supabase import create_client, Client

# ── Bootstrap ───────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("add_images")

BUCKET = "ncert-diagrams"
PDF_DIR = Path(__file__).parent / "pdfs"

# Region render settings
RENDER_DPI = 200
MIN_REGION_AREA = 15000        # skip tiny crops
MIN_DIAGRAM_SCORE = 1.5        # threshold: score = drawings / (chars/100)

# Candidate region shapes (dx0, dy0, dx1, dy1) relative to caption
# caption anchor: (x0, y0) = top-left, (x1, y1) = bottom-right of caption line
CANDIDATES = [
    # (name, x0_offset, y0_offset, x1_offset, y1_offset)
    # ABOVE caption — the classic case
    ("above_tight",    -10, -280,   10, -5),
    ("above_wide",     -40, -400,   40, -5),
    # BELOW caption — sometimes diagram after caption
    ("below_tight",    -10,   5,    10, 280),
    ("below_wide",     -40,   5,    40, 400),
    # AROUND caption — for small diagrams embedded in text
    ("around_left",  -300, -150,   -5,  150),
    ("around_right",    5, -150,  300,  150),
]


# ── Supabase ────────────────────────────────────────────────────
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        log.error("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)
    return create_client(url, key)


# ── Folder parsing ──────────────────────────────────────────────
FOLDER_PATTERN = re.compile(
    r"^Class\s*(?P<class>\d{1,2})[_ ]+(?P<subject>.+)$", re.IGNORECASE
)

FOLDER_OVERRIDES = {
    "First_Flight_Poems":       ("10", "English"),
    "First_Flight_Prose":       ("10", "English"),
    "Footprints_Without_Feet":  ("10", "English"),
    "First_Flight":             ("10", "English"),
    "Vistas":                   ("10", "English"),
    "Flamingo":                 ("12", "English"),
    "Vistas_Class12":           ("12", "English"),
}

SUBJECT_ALIASES = {
    "maths": "Mathematics", "math": "Mathematics",
    "mathematics": "Mathematics", "science": "Science",
    "physics - i": "Physics", "physics - ii": "Physics", "physics": "Physics",
    "chemistry - i": "Chemistry", "chemistry - ii": "Chemistry", "chemistry": "Chemistry",
    "biology": "Biology", "english": "English",
    "history": "History", "geography": "Geography",
    "political_science": "Political Science", "political science": "Political Science",
    "economics": "Economics", "accountancy": "Accountancy",
    "business_studies": "Business Studies", "business studies": "Business Studies",
}


def normalize_subject(raw: str) -> str:
    key = raw.lower().strip()
    if key in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[key]
    for alias, canonical in SUBJECT_ALIASES.items():
        if key.startswith(alias):
            return canonical
    return raw.replace("_", " ").strip().title()


@dataclass
class PDFMeta:
    path: Path
    class_grade: str
    subject: str
    chapter_name: str
    folder_name: str


def parse_pdf_path(pdf_path: Path) -> Optional[PDFMeta]:
    folder = pdf_path.parent.name
    if folder in FOLDER_OVERRIDES:
        cls, subj = FOLDER_OVERRIDES[folder]
    else:
        m = FOLDER_PATTERN.match(folder)
        if not m:
            return None
        cls = m.group("class")
        subj = normalize_subject(m.group("subject"))
    return PDFMeta(
        path=pdf_path,
        class_grade=cls,
        subject=subj,
        chapter_name=pdf_path.stem,
        folder_name=folder,
    )


# ── Figure caption detection ────────────────────────────────────
FIG_PATTERN = re.compile(r"(?:fig(?:ure)?\.?\s*)(\d+\.\d+)", re.IGNORECASE)


def find_figure_captions(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    """Search for figure captions. Unique by figure_ref; first occurrence wins."""
    seen: dict[str, pymupdf.Rect] = {}
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return []

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            m = FIG_PATTERN.search(line_text)
            if not m:
                continue
            fig_ref = m.group(1)
            if fig_ref in seen:
                continue
            bbox = line.get("bbox")
            if bbox:
                # Skip captions that appear inside a table row (heuristic: bbox width < 200pt)
                rect = pymupdf.Rect(bbox)
                if rect.width < 80:
                    continue
                seen[fig_ref] = rect
    return list(seen.items())


# ── Region scoring ──────────────────────────────────────────────
@dataclass
class RegionScore:
    name: str
    rect: pymupdf.Rect
    drawings: int
    text_chars: int
    area: float
    score: float


def _count_drawings_in_rect(page: pymupdf.Page, rect: pymupdf.Rect) -> int:
    """Count vector drawing ops whose center falls inside rect."""
    count = 0
    try:
        drawings = page.get_drawings()
    except Exception:
        return 0
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
            count += 1
    return count


def _count_text_chars_in_rect(page: pymupdf.Page, rect: pymupdf.Rect) -> int:
    """Count text characters whose center falls inside rect. Used as 'table penalty'."""
    total = 0
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return 0
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            bbox = line.get("bbox")
            if not bbox:
                continue
            lx = (bbox[0] + bbox[2]) / 2
            ly = (bbox[1] + bbox[3]) / 2
            if rect.x0 <= lx <= rect.x1 and rect.y0 <= ly <= rect.y1:
                for span in line.get("spans", []):
                    total += len(span.get("text", ""))
    return total


def score_region(
    page: pymupdf.Page,
    caption_rect: pymupdf.Rect,
    offset: tuple[int, int, int, int],
    name: str,
) -> Optional[RegionScore]:
    """Build candidate region, count drawings + text, return score."""
    page_rect = page.rect
    x0o, y0o, x1o, y1o = offset

    x0 = max(0, caption_rect.x0 + x0o)
    y0 = max(0, caption_rect.y0 + y0o)
    x1 = min(page_rect.width, caption_rect.x1 + x1o)
    y1 = min(page_rect.height, caption_rect.y1 + y1o)

    if x1 <= x0 or y1 <= y0:
        return None

    region = pymupdf.Rect(x0, y0, x1, y1)
    area = region.width * region.height
    if area < MIN_REGION_AREA:
        return None

    drawings = _count_drawings_in_rect(page, region)
    text_chars = _count_text_chars_in_rect(page, region)

    # Score: drawings per 100 chars of text
    score = drawings / (text_chars / 100.0 + 1.0)

    return RegionScore(
        name=name,
        rect=region,
        drawings=drawings,
        text_chars=text_chars,
        area=area,
        score=score,
    )


def find_best_region(
    page: pymupdf.Page,
    caption_rect: pymupdf.Rect,
    debug: bool = False,
) -> Optional[RegionScore]:
    """Try all candidate regions, return highest-scoring one."""
    candidates: list[RegionScore] = []
    for name, *offset in CANDIDATES:
        s = score_region(page, caption_rect, tuple(offset), name)
        if s:
            candidates.append(s)

    if not candidates:
        return None

    if debug:
        for c in candidates:
            log.info(
                f"      [{c.name}] drawings={c.drawings} chars={c.text_chars} "
                f"area={int(c.area)} score={c.score:.2f}"
            )

    best = max(candidates, key=lambda c: c.score)
    if best.score < MIN_DIAGRAM_SCORE:
        if debug:
            log.info(f"      ❌ Best score {best.score:.2f} < threshold {MIN_DIAGRAM_SCORE}")
        return None
    return best


# ── Rendering ───────────────────────────────────────────────────
@dataclass
class RenderedDiagram:
    figure_ref: str
    page_num: int
    bytes_data: bytes
    ext: str
    width: int
    height: int
    region_name: str
    score: float


def render_region(
    page: pymupdf.Page,
    region: pymupdf.Rect,
    page_num: int,
    figure_ref: str,
    region_name: str,
    score: float,
) -> Optional[RenderedDiagram]:
    try:
        pix = page.get_pixmap(clip=region, dpi=RENDER_DPI, alpha=False)
        img_bytes = pix.tobytes("png")
    except Exception as e:
        log.error(f"  Render failed for Fig. {figure_ref}: {e}")
        return None

    return RenderedDiagram(
        figure_ref=figure_ref,
        page_num=page_num,
        bytes_data=img_bytes,
        ext="png",
        width=pix.width,
        height=pix.height,
        region_name=region_name,
        score=score,
    )


def extract_diagrams_from_pdf(
    pdf_path: Path, debug: bool = False
) -> list[RenderedDiagram]:
    results: list[RenderedDiagram] = []
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        log.error(f"Cannot open {pdf_path.name}: {e}")
        return results

    try:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            captions = find_figure_captions(page)
            for fig_ref, rect in captions:
                if debug:
                    log.info(f"   📌 Fig. {fig_ref} at page {page_num}")
                best = find_best_region(page, rect, debug=debug)
                if not best:
                    continue
                diag = render_region(
                    page, best.rect, page_num, fig_ref, best.name, best.score
                )
                if diag:
                    results.append(diag)
                    if debug:
                        log.info(
                            f"      ✅ Selected [{best.name}] "
                            f"{diag.width}x{diag.height} score={best.score:.2f}"
                        )
    finally:
        doc.close()

    return results


# ── Upload ──────────────────────────────────────────────────────
def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_") or "unnamed"


def upload_image(
    supabase: Client, diag: RenderedDiagram, meta: PDFMeta, dry_run: bool
) -> Optional[str]:
    chap = sanitize(meta.chapter_name)[:60]
    subj = sanitize(meta.subject)
    filename = f"fig_{diag.figure_ref.replace('.', '_')}_p{diag.page_num}.{diag.ext}"
    storage_path = f"class{meta.class_grade}/{subj}/{chap}/{filename}"

    if dry_run:
        log.info(
            f"  [DRY] {storage_path} "
            f"({diag.width}x{diag.height}, {len(diag.bytes_data)}B, region={diag.region_name}, score={diag.score:.2f})"
        )
        return f"[DRY] {storage_path}"

    try:
        supabase.storage.from_(BUCKET).upload(
            path=storage_path,
            file=diag.bytes_data,
            file_options={"content-type": "image/png", "upsert": "true"},
        )
        return supabase.storage.from_(BUCKET).get_public_url(storage_path)
    except Exception as e:
        log.error(f"  Upload failed: {storage_path} → {e}")
        return None


# ── DB matching ─────────────────────────────────────────────────
def find_matching_questions(
    supabase: Client, meta: PDFMeta, figure_ref: str
) -> list[dict]:
    candidates = [
        figure_ref,
        f"Fig. {figure_ref}",
        f"Fig.{figure_ref}",
        f"Figure {figure_ref}",
    ]
    try:
        resp = (
            supabase.table("ncert_questions")
            .select("id, figure_ref, image_url")
            .eq("class_grade", meta.class_grade)
            .eq("subject", meta.subject)
            .in_("figure_ref", candidates)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        log.error(f"  DB query failed: {e}")
        return []


def update_question_image(
    supabase: Client, qid: int, url: str, dry_run: bool
) -> bool:
    if dry_run:
        log.info(f"  [DRY] DB update q{qid} → {url}")
        return True
    try:
        supabase.table("ncert_questions").update({"image_url": url}).eq("id", qid).execute()
        return True
    except Exception as e:
        log.error(f"  DB update failed q{qid}: {e}")
        return False


# ── Stats ───────────────────────────────────────────────────────
@dataclass
class Stats:
    pdfs_processed: int = 0
    diagrams_extracted: int = 0
    diagrams_uploaded: int = 0
    questions_matched: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n📊 SUMMARY\n{'='*60}\n"
            f"  PDFs processed:       {self.pdfs_processed}\n"
            f"  Diagrams extracted:   {self.diagrams_extracted}\n"
            f"  Diagrams uploaded:    {self.diagrams_uploaded}\n"
            f"  Questions matched:    {self.questions_matched}\n"
            f"  Errors:               {len(self.errors)}\n"
            f"{'='*60}"
        )


# ── Pipeline per PDF ────────────────────────────────────────────
def process_pdf(
    supabase: Client, meta: PDFMeta, stats: Stats, dry_run: bool, debug: bool
):
    log.info(f"\n📄 {meta.folder_name}/{meta.path.name}")
    log.info(f"   → Class {meta.class_grade} | {meta.subject} | {meta.chapter_name}")

    diagrams = extract_diagrams_from_pdf(meta.path, debug=debug)
    stats.diagrams_extracted += len(diagrams)
    log.info(f"   → {len(diagrams)} diagrams rendered")

    for i, diag in enumerate(diagrams):
        log.info(
            f"   [{i+1}/{len(diagrams)}] Fig. {diag.figure_ref} "
            f"(page {diag.page_num}, {diag.width}x{diag.height}, "
            f"region={diag.region_name}, score={diag.score:.2f})"
        )
        url = upload_image(supabase, diag, meta, dry_run=dry_run)
        if not url:
            stats.errors.append(f"{meta.path.name} Fig.{diag.figure_ref}: upload failed")
            continue
        stats.diagrams_uploaded += 1

        matches = find_matching_questions(supabase, meta, diag.figure_ref)
        for q in matches:
            # Overwrite even if image_url exists (v3 better than v2)
            if update_question_image(supabase, q["id"], url, dry_run=dry_run):
                stats.questions_matched += 1
                log.info(f"      ✅ q{q['id']} ← Fig. {diag.figure_ref}")

    stats.pdfs_processed += 1


# ── CLI ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NCERT diagram pipeline (v3 — smart)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pdf", help="Only this PDF filename")
    parser.add_argument("--class", dest="class_filter")
    parser.add_argument("--subject")
    parser.add_argument("--debug", action="store_true", help="Verbose region scores")
    args = parser.parse_args()

    if not PDF_DIR.exists():
        log.error(f"❌ PDF folder not found: {PDF_DIR}")
        sys.exit(1)

    all_pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    if not all_pdfs:
        log.error(f"❌ No PDFs found under {PDF_DIR}")
        sys.exit(1)

    parsed: list[PDFMeta] = []
    for p in all_pdfs:
        meta = parse_pdf_path(p)
        if not meta:
            log.warning(f"⚠️  Cannot parse folder: {p.parent.name} (skipping {p.name})")
            continue
        parsed.append(meta)

    if args.pdf:
        parsed = [m for m in parsed if m.path.name == args.pdf]
    if args.class_filter:
        parsed = [m for m in parsed if m.class_grade == args.class_filter]
    if args.subject:
        parsed = [m for m in parsed if m.subject.lower() == args.subject.lower()]

    log.info(f"🔍 {len(parsed)} PDFs to process")
    if args.dry_run:
        log.info("🧪 DRY-RUN mode")
    if args.debug:
        log.info("🔬 DEBUG mode")

    if not parsed:
        log.error("❌ No PDFs after filtering")
        sys.exit(1)

    supabase = get_supabase()
    stats = Stats()

    for meta in parsed:
        try:
            process_pdf(supabase, meta, stats, dry_run=args.dry_run, debug=args.debug)
        except KeyboardInterrupt:
            log.warning("\n⚠️  Interrupted")
            break
        except Exception as e:
            log.exception(f"❌ Failed: {meta.path.name}")
            stats.errors.append(f"{meta.path.name}: {e}")

    print(stats.summary())
    if stats.errors:
        log.warning("\n⚠️  Errors:")
        for err in stats.errors[:20]:
            log.warning(f"   • {err}")


if __name__ == "__main__":
    main()