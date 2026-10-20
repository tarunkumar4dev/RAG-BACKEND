"""
recrop_diagrams.py — re-cut the NCERT figure images that ncert_questions.image_url points at.

batch_extract_diagrams.py cropped the bounding box of EVERY drawing on the page above the caption
(page header rule, "QUESTIONS" box, neighbouring figures), so many images were half a page of
text with two or three figures in it. This script cuts only the figure next to its caption:

  1. caption = a line that starts with "Fig./Figure X.Y" (exact number, so 1.1 ≠ 1.10); when the
     number appears several times, the line with the most figure graphics around it wins;
  2. graphics = vector drawings + raster images, minus page furniture (header/footer rules,
     full-width lines, page-sized backgrounds, coloured boxes that hold body text);
  3. start from the graphic nearest the caption and grow through graphics and short labels
     that touch it (gap ≤ GAP pt) — a separate figure or a paragraph is not reached;
  4. crop = that cluster + the caption line when it sits right next to it.

New images go to ncert-diagrams/v2/… (old files stay, so --rollback is just the old URLs).

  python recrop_diagrams.py --dry-run --out previews/                 # PNGs + before/after sheets
  python recrop_diagrams.py --dry-run --out previews/ --only class10/Science/Electricity
  python recrop_diagrams.py --apply --out previews/                   # upload + update image_url
  python recrop_diagrams.py --rollback previews/recrop_applied_….json

Reading works with SUPABASE_URL + SUPABASE_KEY (anon). --apply / --rollback need
SUPABASE_SERVICE_KEY set in the terminal ($env:SUPABASE_SERVICE_KEY = "…").
"""

import argparse
import io
import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from supabase import create_client

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / "test-generator-backend" / ".env")

BUCKET = "ncert-diagrams"
PDF_DIR = Path(__file__).parent / "pdfs"
RENDER_DPI = 170
GAP = 14            # pt: graphics/labels closer than this belong to the same figure
SEED_MAX_DIST = 140  # pt: the figure must start this close to its caption
LABEL_MAX_CHARS = 28  # longer text lines are body text, never part of a figure

_CAPTION = re.compile(r"^\s*fig(?:ure)?\s*\.?\s*(\d+)\s*\.\s*(\d+)(?!\s*\d)", re.IGNORECASE)
_SUBJECT_FOLDER = {"Mathematics": "Maths"}


# ── PDF lookup ────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def find_pdf(class_grade: str, subject: str, chapter: str):
    folder_subject = _SUBJECT_FOLDER.get(subject, subject)
    want = _norm(chapter)
    for folder in PDF_DIR.glob(f"Class{class_grade}_*"):
        if not _norm(folder.name.split("_", 1)[1]).startswith(_norm(folder_subject)):
            continue
        for pdf in folder.glob("*.pdf"):
            stem = _norm(pdf.stem)
            if stem == want or stem.startswith(want[:20]) or want.startswith(stem[:20]):
                return pdf
    return None


# ── page analysis ─────────────────────────────────────────────────────────────
def text_lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            t = "".join(s.get("text", "") for s in spans).strip()
            if t:
                bold = any(s.get("flags", 0) & 16 or "bold" in s.get("font", "").lower() for s in spans)
                out.append((pymupdf.Rect(line["bbox"]), t, bold))
    return out


def graphics(page, lines):
    W, H = page.rect.width, page.rect.height
    page_area = W * H
    rects = []

    def furniture(r: pymupdf.Rect, max_share: float) -> bool:
        if r.is_empty or r.is_infinite or r.width < 1 and r.height < 1:
            return True
        if r.width * r.height > max_share * page_area:
            return True
        if r.y1 < 45 or r.y0 > H - 35:  # running header / footer
            return True
        if r.width > 0.8 * W and r.height < 4:  # section rules (circuit wires are shorter)
            return True
        return False

    body_lines = [lr for lr, t, _ in lines if len(t) > LABEL_MAX_CHARS]

    all_lines = [lr for lr, _, _ in lines]

    def holds_paragraphs(r: pymupdf.Rect) -> bool:
        # coloured panel behind an Activity / QUESTIONS box (often one strip per text line), or
        # the "not to be republished" watermark image laid over the text
        if r.width * r.height > 12000 and sum(
                len(t) for lr, t, _ in lines if r.contains(lr) and len(t) > LABEL_MAX_CHARS) > 120:
            return True
        return r.height > 6 and sum(1 for lr in body_lines if r.intersects(lr)) >= 2

    def inline(r: pymupdf.Rect) -> bool:
        # fraction bars / brackets inside a paragraph line
        c = pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        if any(pymupdf.Rect(lr.x0 - 2, lr.y0 - 3, lr.x1 + 2, lr.y1 + 3).contains(c) for lr in body_lines):
            return True
        # a displayed fraction bar: a short flat line with text right above and right below it
        if r.height < 2 and r.width < 90:
            def over(lr):
                return lr.x0 < r.x1 and lr.x1 > r.x0
            above = any(over(lr) and 0 <= r.y0 - lr.y1 <= 6 for lr in all_lines)
            below = any(over(lr) and 0 <= lr.y0 - r.y1 <= 6 for lr in all_lines)
            return above and below
        return False

    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if furniture(r, 0.30) or holds_paragraphs(r) or inline(r):
            continue
        rects.append(r)
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if not furniture(r, 0.60) and not holds_paragraphs(r):
            rects.append(r)
    return rects


def _dist(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return (dx * dx + dy * dy) ** 0.5


def grow_cluster(seed: pymupdf.Rect, rects, labels):
    """Graphics chain through each other; labels are only attached at the end (one pass), so a
    run of short lines ("So, ∠P = ∠S", "(Given)") can't walk the crop into a worked solution."""
    box = pymupdf.Rect(seed)
    pool = list(rects)
    changed = True
    while changed:
        changed = False
        grown = pymupdf.Rect(box.x0 - GAP, box.y0 - GAP, box.x1 + GAP, box.y1 + GAP)
        keep = []
        for r in pool:
            if grown.intersects(r):
                box |= r
                changed = True
            else:
                keep.append(r)
        pool = keep
    grown = pymupdf.Rect(box.x0 - GAP, box.y0 - GAP, box.x1 + GAP, box.y1 + GAP)
    for r in labels:
        if grown.intersects(r):
            box |= r
    return box


def figure_on_page(page, caption: pymupdf.Rect, lines, rects):
    if not rects:
        return None
    seed = min(rects, key=lambda r: _dist(r, caption))
    if _dist(seed, caption) > SEED_MAX_DIST:
        return None
    labels = [lr for lr, t, _ in lines if len(t) <= LABEL_MAX_CHARS and not _CAPTION.match(t)]
    # The caption ends its figure: a figure above its caption never continues below it.
    if seed.y1 <= caption.y0 + 2:
        rects = [r for r in rects if r.y0 < caption.y0]
        labels = [r for r in labels if r.y0 < caption.y0]
    elif seed.y0 >= caption.y1 - 2:
        rects = [r for r in rects if r.y1 > caption.y1]
        labels = [r for r in labels if r.y1 > caption.y1]
    box = grow_cluster(seed, rects, labels)
    if _dist(box, caption) <= 30:
        box |= caption
    pad = 6
    box = pymupdf.Rect(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad) & page.rect
    return box


_page_cache: dict = {}


def page_info(doc, pi):
    """(text lines, figure graphics) of one page, computed once per run."""
    key = (doc.name, pi)
    if key not in _page_cache:
        lines = text_lines(doc[pi])
        _page_cache[key] = (lines, None)
    lines, rects = _page_cache[key]
    return lines, rects


def page_graphics(doc, pi):
    lines, rects = page_info(doc, pi)
    if rects is None:
        rects = graphics(doc[pi], lines)
        _page_cache[(doc.name, pi)] = (lines, rects)
    return rects


def locate_figure(doc, fig_ref: str):
    """(page_index, crop_rect) for 'X.Y', or None. Picks the caption with most graphics around it."""
    want = tuple(int(p) for p in fig_ref.split("."))
    best = None
    for pi in range(len(doc)):
        lines, _ = page_info(doc, pi)
        for cap, t, _bold in lines:
            m = _CAPTION.match(t)
            if not m or (int(m.group(1)), int(m.group(2))) != want:
                continue
            rects = page_graphics(doc, pi)
            box = figure_on_page(doc[pi], cap, lines, rects)
            if box is None:
                continue
            n = sum(1 for r in rects if box.intersects(r))
            page = doc[pi].rect
            share = box.width * box.height / (page.width * page.height)
            score = n - (500 if share > 0.55 else 0)
            if best is None or score > best[0]:
                best = (score, pi, box)
    return (best[1], best[2]) if best else None


# ── DB / storage ──────────────────────────────────────────────────────────────
def client(write: bool):
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") if write else (os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY"))
    if not url or not key:
        sys.exit("Set SUPABASE_URL and " + ("SUPABASE_SERVICE_KEY" if write else "SUPABASE_KEY") + " in the terminal.")
    return create_client(url, key)


def current_images(sb):
    """{old_url: {class_grade, subject, chapter, fig, ids:[…]}} for every figure image in use."""
    out, start = {}, 0
    while True:
        rows = (sb.table("ncert_questions").select("id, class_grade, subject, chapter, image_url")
                .like("image_url", "%/ncert-diagrams/class%/fig\\_%").range(start, start + 999).execute().data or [])
        for r in rows:
            m = re.search(r"/fig_(\d+)_(\d+)\.png", r["image_url"])
            if not m:
                continue
            e = out.setdefault(r["image_url"], {"class_grade": r["class_grade"], "subject": r["subject"],
                                                "chapter": r["chapter"], "fig": f"{m.group(1)}.{m.group(2)}", "ids": []})
            e["ids"].append(r["id"])
        if len(rows) < 1000:
            return out
        start += 1000


def new_path(old_url: str) -> str:
    rel = old_url.split(f"/{BUCKET}/", 1)[1].split("?", 1)[0]
    return "v2/" + rel


# ── preview sheet ─────────────────────────────────────────────────────────────
def contact_sheet(pairs, path: Path):
    from PIL import Image, ImageDraw
    W, H = 380, 300
    sheet = Image.new("RGB", (W * 2, H * len(pairs)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (title, old_png, new_png) in enumerate(pairs):
        for j, data in enumerate((old_png, new_png)):
            x, y = j * W, i * H
            dr.rectangle([x, y, x + W - 1, y + H - 1], outline="gray")
            dr.text((x + 4, y + 2), ("OLD " if j == 0 else "NEW ") + title, fill="red" if j == 0 else "green")
            if data:
                im = Image.open(io.BytesIO(data)).convert("RGB")
                im.thumbnail((W - 8, H - 20))
                sheet.paste(im, (x + 4, y + 16))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="render previews only (default)")
    mode.add_argument("--apply", action="store_true", help="upload v2 images and point image_url at them")
    mode.add_argument("--rollback", metavar="APPLIED_JSON", help="restore the old image_url values")
    ap.add_argument("--out", default="previews", help="folder for previews / applied json")
    ap.add_argument("--only", help="only image paths containing this text, e.g. class10/Science/Electricity")
    ap.add_argument("--no-sheet", action="store_true", help="skip downloading old images for the sheets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.rollback:
        sb = client(write=True)
        applied = json.loads(Path(args.rollback).read_text(encoding="utf-8"))
        for item in applied:
            sb.table("ncert_questions").update({"image_url": item["old_url"]}).in_("id", item["ids"]).execute()
        print(f"Restored {sum(len(i['ids']) for i in applied)} rows.")
        return

    sb_read = client(write=False)
    sb_write = client(write=True) if args.apply else None
    images = current_images(sb_read)
    if args.only:
        images = {u: e for u, e in images.items() if args.only in u}
    print(f"{len(images)} figure images in use")

    applied, skipped, pairs, docs = [], [], [], {}
    for n, (old_url, e) in enumerate(sorted(images.items())):
        pdf = find_pdf(e["class_grade"], e["subject"], e["chapter"])
        if not pdf:
            skipped.append((old_url, "no PDF"))
            continue
        doc = docs.get(pdf) or docs.setdefault(pdf, pymupdf.open(str(pdf)))
        found = locate_figure(doc, e["fig"])
        if not found:
            skipped.append((old_url, f"Fig. {e['fig']} not found in {pdf.name}"))
            continue
        pi, box = found
        png = doc[pi].get_pixmap(clip=box, dpi=RENDER_DPI, alpha=False).tobytes("png")
        rel = new_path(old_url)
        local = out / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(png)

        if not args.no_sheet:
            try:
                old_png = urllib.request.urlopen(old_url, timeout=20).read()
            except Exception:
                old_png = None
            pairs.append((f"{e['subject'][:5]} {e['chapter'][:24]} Fig {e['fig']}", old_png, png))

        if args.apply:
            sb_write.storage.from_(BUCKET).upload(rel, png, {"content-type": "image/png", "upsert": "true"})
            url = sb_write.storage.from_(BUCKET).get_public_url(rel)
            sb_write.table("ncert_questions").update({"image_url": url}).in_("id", e["ids"]).eq("image_url", old_url).execute()
            applied.append({"old_url": old_url, "new_url": url, "ids": e["ids"]})
        print(f"[{n + 1}/{len(images)}] Fig {e['fig']} p{pi + 1} {int(box.width)}x{int(box.height)}pt → {rel}")

    for i in range(0, len(pairs), 8):
        contact_sheet(pairs[i:i + 8], out / f"sheet_{i // 8:02d}.png")
    for u, why in skipped:
        print(f"SKIPPED (old image kept): {why} — {u}")
    if args.apply:
        p = out / f"recrop_applied_{datetime.now():%Y%m%d_%H%M%S}.json"
        p.write_text(json.dumps(applied, indent=1), encoding="utf-8")
        print(f"Updated {sum(len(a['ids']) for a in applied)} rows. Rollback file: {p}")
    else:
        print(f"Dry run: {len(images) - len(skipped)} re-cut, {len(skipped)} skipped. Previews in {out}/")


if __name__ == "__main__":
    main()
