"""
add_question_block_images.py — images for NCERT questions whose content is a drawing.

Chemistry questions like "Give the IUPAC names of the following compounds : (a) (b) (c)" were
extracted as text only: the structures are vector drawings, so the options came out empty.
Those questions have no "Fig. X.Y" caption, so add_images.py never picks them up.

This script finds each such question in its chapter PDF (by its number in the left column, e.g.
"8.4", or by its opening words), renders the block from that question down to the next question
number, uploads it to the ncert-diagrams bucket and sets ncert_questions.image_url.
Only rows with an empty image_url are touched.

Two kinds of rows:
  --mode labels  text has "(a) (b)" with nothing between, or an "Image of structure" placeholder
  --mode items   "… the following …?" with no options; the PDF decides whether the items are missing
                 (rows whose PDF block matches the stored text are left alone)

  python add_question_block_images.py --class 11 --subject Chemistry --dry-run --out previews/
  python add_question_block_images.py --class 11 --subject Chemistry [--mode items]   # upload + update
  python add_question_block_images.py --class 11 --subject Chemistry --rollback previews/applied_….json

Needs SUPABASE_URL + SUPABASE_SERVICE_KEY in .env (same as add_images.py).
"""

import argparse
import json
import os
import re
import sys
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
# "(a) (b)" / "(i) (ii)" with nothing between = the items were drawings.
# Only consecutive labels count: "NO (g) (b) …" is a state symbol followed by the next item.
_PAIR = re.compile(r"\(([a-h]|i{1,3}|iv|v|vi)\)\s*\(([a-h]|i{1,3}|iv|v|vi)\)")
_ROMAN = ["i", "ii", "iii", "iv", "v", "vi"]


def has_empty_labels(text: str) -> bool:
    if re.search(r"\bimage of (the )?(structure|figure|diagram|compound)", text or "", re.I):
        return True  # extraction left a placeholder where the drawing was
    for a, b in _PAIR.findall(text or ""):
        if a in _ROMAN and b in _ROMAN and _ROMAN.index(b) == _ROMAN.index(a) + 1:
            return True
        if len(a) == len(b) == 1 and ord(b) == ord(a) + 1 and a not in ("g", "i"):
            return True
    return False


def chapter_pdfs(class_grade: str, subject: str) -> dict:
    """chapter name (pdf stem) → path, across 'Class12_Chemistry - I' / '- II' folders."""
    out = {}
    for folder in PDF_DIR.glob(f"Class{class_grade}_{subject}*"):
        for pdf in folder.glob("*.pdf"):
            out[pdf.stem] = pdf
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def find_pdf(chapter: str, pdfs: dict):
    want = _norm(chapter)
    for stem, path in pdfs.items():
        if _norm(stem) == want or _norm(stem).startswith(want[:25]) or want.startswith(_norm(stem)[:25]):
            return path
    return None


class PdfIndex:
    """One chapter PDF: cached text lines per page + where the end-of-chapter EXERCISES start."""

    def __init__(self, path: Path):
        self.doc = pymupdf.open(str(path))
        self._lines = {}
        self.ex_start = next((i for i in range(len(self.doc) - 1, -1, -1)
                              if any(t.strip().upper() == "EXERCISES" for t, _ in self.lines(i))), 0)

    def lines(self, pno: int):
        if pno not in self._lines:
            out = []
            for b in self.doc[pno].get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if text:
                        out.append((text, pymupdf.Rect(line["bbox"])))
            self._lines[pno] = sorted(out, key=lambda x: (x[1].y0, x[1].x0))
        return self._lines[pno]


def _column(page, lines, at: pymupdf.Rect):
    """x-range of the text column containing `at` (NCERT theory pages are two-column)."""
    w = page.rect.width
    mid = w / 2
    wide = [r for _, r in lines if r.width > w * 0.2]
    left_col = sum(1 for r in wide if r.x1 < mid + 8)
    right_col = sum(1 for r in wide if r.x0 > mid - 20)
    xs = [r.x0 for _, r in lines] or [at.x0]
    x1s = [r.x1 for _, r in lines] or [at.x1]
    if left_col >= 3 and right_col >= 3:
        if at.x0 < mid - 10:
            return max(0, min(xs) - 6), mid + 4
        return mid - 8, min(w, max(x1s) + 6)
    return max(0, min(xs) - 6), min(w, max(x1s) + 6)


def _words(s: str, n: int):
    return [w for w in _norm(s).split() if len(w) > 2][:n]


_LABEL = re.compile(r"\((?:[a-h]|i{1,3}|iv|vi?)\)")


def _clip_has_question(page, clip, q) -> bool:
    """The crop holds this question's stem. Compared without spaces: stored texts have split words
    ("pr oducts", "r eactions"), and several questions open alike ("In the following pairs of halogen
    compounds, which would undergo SN2…" vs "…which compound undergoes faster SN1…"), hence 90%."""
    from difflib import SequenceMatcher
    # the stem only: items after "(a)" / "(i)" may be placeholders ("Image of structure") or garbled formulas
    stem = _compact(_LABEL.split(q["question_text"], maxsplit=1)[0])[:120]
    if len(stem) < 12:
        stem = _compact(q["question_text"])[:60]
    text = _compact(page.get_text("text", clip=clip))
    if not stem or not text:
        return False
    sm = SequenceMatcher(None, stem, text, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks()) / len(stem) >= 0.9


FOOTER = 45  # page number / running footer at the bottom of every NCERT page
_FOOTER_TEXT = re.compile(r"^(chemistry|physics|biology|mathematics|science)\b\s*\d*$|^\d{1,3}$", re.I)


_HEADINGS = {"summary", "exercises", "intext questions", "intext question"}


def _stop(t: str, r: pymupdf.Rect, start: pymupdf.Rect, example: bool, page_h: float = 1e9) -> bool:
    # subscripts ("2" in CH2) are separate lines too, so footer text only counts near the page bottom
    if _FOOTER_TEXT.match(t) and r.y0 > page_h - 90:
        return True
    if t.strip().lower() in _HEADINGS or t.lower().startswith("answers to"):
        return True
    if example:
        return t.lower().startswith("solution")
    # next question number, or text that starts left of the question (a section heading after the box)
    return (bool(NUM_START.match(t)) and abs(r.x0 - start.x0) < 25) or r.x0 < start.x0 - 12


def _box_parts(idx: PdfIndex, pno: int, start: pymupdf.Rect, example: bool):
    """[(page, clip)] from the question start to the next question / 'Solution', continuing onto the
    next page (same column) when the page ends first."""
    page = idx.doc[pno]
    lines = idx.lines(pno)
    x0, x1 = _column(page, lines, start)
    limit = page.rect.height - FOOTER
    below = [(t, r) for t, r in lines if start.y1 + 2 < r.y0 < limit and r.x0 >= x0 - 2 and r.x1 <= x1 + 2]
    stop = next((r for t, r in below if _stop(t, r, start, example, page.rect.height)), None)
    bottom = stop.y0 - 3 if stop else limit
    parts = [(page, pymupdf.Rect(x0, start.y0 - 4, x1, bottom))]
    if stop is None and pno + 1 < len(idx.doc):
        nxt = idx.doc[pno + 1]
        nlines = [(t, r) for t, r in idx.lines(pno + 1)
                  if FOOTER < r.y0 < nxt.rect.height - FOOTER and r.x0 >= x0 - 30 and r.x1 <= x1 + 30]
        if nlines:
            top = nlines[0][1].y0 - 4
            nstop = next((r for t, r in nlines if _stop(t, r, pymupdf.Rect(start.x0, 0, 0, 0), example, nxt.rect.height)), None)
            nbottom = nstop.y0 - 3 if nstop else None
            # only continue when the next page clearly carries on this question (stop found, not at the very top)
            if nbottom and nbottom - top > 20:
                parts.append((nxt, pymupdf.Rect(x0, top, x1, nbottom)))
    return parts


NUM_START = re.compile(r"^(\d{1,2}\.\d{1,3})(\s|$)")


def locate(idx: PdfIndex, q: dict):
    """(page, clip rect) for the question, or None. Every crop must contain the question's opening words."""
    raw = (q.get("question_number") or "").strip()
    m_num = re.search(r"(\d{1,2})\s*\.\s*(\d{1,3})", raw)  # "Problem 9.7", "Intext 6.8", "3. 37"
    num = f"{m_num.group(1)}.{m_num.group(2)}" if m_num else ""
    is_example = q.get("question_type") == "example" or re.match(r"^(problem|example)", raw, re.I) is not None
    tries = []
    if num:
        if is_example:
            for pno in range(len(idx.doc)):
                for t, r in idx.lines(pno):
                    if re.match(rf"^(problem|example)\s*{re.escape(num)}\b", t, re.I):
                        tries.append((pno, r, True))
        else:
            # end-of-chapter exercises first, then intext questions anywhere in the chapter
            order = list(range(idx.ex_start, len(idx.doc))) + list(range(0, idx.ex_start))
            for pno in order:
                for t, r in idx.lines(pno):
                    m = NUM_START.match(t)
                    if m and m.group(1) == num and r.x0 < idx.doc[pno].rect.width * 0.3:
                        tries.append((pno, r, False))
    phrase = " ".join(q["question_text"].split()[:6])
    if len(phrase) > 15:
        for pno in range(len(idx.doc)):
            for r in idx.doc[pno].search_for(phrase)[:1]:
                # widen the hit to its whole line so the question number on the left is the anchor
                line = next((lr for _, lr in idx.lines(pno)
                             if abs(lr.y0 - r.y0) < 4 and lr.x0 <= r.x0 + 1 and lr.x1 >= r.x0), r)
                tries.append((pno, line, is_example))
    for pno, start, example in tries:
        parts = _box_parts(idx, pno, start, example)
        page, clip = parts[0]
        if clip.height >= 25 and _clip_has_question(page, clip, q):
            return parts
    return None


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def text_is_complete(parts, q: dict) -> bool:
    """True when the stored question text already holds what the PDF block shows: the block's text
    matches the question (compared without spaces — PDFs split 'H2SO4' as 'H 2SO4') and the block has
    no drawings (structures / reaction schemes) of its own."""
    from difflib import SequenceMatcher
    block = "".join(_compact(p.get_text("text", clip=c)) for p, c in parts)
    block = re.sub(r"^(problem|example|intextquestion)?\d{1,5}", "", block)  # "Problem 6.12" / "4.26" label
    question = _compact(q["question_text"])
    if not block:
        return True
    sm = SequenceMatcher(None, question, block, autojunk=False)
    unmatched = len(block) - sum(m.size for m in sm.get_matching_blocks())
    for page, clip in parts:
        # reaction schemes are sometimes embedded as pictures
        for info in page.get_image_info():
            r = pymupdf.Rect(info["bbox"])
            inside = (r & clip).get_area() / max(1.0, r.get_area())
            # page borders / watermarks reach far outside the question block
            if inside > 0.8 and r.height > 6 and r.width < clip.width * 0.95:
                return False
        for d in page.get_drawings():
            r = d.get("rect")
            # skip box backgrounds, borders and coloured side bars
            if (r is None or not clip.contains(r.tl) or r.width > clip.width * 0.8
                    or r.height > clip.height * 0.6 or (r.width < 2 and r.height < 2)):
                continue
            if d.get("fill") and not d.get("color") and r.width > 40 and r.height > 40:
                continue
            return False
    # a missing reaction line or option is at least ~10 characters the stored text doesn't have
    return unmatched < 10


def render_parts(parts) -> bytes:
    """Render each (page, clip) and stack them vertically into one PNG."""
    import io
    from PIL import Image
    ims = [Image.open(io.BytesIO(p.get_pixmap(clip=c, dpi=RENDER_DPI, alpha=False).tobytes("png"))) for p, c in parts]
    out = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)), "white")
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="class_grade", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="previews")
    ap.add_argument("--ids", help="comma-separated question ids (default: all with empty labels)")
    ap.add_argument("--rollback", help="json file of applied ids → set image_url back to NULL")
    ap.add_argument("--rows", help="dry run from a JSON export of the rows instead of the DB")
    ap.add_argument("--mode", choices=["labels", "items"], default="labels",
                    help="labels: '(a) (b)' with nothing between → whole question block; "
                         "items: 'the following …?' with no options → only the part below the stem")
    args = ap.parse_args()

    sb = None if (args.dry_run and args.rows) else create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    if args.rollback:
        ids = json.loads(Path(args.rollback).read_text())
        for i in range(0, len(ids), 100):
            sb.table("ncert_questions").update({"image_url": None}).in_("id", ids[i:i + 100]).execute()
        print(f"rolled back {len(ids)} rows")
        return

    rows = json.loads(Path(args.rows).read_text(encoding="utf-8")) if args.rows else []
    q = None if args.rows else (sb.table("ncert_questions").select(
        "id, class_grade, chapter, question_number, question_type, question_text, image_url, options")
         .eq("class_grade", args.class_grade).ilike("subject", args.subject))
    start = 0
    while q is not None:
        page = q.range(start, start + 999).execute().data
        rows += page
        if len(page) < 1000:
            break
        start += 1000
    if args.mode == "labels":
        targets = [r for r in rows if not r.get("image_url") and has_empty_labels(r["question_text"])]
    else:  # "items": 'the following …' questions with no options; the PDF decides whether items are missing
        def empty_opts(o):
            return o in (None, "", "[]", []) or (isinstance(o, str) and o.strip() in ('[]', '"[]"', "null"))
        targets = [r for r in rows if not r.get("image_url") and str(r.get("class_grade", args.class_grade)) == args.class_grade
                   and empty_opts(r.get("options", "[]")) and re.search(r"following", r["question_text"] or "", re.I)
                   and re.search(r"[:?]\s*$", r["question_text"] or "") and len(r["question_text"]) < 220
                   # "9.14 (ii)" rows are one sub-part each; their text is the whole sub-part
                   and "(" not in (r.get("question_number") or "")]
    if args.ids:
        keep = {int(x) for x in args.ids.split(",")}
        targets = [r for r in rows if r["id"] in keep]
    print(f"{len(rows)} rows, {len(targets)} targets")

    pdfs = chapter_pdfs(args.class_grade, args.subject)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs, applied, missed, skipped = {}, [], [], []
    for r in targets:
        pdf = find_pdf(r["chapter"], pdfs)
        if not pdf:
            missed.append((r["id"], "no pdf for chapter " + r["chapter"]))
            continue
        idx = docs.setdefault(pdf, PdfIndex(pdf))
        hit = locate(idx, r)
        if not hit:
            missed.append((r["id"], f"not found in {pdf.name}: {r['question_number']}"))
            continue
        if args.mode == "items" and text_is_complete(hit, r):
            skipped.append(r["id"])
            continue
        png = render_parts(hit)
        name = f"q_{r['id']}.png"
        if args.dry_run:
            (out_dir / name).write_bytes(png)
            print(f"[DRY] {r['id']} {r['question_number']} p{hit[0][0].number + 1} parts={len(hit)} → {out_dir / name}")
            continue
        path = f"class{args.class_grade}/{args.subject}/question_blocks/{name}"
        sb.storage.from_(BUCKET).upload(path=path, file=png,
                                        file_options={"content-type": "image/png", "upsert": "true"})
        url = sb.storage.from_(BUCKET).get_public_url(path)
        # never overwrite an image someone already set
        sb.table("ncert_questions").update({"image_url": url}).eq("id", r["id"]).is_("image_url", "null").execute()
        applied.append(r["id"])
        print(f"[OK] {r['id']} {r['question_number']} → {path}")

    if applied:
        log = out_dir / f"applied_class{args.class_grade}_{args.subject}_{args.mode}.json"
        log.write_text(json.dumps(applied))
        print(f"applied {len(applied)} (ids in {log}, use --rollback to undo)")
    for m in missed:
        print("[MISS]", *m)
    if skipped:
        print(f"[SKIP] {len(skipped)} rows: nothing below the stem in the PDF (text is complete): {skipped}")
    print(f"done: {len(targets) - len(missed) - len(skipped)} imaged, {len(skipped)} complete, {len(missed)} missed")


if __name__ == "__main__":
    main()
