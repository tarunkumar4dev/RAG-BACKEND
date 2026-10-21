"""
Anonymise real handwritten copies for the G3 test set (tests/fixtures/real/README.md). No network, no Gemini.

  $env:REAL_COPIES_DIR = "C:/Users/login/a4ai_real_copies"     # outside every repo
  python scripts/anonymise_copies.py [--raw DIR] [--out DIR] [--header 0.22] [--every-page-header] [--force]

Defaults: --raw $REAL_COPIES_DIR/raw, --out $REAL_COPIES_DIR/anonymised. Refuses any folder inside a git repo:
real copies are never committed, not even anonymised.

For every folder in raw/: pages → normalised JPEGs (Stage A, same as the checker sees them), the page-1 header
band (and any redact.json boxes) filled solid, metadata dropped, identifiers stripped from the JSON files.
Output folders are named copy_01, copy_02 …; the mapping to the original folder goes to raw/_mapping.json only.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.answer_checker import preprocess  # noqa: E402

VERSION = "anonymise-1"
PAGE_TYPES = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}
PII_KEYS = {"name", "student_name", "roll", "roll_no", "school", "institute", "teacher", "teacher_name",
            "phone", "email", "parent_phone", "parent_name"}


def strip_pii(value):
    if isinstance(value, dict):
        return {k: strip_pii(v) for k, v in value.items() if k.lower() not in PII_KEYS}
    if isinstance(value, list):
        return [strip_pii(v) for v in value]
    return value


def black_out(jpeg: bytes, w: int, h: int, boxes: list) -> bytes:
    """Fill normalised boxes [(x0, y0, x1, y1)] solid and re-encode (no EXIF survives)."""
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=jpeg)
        for x0, y0, x1, y1 in boxes:
            r = pymupdf.Rect(x0 * w, y0 * h, x1 * w, y1 * h)
            page.draw_rect(r, color=None, fill=(0.35, 0.35, 0.35))
            page.insert_text((r.x0 + 12, r.y0 + min(40, r.height - 8)), "REDACTED", fontsize=min(28, r.height / 2),
                             color=(1, 1, 1))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), colorspace=pymupdf.csRGB, alpha=False)
        return pix.tobytes("jpeg", jpg_quality=preprocess.JPEG_QUALITY)
    finally:
        doc.close()


def anonymise_copy(src: Path, dst: Path, header: float, every_page: bool) -> dict:
    page_files = sorted(p for p in (src / "pages").iterdir() if p.suffix.lower() in PAGE_TYPES)
    if not page_files:
        raise ValueError(f"{src.name}: no pages in pages/")
    pages = preprocess.normalize_sources([str(p) for p in page_files], max_pages=preprocess.MAX_PAGES)
    extra = json.loads((src / "redact.json").read_text("utf-8")) if (src / "redact.json").exists() else []

    (dst / "pages").mkdir(parents=True)
    log = []
    for i, p in enumerate(pages, 1):
        boxes = []
        if i == 1 or every_page:
            boxes.append((0.0, 0.0, 1.0, header))
        boxes += [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in extra if int(b.get("page", 0)) == i]
        (dst / "pages" / f"p{i:02d}.jpg").write_bytes(black_out(p.jpeg, p.w, p.h, boxes))
        log.append({"page": i, "boxes": [list(b) for b in boxes]})

    found = [n for n in ("rubric.json", "test.json", "expected.json") if (src / n).exists()]
    if "expected.json" not in found or not ({"rubric.json", "test.json"} & set(found)):
        raise ValueError(f"{src.name}: needs expected.json and rubric.json or test.json")
    for n in found:
        data = strip_pii(json.loads((src / n).read_text("utf-8")))
        (dst / n).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    (dst / "anonymise.json").write_text(json.dumps({"version": VERSION, "pages": len(pages), "blacked_out": log},
                                                   indent=1), encoding="utf-8")
    return {"pages": len(pages)}


def inside_git_repo(path: Path) -> bool:
    p = path.resolve()
    return any((d / ".git").exists() for d in [p, *p.parents])


def main() -> None:
    base = Path(os.environ["REAL_COPIES_DIR"]) if os.getenv("REAL_COPIES_DIR") else None
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=base / "raw" if base else None)
    ap.add_argument("--out", type=Path, default=base / "anonymised" if base else None)
    ap.add_argument("--header", type=float, default=0.22, help="share of page 1 height blacked out (0-1)")
    ap.add_argument("--every-page-header", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild copies that already exist")
    args = ap.parse_args()
    if not 0 < args.header < 0.6:
        sys.exit("--header must be between 0 and 0.6")
    if args.raw is None or args.out is None:
        sys.exit("Set $env:REAL_COPIES_DIR (a folder outside the repos) or pass --raw and --out.")
    for d in (args.raw, args.out):
        if inside_git_repo(d):
            sys.exit(f"{d} is inside a git repo. Real copies (even anonymised) never go in a repo.")

    mapping_path = args.raw / "_mapping.json"
    mapping = json.loads(mapping_path.read_text("utf-8")) if mapping_path.exists() else {}
    by_source = {v: k for k, v in mapping.items()}
    for src in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        copy_id = by_source.get(src.name) or f"copy_{len(mapping) + 1:02d}"
        dst = args.out / copy_id
        if dst.exists():
            if not args.force:
                print(f"skip {copy_id} (exists; --force to rebuild)")
                continue
            shutil.rmtree(dst)
        try:
            info = anonymise_copy(src, dst, args.header, args.every_page_header)
        except (ValueError, preprocess.BadFile) as e:
            shutil.rmtree(dst, ignore_errors=True)
            print(f"FAILED {copy_id}: {e}")
            continue
        mapping[copy_id] = src.name
        mapping_path.write_text(json.dumps(mapping, indent=1), encoding="utf-8")
        print(f"{copy_id}: {info['pages']} pages → {dst}  — CHECK pages/p01.jpg before committing")


if __name__ == "__main__":
    main()
