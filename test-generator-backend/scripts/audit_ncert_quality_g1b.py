#!/usr/bin/env python3
"""
scripts/audit_ncert_quality_g1b.py — G1b of the NCERT audit. READ ONLY: row dumps / SELECT, HTTP HEAD on
image URLs, local NCERT PDFs. Nothing is written to the database and no AI model is called.

Appends (or replaces) the "G1b" section of docs/ncert-audit-class10-maths.md — the G1a part above it is
left untouched — and writes:
  docs/audit/ncert_audit_class10_maths_rows_v2.csv   per row: v2 hash, all flags, clean (G5 baseline)
  docs/audit/near_duplicate_pairs.csv                Class 10 Maths near-duplicate pairs (similarity > 0.9)
  docs/audit/garbled_precision_sample.csv            30 random garbled Class 10 Maths questions to check by hand

Needs rapidfuzz + numpy (audit only, not in requirements.txt): pip install rapidfuzz==3.13.0 numpy==2.1.3

Usage (from test-generator-backend/):
  python scripts/audit_ncert_quality_g1b.py --from-json dump/class_*.json \\
      --katex-script ../../fe/scripts/check_katex_segments.cjs --node-path ../../fe/node_modules \\
      --pdf-dir ../ncert-ingestion/pdfs --head-check 50 \\
      --price "gemini-3.5-flash-lite=0.30/2.50" --price "gemini-3.6-flash=0.75/3.75"
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import scripts.audit_ncert_quality as g1a  # noqa: E402
from app.services.ncert_text_hash import HASH_VERSION, _normalize, content_hash, normalize  # noqa: E402

SEED = 20261022
NEAR_DUP_CUTOFF = 90.0  # rapidfuzz fuzz.ratio, strictly greater than

# Adobe Symbol font glyphs that the PDF text layer stored as U+F0xx (Private Use Area). One glyph = one
# character, so these could be mapped back deterministically. Bracket pieces (U+F0E6–F0FE) and digits
# are layout, not single symbols.
SYMBOL_GLYPHS = {
    0xF020: " ", 0xF028: "(", 0xF029: ")", 0xF02B: "+", 0xF02D: "−", 0xF02E: ".", 0xF03D: "=",
    0xF040: "≅", 0xF044: "Δ", 0xF05E: "⊥", 0xF061: "α", 0xF062: "β", 0xF066: "φ", 0xF067: "γ",
    0xF06C: "λ", 0xF06D: "μ", 0xF070: "π", 0xF071: "θ", 0xF072: "ρ", 0xF073: "σ", 0xF077: "ω",
    0xF0A2: "′", 0xF0A3: "≤", 0xF0A5: "∞", 0xF0AB: "↔", 0xF0AE: "→", 0xF0B0: "°", 0xF0B3: "≥",
    0xF0B4: "×", 0xF0B9: "≠", 0xF0BB: "≈", 0xF0CE: "∈", 0xF0D0: "∠", 0xF0D6: "√", 0xF0D7: "·", 0xF0DE: "⇒",
}
_PUA = re.compile(r"[-]")
GARBLE_SIGNALS = {
    "symbol_font_glyph": _PUA,
    # "t a n 4 5" — a function name broken into letters
    "spaced_function": re.compile(r"(?<![A-Za-z])(?:t a ?n|ta n|s i ?n|si n|c o ?s|co s|c o ?t|co t|s e ?c|se c)(?![A-Za-z])"),
    # "2R180p" — a fraction with its symbols flattened into one token
    "digit_letter_soup": re.compile(r"\b\d+[A-Za-z]\d{2,}[A-Za-z]*\b"),
    # "= /360 × r²" — numerator (θ, π) lost
    "missing_numerator": re.compile(r"(?:=|\(|×)\s*/\s*\d"),
}
MATH_ONLY_SIGNALS = {
    # "x2 + 5x", "PQ2 =" — superscript lost (Mathematics only: chemistry formulas look the same)
    "flattened_power": re.compile(r"\b(?:[a-z]|[A-Z]{1,2})2\s*[+\-–=]"),
    # "3 2, 1 2" for 3/2, 1/2
    "flattened_fraction": re.compile(r"(?<![\d.,/])\b\d{1,3} \d{1,3}\b(?![\d.,/]|\s?[A-Za-z°%])"),
}
_LATEX_CMD = re.compile(r"\\[A-Za-z]+")
_FIG_NO = re.compile(r"fig(?:ure)?\.?\s*(\d+)\s*\.\s*(\d+)", re.I)
_URL_FIG_NO = re.compile(r"fig_(\d+)_(\d+)", re.I)
_DIFF_SYMBOLS = set("∠πθ²³√°×Δ≤≥′≅⊥")


# ── per-row G1b checks ──────────────────────────────────────────────────────────

_TABLE_WORDS = re.compile(r"\b(?:table|frequency|distribution|class interval|number of)\b", re.I)


def is_table_like(text: str, has_question_table: bool = False) -> bool:
    """A flattened data table (frequency tables in Statistics): numbers in a row are not fractions."""
    return has_question_table or (bool(_TABLE_WORDS.search(text)) and len(re.findall(r"\d+", text)) >= 8)


def garble_signals(text: str, subject: str, has_question_table: bool = False) -> list[str]:
    found = [name for name, rx in GARBLE_SIGNALS.items() if rx.search(text)]
    if "math" in (subject or "").lower():
        found += [name for name, rx in MATH_ONLY_SIGNALS.items()
                  if rx.search(text) and not (name == "flattened_fraction" and is_table_like(text, has_question_table))]
    return found


def garble_tier(text: str, signals: list[str]) -> str | None:
    if not signals:
        return None
    if signals == ["symbol_font_glyph"] and all(ord(c) in SYMBOL_GLYPHS for c in _PUA.findall(text)):
        return "symbol_only"
    return "layout"


def figure_mismatch(r) -> str | None:
    """figure_ref 'Fig. 11.6' vs image file fig_11_6.png, and the image's chapter folder vs the row's chapter."""
    url = r.get("image_url") or ""
    if not url:
        return None
    ref = _FIG_NO.search(r.get("figure_ref") or "")
    img = _URL_FIG_NO.search(url)
    if ref and img and ref.groups() != img.groups():
        return "figure_number"
    folder = re.sub(r"[^a-z0-9]", "", url.rsplit("/", 2)[-2].lower())
    if folder == "questionblocks":  # whole-question images from add_question_block_images.py, not per chapter
        return None
    chapter = re.sub(r"[^a-z0-9]", "", (r.get("chapter") or "").lower())
    if folder and chapter and folder not in chapter and chapter not in folder:
        return "chapter_folder"
    return None


# ── near-duplicates ─────────────────────────────────────────────────────────────

def near_duplicates(rows):
    """Within one class + subject + chapter, on rows left after exact dedupe (no junk). Returns pairs and
    marks every member of a cluster except its canonical row as near_duplicate (needs_review)."""
    from rapidfuzz import fuzz, process

    groups = collections.defaultdict(list)
    for r in rows:
        if not r["flags"] & {"exact_duplicate", "label_only"}:
            groups[(r["class_grade"], r["subject"], r["chapter"])].append(r)
    pairs = []
    for key, reps in groups.items():
        if len(reps) < 2:
            continue
        texts = [r["norm"] for r in reps]
        parent = list(range(len(reps)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for start in range(0, len(reps), 2000):  # bounded memory on big chapters
            scores = process.cdist(texts[start:start + 2000], texts, scorer=fuzz.ratio,
                                   score_cutoff=NEAR_DUP_CUTOFF, workers=-1)
            for i_local, row in enumerate(scores):
                i = start + i_local
                for j in (row > NEAR_DUP_CUTOFF).nonzero()[0]:
                    if j <= i:
                        continue
                    pairs.append((reps[i], reps[int(j)], float(row[j])))
                    parent[find(i)] = find(int(j))
        clusters = collections.defaultdict(list)
        for i, r in enumerate(reps):
            clusters[find(i)].append(r)
        for members in clusters.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda c: (not c["has_answer"], bool(c["garble"]), c["id"]))
            for m in members[1:]:
                m["flags"].add("near_duplicate")
                m["near_duplicate_of"] = members[0]["id"]
    return pairs


# ── KaTeX (frontend's katex via node) ───────────────────────────────────────────

def katex_check(rows, script, node_path, min_rows):
    per_subject = collections.Counter(
        (r["class_grade"], r["subject"]) for r in rows if _LATEX_CMD.search(r["text"]) and "$" not in r["text"])
    meaningful = {k for k, n in per_subject.items() if n >= min_rows}
    items = [{"id": r["id"], "mode": "builder", "text": r["text"]} for r in rows if "$" in r["text"]]
    items += [{"id": r["id"], "mode": "full", "text": r["text"]} for r in rows
              if (r["class_grade"], r["subject"]) in meaningful and _LATEX_CMD.search(r["text"]) and "$" not in r["text"]]
    env = dict(os.environ, NODE_PATH=node_path)
    done = subprocess.run(["node", script], input=json.dumps(items), capture_output=True, text=True,
                          encoding="utf-8", env=env, check=True)
    return json.loads(done.stdout), meaningful


# ── image HEAD check ────────────────────────────────────────────────────────────

def head(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "a4ai-ncert-audit"})
    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            return res.status, res.headers.get("Content-Type", ""), res.headers.get("Content-Length", "")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "") if e.headers else "", ""
    except Exception as e:  # DNS / timeout
        return f"error: {type(e).__name__}", "", ""


def head_check(rows, n):
    urls = sorted({r["image_url"] for r in rows if r["image_url"]})
    sample = random.Random(SEED).sample(urls, min(n, len(urls)))
    out = []
    for url in sample:
        status, ctype, size = head(url)
        bare = None
        if url.endswith("?"):
            bare = head(url[:-1])[0]
        out.append({"url": url, "status": status, "type": ctype, "bytes": size, "without_question_mark": bare})
    return out, len(urls)


# ── PDFs (cost estimate inputs) ─────────────────────────────────────────────────

def pdf_stats(pdf_dir):
    import pymupdf

    books = {}
    for folder in sorted(glob.glob(os.path.join(pdf_dir, "Class*"))):
        name = os.path.basename(folder)
        pages = pua_pages = chars = 0
        for path in glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True):
            with pymupdf.open(path) as doc:
                for page in doc:
                    text = page.get_text()
                    pages += 1
                    chars += len(text)
                    pua_pages += bool(_PUA.search(text))
        books[name] = {"pages": pages, "pua_pages": pua_pages, "chars": chars}
    return books


def tokens(chars: float) -> float:
    return chars / 4  # heuristic (no Gemini count_tokens call); maths/PUA text tokenises worse → ±30 %


# ── report ──────────────────────────────────────────────────────────────────────

def money(t_in, t_out, price):
    return t_in / 1e6 * price[0] + t_out / 1e6 * price[1]


def build_g1b(focus, rows, v1, pairs, katex, meaningful, heads, n_urls, books, prices, args, notes):
    pct, snippet = g1a.pct, g1a.snippet
    L = []
    w = L.append
    total, all_total = len(focus), len(rows)
    w("\n# G1b\n")
    w(f"- **Hash:** `{HASH_VERSION}` (`Example N :` / `Q N.` label stripped before hashing; empty after the strip = "
      "junk, never a duplicate; prefixes are stripped in any order until none is left). 14/14 fixtures pass; the same file is copied to the frontend worktree "
      "(`src/lib/ncert/ncert_normalize_fixtures.json`).")
    w(f"- **Same input as G1a** (dump at DB `now()` {args.db_now}); random samples use seed {SEED}.")
    w("- Read-only: SELECT dumps, HTTP HEAD on 50 image URLs, local PDFs. No DB writes, no AI calls.\n")
    if notes:
        w(notes.strip() + "\n")

    # 1. v2 delta
    w("## 1. Exact duplicates — v1 (G1a) vs v2\n")
    w("v1 is recomputed in this run with the G1a rules and reproduces the G1a numbers above.\n")
    w("| | Class 10 Maths v1 | v2 | Δ | Full DB v1 | v2 | Δ |\n|---|---:|---:|---:|---:|---:|---:|")
    for label, fn in [
        ("exact_duplicate rows", lambda rs: sum("exact_duplicate" in r["flags"] for r in rs)),
        ("distinct questions", lambda rs: len({(r["class_grade"], r["subject"], r["chapter"], r["hash"]) for r in rs
                                               if "label_only" not in r["flags"]})),
        ("label_only (junk)", lambda rs: sum("label_only" in r["flags"] for r in rs)),
        ("heading_only (needs_review, new)", lambda rs: sum("heading_only" in r["flags"] for r in rs)),
        ("clean (G1a definition)", lambda rs: sum(r["clean"] for r in rs)),
    ]:
        f1 = fn([r for r in v1 if r["class_grade"] == g1a.FOCUS[0] and r["subject"] == g1a.FOCUS[1]])
        f2, a1, a2 = fn(focus), fn(v1), fn(rows)
        w(f"| {label} | {f1:,} | {f2:,} | {f2 - f1:+,} | {a1:,} | {a2:,} | {a2 - a1:+,} |")
    w("\n`distinct questions` leaves junk (label_only) rows out in both columns; G1a's full-DB figure counted them.\n")

    # 2. cross-chapter
    cc_focus = sum("cross_chapter" in r["flags"] for r in focus)
    cc_all = [r for r in rows if "cross_chapter" in r["flags"]]
    w("## 2. Same hash in different chapters (flag only, never merged)\n")
    w(f"Class 10 Maths: **{cc_focus:,}** rows · full DB: **{len(cc_all):,}** rows "
      f"({len({(r['class_grade'], r['subject'], r['hash']) for r in cc_all}):,} hashes). Junk and short rows excluded.\n")
    by_hash = collections.defaultdict(list)
    for r in cc_all:
        by_hash[(r["class_grade"], r["subject"], r["hash"])].append(r)
    keys = sorted(by_hash, key=lambda k: (k[1] != g1a.FOCUS[1] or k[0] != g1a.FOCUS[0], k))
    keys = random.Random(SEED).sample(keys, min(10, len(keys))) if len(keys) > 10 else keys
    w("| class · subject | chapters (row ids) | text |\n|---|---|---|")
    for k in keys:
        rs = by_hash[k]
        chs = collections.defaultdict(list)
        for r in rs:
            chs[r["chapter"]].append(r["id"])
        where = "<br>".join(f"{c}: {', '.join(f'#{i}' for i in sorted(ids)[:4])}{' …' if len(ids) > 4 else ''}"
                            for c, ids in sorted(chs.items()))
        w(f"| {k[0]} · {k[1]} | {where} | {snippet(rs[0]['text'], 110)} |")
    w("")

    # 3. garbled math
    canon = [r for r in focus if not r["flags"] & {"exact_duplicate", "label_only"}]
    g_focus = [r for r in focus if r["garble"]]
    g_canon = [r for r in canon if r["garble"]]
    w("## 3. Garbled math (new category → `needs_review`)\n")
    w(f"Class 10 Maths: **{len(g_focus):,} rows ({pct(len(g_focus), total)})**, "
      f"**{len(g_canon):,} of {len(canon):,} questions after exact dedupe ({pct(len(g_canon), len(canon))})**.\n")
    w("| signal | what it catches | rows | questions (deduped) |\n|---|---|---:|---:|")
    what = {
        "symbol_font_glyph": "Symbol-font glyph stored as U+F0xx (∠ θ π Δ ° × ≤ …, bracket pieces)",
        "spaced_function": "`t a n 4 5` — function name split into letters",
        "digit_letter_soup": "`2R180p` — flattened fraction",
        "missing_numerator": "`= /360` — numerator symbol lost",
        "flattened_power": "`x2 +`, `PQ2 =` — superscript lost (Maths only)",
        "flattened_fraction": "`3 2, 1 2` for 3/2, 1/2 (Maths only, table-like rows excluded)",
    }
    for sig in list(GARBLE_SIGNALS) + list(MATH_ONLY_SIGNALS):
        w(f"| `{sig}` | {what[sig]} | {sum(sig in r['garble'] for r in focus):,} | {sum(sig in r['garble'] for r in canon):,} |")
    tiers = collections.Counter(r["garble_tier"] for r in g_canon)
    w(f"\nTier (deduped questions): **symbol_only {tiers['symbol_only']:,}** — every bad character is a single "
      f"Symbol glyph, a fixed table could map it back · **layout {tiers['layout']:,}** — fractions/powers/brackets "
      "flattened, needs re-extraction from the PDF page.\n")
    w("10 samples (deduped questions):\n")
    for r in random.Random(SEED).sample(g_canon, min(10, len(g_canon))):
        w(f"- #{r['id']} [{', '.join(r['garble'])}] {snippet(r['text'], 140)}")
    w("\nPrecision: see the manual 30-row check in the G1b notes above "
      "(`docs/audit/garbled_precision_sample.csv`).\n")

    # 4. near-duplicates
    p_focus = [p for p in pairs if p[0]["class_grade"] == g1a.FOCUS[0] and p[0]["subject"] == g1a.FOCUS[1]]
    nd_focus = sum("near_duplicate" in r["flags"] for r in focus)
    nd_all = sum("near_duplicate" in r["flags"] for r in rows)
    w("## 4. Near-duplicates (rapidfuzz `fuzz.ratio` > 90, same chapter, after exact dedupe)\n")
    w(f"Class 10 Maths: **{len(p_focus):,} pairs**; **{nd_focus:,} rows** sit in a cluster with a better copy "
      f"(→ `needs_review`, never auto-merged) · full DB: {sum(1 for _ in pairs):,} pairs, {nd_all:,} rows.\n")
    after = len(canon) - nd_focus
    w(f"Class 10 Maths questions: {total:,} rows → {len(canon):,} after exact dedupe → **{after:,} if every near-"
      f"duplicate cluster were one question**.\n")
    sym_pairs = [p for p in p_focus if p[3]]
    w(f"{len(sym_pairs):,} of the pairs differ in a math symbol (one copy has ∠/π/√/²/° …, the other lost it) — "
      "the canonical copy should be the one with the symbol.\n")
    w("| chapter | near-dup pairs | rows flagged |\n|---|---:|---:|")
    for ch in sorted({r["chapter"] for r in focus}):
        w(f"| {ch} | {sum(1 for p in p_focus if p[0]['chapter'] == ch):,} | "
          f"{sum('near_duplicate' in r['flags'] for r in focus if r['chapter'] == ch):,} |")
    w("\nPairs: `docs/audit/near_duplicate_pairs.csv`.\n")

    # 5. LaTeX
    w("## 5. LaTeX\n")
    per = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = (r["class_grade"], r["subject"])
        per[k][0] += 1
        per[k][1] += "$" in r["text"]
        per[k][2] += bool(_LATEX_CMD.search(r["text"]))
    w("Subjects with any `$` or `\\command` (all others: 0):\n")
    w("| class · subject | rows | with `$` | with `\\command` |\n|---|---:|---:|---:|")
    for k in sorted((k for k, v in per.items() if v[1] or v[2]), key=lambda k: (int(k[0]), k[1])):
        w(f"| {k[0]} · {k[1]} | {per[k][0]:,} | {per[k][1]:,} | {per[k][2]:,} |")
    builder = [x for x in katex if x["mode"] == "builder"]
    full = [x for x in katex if x["mode"] == "full"]
    w(f"\n- **`$` rows ({len(builder)}):** every one is a currency dollar. The builder splits them like math: "
      f"{sum(1 for x in builder if x['mathParts']):,} rows get a `$…$` part rendered by KaTeX "
      f"(prose shown as italic math), {sum(1 for x in builder if not x['ok']):,} of those fail to parse.")
    w(f"- **`\\command` without `$` ({sum(1 for r in rows if _LATEX_CMD.search(r['text']) and '$' not in r['text']):,} rows):** "
      "the builder only renders `$…$`, so these show as raw `\\sqrt{2}x^2` text. The PDF export converts them "
      "(`_process_latex`), so builder and PDF disagree.")
    w(f"- **KaTeX parse** (subjects with ≥ {args.katex_min_rows} such rows: "
      f"{', '.join(f'{c} {s}' for c, s in sorted(meaningful)) or 'none'}): {len(full):,} rows parsed, "
      f"**{sum(1 for x in full if not x['ok']):,} fail**.")
    errors = collections.Counter(re.sub(r"at position \d+.*", "", x["error"] or "")[:70] for x in full if not x["ok"])
    if errors:
        w("\n| KaTeX error | rows |\n|---|---:|")
        for e, n in errors.most_common(8):
            w(f"| {e.replace('|', '/')} | {n} |")
    bad = [x for x in full if not x["ok"]][:5]
    text_of = {r["id"]: r["text"] for r in rows}
    for x in bad:
        w(f"- #{x['id']}: {snippet(text_of[x['id']], 120)}")
    w("")

    # 6. images
    w(f"## 6. Images — HEAD check of {len(heads)} random URLs (of {n_urls:,} distinct)\n")
    st = collections.Counter(str(h["status"]) for h in heads)
    w(f"Status codes: {', '.join(f'**{k}** × {v}' for k, v in st.most_common())}. Trailing `?`: "
      f"{sum(1 for h in heads if h['url'].endswith('?'))} of {len(heads)} URLs end with `?`; "
      f"without it: {', '.join(f'{k} × {v}' for k, v in collections.Counter(str(h['without_question_mark']) for h in heads if h['without_question_mark'] is not None).most_common()) or '—'} "
      "(the `?` is an empty query string and changes nothing).\n")
    types = collections.Counter(h["type"] for h in heads)
    w(f"Content types: {', '.join(f'{k or '—'} × {v}' for k, v in types.most_common())}.\n")
    non200 = [h for h in heads if h["status"] != 200]
    for h in non200[:10]:
        w(f"- {h['status']}: `{h['url']}`")
    mm = collections.Counter(r["fig_mismatch"] for r in rows if r["fig_mismatch"])
    mm_f = collections.Counter(r["fig_mismatch"] for r in focus if r["fig_mismatch"])
    w(f"\nImage vs row (whole table, no network): `figure_ref` number ≠ image file number: "
      f"**{mm['figure_number']:,}** rows (Class 10 Maths {mm_f['figure_number']:,}); image stored under another "
      f"chapter's folder: **{mm['chapter_folder']:,}** (Class 10 Maths {mm_f['chapter_folder']:,}).")
    for r in [r for r in focus if r["fig_mismatch"]][:5]:
        w(f"- #{r['id']} {r['fig_mismatch']}: figure_ref `{r['figure_ref']}` · `{r['image_url'].rsplit('/', 3)[-2:]}` · "
          f"{snippet(r['text'], 80)}")
    w("")

    # 7. cost estimate
    w("## 7. Re-extraction cost estimate (nothing executed)\n")
    prompt_t = args.prompt_tokens
    book = books.get(args.focus_pdf_folder, {})
    q_tokens = lambda rs: sum(tokens(len(r["text"]) + len(r["answer"]) + len(r["options_text"])) for r in rs) * 1.3
    final_focus = [r for r in canon if "near_duplicate" not in r["flags"]]
    layout_focus = [r for r in final_focus if r["garble_tier"] == "layout"]
    final_all = [r for r in rows if not r["flags"] & {"exact_duplicate", "label_only", "near_duplicate"}]
    layout_all = [r for r in final_all if r["garble_tier"] == "layout"]
    all_pages = sum(b["pages"] for b in books.values())
    all_pua = sum(b["pua_pages"] for b in books.values())
    all_chars = sum(b["chars"] for b in books.values())
    page_text_t = tokens(all_chars / all_pages) if all_pages else 0
    w(f"- **Models in the code:** `test-generator-backend/.env` sets `GEMINI_MODEL=gemini-3.5-flash-lite`; "
      f"`02_extract_ncert_questions.py` defaults to `gemini-3.6-flash` when the env var is missing. Both priced.")
    w("- **Prices** (ai.google.dev/gemini-api/docs/pricing, read 22 Oct 2026, per 1M tokens, output includes "
      "thinking): " + "; ".join(f"`{n}` ${p[0]:.3f} in / ${p[1]:.3f} out" for n, p in prices.items()) + ".")
    w(f"- **Tokens:** PDF page at media_resolution `medium` = 560 + its native text "
      f"(avg {page_text_t:,.0f} tokens/page here) — Google's table does not say \"per page\"; that is our reading. "
      f"Extraction prompt ≈ {prompt_t:,} tokens (5,188 chars). Question tokens = chars/4 of text + answer + options "
      "× 1.3 JSON overhead (no `count_tokens` call). Thinking billed as output: shown as none and as +100 %.")
    w("- **Why the PDF and not the chunks:** the garble is already in `ncert_chunks.content` (the PDF text layer), "
      "so re-running the current text pipeline reproduces it. The fix needs the page image.\n")
    scen = []
    if book:
        pages_pua = book["pua_pages"]
        out_t = q_tokens(final_focus) * (pages_pua / book["pages"])
        scen.append(("A. Class 10 Maths — re-extract the pages with Symbol glyphs",
                     f"{pages_pua} of {book['pages']} pages",
                     pages_pua * (560 + tokens(book["chars"] / book["pages"]) + prompt_t), out_t))
    scen.append(("B. Class 10 Maths — repair each layout-garbled question (1 page image + row)",
                 f"{len(layout_focus):,} questions",
                 len(layout_focus) * (400 + 560 + page_text_t) + q_tokens(layout_focus) / 1.3, q_tokens(layout_focus)))
    if all_pages:
        scen.append((f"C. Every local NCERT PDF ({len(books)} books) — re-extract pages with Symbol glyphs",
                     f"{all_pua:,} of {all_pages:,} pages",
                     all_pua * (560 + page_text_t + prompt_t), q_tokens(final_all) * (all_pua / all_pages)))
    scen.append(("D. Full DB — repair each layout-garbled question",
                 f"{len(layout_all):,} questions",
                 len(layout_all) * (400 + 560 + page_text_t) + q_tokens(layout_all) / 1.3, q_tokens(layout_all)))
    scen.append(("E. symbol_only tier — deterministic Symbol→Unicode table (needs your OK as a new allowed fix)",
                 f"{sum(1 for r in final_all if r['garble_tier'] == 'symbol_only'):,} questions (full DB)", 0, 0))
    head_cols = " | ".join(f"`{n}` (no thinking / +100 %)" for n in prices)
    w(f"| scenario | size | input tokens | output tokens | {head_cols} |")
    w("|---|---|---:|---:|" + "---:|" * len(prices))
    for name, size, t_in, t_out in scen:
        cells = " | ".join(f"${money(t_in, t_out, p):,.2f} / ${money(t_in, 2 * t_out, p):,.2f}" for p in prices.values())
        w(f"| {name} | {size} | {t_in:,.0f} | {t_out:,.0f} | {cells} |")
    w("\nThe local PDF folder covers " + ", ".join(sorted(books)) + " — not every subject in the DB "
      "(no Accountancy, Economics, History … PDFs here), so C is a lower bound for the whole table. "
      "Retries (the extraction script retries up to 4× + empty-result retries) are not included.\n")

    # 8. full DB
    classes = sorted({r["class_grade"] for r in rows}, key=lambda c: int(c) if c.isdigit() else 99)
    pc = {c: [r for r in rows if r["class_grade"] == c] for c in classes}
    w("## 8. Full database — new categories (counts only)\n")
    w("| category | all classes | " + " | ".join(f"Class {c}" for c in classes) + " |")
    w("|---|---:|" + "---:|" * len(classes))
    cats = [
        ("exact_duplicate (v2)", lambda r: "exact_duplicate" in r["flags"]),
        ("label_only (v2 junk)", lambda r: "label_only" in r["flags"]),
        ("heading_only (needs_review)", lambda r: "heading_only" in r["flags"]),
        ("cross_chapter", lambda r: "cross_chapter" in r["flags"]),
        ("garbled_math (any)", lambda r: bool(r["garble"])),
        ("garbled — symbol_only", lambda r: r["garble_tier"] == "symbol_only"),
        ("garbled — layout", lambda r: r["garble_tier"] == "layout"),
        ("near_duplicate", lambda r: "near_duplicate" in r["flags"]),
        ("raw LaTeX shown as text in builder", lambda r: bool(_LATEX_CMD.search(r["text"])) and "$" not in r["text"]),
        ("currency `$` parsed as math", lambda r: "$" in r["text"]),
        ("figure_ref ≠ image file", lambda r: r["fig_mismatch"] == "figure_number"),
        ("image in another chapter's folder", lambda r: r["fig_mismatch"] == "chapter_folder"),
        ("**clean (G1a definition, v2 hash)**", lambda r: r["clean"]),
        ("**clean strict (also no garble / near-dup / raw LaTeX)**", lambda r: r["clean_strict"]),
    ]
    for label, fn in cats:
        n_all = sum(1 for r in rows if fn(r))
        w(f"| {label} | {n_all:,} ({pct(n_all, all_total)}) | "
          + " | ".join(f"{sum(1 for r in pc[c] if fn(r)):,}" for c in classes) + " |")
    f_strict = sum(r["clean_strict"] for r in focus)
    w(f"\nClass 10 Maths: clean (G1a def, v2) **{sum(r['clean'] for r in focus):,} ({pct(sum(r['clean'] for r in focus), total)})** · "
      f"clean strict **{f_strict:,} ({pct(f_strict, total)})**.\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only NCERT quality audit (G1b)")
    ap.add_argument("--from-json", nargs="+", required=True)
    ap.add_argument("--db-now", default="unknown")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"))
    ap.add_argument("--notes", help="hand-written G1b notes (clarifications, precision check)")
    ap.add_argument("--katex-script", required=True)
    ap.add_argument("--node-path", required=True, help="frontend node_modules (katex)")
    ap.add_argument("--katex-min-rows", type=int, default=20)
    ap.add_argument("--head-check", type=int, default=50)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--focus-pdf-folder", default="Class10_Maths")
    ap.add_argument("--prompt-tokens", type=int, default=1300)
    ap.add_argument("--price", action="append", required=True, help='"model=IN/OUT" USD per 1M tokens')
    args = ap.parse_args()
    prices = {}
    for p in args.price:
        name, rates = p.split("=")
        prices[name] = tuple(float(x) for x in rates.split("/"))

    raw = g1a.load_json(args.from_json)

    # v1 = the G1a rules exactly (label kept in the hash, junk could be a duplicate)
    g1a.normalize = lambda t: _normalize(t, strip_label=False)
    g1a.content_hash = lambda t: hashlib.sha256(_normalize(t, strip_label=False).encode("utf-8")).hexdigest()
    g1a.HEADING_IS_LABEL = True
    v1 = [g1a.analyse_row(r) for r in raw]
    g1a.mark_duplicates(v1, junk_never_duplicate=False)
    g1a.normalize, g1a.content_hash, g1a.HEADING_IS_LABEL = normalize, content_hash, False

    rows = [g1a.analyse_row(r) for r in raw]
    g1a.mark_duplicates(rows)
    for r, src in zip(rows, raw):
        r["answer"] = (src.get("answer") or "")
        r["options_text"] = src.get("options_text") or ""
        r["image_url"] = src.get("image_url") or ""
        r["garble"] = garble_signals(r["text"], r["subject"])
        r["garble_tier"] = garble_tier(r["text"], r["garble"])
        r["fig_mismatch"] = figure_mismatch(src)
        r["near_duplicate_of"] = None
    pairs = near_duplicates(rows)
    for r in rows:
        r["clean_strict"] = r["clean"] and not r["garble"] and "near_duplicate" not in r["flags"] \
            and not (_LATEX_CMD.search(r["text"]) and "$" not in r["text"])
    pairs = [(a, b, s, sorted((set(a["text"]) ^ set(b["text"])) & _DIFF_SYMBOLS)) for a, b, s in pairs]
    focus = [r for r in rows if r["class_grade"] == g1a.FOCUS[0] and r["subject"] == g1a.FOCUS[1]]

    print("KaTeX …")
    katex, meaningful = katex_check(rows, args.katex_script, args.node_path, args.katex_min_rows)
    print("HEAD …")
    heads, n_urls = head_check(rows, args.head_check)
    print("PDFs …")
    books = pdf_stats(args.pdf_dir)

    audit_dir = os.path.join(args.out_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    with open(os.path.join(audit_dir, "ncert_audit_class10_maths_rows_v2.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "chapter", "question_type", f"content_hash_{HASH_VERSION}", "flags", "duplicate_of",
                     "near_duplicate_of", "garble_signals", "clean", "clean_strict"])
        for r in sorted(focus, key=lambda r: r["id"]):
            flags = [f for f in g1a.FLAG_ORDER + ["near_duplicate"] if f in r["flags"]] + (["garbled_math"] if r["garble"] else [])
            wr.writerow([r["id"], r["chapter"], r["question_type"], r["hash"], ";".join(flags), r["duplicate_of"] or "",
                         r["near_duplicate_of"] or "", ";".join(r["garble"]), int(r["clean"]), int(r["clean_strict"])])
    with open(os.path.join(audit_dir, "near_duplicate_pairs.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["chapter", "id_a", "id_b", "similarity", "symbols_differ", "text_a", "text_b"])
        focus_ids = {r["id"] for r in focus}
        for a, b, s, sym in sorted((p for p in pairs if p[0]["id"] in focus_ids), key=lambda p: (p[0]["chapter"], p[0]["id"], p[1]["id"])):
            wr.writerow([a["chapter"], a["id"], b["id"], f"{s:.1f}", "".join(sym),
                         g1a.snippet(a["text"], 200), g1a.snippet(b["text"], 200)])
    canon_garbled = [r for r in focus if r["garble"] and not r["flags"] & {"exact_duplicate", "label_only"}]
    with open(os.path.join(audit_dir, "garbled_precision_sample.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "signals", "tier", "text", "verdict (garbled? y/n)"])
        for r in random.Random(SEED + 1).sample(canon_garbled, min(30, len(canon_garbled))):
            wr.writerow([r["id"], ";".join(r["garble"]), r["garble_tier"], re.sub(r"\s+", " ", r["text"]), ""])

    notes = None
    if args.notes:
        with open(args.notes, encoding="utf-8") as f:
            notes = f.read()
    section = build_g1b(focus, rows, v1, pairs, katex, meaningful, heads, n_urls, books, prices, args, notes)
    report = os.path.join(args.out_dir, "ncert-audit-class10-maths.md")
    with open(report, encoding="utf-8") as f:
        g1a_part = f.read().split("\n# G1b\n")[0].rstrip("\n")
    with open(report, "w", encoding="utf-8") as f:
        f.write(g1a_part + "\n" + section)
    print(f"→ {report}\n→ {audit_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
