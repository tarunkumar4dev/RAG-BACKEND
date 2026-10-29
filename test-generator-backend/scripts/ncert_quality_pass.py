#!/usr/bin/env python3
"""
scripts/ncert_quality_pass.py — writes quality_status / duplicate_of / quality_flags / content_hash
(migration 19) and the allowed deterministic fixes for NCERT questions. DRY RUN unless told otherwise.

Per row (in one class + subject, or everything):
  fixes (logged field by field in ncert_question_fixes, originals stay in a4_backup_20261022):
    symbol_font_map      Adobe Symbol glyphs stored as U+F0xx → Unicode (∠ θ π Δ ° × ≤ …), text/answer/options
    figure_ref_from_image  figure_ref ← "Fig. X.Y" of the image file, only when the question text names X.Y too
    trim_whitespace / literal_newline / doubled_escape   (the brief's allowed list; "balance a single $" is
                         not done: every $ in the table is a currency sign)
  status (on the fixed text):
    junk          label only / empty after normalize
    duplicate     same chapter and (same content_hash, or same skeleton = text without math symbols,
                  whitespace and punctuation). Canonical = most math symbols > has answer > figure OK > lowest id.
                  Skeleton-only merges need ≥ 20 skeleton chars and the same operator sequence (+ − = < > …) in
                  every ungarbled copy — otherwise needs_review ("x + y = 5" vs "x − y = 5" share a skeleton).
    needs_review  near-duplicate (rapidfuzz > 90, same chapter), short text, heading, garbled math left after the
                  fixes, MCQ answer problems, figure missing, raw LaTeX, cross-chapter, currency $  (all visible)
    fixed         a fix was applied and nothing needs review
    clean         nothing to do

Modes (rows from --from-json dumps or DATABASE_URL):
  (default)            dry run: report + CSVs, nothing written
  --emit-sql FILE      SQL for --table / --fixes-table (run it yourself, e.g. on a copy table)
  --emit-dir DIR       the same, one statement per file (the Supabase Management API rejects big bodies: 413)
  --apply              DATABASE_URL: same statements, one transaction per chunk of --chunk rows
  --apply --rollback   DATABASE_URL: everything in ONE transaction, timed, then ROLLBACK (G4.5 timing run)

  python scripts/ncert_quality_pass.py --from-json dump.json --class-grade 10 --subject Mathematics \\
      --pdf-dir ../ncert-ingestion/pdfs --emit-sql out.sql --table a4_backup_20261022.ncert_q_c10m_copy \\
      --fixes-table a4_backup_20261022.ncert_q_c10m_copy_fixes
Needs scripts/requirements-audit.txt (rapidfuzz, numpy).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import scripts.audit_ncert_quality as g1a  # noqa: E402
import scripts.audit_ncert_quality_g1b as g1b  # noqa: E402
from app.services.ncert_text_hash import HASH_VERSION  # noqa: E402

SEED = 20261022
SKELETON_MIN = 20
NEAR_DUP_CUTOFF = 90.0
HIDDEN = {"duplicate", "junk"}
# counted for "most math symbols intact"; ₹ (a backtick in garbled copies) and ^ (n^2 vs n2) included
MATH_SYMBOLS = set("πθΔδαβγλμρσφωΩΣ∠√∛²³¹⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄°×÷≤≥≠≈≅≡⊥∥′″∞∈∉⊂⇒→↔·±∓∴∵△₹^")
_OPERATORS = re.compile(r"[+\-−–=<>≤≥×÷]")
_LITERAL_NEWLINE = re.compile(r"\\n(?![A-Za-z])")  # "\n" text artifact, not \neq / \nabla
_DOUBLED_ESCAPE = re.compile(r"\\\\(?=(?:frac|sqrt|sin|cos|tan|cot|sec|csc|log|ln|lim|int|sum|pi|theta|alpha|beta|"
                             r"gamma|delta|times|div|pm|leq|geq|neq|angle|circ|cdot|left|right|text|mathrm)\b)")
_FIG_IN_TEXT = re.compile(r"fig(?:ure)?\.?\s*(\d+)\s*\.\s*(\d+)", re.I)


# ── fixes ───────────────────────────────────────────────────────────────────────

def map_symbols(s: str) -> str:
    return "".join(g1b.SYMBOL_GLYPHS.get(ord(c), c) for c in s)


def fix_text(s: str) -> tuple[str, list[str]]:
    rules = []
    new = map_symbols(s)
    if new != s:
        rules.append("symbol_font_map")
    t = _LITERAL_NEWLINE.sub(" ", new)
    if t != new:
        rules.append("literal_newline")
    new, t = t, _DOUBLED_ESCAPE.sub(r"\\", t)
    if t != new:
        rules.append("doubled_escape")
    new, t = t, t.strip()
    if t != new:
        rules.append("trim_whitespace")
    return t, rules


def figure_ref_fix(text: str, figure_ref: str | None, image_url: str | None) -> str | None:
    img = g1b._URL_FIG_NO.search(image_url or "")
    if not img:
        return None
    if img.groups() not in {m.groups() for m in _FIG_IN_TEXT.finditer(text)}:
        return None  # only when the question text names the same figure as the image file
    ref = _FIG_IN_TEXT.search(figure_ref or "")
    if ref and ref.groups() == img.groups():
        return None
    return f"Fig. {img.group(1)}.{img.group(2)}"


def apply_fixes(src):
    """Returns the row with fixed fields and the field-level change log."""
    row, log = dict(src), []
    for field in ("question_text", "answer"):
        if src.get(field):
            new, rules = fix_text(src[field])
            if rules:
                row[field] = new
                log.append((field, src[field], new, "+".join(rules)))
    opts = g1a.parse_options(src.get("options_text"))
    if opts:
        fixed = [fix_text(o)[0] for o in opts]
        if fixed != opts:
            row["options_text"] = json.dumps(fixed, ensure_ascii=False)
            log.append(("options", src.get("options_text"), row["options_text"], "symbol_font_map"))
    new_ref = figure_ref_fix(row["question_text"], src.get("figure_ref"), src.get("image_url"))
    if new_ref:
        row["figure_ref"] = new_ref
        log.append(("figure_ref", src.get("figure_ref"), new_ref, "figure_ref_from_image"))
    return row, log


# ── classification ──────────────────────────────────────────────────────────────

def skeleton(norm: str) -> str:
    return "".join(c for c in norm if c.isalnum() and c not in MATH_SYMBOLS and not ("" <= c <= ""))


def symbol_count(r) -> int:
    return sum(c in MATH_SYMBOLS for c in r["text"] + r["options_text"])


def canonical_key(r):
    figure_bad = bool(r["flags"] & {"diagram_type_no_image", "diagram_text_no_image"})
    return (-symbol_count(r), not r["has_answer"], figure_bad, r["id"])


def canonical_reason(winner, others) -> str:
    parts = [f"{symbol_count(winner)} math symbols"]
    best_other = min(others, key=canonical_key)
    if symbol_count(winner) == symbol_count(best_other):
        if winner["has_answer"] and not best_other["has_answer"]:
            parts.append("has the answer")
        elif canonical_key(winner)[2] != canonical_key(best_other)[2]:
            parts.append("figure OK")
        else:
            parts.append("lowest id")
    else:
        parts.append(f"vs {symbol_count(best_other)} in the next best copy")
    return ", ".join(parts)


def classify(rows):
    """Sets status / duplicate_of / review flags in place. Returns cluster info for the report."""
    by_chapter = collections.defaultdict(list)
    for r in rows:
        r.update(status=None, duplicate_of=None, review=set(), dup_reason=None, near_duplicate_of=None)
        if "label_only" in r["flags"]:
            r["status"] = "junk"
        else:
            by_chapter[(r["class_grade"], r["subject"], r["chapter"])].append(r)

    clusters = []
    for members in by_chapter.values():
        groups = collections.defaultdict(list)
        for r in members:
            sk = r["skeleton"]
            groups[("sk", sk) if len(sk) >= SKELETON_MIN else ("hash", r["hash"])].append(r)
        for (kind, _), grp in groups.items():
            if len(grp) < 2:
                continue
            sigs = {r["operators"] for r in grp if not r["garble"]}
            if kind == "sk" and len(sigs) > 1:
                # same skeleton, different operators: only exact copies merge, the rest is reviewed
                by_hash = collections.defaultdict(list)
                for r in grp:
                    by_hash[r["hash"]].append(r)
                for sub in by_hash.values():
                    if len(sub) > 1:
                        _merge(sub, "exact_hash", clusters)
                for r in grp:
                    if r["status"] != "duplicate":
                        r["review"].add("skeleton_operator_conflict")
                continue
            exact = len({r["hash"] for r in grp}) == 1
            _merge(grp, "exact_hash" if exact else "same_skeleton", clusters)
    _near_duplicates(rows)

    for r in rows:
        if r["status"]:
            continue
        r["review"] |= r["flags"] & {"short_text", "heading_only", "mcq_no_answer", "mcq_answer_not_in_options",
                                     "mcq_type_no_options", "diagram_type_no_image", "diagram_text_no_image",
                                     "cross_chapter"}
        if r["garble"]:
            r["review"].add("garbled_math")
        if g1b._LATEX_CMD.search(r["text"]) and "$" not in r["text"]:
            r["review"].add("raw_latex")
        if "$" in r["text"]:
            r["review"].add("currency_dollar")
        r["status"] = "needs_review" if r["review"] else ("fixed" if r["fix_log"] else "clean")
    return clusters


def _merge(grp, reason, clusters):
    grp = sorted(grp, key=canonical_key)
    winner = grp[0]
    for r in grp[1:]:
        r["status"], r["duplicate_of"], r["dup_reason"] = "duplicate", winner["id"], reason
    clusters.append({"winner": winner, "members": grp, "reason": reason, "why": canonical_reason(winner, grp[1:])})


def _near_duplicates(rows):
    from rapidfuzz import fuzz, process

    by_chapter = collections.defaultdict(list)
    for r in rows:
        if r["status"] is None:
            by_chapter[(r["class_grade"], r["subject"], r["chapter"])].append(r)
    for reps in by_chapter.values():
        if len(reps) < 2:
            continue
        texts = [r["norm"] for r in reps]
        parent = list(range(len(reps)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        scores = process.cdist(texts, texts, scorer=fuzz.ratio, score_cutoff=NEAR_DUP_CUTOFF, workers=-1)
        for i, row in enumerate(scores):
            for j in (row > NEAR_DUP_CUTOFF).nonzero()[0]:
                if j > i:
                    parent[find(i)] = find(int(j))
        groups = collections.defaultdict(list)
        for i, r in enumerate(reps):
            groups[find(i)].append(r)
        for grp in groups.values():
            if len(grp) < 2:
                continue
            grp.sort(key=canonical_key)
            for r in grp:
                r["review"].add("near_duplicate")
            for r in grp[1:]:
                r["near_duplicate_of"] = grp[0]["id"]


def prepare(src):
    fixed, log = apply_fixes(src)
    r = g1a.analyse_row(fixed)
    r.update(
        answer=fixed.get("answer") or "", options_text=fixed.get("options_text") or "",
        image_url=fixed.get("image_url") or "", figure_ref=fixed.get("figure_ref"),
        fix_log=log, skeleton=skeleton(r["norm"]),
        operators="".join("-" if c in "−–" else c for c in _OPERATORS.findall(fixed["question_text"] or "")),
        garble=g1b.garble_signals(fixed["question_text"] or "", fixed["subject"], bool(src.get("has_question_table"))),
    )
    return r


def quality_flags(r) -> dict:
    out = {"v": HASH_VERSION, "flags": sorted(r["review"])}
    if r["garble"]:
        out["garble"] = r["garble"]
    if r["fix_log"]:
        out["fixes"] = sorted({rule for *_, rule in r["fix_log"]})
    if r["dup_reason"]:
        out["duplicate_reason"] = r["dup_reason"]
    if r["near_duplicate_of"]:
        out["near_duplicate_of"] = r["near_duplicate_of"]
    if r["flags"] & {"diagram_type_no_image", "diagram_text_no_image"}:
        out["figure_missing"] = True
    return out


# ── PDF sanity counts ───────────────────────────────────────────────────────────

def pdf_question_counts(folder):
    """Per chapter PDF: number of solved examples and of numbered exercise questions (sub-parts not counted)."""
    import pymupdf

    out = {}
    for path in sorted(os.listdir(folder)):
        if not path.lower().endswith(".pdf"):
            continue
        with pymupdf.open(os.path.join(folder, path)) as doc:
            text = "\n".join(p.get_text() for p in doc)
        examples = len({int(n) for n in re.findall(r"Example\s+(\d+)\s*:", text)})
        exercise_q = 0
        heads = list(re.finditer(r"EXERCISE\s+\d+\.\d+", text))
        for k, h in enumerate(heads):
            end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
            block = text[h.end():end]
            # stop at the next section heading ("1.3 Revisiting …", "2.4 Summary"): the summary's numbered
            # points and later theorems are not exercise questions
            nxt = re.search(r"(?m)^\s*\d+\.\d+\s+[A-Z][a-z]", block)
            block = block[:nxt.start()] if nxt else block
            n = 0
            for m in re.finditer(r"(?m)^\s*(\d{1,2})\.\s", block):
                if int(m.group(1)) == n + 1:
                    n += 1
            exercise_q += n
        out[os.path.splitext(path)[0].lower()] = {"examples": examples, "exercise": exercise_q}
    return out


# ── SQL ─────────────────────────────────────────────────────────────────────────

def lit(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, int):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def sql_statements(rows, table, fixes_table, chunk):
    stmts = []
    ordered = sorted(rows, key=lambda r: r["id"])
    for i in range(0, len(ordered), chunk):
        part = ordered[i:i + chunk]
        values = ",\n  ".join(
            f"({r['id']}, {lit(r['status'])}, {lit(r['duplicate_of'])}::bigint, {lit(json.dumps(quality_flags(r), ensure_ascii=False))}, "
            f"{lit(r['hash'])})" for r in part)
        stmts.append(f"update {table} t set quality_status = v.s, duplicate_of = v.d, quality_flags = v.f::jsonb, "
                     f"content_hash = v.h, hash_version = {lit(HASH_VERSION)}\nfrom (values\n  {values}\n) v(id, s, d, f, h)\n"
                     f"where t.id = v.id;")
    log = [(r["id"], *entry) for r in ordered for entry in r["fix_log"]]
    for i in range(0, len(log), chunk):
        part = log[i:i + chunk]
        for field in ("question_text", "answer", "figure_ref", "options"):
            vals = [(qid, new) for qid, f, _old, new, _rule in part if f == field]
            if not vals:
                continue
            setter = "to_jsonb(v.x::text)" if field == "options" else "v.x"
            body = ",\n  ".join(f"({qid}, {lit(new)})" for qid, new in vals)
            stmts.append(f"update {table} t set {field} = {setter}\nfrom (values\n  {body}\n) v(id, x)\nwhere t.id = v.id;")
        body = ",\n  ".join(f"({qid}, {lit(f)}, {lit(old)}, {lit(new)}, {lit(rule)})" for qid, f, old, new, rule in part)
        stmts.append(f"insert into {fixes_table} (question_id, field, old_value, new_value, rule) values\n  {body};")
    return stmts


# ── report ──────────────────────────────────────────────────────────────────────

def report(rows, clusters, args, pdf_counts, g1a_clean):
    pct = g1a.pct
    L = []
    w = L.append
    total = len(rows)
    st = collections.Counter(r["status"] for r in rows)
    w(f"# NCERT quality pass — dry run, Class {args.class_grade} {args.subject} (G4.4)\n")
    w("- **Repo / branch:** RAG-BACKEND `fix/ncert-builder-quality` · `scripts/ncert_quality_pass.py`")
    w(f"- **Input:** {args.source} · {total:,} rows · hash `{HASH_VERSION}` · seed {SEED}")
    w(f"- **Target of the emitted SQL:** `{args.table}` (+ `{args.fixes_table}`) — production rows untouched.\n")
    if args.notes:
        with open(args.notes, encoding="utf-8") as f:
            w(f.read().strip() + "\n")
    w("## Statuses\n")
    w("| status | rows | % | visible to teachers |\n|---|---:|---:|---|")
    for s in ("clean", "fixed", "needs_review", "duplicate", "junk"):
        w(f"| `{s}` | {st[s]:,} | {pct(st[s], total)} | {'no' if s in HIDDEN else 'yes'} |")
    dup = collections.Counter(r["dup_reason"] for r in rows if r["status"] == "duplicate")
    w(f"\nDuplicates: {dup['exact_hash']:,} exact content_hash · {dup['same_skeleton']:,} same skeleton (rule 5). "
      f"Skeleton groups held back by the operator guard: "
      f"{len({r['skeleton'] for r in rows if 'skeleton_operator_conflict' in r['review']}):,} "
      f"({sum('skeleton_operator_conflict' in r['review'] for r in rows):,} rows → needs_review).\n")
    rev = collections.Counter(f for r in rows if r["status"] == "needs_review" for f in r["review"])
    w("needs_review reasons (a row can have several): "
      + ", ".join(f"`{k}` {v:,}" for k, v in rev.most_common()) + ".\n")
    fixes = collections.Counter(rule for r in rows for *_, rule in r["fix_log"])
    fx_rows = sum(bool(r["fix_log"]) for r in rows)
    w(f"Deterministic fixes: {fx_rows:,} rows changed — " + ", ".join(f"`{k}` {v:,}" for k, v in fixes.most_common())
      + ". Every change is in `docs/audit/quality_pass_dryrun_fixes.csv` (and `ncert_question_fixes` when applied). "
      "Not done: *balance a single `$`* — every `$` in the table is a currency sign.\n")

    w("## Visible questions per chapter\n")
    names = sorted({r["chapter"] for r in rows})
    w("| chapter | before | after | Δ | clean | fixed | needs_review | of which near-dup copies |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ch in names:
        rs = [r for r in rows if r["chapter"] == ch]
        vis = [r for r in rs if r["status"] not in HIDDEN]
        c = collections.Counter(r["status"] for r in vis)
        w(f"| {ch} | {len(rs):,} | {len(vis):,} | {len(vis) - len(rs):+,} | {c['clean']:,} | {c['fixed']:,} "
          f"| {c['needs_review']:,} | {sum(1 for r in vis if r['near_duplicate_of']):,} |")
    vis_all = sum(r["status"] not in HIDDEN for r in rows)
    w(f"| **total** | **{total:,}** | **{vis_all:,}** | **{vis_all - total:+,}** | | | | "
      f"**{sum(1 for r in rows if r['status'] not in HIDDEN and r['near_duplicate_of']):,}** |\n")

    w("### Sanity check against the NCERT book\n")
    w("Like for like: visible rows **with near-duplicate copies collapsed** (one row per near-dup cluster), split "
      "into solved examples (`Example N`) and exercise-type rows (exercise / mcq / fillblank / diagram that is not an "
      "example). `NCERT` = counted in the chapter PDF: distinct `Example N :` headings and numbered questions under "
      "`EXERCISE x.y` (sub-parts (i), (ii) count once — the table often stores them as separate rows, so exercise "
      "ratios above 1 are expected). In-text questions and activities are not in either column. "
      "⚠️ = ratio outside 0.75–2.0. `Ex N >1` = Example numbers still shown more than once after collapsing.\n")
    w("| chapter | examples visible / NCERT | exercise rows visible / NCERT questions | Ex N >1 |")
    w("|---|---:|---:|---:|")

    def ratio_cell(a, b):
        if not b:
            return f"{a} / —"
        r = a / b
        return f"{a} / {b} = {r:.2f}{' ⚠️' if not 0.75 <= r <= 2.0 else ''}"

    for ch in names:
        heads = [r for r in rows if r["chapter"] == ch and r["status"] not in HIDDEN and not r["near_duplicate_of"]]
        ex = [r for r in heads if r["example_no"] is not None or r["question_type"] == "example"]
        exr = [r for r in heads if r not in ex and r["question_type"] in ("exercise", "mcq", "fillblank", "diagram")]
        repeats = sum(1 for n, k in collections.Counter(r["example_no"] for r in ex if r["example_no"]).items() if k > 1)
        n = pdf_counts.get(ch.lower(), {})
        w(f"| {ch} | {ratio_cell(len(ex), n.get('examples'))} | {ratio_cell(len(exr), n.get('exercise'))} | {repeats} |")
    w("")

    w("## G5 pre-check — rows that were clean in G1a and are now hidden\n")
    hidden_clean = [r for r in rows if r["id"] in g1a_clean and r["status"] in HIDDEN]
    by_id = {r["id"]: r for r in rows}
    orphan = [r for r in hidden_clean if r["status"] != "duplicate" or by_id.get(r["duplicate_of"], {}).get("status") in HIDDEN]
    reasons = collections.Counter(r["dup_reason"] or r["status"] for r in hidden_clean)
    w(f"**{len(hidden_clean):,}** G1a-clean rows become hidden ({', '.join(f'{k} {v:,}' for k, v in reasons.items()) or '—'}). "
      f"Every one is a `duplicate` whose canonical row stays visible: **{'yes' if not orphan else f'NO — {len(orphan)} orphans'}**. "
      "G1a had no near-duplicate / skeleton rule, so its \"clean\" counted these copies as separate questions; the "
      "G5 criterion should read \"no G1a-clean *question* disappears\" (its canonical copy is visible).\n")

    w("## 20 sample clusters — which copy was kept and why\n")
    rnd = random.Random(SEED)
    sk = [c for c in clusters if c["reason"] == "same_skeleton"]
    ex = [c for c in clusters if c["reason"] == "exact_hash"]
    pick = rnd.sample(sk, min(12, len(sk))) + rnd.sample(ex, min(20 - min(12, len(sk)), len(ex)))
    for n_c, c in enumerate(pick, 1):
        win = c["winner"]
        w(f"**{n_c}. {win['chapter']}** — {c['reason']} · {len(c['members'])} rows · kept **#{win['id']}** ({c['why']})\n")
        shown = collections.OrderedDict()
        for m in c["members"]:
            shown.setdefault(m["text"], m)
        for text, m in list(shown.items())[:3]:
            tag = "kept" if m is win else f"dup of #{m['duplicate_of']}"
            w(f"- #{m['id']} ({tag}, {symbol_count(m)} sym{', answer' if m['has_answer'] else ''}): {g1a.snippet(text, 150)}")
        if len(shown) > 3:
            w(f"- … {len(shown) - 3} more distinct texts")
        w("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="NCERT quality pass (dry run by default)")
    ap.add_argument("--from-json", nargs="+", help="dumps with id, class_grade, subject, chapter, section, question_number, "
                    "question_text, question_type, answer, options_text, figure_ref, image_url, has_question_table")
    ap.add_argument("--class-grade")
    ap.add_argument("--subject")
    ap.add_argument("--table", default="public.ncert_questions")
    ap.add_argument("--fixes-table", default="public.ncert_question_fixes")
    ap.add_argument("--emit-sql")
    ap.add_argument("--emit-dir", help="write each statement to its own NNNN.sql (own transaction)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rollback", action="store_true", help="with --apply: one transaction, timed, rolled back")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--pdf-dir", help="folder of the chapter PDFs for the sanity check")
    ap.add_argument("--g1a-csv", help="G1a per-row CSV (rows clean in G1a)")
    ap.add_argument("--notes", help="hand-written markdown put under the report header")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"))
    args = ap.parse_args()

    if args.from_json:
        raw = g1a.load_json(args.from_json)
        args.source = "JSON dump"
    elif os.environ.get("DATABASE_URL"):
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        conn.set_session(readonly=True)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT {g1a.SQL_COLUMNS}, question_table IS NOT NULL AS has_question_table FROM {args.table} ORDER BY id")
        raw = [dict(r) for r in cur.fetchall()]
        conn.close()
        args.source = f"DATABASE_URL {args.table}"
    else:
        print("Pass --from-json or set DATABASE_URL in the terminal.", file=sys.stderr)
        return 2
    raw = [r for r in raw if (not args.class_grade or str(r["class_grade"]) == args.class_grade)
           and (not args.subject or r["subject"] == args.subject)]
    if not raw:
        print("No rows in scope.", file=sys.stderr)
        return 1

    rows = [prepare(r) for r in raw]
    g1a.mark_duplicates(rows)  # only for the cross_chapter flag (same hash in another chapter)
    for r in rows:
        r["flags"].discard("exact_duplicate")
    clusters = classify(rows)

    pdf_counts = pdf_question_counts(args.pdf_dir) if args.pdf_dir else {}
    g1a_clean = set()
    if args.g1a_csv:
        with open(args.g1a_csv, encoding="utf-8") as f:
            g1a_clean = {int(x["id"]) for x in csv.DictReader(f) if x["clean"] == "1"}

    audit_dir = os.path.join(args.out_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    tag = f"class{args.class_grade or 'all'}_{(args.subject or 'all').lower().replace(' ', '_')}"
    with open(os.path.join(audit_dir, f"quality_pass_dryrun_{tag}_rows.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "chapter", "status", "duplicate_of", "duplicate_reason", "near_duplicate_of", "review", "fixes", "content_hash"])
        for r in sorted(rows, key=lambda r: r["id"]):
            wr.writerow([r["id"], r["chapter"], r["status"], r["duplicate_of"] or "", r["dup_reason"] or "",
                         r["near_duplicate_of"] or "", ";".join(sorted(r["review"])),
                         ";".join(sorted({x[3] for x in r["fix_log"]})), r["hash"]])
    with open(os.path.join(audit_dir, f"quality_pass_dryrun_{tag}_fixes.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "field", "old", "new", "rule"])
        for r in sorted(rows, key=lambda r: r["id"]):
            for field, old, new, rule in r["fix_log"]:
                wr.writerow([r["id"], field, old, new, rule])
    with open(os.path.join(args.out_dir, f"ncert-quality-pass-dryrun-{tag}.md"), "w", encoding="utf-8") as f:
        f.write(report(rows, clusters, args, pdf_counts, g1a_clean))

    stmts = sql_statements(rows, args.table, args.fixes_table, args.chunk)
    if args.emit_sql:
        with open(args.emit_sql, "w", encoding="utf-8") as f:
            f.write("begin;\n" + "\n".join(stmts) + "\ncommit;\n")
    if args.emit_dir:  # one file per statement, for runners with a request-size limit (Management API: 413)
        os.makedirs(args.emit_dir, exist_ok=True)
        for i, s in enumerate(stmts, 1):
            with open(os.path.join(args.emit_dir, f"{i:04d}.sql"), "w", encoding="utf-8") as f:
                f.write("begin;\n" + s + "\ncommit;\n")
    if args.apply:
        import psycopg2
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()
        t0 = time.time()
        for i, s in enumerate(stmts, 1):
            cur.execute(s)
            if not args.rollback:
                conn.commit()
            print(f"  statement {i}/{len(stmts)} {time.time() - t0:.1f}s")
        if args.rollback:
            conn.rollback()
            print(f"ROLLED BACK after {time.time() - t0:.1f}s (nothing kept)")
        conn.close()

    st = collections.Counter(r["status"] for r in rows)
    print(f"{len(rows):,} rows → " + ", ".join(f"{k} {v:,}" for k, v in st.most_common())
          + f" · {len(stmts)} SQL statements{' (written to ' + args.emit_sql + ')' if args.emit_sql else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
