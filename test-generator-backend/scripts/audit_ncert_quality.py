#!/usr/bin/env python3
"""
scripts/audit_ncert_quality.py — G1a quality audit of ncert_questions. READ ONLY (SELECT only).

Rows come from either
  * DATABASE_URL set in the terminal → psycopg2, session READ ONLY, one SELECT; or
  * --from-json FILE ... → dumps of the same SELECT, e.g.
      supabase db query --linked -o json "select <SQL_COLUMNS> from ncert_questions where class_grade='10'"

Writes (default --out-dir docs):
  ncert-audit-class10-maths.md                 report: Class 10 Maths detail + full-DB counts
  audit/ncert_audit_class10_maths_rows.csv     one line per Class 10 Maths row: id, hash, flags, clean
                                               → G5 baseline ("no row clean in G1 is hidden later")

Categories (G1a). Near-duplicates (similarity > 0.9), KaTeX parsing, truncated/merged rows and
image HEAD checks are G1b and are NOT computed here.
  exact_duplicate   same content_hash in the same class + subject + chapter (section NOT in the key);
                    every copy except the canonical one (has answer > figure has image > lowest id);
                    label_only rows are junk and never duplicates
  cross_chapter     same content_hash in another chapter of the same class + subject — counted only,
                    never a duplicate
  label_only        nothing but "Example 1" / "Exercise 3.2", or empty after normalize (junk)
  heading_only      text == its section heading / question_number ("What is a mineral?") → needs_review
  short_text        < 15 letters/digits after the label — a candidate, read the samples
  mcq_no_answer     options present, answer empty
  mcq_answer_not_in_options   answer matches no option text and no option letter
  mcq_type_no_options         question_type 'mcq' without options
  diagram_text_no_image       text says figure/fig/diagram, type != 'diagram', no image_url
  diagram_type_no_image       question_type 'diagram' without image_url
  clean             none of the above flags except cross_chapter (answer NULL on non-MCQs is normal)

Usage (from test-generator-backend/):
  python scripts/audit_ncert_quality.py
  python scripts/audit_ncert_quality.py --from-json dump/class_*.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.ncert_text_hash import content_hash, normalize  # noqa: E402

SQL_COLUMNS = (
    "id, class_grade, subject, chapter, section, question_number, question_text, question_type, answer, "
    "options #>> '{}' as options_text, jsonb_typeof(options) as options_jsonb_type, figure_ref, image_url, created_at"
)
FOCUS = ("10", "Mathematics")

_LABEL = r"(?:example|exercise|ex|q|question)\s*[\d.]*\s*[:.\-–]?"
_LABEL_ONLY = re.compile(rf"^(?:{_LABEL}|activity|solution|answer)$")
_LEADING_LABEL = re.compile(rf"^{_LABEL}\s*")
_EXAMPLE_NO = re.compile(r"^\s*example\s*(\d+)", re.I)
_FIGURE_WORD = re.compile(r"\b(?:figure|fig|diagram)s?\b", re.I)
_OPTION_LABEL = re.compile(r"^\s*(?:\(\s*[a-fA-F]\s*\)|[a-fA-F]\s*[.)])\s*")
_ANSWER_PREFIX = re.compile(r"^(?:(?:the\s+)?(?:correct\s+)?(?:answer|ans|option)\s*(?:is)?\s*[:.\-]?\s*)")
_ANSWER_LETTER = re.compile(r"^(?:\(\s*([a-f])\s*\)|([a-f])\s*[.)]|([a-f])$)")

FLAG_ORDER = [
    "exact_duplicate", "label_only", "heading_only", "short_text", "mcq_no_answer", "mcq_answer_not_in_options",
    "mcq_type_no_options", "diagram_text_no_image", "diagram_type_no_image", "cross_chapter",
]
NOT_CLEAN = set(FLAG_ORDER) - {"cross_chapter"}
# G1a counted "text == its section / question_number" as label_only; G1b found those are real headings
# ("What is a mineral?") → heading_only (needs_review). True only to reproduce G1a.
HEADING_IS_LABEL = False


# ── loading ─────────────────────────────────────────────────────────────────────

def load_json(paths):
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            rows.extend(data["rows"] if isinstance(data, dict) else data)
    return rows


def load_db():
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT {SQL_COLUMNS} FROM ncert_questions ORDER BY id")
        rows = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT now()")
        db_now = cur.fetchone()["now"]
    finally:
        conn.close()
    return rows, db_now


def parse_options(raw):
    # options is jsonb but stored as a JSON *string* holding the list ("[\"(A) 60°\", …]")
    value = raw
    for _ in range(3):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except Exception:
            return []
    return [str(o) for o in value] if isinstance(value, list) else []


# ── per-row checks ──────────────────────────────────────────────────────────────

def answer_matches(answer: str, options: list[str]) -> bool:
    a = _ANSWER_PREFIX.sub("", answer.strip().lower()).strip()
    m = _ANSWER_LETTER.match(a)
    if m:
        letter = next(g for g in m.groups() if g)
        if ord(letter) - ord("a") < len(options):
            return True
    a_norm = normalize(_OPTION_LABEL.sub("", a))
    for opt in options:
        o_norm = normalize(_OPTION_LABEL.sub("", opt))
        if not o_norm:
            continue
        if o_norm == a_norm:
            return True
        if len(o_norm) >= 3 and re.search(rf"(?<!\w){re.escape(o_norm)}(?!\w)", a_norm):
            return True
    return False


def analyse_row(r):
    text = r.get("question_text") or ""
    norm = normalize(text)
    qtype = (r.get("question_type") or "").lower()
    options = parse_options(r.get("options_text"))
    answer = (r.get("answer") or "").strip()
    image = bool((r.get("image_url") or "").strip())
    flags = set()

    heading = bool(norm) and norm in {normalize(r.get("section")), normalize(r.get("question_number"))}
    if not norm or _LABEL_ONLY.match(norm) or (heading and HEADING_IS_LABEL):
        flags.add("label_only")
    else:
        if heading:
            flags.add("heading_only")
        body = _LEADING_LABEL.sub("", norm)
        if sum(ch.isalnum() for ch in body) < 15:
            flags.add("short_text")

    if options:
        if not answer:
            flags.add("mcq_no_answer")
        elif not answer_matches(answer, options):
            flags.add("mcq_answer_not_in_options")
    elif qtype == "mcq":
        flags.add("mcq_type_no_options")

    if not image:
        if qtype == "diagram":
            flags.add("diagram_type_no_image")
        elif _FIGURE_WORD.search(text):
            flags.add("diagram_text_no_image")

    example = _EXAMPLE_NO.match(r.get("question_number") or "") or _EXAMPLE_NO.match(text)
    return {
        "id": r["id"], "class_grade": str(r["class_grade"]), "subject": r["subject"], "chapter": r["chapter"],
        "section": r.get("section"), "question_type": qtype, "text": text, "norm": norm,
        "hash": content_hash(text), "has_answer": bool(answer), "has_image": image, "has_options": bool(options),
        "figure_ref": r.get("figure_ref"), "options_jsonb_type": r.get("options_jsonb_type"),
        "example_no": int(example.group(1)) if example else None, "created_at": r.get("created_at"),
        "flags": flags, "duplicate_of": None,
    }


def mark_duplicates(rows, junk_never_duplicate=True):
    by_chapter = collections.defaultdict(lambda: collections.defaultdict(list))
    chapters_of_hash = collections.defaultdict(set)
    for r in rows:
        if junk_never_duplicate and "label_only" in r["flags"]:
            continue  # empty after normalize (bare "Example 1 :") is junk, never a duplicate (G1a: False)
        by_chapter[(r["class_grade"], r["subject"], r["chapter"])][r["hash"]].append(r)
        if "short_text" not in r["flags"]:
            chapters_of_hash[(r["class_grade"], r["subject"], r["hash"])].add(r["chapter"])
    for groups in by_chapter.values():
        for copies in groups.values():
            if len(copies) < 2:
                continue
            # canonical: has answer > figure has an image > lowest id (LaTeX validity is G1b)
            copies.sort(key=lambda c: (not c["has_answer"], "diagram_text_no_image" in c["flags"]
                                       or "diagram_type_no_image" in c["flags"], c["id"]))
            for dup in copies[1:]:
                dup["flags"].add("exact_duplicate")
                dup["duplicate_of"] = copies[0]["id"]
    for r in rows:
        if len(chapters_of_hash.get((r["class_grade"], r["subject"], r["hash"]), ())) > 1:
            r["flags"].add("cross_chapter")
    for r in rows:
        r["clean"] = not (r["flags"] & NOT_CLEAN)


# ── report helpers ──────────────────────────────────────────────────────────────

def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "—"


def snippet(text, n=150):
    one_line = re.sub(r"\s+", " ", text or "").strip()
    one_line = one_line if len(one_line) <= n else one_line[: n - 1] + "…"
    return one_line.replace("|", "\\|") or "∅ (empty)"


def samples(rows, flag, k=5):
    picked = [r for r in rows if flag in r["flags"]]
    step = max(1, len(picked) // k)  # spread over the category, not just the lowest ids
    out = []
    for r in picked[::step][:k]:
        extra = f" (dup of #{r['duplicate_of']})" if flag == "exact_duplicate" else ""
        out.append(f"#{r['id']}{extra}: {snippet(r['text'])}")
    return "<br>".join(out)


def example_verdict(ex_rows):
    """Is every 'Example 1' text of a chapter a copy of ONE question? Bodies are compared without the
    leading 'Example 1 :' (some runs kept it); a text inside the longest one is a split part or a cut stem."""
    hashes = {r["hash"] for r in ex_rows}
    if len(ex_rows) <= 1:
        return "single row"
    if len(hashes) == 1:
        return "retry duplication — identical copies"
    bodies = {_LEADING_LABEL.sub("", r["norm"]) for r in ex_rows}
    longest = max(bodies, key=len)
    similar, part, other = 0, 0, []
    for b in bodies:
        if difflib.SequenceMatcher(None, b, longest).ratio() >= 0.9:
            similar += 1
        elif len(b) >= 20 and b in longest:
            part += 1
        else:
            other.append(b)
    verdict = f"retry duplication — {len(hashes)} texts: {similar} re-extracted variants"
    if part:
        verdict += f", {part} split part / cut stem"
    if other:
        verdict += f", {len(other)} other ({'; '.join(repr(o[:40]) for o in other[:2])})"
    return verdict


# ── report ──────────────────────────────────────────────────────────────────────

def build_report(focus, all_rows, meta):
    total = len(focus)
    counts = collections.Counter(f for r in focus for f in r["flags"])
    clean = sum(r["clean"] for r in focus)
    distinct = len({(r["chapter"], r["hash"]) for r in focus})
    L = []
    w = L.append

    w("# NCERT audit — Class 10 Mathematics (G1a)\n")
    w(f"- **Repo / branch:** RAG-BACKEND `fix/ncert-builder-quality` (from `main` {meta['base']}) — not committed")
    w(f"- **Run:** {meta['run_at']} · DB `now()` at dump: {meta['db_now']}")
    w(f"- **Source:** {meta['source']} (SELECT only, no writes)")
    w(f"- **Rows:** Class 10 Mathematics {total:,} · full table {len(all_rows):,}")
    w(f"- **Newest row:** Class 10 Maths {meta['focus_max_created']} · whole table {meta['all_max_created']}")
    w(f"- **Caveat — rows inserted in the 24 h before the run:** {meta['last24h']}")
    w("- **Normalization:** pinned G0 definition (`app/services/ncert_text_hash.py`, 10/10 fixtures pass) — "
      "NFC, lowercase, strip `$ \\( \\) \\[ \\]`, collapse whitespace, strip leading `Q1.` / `1)` / `(i)` / `(a)`, "
      "strip trailing punctuation. `Example 1` is **not** stripped (it is part of the text).")
    w("- **Not in G1a (→ G1b):** near-duplicates (> 0.9), KaTeX parse, truncated/merged rows, image HEAD checks, "
      "garbled-math cost estimate.\n")
    if meta.get("notes"):
        w(meta["notes"].strip() + "\n")

    w("## Summary\n")
    w(f"| | rows | % of Class 10 Maths |\n|---|---:|---:|")
    w(f"| total rows | {total:,} | 100% |")
    w(f"| distinct questions (chapter + hash) | {distinct:,} | {pct(distinct, total)} |")
    w(f"| **clean rows** | **{clean:,}** | **{pct(clean, total)}** |")
    w(f"| rows that would be hidden (duplicate / label / short) | "
      f"{sum(bool(r['flags'] & {'exact_duplicate', 'label_only', 'short_text'}) for r in focus):,} | |\n")
    w("A row can be in several categories, so the category counts do not add up to the total. "
      "**clean** = no flag except `cross_chapter`. A NULL answer on a non-MCQ is not a flag (NCERT exercises "
      "have no answers in the text) — see *Answers*.\n")

    w("## Categories\n")
    w("| category | count | % of Class 10 Maths | 5 sample rows (id: text) |\n|---|---:|---:|---|")
    for flag in FLAG_ORDER:
        w(f"| `{flag}` | {counts[flag]:,} | {pct(counts[flag], total)} | {samples(focus, flag) or '—'} |")
    w("")

    w("## Per chapter\n")
    w("`UI rows` = what the builder shows when the chapter is picked today: the backend filters "
      "`chapter ILIKE '%<name>%'`, so a chapter whose name is inside another one also gets that chapter's rows.\n")
    w("| chapter | rows | distinct | exact dup | label | short | MCQ problem | diagram | clean | clean % | UI rows |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_ch = collections.defaultdict(list)
    for r in focus:
        by_ch[r["chapter"]].append(r)
    names = sorted(by_ch)
    for ch in names:
        rs = by_ch[ch]
        c = collections.Counter(f for r in rs for f in r["flags"])
        ui = sum(len(by_ch[o]) for o in names if ch.lower() in o.lower())
        w(f"| {ch} | {len(rs):,} | {len({r['hash'] for r in rs}):,} | {c['exact_duplicate']:,} | {c['label_only']:,} "
          f"| {c['short_text']:,} | {c['mcq_no_answer'] + c['mcq_answer_not_in_options'] + c['mcq_type_no_options']:,} "
          f"| {c['diagram_text_no_image'] + c['diagram_type_no_image']:,} | {sum(r['clean'] for r in rs):,} "
          f"| {pct(sum(r['clean'] for r in rs), len(rs))} | {ui:,}{' ⚠️' if ui != len(rs) else ''} |")
    w("")

    w("## \"Example N\" analysis\n")
    w("Rows whose `question_number` or text starts with `Example N`. The verdict compares the `Example 1` texts of "
      "one chapter (without the `Example 1 :` prefix) with the longest one: ≥ 0.9 similar = re-extracted variant; "
      "contained in it = split part / cut stem; *other* = neither — a different question labelled Example 1 "
      "(Circles: a concentric-circles proof) or a variant too different for this check (→ G1b near-duplicates).\n")
    w("| chapter | rows labelled Example | distinct N | `Example 1` rows | `Example 1` distinct texts | "
      "`Example 1` rows in UI | verdict |")
    w("|---|---:|---:|---:|---:|---:|---|")
    for ch in names:
        ex = [r for r in by_ch[ch] if r["example_no"] is not None]
        ex1 = [r for r in ex if r["example_no"] == 1]
        ui_ex1 = sum(1 for o in names if ch.lower() in o.lower() for r in by_ch[o] if r["example_no"] == 1)
        w(f"| {ch} | {len(ex):,} | {len({r['example_no'] for r in ex})} | {len(ex1)} | {len({r['hash'] for r in ex1})} "
          f"| {ui_ex1}{' ⚠️' if ui_ex1 != len(ex1) else ''} | {example_verdict(ex1)} |")
    w("")
    worst = max(names, key=lambda ch: sum(1 for r in by_ch[ch] if r["example_no"] == 1))
    w(f"Example — all `Example 1` rows of **{worst}** (id: text):\n")
    for r in sorted((r for r in by_ch[worst] if r["example_no"] == 1), key=lambda r: r["id"])[:12]:
        w(f"- #{r['id']} `{r['hash'][:10]}` {snippet(r['text'], 120)}")
    w("")

    w("## Answers\n")
    w("| question_type | rows | answer NULL/empty | with options | MCQ answer missing | MCQ answer not in options |")
    w("|---|---:|---:|---:|---:|---:|")
    by_type = collections.defaultdict(list)
    for r in focus:
        by_type[r["question_type"]].append(r)
    for t in sorted(by_type, key=lambda t: -len(by_type[t])):
        rs = by_type[t]
        w(f"| {t} | {len(rs):,} | {sum(not r['has_answer'] for r in rs):,} "
          f"| {sum(r['has_options'] for r in rs):,} "
          f"| {sum('mcq_no_answer' in r['flags'] for r in rs):,} | {sum('mcq_answer_not_in_options' in r['flags'] for r in rs):,} |")
    jt = collections.Counter(r["options_jsonb_type"] for r in all_rows)
    w(f"\n`options` jsonb type over the whole table: {', '.join(f'{k}: {v:,}' for k, v in jt.items())} "
      "(a JSON string holding the list — the script decodes it twice).\n")

    w("## Full database — counts only\n")
    classes = sorted({r["class_grade"] for r in all_rows}, key=lambda c: int(c) if c.isdigit() else 99)
    w("| category | all classes | " + " | ".join(f"Class {c}" for c in classes) + " |")
    w("|---|---:|" + "---:|" * len(classes))
    per_class = {c: [r for r in all_rows if r["class_grade"] == c] for c in classes}
    w("| rows | " + f"{len(all_rows):,} | " + " | ".join(f"{len(per_class[c]):,}" for c in classes) + " |")
    w("| distinct (subject+chapter+hash) | "
      + f"{len({(r['class_grade'], r['subject'], r['chapter'], r['hash']) for r in all_rows}):,} | "
      + " | ".join(f"{len({(r['subject'], r['chapter'], r['hash']) for r in per_class[c]}):,}" for c in classes) + " |")
    for flag in FLAG_ORDER:
        w(f"| `{flag}` | {sum(flag in r['flags'] for r in all_rows):,} | "
          + " | ".join(f"{sum(flag in r['flags'] for r in per_class[c]):,}" for c in classes) + " |")
    all_clean = sum(r["clean"] for r in all_rows)
    w(f"| **clean** | **{all_clean:,} ({pct(all_clean, len(all_rows))})** | "
      + " | ".join(f"{sum(r['clean'] for r in per_class[c]):,} ({pct(sum(r['clean'] for r in per_class[c]), len(per_class[c]))})"
                   for c in classes) + " |")
    w("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only NCERT quality audit (G1a)")
    ap.add_argument("--from-json", nargs="+", help="JSON dumps of the SELECT instead of DATABASE_URL")
    ap.add_argument("--db-now", help="DB now() at dump time (with --from-json)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"))
    ap.add_argument("--base", default="", help="base commit, printed in the report header")
    ap.add_argument("--notes", help="hand-written markdown (findings, pre-flight) put under the report header")
    args = ap.parse_args()

    if args.from_json:
        raw, db_now, source = load_json(args.from_json), args.db_now or "unknown", "JSON dump of `supabase db query --linked`"
    elif os.environ.get("DATABASE_URL"):
        raw, db_now = load_db()
        source = "DATABASE_URL (psycopg2, read-only session)"
    else:
        print("Set DATABASE_URL in the terminal or pass --from-json.", file=sys.stderr)
        return 2

    rows = [analyse_row(r) for r in raw]
    mark_duplicates(rows)
    focus = [r for r in rows if r["class_grade"] == FOCUS[0] and r["subject"] == FOCUS[1]]
    if not focus:
        print("No Class 10 Mathematics rows in the input.", file=sys.stderr)
        return 1

    def newest(rs):
        return max((str(r["created_at"]) for r in rs if r["created_at"]), default="—")

    last24h = "unknown"
    try:
        cutoff = datetime.fromisoformat(str(db_now).replace(" ", "T").replace("+00", "+00:00")) if db_now != "unknown" else None
        if cutoff:
            n = sum(1 for r in rows if r["created_at"]
                    and (cutoff - datetime.fromisoformat(str(r["created_at"]).replace(" ", "T").replace("Z", "+00:00"))).total_seconds() < 86400)
            last24h = f"{n} rows" + (" — none, the data did not change around the audit" if n == 0 else " — counts may move")
    except ValueError:
        pass

    meta = {
        "base": args.base, "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "db_now": db_now,
        "source": source, "focus_max_created": newest(focus), "all_max_created": newest(rows), "last24h": last24h,
    }
    if args.notes:
        with open(args.notes, encoding="utf-8") as f:
            meta["notes"] = f.read()
    os.makedirs(os.path.join(args.out_dir, "audit"), exist_ok=True)
    report_path = os.path.join(args.out_dir, "ncert-audit-class10-maths.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report(focus, rows, meta))

    csv_path = os.path.join(args.out_dir, "audit", "ncert_audit_class10_maths_rows.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "chapter", "question_type", "content_hash", "flags", "duplicate_of", "clean"])
        for r in sorted(focus, key=lambda r: r["id"]):
            wr.writerow([r["id"], r["chapter"], r["question_type"], r["hash"],
                         ";".join(f for f in FLAG_ORDER if f in r["flags"]), r["duplicate_of"] or "", int(r["clean"])])

    clean = sum(r["clean"] for r in focus)
    print(f"Class 10 Maths: {len(focus):,} rows, clean {clean:,} ({pct(clean, len(focus))}) · full table {len(rows):,}")
    print(f"→ {report_path}\n→ {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
