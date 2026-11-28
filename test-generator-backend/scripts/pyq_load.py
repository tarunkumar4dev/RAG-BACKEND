"""
Load reviewed PYQ JSON (from scripts/pyq_extract.py) into ncert_questions — step 2. Needs migration 37.

    .venv/bin/python scripts/pyq_load.py data/pyq/accountancy_12_2026            # dry run: checks + summary
    .venv/bin/python scripts/pyq_load.py data/pyq/accountancy_12_2026 --apply    # writes to the database

- A question printed in several sets (67/5/1 and 67/5/2 share most questions) is stored once, with every
  Q.P. code in pyq_papers; its number is the one in the lowest set code. "Same question" = same English text
  and options once case, spaces and punctuation are ignored.
- Each question becomes an English row and a Hindi row with the same pair_id (sha1 of exam, year, subject,
  class and the English text), so the Test Builder can filter by language.
- Re-running updates the rows in place (matched on pair_id + language), so ids — and with them the questions
  in teachers' saved papers — stay the same. It never deletes rows.
- Nothing is written without --apply, and nothing at all if any check fails.
- Checks beyond the per-question ones: a number twice in one paper; a merged question whose sets disagree on
  marks / chapter / MCQ answer (one answer key is wrong); near-identical questions that did not merge (a
  transcription difference — fix the JSON so they match). Disagreements stop the load; near-matches are warnings.
- Editing a question's English text in the JSON gives it a new pair_id. --apply then hides the row with the old
  text (quality_status 'duplicate', like the textbook duplicates) instead of leaving both visible; ids stay valid.
"""

import argparse
import difflib
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pyq_extract import CHAPTERS  # noqa: E402

TYPES = {"mcq", "short_answer", "long_answer", "case_study"}
DIFFICULTIES = {"easy", "medium", "hard"}
_NOT_WORD = re.compile(r"[^0-9a-z]+")
_OPTION_LABEL = re.compile(r"^\s*\(?[A-Da-d][\).:]\s+")


def norm(text: str) -> str:
    return _NOT_WORD.sub("", (text or "").lower())


def question_key(q: dict) -> str:
    """Same question in two sets: same English text and the same options in any order."""
    en = q.get("en") or {}
    return norm(en.get("text")) + "#" + "|".join(sorted(norm(o) for o in clean_options(en.get("options"))))


def mcq_choice(q: dict) -> str:
    """The text of the option an MCQ answer picks, so sets with shuffled options can be compared."""
    m = re.match(r"^\s*([A-D])", q.get("answer") or "")
    options = clean_options((q.get("en") or {}).get("options"))
    i = "ABCD".index(m.group(1)) if m else -1
    return norm(options[i]) if 0 <= i < len(options) else ""


def paper_sort_key(code: str) -> tuple:
    return tuple(int(p) for p in code.split("/"))


def clean_options(options) -> list:
    return [_OPTION_LABEL.sub("", str(o)).strip() for o in (options or []) if str(o).strip()]


def labelled(options: list) -> str:
    """Stored like the textbook rows: a JSON list of "A. …" strings."""
    return json.dumps([f"{'ABCD'[i]}. {o}" for i, o in enumerate(options)], ensure_ascii=False)


def load_papers(folder: Path) -> list:
    papers = []
    for f in sorted(folder.glob("*.json")):
        data = json.loads(f.read_text())
        data["_file"] = f.name
        papers.append(data)
    if not papers:
        raise SystemExit(f"No JSON in {folder}")
    papers.sort(key=lambda p: paper_sort_key(p["paper"]))
    return papers


def check_question(q: dict, where: str, chapters: set) -> list:
    errors = []
    if q.get("chapter") not in chapters:
        errors.append(f"{where}: unknown chapter {q.get('chapter')!r}")
    if q.get("question_type") not in TYPES:
        errors.append(f"{where}: unknown question_type {q.get('question_type')!r}")
    if q.get("difficulty") not in DIFFICULTIES:
        errors.append(f"{where}: unknown difficulty {q.get('difficulty')!r}")
    if not isinstance(q.get("marks"), int) or not 1 <= q["marks"] <= 10:
        errors.append(f"{where}: marks {q.get('marks')!r}")
    for lang in ("en", "hi"):
        v = q.get(lang) or {}
        if len((v.get("text") or "").strip()) < 5:
            errors.append(f"{where}: {lang} text missing")
        n = len(clean_options(v.get("options")))
        if q.get("question_type") == "mcq" and n != 4:
            errors.append(f"{where}: {lang} mcq has {n} options")
    if q.get("question_type") == "mcq" and not re.match(r"^\s*[A-D]\b", q.get("answer") or ""):
        errors.append(f"{where}: mcq answer should start with A–D: {q.get('answer')!r}")
    if "hi" in q and q["hi"].get("text") and not re.search(r"[ऀ-ॿ]", q["hi"]["text"]):
        errors.append(f"{where}: hi text has no Devanagari")
    return errors


def build_rows(papers: list) -> tuple:
    """(rows, errors). One English + one Hindi row per distinct question."""
    first = papers[0]
    exam, year = first["exam"], int(first["year"])
    subject, class_grade = first["subject"], str(first["class_grade"])
    chapters = set(CHAPTERS[(subject.lower(), class_grade)])
    errors, merged = [], {}

    for p in papers:
        numbers = Counter((str(q.get("question_number")).replace(" ", ""), q.get("part")) for q in p["questions"])
        errors.extend(f"{p['_file']}: Q{n} ({part}) appears {c} times" for (n, part), c in numbers.items() if c > 1)
        if (p["exam"], int(p["year"]), p["subject"], str(p["class_grade"])) != (exam, year, subject, class_grade):
            errors.append(f"{p['_file']}: exam/year/subject/class differ from {papers[0]['_file']}")
            continue
        for q in p["questions"]:
            where = f"{p['paper']} Q{q.get('question_number')} ({q.get('part')})"
            errors.extend(check_question(q, where, chapters))
            key = question_key(q)
            if key in merged:
                first_q, first_where = merged[key]["q"], merged[key]["where"]
                for field in ("marks", "chapter", "question_type"):
                    if q.get(field) != first_q.get(field):
                        errors.append(f"{where}: {field} {q.get(field)!r} but {first_where} has {first_q.get(field)!r}")
                if q.get("question_type") == "mcq" and mcq_choice(q) != mcq_choice(first_q):
                    errors.append(f"{where}: answer {q.get('answer', '')[:1]!r} picks a different option than "
                                  f"{first_where} ({first_q.get('answer', '')[:1]!r}) — check the answer key")
                merged[key]["papers"].append(p["paper"])
            else:
                merged[key] = {"q": q, "papers": [p["paper"]], "where": where}

    rows = []
    for key, item in merged.items():
        q = item["q"]
        pair_id = "pyq-" + hashlib.sha1(f"{exam}|{year}|{subject.lower()}|{class_grade}|{key}".encode()).hexdigest()[:20]
        for lang in ("en", "hi"):
            v = q.get(lang) or {}
            options = clean_options(v.get("options"))
            rows.append({
                "class_grade": class_grade,
                "subject": subject,
                "chapter": q["chapter"],
                "section": f"{exam} {year}" + (" · Part B Option II" if q.get("part") == "B-II" else ""),
                "question_number": str(q["question_number"]).replace(" ", ""),
                "question_text": (v.get("text") or "").strip(),
                "question_type": q["question_type"],
                "options": labelled(options) if options else json.dumps([]),
                "answer": (q.get("answer") or "").strip() or None,
                "marks": q["marks"],
                "difficulty": q["difficulty"],
                "quality_status": None,
                "source": "pyq",
                "language": lang,
                "pyq_year": year,
                "pyq_exam": exam,
                "pyq_papers": sorted(set(item["papers"]), key=paper_sort_key),
                "pair_id": pair_id,
            })
    return rows, errors


def near_duplicates(papers: list, threshold: float = 0.9) -> list:
    """Questions from different sets that almost match but were not merged."""
    items = [(p["paper"], q, norm((q.get("en") or {}).get("text"))) for p in papers for q in p["questions"]]
    found = []
    for i, (pa, qa, ta) in enumerate(items):
        for pb, qb, tb in items[i + 1:]:
            if pa == pb or question_key(qa) == question_key(qb):
                continue
            if re.findall(r"\d+", ta) != re.findall(r"\d+", tb):
                continue  # same template, other figures: a different question
            m = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
            if m.real_quick_ratio() > threshold and m.quick_ratio() > threshold and m.ratio() > threshold:
                found.append(f"{pa} Q{qa.get('question_number')} ≈ {pb} Q{qb.get('question_number')} "
                             f"({m.ratio():.0%} alike, not merged)")
    return found


def summary(rows: list) -> None:
    en = [r for r in rows if r["language"] == "en"]
    print(f"{len(en)} questions × 2 languages = {len(rows)} rows")
    print("by chapter:")
    for k, v in sorted(Counter(r["chapter"] for r in en).items()):
        print(f"  {v:4}  {k}")
    print("by marks:", dict(sorted(Counter(r["marks"] for r in en).items())))
    print("by type:", dict(Counter(r["question_type"] for r in en)))
    print("by difficulty:", dict(Counter(r["difficulty"] for r in en)))
    print("in more than one set:", sum(1 for r in en if len(r["pyq_papers"]) > 1))


def apply(rows: list) -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from app.core.database import get_supabase_admin
    sb = get_supabase_admin()

    existing = {}
    pair_ids = sorted({r["pair_id"] for r in rows})
    for i in range(0, len(pair_ids), 100):
        got = sb.table("ncert_questions").select("id,pair_id,language").eq("source", "pyq") \
            .in_("pair_id", pair_ids[i:i + 100]).execute().data or []
        existing.update({(g["pair_id"], g["language"]): g["id"] for g in got})

    updates = [dict(r, id=existing[(r["pair_id"], r["language"])]) for r in rows
               if (r["pair_id"], r["language"]) in existing]
    inserts = [r for r in rows if (r["pair_id"], r["language"]) not in existing]
    for i in range(0, len(updates), 200):
        sb.table("ncert_questions").upsert(updates[i:i + 200], on_conflict="id").execute()
    for i in range(0, len(inserts), 200):
        sb.table("ncert_questions").insert(inserts[i:i + 200]).execute()

    first = rows[0]
    loaded = set(pair_ids)
    current = sb.table("ncert_questions").select("id,pair_id,pyq_papers,quality_status").eq("source", "pyq") \
        .eq("pyq_exam", first["pyq_exam"]).eq("pyq_year", first["pyq_year"]).eq("subject", first["subject"]) \
        .eq("class_grade", first["class_grade"]).execute().data or []
    papers_in_load = {p for r in rows for p in r["pyq_papers"]}
    # only rows that came entirely from the papers being loaded: a paper left out of this folder is untouched
    stale = [r["id"] for r in current if r["pair_id"] not in loaded and r.get("quality_status") != "duplicate"
             and set(r.get("pyq_papers") or []) <= papers_in_load]
    for i in range(0, len(stale), 200):
        sb.table("ncert_questions").update({"quality_status": "duplicate"}).in_("id", stale[i:i + 200]).execute()
    print(f"updated {len(updates)} rows, inserted {len(inserts)} rows, hid {len(stale)} rows no longer in the JSON")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--apply", action="store_true", help="write to the database (default: dry run)")
    args = ap.parse_args()

    papers = load_papers(args.folder)
    rows, errors = build_rows(papers)
    summary(rows)
    warnings = near_duplicates(papers)
    if warnings:
        print(f"\n{len(warnings)} near-identical question(s) not merged — same printed question? make the JSON match:")
        for w in warnings:
            print("  " + w)
    if errors:
        print(f"\n{len(errors)} problem(s) — fix the JSON, nothing written:")
        for e in errors:
            print("  " + e)
        raise SystemExit(1)
    if args.apply:
        apply(rows)
    else:
        print("\nDry run: nothing written. Add --apply to load.")


if __name__ == "__main__":
    main()
