#!/usr/bin/env python3
"""
scripts/ncert_quality_pass.py — set ncert_questions.quality_flag (needs migration 18 on the DB).

Every NCERT question was extracted several times, so the Test Builder showed most of them 3–8
times, and some copies lost the content that was a drawing or a special font. For every row:

  'garbled'    PDF glyph names instead of characters ("AlCl34/unif02b/unif0ae …")
  'incomplete' "… the following reaction?" / "(a) (b) (c)" with the items missing, no image
  'duplicate'  the same question (same text after cleanup, same class/subject/chapter) is kept
               visible in a better copy (image > complete > options > line breaks > lowest id)
  NULL         visible

Run add_question_block_images.py first: a row that gets an image of its PDF block is no
longer incomplete, and the image copy is the one kept among duplicates.

Safe by default:
  * dry run unless --apply; --report writes every planned flag as JSONL;
  * --apply first writes scripts/backups/quality_flags_<ts>.json with the current flags of every
    row it changes; --rollback BACKUP.json restores them;
  * only quality_flag is written — question text, options and images are never touched.

Usage (from test-generator-backend/, DATABASE_URL set in the terminal):
  python scripts/ncert_quality_pass.py                       # dry run, summary per class/subject
  python scripts/ncert_quality_pass.py --class-grade 11 --show 20
  python scripts/ncert_quality_pass.py --apply
  python scripts/ncert_quality_pass.py --rollback scripts/backups/quality_flags_YYYYmmdd_HHMMSS.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db_pool import get_db_connection  # noqa: E402
from app.services.ncert_cleanup import _alnum, clean_question, quality_issue  # noqa: E402

BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backups")


def _options(raw):
    # stored as a JSON string inside jsonb ("[\"(a) …\"]") on most rows
    while isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    return raw if isinstance(raw, list) else []


def _subject_family(subject: str) -> str:
    """'Chemistry-I' / 'Chemistry' and 'Political_Science' / 'Political Science' are one book list."""
    s = (subject or "").lower().replace("_", " ")
    return re.sub(r"[\s\-]+(i|ii)$", "", s).strip()


def fetch_rows(conn, class_grade):
    sql = """SELECT id, class_grade, subject, chapter, question_text, options, image_url,
                    question_table IS NOT NULL, quality_flag
               FROM ncert_questions"""
    params = []
    if class_grade:
        sql += " WHERE class_grade = %s"
        params.append(class_grade)
    cur = conn.cursor()
    cur.execute(sql + " ORDER BY id", params)
    rows = [
        {"id": r[0], "class_grade": r[1], "subject": r[2], "chapter": r[3], "question_text": r[4] or "",
         "options": _options(r[5]), "image_url": r[6], "has_table": r[7], "quality_flag": r[8]}
        for r in cur.fetchall()
    ]
    cur.close()
    return rows


def _rank(r) -> tuple:
    """Higher is better: the copy that stays visible."""
    return (
        bool(r["image_url"]),
        r["issue"] is None,
        len(r["clean_options"]) >= 2,
        r["has_table"],
        "\n" in r["clean_text"],
        -r["id"],
    )


def plan(rows):
    """Returns {id: new_flag_or_None}."""
    groups = collections.defaultdict(list)
    for r in rows:
        q = clean_question({"question_text": r["question_text"], "options": list(r["options"])})
        r["clean_text"] = q["question_text"]
        r["clean_options"] = q["options"] or []
        r["issue"] = quality_issue(q["question_text"], q["options"], bool(r["image_url"]), r["has_table"])
        key = _alnum(q["question_text"])
        if len(key) < 15:  # "Why?" / "Explain." — too short to call two rows the same question
            key = f"#{r['id']}"
        groups[(r["class_grade"], _subject_family(r["subject"]), r["chapter"], key)].append(r)

    flags = {}
    for members in groups.values():
        best = max(members, key=_rank)
        for r in members:
            flags[r["id"]] = r["issue"] if r is best else "duplicate"
    return flags


def apply_flags(conn, changes, batch_size):
    by_flag = collections.defaultdict(list)
    for qid, flag in changes.items():
        by_flag[flag].append(qid)
    cur = conn.cursor()
    try:
        for flag, ids in by_flag.items():
            for i in range(0, len(ids), batch_size):
                cur.execute("UPDATE ncert_questions SET quality_flag = %s WHERE id = ANY(%s)", (flag, ids[i:i + batch_size]))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write the flags (default is a dry run)")
    mode.add_argument("--dry-run", action="store_true", help="report only (default)")
    mode.add_argument("--rollback", metavar="BACKUP_JSON", help="restore flags from a backup written by --apply")
    ap.add_argument("--class-grade", help="only this class (e.g. 11)")
    ap.add_argument("--show", type=int, default=0, help="print N incomplete/garbled rows")
    ap.add_argument("--report", help="write {id, flag, text} JSONL to this path")
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    conn = get_db_connection()
    if not conn:
        print("No database connection (check DATABASE_URL / DB_* env vars).")
        return 1
    try:
        if args.rollback:
            with open(args.rollback, encoding="utf-8") as f:
                old = {int(k): v for k, v in json.load(f).items()}
            apply_flags(conn, old, args.batch_size)
            print(f"Restored quality_flag on {len(old)} rows from {args.rollback}")
            return 0

        rows = fetch_rows(conn, args.class_grade)
        flags = plan(rows)
        by_id = {r["id"]: r for r in rows}

        summary = collections.defaultdict(collections.Counter)
        for r in rows:
            summary[(r["class_grade"], r["subject"])][flags[r["id"]] or "visible"] += 1
        print(f"{'class':>5}  {'subject':<22} {'rows':>6} {'visible':>8} {'dup':>6} {'incompl':>8} {'garbled':>8}")
        for (cg, subj), c in sorted(summary.items(), key=lambda x: (int(x[0][0]) if x[0][0].isdigit() else 0, x[0][1])):
            print(f"{cg:>5}  {subj:<22} {sum(c.values()):>6} {c['visible']:>8} {c['duplicate']:>6} "
                  f"{c['incomplete']:>8} {c['garbled']:>8}")
        total = collections.Counter(f or "visible" for f in flags.values())
        print(f"TOTAL {len(rows)} rows → {dict(total)}")

        if args.show:
            shown = 0
            for r in rows:
                if flags[r["id"]] in ("incomplete", "garbled") and shown < args.show:
                    print(f"  [{flags[r['id']]}] #{r['id']} {r['class_grade']} {r['subject']} | {r['clean_text'][:160]!r}")
                    shown += 1

        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                for qid, flag in flags.items():
                    f.write(json.dumps({"id": qid, "flag": flag, "text": by_id[qid]["clean_text"][:300]},
                                       ensure_ascii=False) + "\n")
            print(f"Report → {args.report}")

        changes = {qid: flag for qid, flag in flags.items() if by_id[qid]["quality_flag"] != flag}
        print(f"Rows whose flag changes: {len(changes)}")
        if not args.apply:
            print("Dry run — nothing written. Re-run with --apply.")
            return 0

        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup = os.path.join(BACKUP_DIR, f"quality_flags_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(backup, "w", encoding="utf-8") as f:
            json.dump({str(qid): by_id[qid]["quality_flag"] for qid in changes}, f)
        print(f"Backup of current flags → {backup}")
        apply_flags(conn, changes, args.batch_size)
        print(f"Updated {len(changes)} rows.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
