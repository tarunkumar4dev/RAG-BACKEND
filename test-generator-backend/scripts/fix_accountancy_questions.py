#!/usr/bin/env python3
"""
scripts/fix_accountancy_questions.py — Rebuild flattened Accountancy tables in ncert_questions.

NCERT extraction stored financial statements as one flat line ("Particulars Amount ` Opening stock
37,500 Purchases 1,05,000 ...") with question_table = NULL. For every Accountancy row this script:

  * rewrites question_text as canonical text: intro + a markdown table (with |---| separator)
    + trailing text. The text stays self-contained, so paths that never read question_table
    (saved test history, browser PDF fallback, search) still show the full question;
  * for questions with exactly one table, also stores question_table as JSON
    ({"headers", "rows", "kind", "amount_cols", "total_rows", "section_rows", "caption"?}).
    The exporter/preview use it in place of the markdown copy for richer layout.

Safe by default:
  * dry run unless --apply is given (--dry-run is accepted for clarity);
  * --apply writes a JSON backup of every row it will touch before changing anything;
  * each batch is one transaction; a row is only updated if question_table is still NULL and
    question_text is unchanged since it was read (md5 guard), otherwise it is skipped;
  * --rollback BACKUP.json restores the saved originals.

Usage (from test-generator-backend/):
  python scripts/fix_accountancy_questions.py                     # dry run + report
  python scripts/fix_accountancy_questions.py --ids 17687 17609   # dry run, show those rows
  python scripts/fix_accountancy_questions.py --apply             # write (backup first)
  python scripts/fix_accountancy_questions.py --rollback scripts/backups/accountancy_fix_YYYYmmdd_HHMMSS.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db_pool import get_db_connection  # noqa: E402
from app.services.accountancy_parser import (  # noqa: E402
    parse_and_structure_accountancy_text,
    segments_to_markdown,
)

BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backups")


def _md5(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def restructure(text: str) -> Tuple[Optional[str], Optional[dict], List[dict]]:
    """Returns (new_question_text, question_table_or_None, tables) or (None, None, []) if the text
    holds no recognisable table. Iterates parse -> markdown -> parse until stable so what is stored
    renders back to exactly the same tables."""
    segs = parse_and_structure_accountancy_text(text)
    if not any(s["type"] == "table" for s in segs):
        return None, None, []
    md = segments_to_markdown(segs)
    for _ in range(3):
        segs2 = parse_and_structure_accountancy_text(md)
        md2 = segments_to_markdown(segs2)
        if md2 == md:
            break
        segs, md = segs2, md2
    tables = [s["table"] for s in segs if s["type"] == "table"]
    if not tables:
        return None, None, []
    qt = dict(tables[0]) if len(tables) == 1 else None
    return md, qt, tables


def fetch_rows(conn, ids: Optional[List[int]], limit: Optional[int], class_grade: Optional[str]):
    sql = """
        SELECT id, class_grade, chapter, question_text
        FROM ncert_questions
        WHERE subject ILIKE %s AND question_table IS NULL AND question_text IS NOT NULL
    """
    params: list = ["%account%"]
    if ids:
        sql += " AND id = ANY(%s)"
        params.append(ids)
    if class_grade:
        sql += " AND class_grade = %s"
        params.append(class_grade)
    sql += " ORDER BY id"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def plan(rows) -> Tuple[list, dict]:
    changes, stats = [], {"scanned": len(rows), "with_tables": 0, "single_table": 0, "multi_table": 0,
                          "unchanged": 0, "kinds": {}, "needs_review": []}
    for rid, cls, chapter, text in rows:
        new_text, qt, tables = restructure(text)
        if new_text is None:
            continue
        stats["with_tables"] += 1
        stats["single_table" if qt else "multi_table"] += 1
        for t in tables:
            stats["kinds"][t.get("kind", "?")] = stats["kinds"].get(t.get("kind", "?"), 0) + 1
            # Ledgers whose Dr/Cr sides were interleaved, or statements whose totals don't add up,
            # deserve a human look before they reach an exam paper.
            if t.get("sides_unknown") or t.get("totals_verified") is False:
                stats["needs_review"].append(rid)
        if new_text == text and qt is None:
            stats["unchanged"] += 1
            continue
        changes.append({"id": rid, "class_grade": cls, "chapter": chapter, "old_text": text,
                        "new_text": new_text, "question_table": qt, "n_tables": len(tables)})
    stats["needs_review"] = sorted(set(stats["needs_review"]))
    return changes, stats


def write_backup(changes: list) -> str:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    path = os.path.join(BACKUP_DIR, f"accountancy_fix_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"id": c["id"], "question_text": c["old_text"], "question_table": None} for c in changes],
                  f, ensure_ascii=False, indent=1)
    return path


def apply_changes(conn, changes: list, batch_size: int) -> Tuple[int, int]:
    updated = skipped = 0
    for start in range(0, len(changes), batch_size):
        batch = changes[start:start + batch_size]
        cur = conn.cursor()
        try:
            for c in batch:
                cur.execute(
                    """
                    UPDATE ncert_questions
                       SET question_text = %s,
                           question_table = %s::jsonb
                     WHERE id = %s
                       AND question_table IS NULL
                       AND md5(question_text) = %s
                    """,
                    (c["new_text"], json.dumps(c["question_table"], ensure_ascii=False) if c["question_table"] else None,
                     c["id"], _md5(c["old_text"])),
                )
                if cur.rowcount == 1:
                    updated += 1
                else:
                    skipped += 1  # row changed since it was read — left alone
            conn.commit()
            print(f"  batch {start // batch_size + 1}: committed {len(batch)} rows")
        except Exception:
            conn.rollback()
            print(f"  batch {start // batch_size + 1}: FAILED — rolled back; stopping. Earlier batches stay committed "
                  f"(restore them with --rollback on the backup file).")
            raise
        finally:
            cur.close()
    return updated, skipped


def rollback(conn, path: str, batch_size: int) -> int:
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    restored = 0
    for start in range(0, len(saved), batch_size):
        batch = saved[start:start + batch_size]
        cur = conn.cursor()
        try:
            for r in batch:
                cur.execute(
                    "UPDATE ncert_questions SET question_text = %s, question_table = %s::jsonb WHERE id = %s",
                    (r["question_text"], json.dumps(r["question_table"]) if r.get("question_table") else None, r["id"]),
                )
                restored += cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
    return restored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write changes (default is a dry run)")
    mode.add_argument("--dry-run", action="store_true", help="report only (default)")
    mode.add_argument("--rollback", metavar="BACKUP_JSON", help="restore rows from a backup written by --apply")
    ap.add_argument("--ids", type=int, nargs="*", help="only these ncert_questions ids")
    ap.add_argument("--class-grade", help="only this class (e.g. 11)")
    ap.add_argument("--limit", type=int, help="scan at most N rows")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--show", type=int, default=5, help="print N before/after samples in a dry run")
    ap.add_argument("--report", help="write the planned changes as JSONL to this path")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    conn = get_db_connection()
    if not conn:
        print("No database connection (check DATABASE_URL / DB_* env vars).")
        return 1
    try:
        if args.rollback:
            n = rollback(conn, args.rollback, args.batch_size)
            print(f"Restored {n} rows from {args.rollback}")
            return 0

        rows = fetch_rows(conn, args.ids, args.limit, args.class_grade)
        changes, stats = plan(rows)
        print(f"Scanned {stats['scanned']} Accountancy rows with question_table IS NULL")
        print(f"  with reconstructable tables : {stats['with_tables']}  "
              f"(single table: {stats['single_table']}, multiple: {stats['multi_table']})")
        print(f"  table kinds                 : {stats['kinds']}")
        print(f"  rows to update              : {len(changes)}")
        if stats["needs_review"]:
            print(f"  needs manual review ({len(stats['needs_review'])}): {stats['needs_review']}")

        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                for c in changes:
                    f.write(json.dumps({k: c[k] for k in ("id", "n_tables", "new_text", "question_table")},
                                       ensure_ascii=False) + "\n")
            print(f"  report written to {args.report}")

        if not args.apply:
            for c in changes[: args.show]:
                print("\n" + "=" * 90 + f"\n#{c['id']}  ({c['class_grade']} · {c['chapter']})")
                print("BEFORE:", c["old_text"][:300].replace("\n", " ⏎ "), "..." if len(c["old_text"]) > 300 else "")
                print("AFTER:\n" + c["new_text"][:1200])
                if c["question_table"]:
                    print("question_table.kind =", c["question_table"].get("kind"))
            print("\nDry run — nothing written. Re-run with --apply to update the database.")
            return 0

        if not changes:
            print("Nothing to update.")
            return 0
        backup = write_backup(changes)
        print(f"Backup of {len(changes)} original rows: {backup}")
        updated, skipped = apply_changes(conn, changes, args.batch_size)
        print(f"Done. Updated {updated} rows, skipped {skipped} (changed since read).")
        print(f"Undo with: python scripts/fix_accountancy_questions.py --rollback \"{backup}\"")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
