#!/usr/bin/env python3
"""
Checks the NCERT library queries in app/api/v1/endpoints/test_generator.py against a fake Supabase
client (no network, no pytest needed).

    python scripts/check_ncert_library_queries.py

  * /ncert-questions filters chapter with an exact eq (no ILIKE substring: "Circles" must not pull in
    "Areas Related to Circles"; "Introduction_2" must survive) and orders by id as the last tiebreaker;
  * the stats fallback (no DB pool) pages through every row instead of stopping at 5,000.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
# config.py requires these at import; the fake client below never uses them
for _name in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_KEY", "GEMINI_API_KEY"):
    os.environ.setdefault(_name, "check-only")

import app.api.v1.endpoints.test_generator as tg  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  — ' + str(detail)}")
    if not cond:
        FAILURES.append(name)


class Result:
    def __init__(self, data, count=None):
        self.data, self.count = data, count


class FakeQuery:
    def __init__(self, client):
        self.client, self.calls = client, []

    def __getattr__(self, method):
        def record(*args, **kwargs):
            self.calls.append((method, args, kwargs))
            return self
        return record

    def execute(self):
        self.client.queries.append(self.calls)
        rng = next((a for m, a, _ in self.calls if m == "range"), None)
        if self.client.rows is None:
            return Result([], 0)
        start, end = rng if rng else (0, len(self.client.rows) - 1)
        return Result(self.client.rows[start:end + 1], len(self.client.rows))


class FakeClient:
    def __init__(self, rows=None):
        self.rows, self.queries = rows, []

    def table(self, name):
        q = FakeQuery(self)
        q.calls.append(("table", (name,), {}))
        return q


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def main() -> int:
    fake = FakeClient(rows=[])
    tg.get_supabase = lambda: fake

    for chapter in ("Circles", "Introduction_2"):
        fake.queries.clear()
        run(tg._ncert_questions_impl("Mathematics", "10", chapter, None, None, None, 30, 0))
        calls = fake.queries[0]
        check(f"{chapter!r}: exact eq on chapter", ("eq", ("chapter", chapter), {}) in calls, calls)
        check(f"{chapter!r}: no ILIKE on chapter", not any(m == "ilike" and a[0] == "chapter" for m, a, _ in calls), calls)
        orders = [a[0] for m, a, _ in calls if m == "order"]
        check(f"{chapter!r}: ORDER BY quality_rank, section, question_number, id",
              orders == ["quality_rank", "section", "question_number", "id"], orders)
        ors = [a[0] for m, a, _ in calls if m == "or_"]
        check(f"{chapter!r}: hides duplicate/junk, keeps NULL", ors == ["quality_status.is.null,quality_status.not.in.(duplicate,junk)"], ors)

    fake.queries.clear()
    run(tg._ncert_questions_impl("Mathematics", "10", "Circles", None, None, None, 30, 0, "missing"))
    calls = fake.queries[0]
    check("figure=missing: image_url IS NULL + diagram/figure_ref",
          ("is_", ("image_url", "null"), {}) in calls and ("or_", ("question_type.eq.diagram,figure_ref.not.is.null",), {}) in calls, calls)
    fake.queries.clear()
    run(tg._ncert_questions_impl("Mathematics", "10", "Circles", None, None, None, 30, 0, "hide"))
    calls = fake.queries[0]
    check("figure=hide: image or (not diagram and no figure_ref)",
          ("or_", ("image_url.not.is.null,and(question_type.neq.diagram,figure_ref.is.null)",), {}) in calls, calls)

    # response shaping: options decoded + own labels stripped, figure_missing, no audit fields
    fake = FakeClient(rows=[
        {"id": 1, "options": json.dumps(["(A) 60°", "(B) 70°", "(C) 80°", "(D) 90°"]), "question_type": "mcq",
         "image_url": None, "figure_ref": "Fig. 10.11", "quality_status": "needs_review", "quality_flags": {"flags": ["x"]},
         "duplicate_of": None, "hash_version": "v2", "quality_rank": 1, "content_hash": "abc"},
        {"id": 2, "options": ["12 cm", "(C) 8.5 cm", "A) x", "b. y"], "question_type": "exercise",
         "image_url": "https://x/fig.png", "figure_ref": "Fig. 1.1"},
        {"id": 3, "options": None, "question_type": "diagram", "image_url": None, "figure_ref": None},
    ])
    tg.get_supabase = lambda: fake
    qs = run(tg._ncert_questions_impl("Mathematics", "10", "Circles", None, None, None, 30, 0))["questions"]
    check("double-encoded options decoded, own labels stripped", qs[0]["options"] == ["60°", "70°", "80°", "90°"], qs[0]["options"])
    check("real array kept; a label that is not the option's own letter stays",
          qs[1]["options"] == ["12 cm", "(C) 8.5 cm", "A) x", "b. y"], qs[1]["options"])
    check("null options → []", qs[2]["options"] == [], qs[2]["options"])
    check("figure_missing", [q["figure_missing"] for q in qs] == [True, False, True], [q["figure_missing"] for q in qs])
    check("audit fields removed, content_hash kept",
          not any(k in qs[0] for k in tg._NCERT_INTERNAL_FIELDS) and qs[0].get("content_hash") == "abc", sorted(qs[0]))
    check("chapter_no from sections", tg._chapter_number({"EXERCISE 5.2": 40, "5.3 nth Term": 20, "9.1": 3}) == 5)
    check("chapter_no None without numbered sections", tg._chapter_number({"Uncategorized": 10, "Poem": 4}) is None)

    # stats fallback: no pool → Supabase client must page through all 2,500 rows
    rows = [{"chapter": "Ch A" if i % 2 else "Ch B", "section": None, "question_type": "exercise"} for i in range(2500)]
    fake = FakeClient(rows=rows)
    tg.get_supabase = lambda: fake
    import app.core.db_pool as pool
    pool.get_db_connection = lambda: None
    stats = run(tg._ncert_question_stats_impl("Mathematics", "10"))
    check("stats fallback counts every row", stats.get("totalQuestions") == 2500, stats.get("totalQuestions"))
    check("stats fallback per-chapter totals", {c["chapter"]: c["total"] for c in stats["chapters"]} == {"Ch A": 1250, "Ch B": 1250},
          stats.get("chapters"))
    check("stats fallback used 3 pages", len(fake.queries) == 3, len(fake.queries))
    check("stats fallback counts visible rows only",
          all(("or_", (tg._NCERT_VISIBLE_OR,), {}) in q for q in fake.queries), fake.queries[0])
    check("stats has chapter_no", all("chapter_no" in c for c in stats["chapters"]), stats["chapters"])

    print(f"\n{'all passed' if not FAILURES else f'{len(FAILURES)} failed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
