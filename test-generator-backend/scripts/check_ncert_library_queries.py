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
        check(f"{chapter!r}: ORDER BY section, question_number, id", orders == ["section", "question_number", "id"], orders)

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

    print(f"\n{'all passed' if not FAILURES else f'{len(FAILURES)} failed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
