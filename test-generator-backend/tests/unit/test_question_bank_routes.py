"""/api/v1/question-banks: auth, owner-from-token, RLS ownership, persistence checks, edits, paper building.

Supabase is an in-memory fake that behaves like PostgREST under the policies of 33_question_bank.sql: a client is
bound to one user; rows of other users are invisible; a forbidden INSERT raises 42501; a forbidden UPDATE / DELETE
changes nothing and returns no rows (the "silent" case). The real policies are tested on a real Postgres in the
frontend repo: supabase/tests/33_question_bank_test.py."""

import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.core import rate_limit
from app.routers import question_bank_router as qbr
from app.routers import smart_assignment_router as sar
from app.services import question_bank_service as svc

OWNER = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
MEMBER = "33333333-3333-3333-3333-333333333333"
INST = "aaaaaaaa-0000-4000-8000-000000000001"
URL = "/api/v1/question-banks"


class ApiErr(Exception):
    def __init__(self, code, message="db error"):
        super().__init__(message)
        self.code = code


class Store:
    def __init__(self):
        self.tables = {"question_banks": [], "bank_questions": [], "tests": [], "questions": []}
        self.members = {MEMBER: {INST}}
        self.drop_question_inserts = False     # RLS swallowing an insert without an error
        self.raise_on = {}                     # (table, op) → exception

    def bank(self, bank_id):
        return next((b for b in self.tables["question_banks"] if b["id"] == bank_id), None)


class Query:
    def __init__(self, store, uid, table):
        self.store, self.uid, self.table = store, uid, table
        self.op, self.payload, self.count, self.cols = "select", None, None, "*"
        self.filters, self.sort, self.n = [], None, None

    # ── builder ──
    def select(self, cols="*", count=None):
        self.op, self.cols, self.count = "select", cols, count
        return self

    def insert(self, rows):
        self.op, self.payload = "insert", rows if isinstance(rows, list) else [rows]
        return self

    def update(self, fields):
        self.op, self.payload = "update", fields
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, k, v):
        self.filters.append(lambda r: str(r.get(k)) == str(v) if not isinstance(v, bool) else r.get(k) is v)
        return self

    def order(self, col, desc=False):
        self.sort = (col, desc)
        return self

    def limit(self, n):
        self.n = n
        return self

    # ── "RLS" ──
    def _readable(self, row):
        if self.uid is None:
            return True
        bank = row if self.table == "question_banks" else self.store.bank(row.get("bank_id"))
        if self.table not in ("question_banks", "bank_questions"):
            return False
        return bool(bank) and (bank["owner_id"] == self.uid
                               or (bank.get("institute_id") in self.store.members.get(self.uid, set())))

    def _writable(self, row):
        if self.uid is None:
            return True
        bank = row if self.table == "question_banks" else self.store.bank(row.get("bank_id"))
        return bool(bank) and bank["owner_id"] == self.uid

    def execute(self):
        err = self.store.raise_on.get((self.table, self.op))
        if err:
            raise err
        rows = self.store.tables[self.table]
        if self.op == "insert":
            out = []
            for payload in self.payload:
                row = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(), **payload}
                if self.table == "question_banks":
                    row = {"institute_id": None, "page_count": None, **row}
                if not self._writable(row):
                    raise ApiErr("42501", "new row violates row-level security policy")
                if self.table == "bank_questions":
                    if self.store.drop_question_inserts:
                        continue
                    if any(r["bank_id"] == row["bank_id"] and r["content_hash"] == row["content_hash"] for r in rows):
                        raise ApiErr("23505", "duplicate key value violates unique constraint")
                rows.append(row)
                out.append(dict(row))
            return SimpleNamespace(data=out, count=None)
        hits = [r for r in rows if all(f(r) for f in self.filters) and self._readable(r)]
        if self.op == "select":
            if self.sort:
                hits = sorted(hits, key=lambda r: r.get(self.sort[0]) or 0, reverse=self.sort[1])
            total = len(hits)
            data = [dict(r) for r in hits[:self.n] if True]
            if "bank_questions(count)" in self.cols:
                for d in data:
                    d["bank_questions"] = [{"count": sum(1 for q in self.store.tables["bank_questions"]
                                                         if q["bank_id"] == d["id"])}]
            return SimpleNamespace(data=data, count=total if self.count else None)
        hits = [r for r in hits if self._writable(r)]
        if self.op == "update":
            for r in hits:
                if "content_hash" in self.payload and any(
                        o is not r and o["bank_id"] == r["bank_id"] and o["content_hash"] == self.payload["content_hash"]
                        for o in rows):
                    raise ApiErr("23505", "duplicate key value violates unique constraint")
                r.update(self.payload)
        elif self.op == "delete":
            for r in hits:
                rows.remove(r)
                if self.table == "question_banks":          # on delete cascade
                    self.store.tables["bank_questions"][:] = [
                        q for q in self.store.tables["bank_questions"] if q["bank_id"] != r["id"]]
        return SimpleNamespace(data=[dict(r) for r in hits], count=None)


class Client:
    def __init__(self, store, uid):
        self.store, self.uid = store, uid

    def table(self, name):
        return Query(self.store, self.uid, name)


@pytest.fixture
def env(monkeypatch):
    store = Store()
    who = {"id": OWNER}
    main.app.dependency_overrides[sar.require_caller] = lambda: SimpleNamespace(
        user=SimpleNamespace(id=who["id"]), db=Client(store, who["id"]), token="t")
    monkeypatch.setattr(qbr, "get_supabase_admin", lambda: Client(store, None))
    rendered = []

    def fake_render(questions, **kw):
        rendered.append((questions, kw))
        return b"%PDF-fake"
    monkeypatch.setattr(svc, "render_paper", fake_render)
    main._rate_store.clear()
    yield SimpleNamespace(client=TestClient(main.app), store=store, who=who, rendered=rendered)
    main.app.dependency_overrides.clear()


def mcq(i, answer="B", source="key"):
    return {"question_text": f"MCQ number {i}?", "question_type": "MCQ",
            "options": [{"label": "a", "text": f"first {i}"}, {"label": "b", "text": f"second {i}"}],
            "correct_answer": answer, "answer_source": source, "marks": 1}


def long(i):
    return {"question_text": f"Explain topic {i}.", "question_type": "Long", "options": [],
            "correct_answer": None, "answer_source": "missing", "marks": 5}


def make_bank(env, questions=None, **extra):
    body = {"name": "Eco bank", "questions": questions or [mcq(i) for i in range(1, 9)] + [long(i) for i in (1, 2, 3)],
            **extra}
    r = env.client.post(URL, json=body)
    assert r.status_code == 201, r.text
    return r.json()["bank"]["id"]


def paper(**over):
    return {"title": "Unit Test 1", "sections": [{"qtype": "mcq", "count": 5, "marks_each": 1},
                                                 {"qtype": "long", "count": 2, "marks_each": 5}],
            "include_answer_key": True, "template": "modern", "shuffle": False, **over}


# ── auth ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method,path", [
    ("post", URL), ("get", URL), ("get", f"{URL}/{uuid.uuid4()}/questions"),
    ("patch", f"{URL}/{uuid.uuid4()}/questions/{uuid.uuid4()}"),
    ("delete", f"{URL}/{uuid.uuid4()}/questions/{uuid.uuid4()}"), ("delete", f"{URL}/{uuid.uuid4()}"),
    ("post", f"{URL}/{uuid.uuid4()}/build-paper"),
])
def test_every_route_needs_a_token(method, path):
    main.app.dependency_overrides.clear()
    main._rate_store.clear()
    r = getattr(TestClient(main.app), method)(path, **({"json": {}} if method in ("post", "patch") else {}))
    assert r.status_code == 401


def test_every_route_has_a_rate_limit():
    routes = [(m, r.path) for r in main.app.routes if getattr(r, "path", "").startswith(URL) for m in r.methods]
    assert len(routes) == 7
    for method, path in routes:
        assert f"{method} {path}" in rate_limit.ROUTE_LIMITS, (method, path)


# ── create ───────────────────────────────────────────────────────────────────

def test_create_persists_the_bank_for_the_token_user(env):
    r = env.client.post(URL, json={"name": "  Eco   bank ", "source_filename": "C:\\papers\\eco<2023>.pdf",
                                   "subject": "Economics", "class_grade": "12",
                                   "questions": [mcq(1), mcq(1), mcq(2, source="ai"), long(1)]})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["stats"]["saved"] == 3 and body["stats"]["duplicates_removed"] == 1
    assert body["stats"]["ai_answers_dropped"] == 1 and body["stats"]["answers_matched"] == 1
    assert body["bank"]["question_count"] == 3 and body["bank"]["status"] == "ready" and body["bank"]["can_edit"]
    # really in the tables, owned by the token's user
    banks, questions = env.store.tables["question_banks"], env.store.tables["bank_questions"]
    assert [(b["owner_id"], b["name"], b["source_filename"]) for b in banks] == [(OWNER, "Eco bank", "eco2023.pdf")]
    assert [(q["position"], q["qtype"], q["answer"], q["answer_source"], q["needs_review"]) for q in questions] == [
        (1, "mcq", "B", "key", False), (2, "mcq", None, None, True), (3, "long", None, None, False)]
    assert all(q["bank_id"] == banks[0]["id"] for q in questions)


@pytest.mark.parametrize("extra", [{"owner_id": OTHER}, {"teacher_id": OTHER}, {"user_id": OTHER},
                                   {"institute_id": INST}])
def test_create_rejects_a_client_sent_owner(env, extra):
    r = env.client.post(URL, json={"name": "x", "questions": [mcq(1)], **extra})
    assert r.status_code == 422 and env.store.tables["question_banks"] == []


def test_create_rejects_unknown_question_fields_and_bad_sizes(env):
    assert env.client.post(URL, json={"name": "x", "questions": [{**mcq(1), "is_admin": True}]}).status_code == 422
    assert env.client.post(URL, json={"name": "x", "questions": []}).status_code == 422
    assert env.client.post(URL, json={"name": "", "questions": [mcq(1)]}).status_code == 422
    assert env.client.post(URL, json={"name": "x", "questions": [mcq(i) for i in range(301)]}).status_code == 422
    assert env.client.post(URL, json={"name": "x", "questions": [{**mcq(1), "question_text": "y" * 6001}]}).status_code == 422
    r = env.client.post(URL, json={"name": "x", "questions": [{**mcq(1), "question_text": "   "}]})
    assert r.status_code == 422 and r.json()["detail"] == "None of these questions has any text."
    assert env.store.tables["question_banks"] == []


def test_create_fails_loudly_when_the_questions_do_not_persist(env):
    """The old bug: the API said OK and the row was never there. Here the bank is removed and the call fails."""
    env.store.drop_question_inserts = True
    r = env.client.post(URL, json={"name": "x", "questions": [mcq(1), mcq(2)]})
    assert r.status_code == 500 and "could not be saved" in r.json()["detail"]
    assert env.store.tables["question_banks"] == [] and env.store.tables["bank_questions"] == []


def test_create_cleans_up_when_the_question_insert_is_refused(env):
    env.store.raise_on[("bank_questions", "insert")] = ApiErr("42501", "row-level security, secret detail")
    r = env.client.post(URL, json={"name": "x", "questions": [mcq(1)]})
    assert r.status_code == 403 and "secret" not in r.text
    assert env.store.tables["question_banks"] == []


def test_database_errors_never_reach_the_client(env):
    env.store.raise_on[("question_banks", "select")] = RuntimeError("postgres://user:pw@db.internal:5432 timeout")
    for r in (env.client.get(URL), env.client.post(URL, json={"name": "x", "questions": [mcq(1)]})):
        assert r.status_code == 500 and "postgres" not in r.text and "internal" not in r.text


def test_missing_table_says_not_available(env):
    env.store.raise_on[("question_banks", "select")] = ApiErr("PGRST205", "Could not find the table")
    r = env.client.get(URL)
    assert r.status_code == 503 and r.json()["detail"] == qbr.NOT_SET_UP


def test_bank_limit(env, monkeypatch):
    monkeypatch.setattr(qbr, "MAX_BANKS", 2)
    make_bank(env)
    make_bank(env)
    r = env.client.post(URL, json={"name": "third", "questions": [mcq(1)]})
    assert r.status_code == 409 and len(env.store.tables["question_banks"]) == 2
    env.who["id"] = OTHER                                    # the limit is per user
    make_bank(env)


# ── read: only your own ──────────────────────────────────────────────────────

def test_list_shows_only_the_callers_banks(env):
    bank_id = make_bank(env)
    mine = env.client.get(URL).json()["banks"]
    assert [(b["id"], b["question_count"], b["can_edit"]) for b in mine] == [(bank_id, 11, True)]
    env.who["id"] = OTHER
    assert env.client.get(URL).json()["banks"] == []


def test_questions_filters(env):
    qs = [mcq(1), mcq(2, answer=None, source="missing"), long(1),
          {**long(2), "question_text": "Explain inflation. [CBSE 2023]"}]
    bank_id = make_bank(env, qs)
    get = lambda **p: [q["position"] for q in env.client.get(f"{URL}/{bank_id}/questions", params=p).json()["questions"]]
    assert get() == [1, 2, 3, 4]
    assert get(qtype="mcq") == [1, 2] and get(qtype="long") == [3, 4]
    assert get(needs_review="true") == [2] and get(needs_review="false") == [1, 3, 4]
    assert get(year=2023) == [4] and get(year=2019) == []
    assert env.client.get(f"{URL}/{bank_id}/questions", params={"qtype": "essay"}).status_code == 422
    assert env.client.get(f"{URL}/{bank_id}/questions", params={"year": 12}).status_code == 422


def test_another_user_gets_404_everywhere(env):
    bank_id = make_bank(env)
    qid = env.store.tables["bank_questions"][0]["id"]
    env.who["id"] = OTHER
    assert env.client.get(f"{URL}/{bank_id}/questions").status_code == 404
    assert env.client.patch(f"{URL}/{bank_id}/questions/{qid}", json={"answer": "A"}).status_code == 404
    assert env.client.delete(f"{URL}/{bank_id}/questions/{qid}").status_code == 404
    assert env.client.delete(f"{URL}/{bank_id}").status_code == 404
    assert env.client.post(f"{URL}/{bank_id}/build-paper", json=paper()).status_code == 404
    assert len(env.store.tables["bank_questions"]) == 11 and env.store.tables["bank_questions"][0]["answer"] == "B"
    assert env.rendered == [] and env.store.tables["tests"] == []


def test_bad_ids_are_404(env):
    assert env.client.get(f"{URL}/not-a-uuid/questions").status_code == 404
    assert env.client.delete(f"{URL}/../../etc").status_code in (404, 405)
    bank_id = make_bank(env)
    assert env.client.patch(f"{URL}/{bank_id}/questions/123", json={"answer": "A"}).status_code == 404
    assert env.client.patch(f"{URL}/{bank_id}/questions/{uuid.uuid4()}", json={"answer": "A"}).status_code == 404


def test_institute_member_reads_but_cannot_change(env):
    bank_id = make_bank(env)
    env.store.bank(bank_id)["institute_id"] = INST
    qid = env.store.tables["bank_questions"][0]["id"]
    env.who["id"] = MEMBER
    assert len(env.client.get(f"{URL}/{bank_id}/questions").json()["questions"]) == 11
    assert env.client.get(URL).json()["banks"][0]["can_edit"] is False
    assert env.client.patch(f"{URL}/{bank_id}/questions/{qid}", json={"answer": "A"}).status_code == 403
    assert env.client.delete(f"{URL}/{bank_id}/questions/{qid}").status_code == 403
    assert env.client.delete(f"{URL}/{bank_id}").status_code == 403
    assert len(env.store.tables["bank_questions"]) == 11


def test_a_write_that_changes_no_row_is_not_reported_as_success(env, monkeypatch):
    """Backend check passes (owner) but the database filters the write out: 403, not 200."""
    bank_id = make_bank(env)
    qid = env.store.tables["bank_questions"][0]["id"]
    monkeypatch.setattr(Query, "_writable", lambda self, row: False)
    assert env.client.patch(f"{URL}/{bank_id}/questions/{qid}", json={"answer": "A"}).status_code == 403
    assert env.client.delete(f"{URL}/{bank_id}/questions/{qid}").status_code == 403
    assert env.client.delete(f"{URL}/{bank_id}").status_code == 403


# ── edit / delete ────────────────────────────────────────────────────────────

def test_patch_answer_marks_and_review(env):
    bank_id = make_bank(env, [mcq(1, answer=None, source="missing"), long(1)])
    q1, q2 = (q["id"] for q in env.store.tables["bank_questions"])
    url = f"{URL}/{bank_id}/questions"
    assert env.store.tables["bank_questions"][0]["needs_review"] is True

    r = env.client.patch(f"{url}/{q1}", json={"answer": "(a)", "marks": 2})
    assert r.status_code == 200, r.text
    assert {k: r.json()["question"][k] for k in ("answer", "answer_source", "marks", "needs_review")} == {
        "answer": "A", "answer_source": "teacher", "marks": 2, "needs_review": False}
    assert env.store.tables["bank_questions"][0]["answer"] == "A"                # persisted

    assert env.client.patch(f"{url}/{q1}", json={"answer": "Z"}).status_code == 422
    assert env.client.patch(f"{url}/{q1}", json={}).status_code == 422
    assert env.client.patch(f"{url}/{q1}", json={"bank_id": str(uuid.uuid4())}).status_code == 422
    assert env.client.patch(f"{url}/{q1}", json={"needs_review": False}).status_code == 422
    assert env.client.patch(f"{url}/{q2}", json={"options": [{"label": "A", "text": "x"}]}).status_code == 422

    r = env.client.patch(f"{url}/{q1}", json={"answer": None})                  # cleared → flagged again
    assert (r.json()["question"]["answer"], r.json()["question"]["answer_source"],
            r.json()["question"]["needs_review"]) == (None, None, True)
    r = env.client.patch(f"{url}/{q1}", json={"reviewed": True})                # an MCQ still needs its answer
    assert r.json()["question"]["needs_review"] is True
    r = env.client.patch(f"{url}/{q2}", json={"answer": "Model answer."})
    assert (r.json()["question"]["answer"], r.json()["question"]["answer_source"]) == ("Model answer.", "teacher")
    r = env.client.patch(f"{url}/{q2}", json={"excluded": True})
    assert r.json()["question"]["excluded"] is True and r.json()["question"]["answer"] == "Model answer."


def test_patch_options_drops_an_answer_that_no_longer_fits(env):
    bank_id = make_bank(env, [{**mcq(1, answer="C"), "options": [{"label": "a", "text": "p"}, {"label": "b", "text": "q"},
                                                                 {"label": "c", "text": "r"}]}])
    qid = env.store.tables["bank_questions"][0]["id"]
    r = env.client.patch(f"{URL}/{bank_id}/questions/{qid}", json={"options": [{"label": "A", "text": "p"},
                                                                              {"label": "B", "text": "q"}]})
    assert r.status_code == 200
    assert (r.json()["question"]["answer"], r.json()["question"]["needs_review"]) == (None, True)
    r = env.client.patch(f"{URL}/{bank_id}/questions/{qid}", json={"options": [{"label": "A", "text": "only"}]})
    assert r.status_code == 422


def test_patch_text_to_an_existing_question_is_409(env):
    bank_id = make_bank(env, [long(1), long(2)])
    qid = env.store.tables["bank_questions"][1]["id"]
    r = env.client.patch(f"{URL}/{bank_id}/questions/{qid}", json={"question_text": "explain  topic 1."})
    assert r.status_code == 409
    r = env.client.patch(f"{URL}/{bank_id}/questions/{qid}", json={"question_text": "  Explain   topic 9. "})
    assert r.json()["question"]["question_text"] == "Explain topic 9."


def test_delete_question_and_bank(env):
    bank_id = make_bank(env)
    qid = env.store.tables["bank_questions"][0]["id"]
    assert env.client.delete(f"{URL}/{bank_id}/questions/{qid}").json() == {"deleted": True}
    assert len(env.store.tables["bank_questions"]) == 10
    assert env.client.delete(f"{URL}/{bank_id}/questions/{qid}").status_code == 404
    assert env.client.delete(f"{URL}/{bank_id}").json() == {"deleted": True}
    assert env.store.tables["question_banks"] == [] and env.store.tables["bank_questions"] == []


# ── build paper ──────────────────────────────────────────────────────────────

def test_build_paper_uses_only_bank_questions_and_saves_history(env):
    bank_id = make_bank(env, subject="Economics", class_grade="XII")
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper(institute_name="Sunrise Classes"))
    assert r.status_code == 200, r.text
    assert r.content == b"%PDF-fake" and r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"] == 'attachment; filename="Unit_Test_1.pdf"'
    meta = json.loads(r.headers["x-paper-meta"])
    assert (meta["questions"], meta["total_marks"], meta["answers_found"], meta["answers_missing"]) == (7, 15, 5, 2)
    assert meta["saved_to_history"] is True

    questions, kw = env.rendered[0]
    bank_texts = {q["question_text"] for q in env.store.tables["bank_questions"]}
    assert len(questions) == 7 and all(q["text"] in bank_texts for q in questions)
    assert [q["section"] for q in questions] == ["A"] * 5 + ["B"] * 2
    assert [q["correctAnswer"] for q in questions] == ["B"] * 5 + ["", ""]          # nothing invented for the long ones
    assert kw["subject"] == "Economics" and kw["class_grade"] == "12" and kw["board"] == "CBSE"
    assert kw["institute_name"] == "Sunrise Classes" and kw["include_answer_key"] is True and kw["fmt"] == "pdf"

    tests, saved = env.store.tables["tests"], env.store.tables["questions"]
    assert [(t["id"], t["teacher_id"], t["exam_title"], t["status"], t["total_questions"], t["total_marks"])
            for t in tests] == [(meta["test_id"], OWNER, "Unit Test 1", "saved", 7, 15)]
    assert [(s["position"], s["section"], s["format"]) for s in saved][:2] == [(1, "A", "mcq"), (2, "A", "mcq")]
    assert [s["text"] for s in saved] == [q["text"] for q in questions]


def test_build_paper_too_few_questions(env):
    bank_id = make_bank(env, [mcq(i) for i in range(1, 7)])
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper(
        sections=[{"qtype": "mcq", "count": 10, "marks_each": 1}, {"qtype": "true_false", "count": 2, "marks_each": 1}]))
    assert r.status_code == 422
    assert r.json()["detail"] == ("Section A needs 10 MCQs, bank has 6. "
                                  "Section B needs 2 True/False questions, bank has 0.")
    assert env.rendered == [] and env.store.tables["tests"] == []


def test_build_paper_skips_needs_review_and_excluded(env):
    bank_id = make_bank(env, [mcq(1), mcq(2, answer=None, source="missing"), mcq(3), mcq(4)])
    q4 = env.store.tables["bank_questions"][3]["id"]
    env.client.patch(f"{URL}/{bank_id}/questions/{q4}", json={"excluded": True})
    one = [{"qtype": "mcq", "count": 3, "marks_each": 1}]
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper(sections=one))
    assert r.status_code == 422 and r.json()["detail"] == "Section A needs 3 MCQs, bank has 2 (1 more needs review)."
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper(sections=one, allow_needs_review=True))
    assert r.status_code == 200
    assert [q["text"] for q in env.rendered[0][0]] == ["MCQ number 1?", "MCQ number 2?", "MCQ number 3?"]
    assert env.rendered[0][0][1]["correctAnswer"] == ""


def test_build_paper_year_filter_and_shuffle(env):
    qs = [{**mcq(i), "question_text": f"MCQ number {i}? [CBSE {2023 if i <= 4 else 2022}]"} for i in range(1, 9)]
    bank_id = make_bank(env, qs)
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper(
        sections=[{"qtype": "mcq", "count": 4, "marks_each": 1}], year_filter=2023, shuffle=True))
    assert r.status_code == 200
    assert sorted(q["text"] for q in env.rendered[0][0]) == [f"MCQ number {i}? [CBSE 2023]" for i in range(1, 5)]
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper(
        sections=[{"qtype": "mcq", "count": 5, "marks_each": 1}], year_filter=2023))
    assert r.json()["detail"] == "Section A needs 5 MCQs, bank has 4 from 2023."


@pytest.mark.parametrize("change,status", [
    ({"template": "../../etc/passwd"}, 422), ({"template": "nope"}, 422), ({"format": "exe"}, 422),
    ({"sections": []}, 422), ({"sections": [{"qtype": "essay", "count": 1, "marks_each": 1}]}, 422),
    ({"sections": [{"qtype": "mcq", "count": 0, "marks_each": 1}]}, 422),
    ({"sections": [{"qtype": "mcq", "count": 1, "marks_each": 0}]}, 422),
    ({"sections": [{"qtype": "mcq", "count": 1, "marks_each": 1}] * 7}, 422),
    ({"sections": [{"qtype": "mcq", "count": 100, "marks_each": 1}] * 3}, 422),
    ({"title": ""}, 422), ({"title": "x" * 201}, 422), ({"teacher_id": OTHER}, 422), ({"year_filter": 3}, 422),
])
def test_build_paper_input_limits(env, change, status):
    bank_id = make_bank(env)
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper(**change))
    assert r.status_code == status and env.rendered == []


def test_build_paper_still_delivers_when_history_cannot_be_saved(env):
    bank_id = make_bank(env)
    env.store.raise_on[("tests", "insert")] = RuntimeError("service key missing")
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper(format="docx"))
    assert r.status_code == 200 and r.headers["content-type"].endswith("wordprocessingml.document")
    meta = json.loads(r.headers["x-paper-meta"])
    assert meta["saved_to_history"] is False and meta["test_id"] is None


def test_build_paper_render_failure_is_a_clean_500(env, monkeypatch):
    bank_id = make_bank(env)

    def boom(*a, **k):
        raise RuntimeError("reportlab internal path C:/secret")
    monkeypatch.setattr(svc, "render_paper", boom)
    r = env.client.post(f"{URL}/{bank_id}/build-paper", json=paper())
    assert r.status_code == 500 and "secret" not in r.text and env.store.tables["tests"] == []
