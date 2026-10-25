"""contest_import_service: answer-key mapping, normalisation and the validator (no Gemini), plus the full
import_paper flow on GEMINI_MOCK recordings (IMPORT-PARSE / IMPORT-SOLVE)."""

import json
import time

import pytest

from app.services import contest_import_service as ci
from app.services import smart_assignment_service as svc


def q(number, text="Which one?", options=None, level="Easy", qtype="MCQ", marks=1, explanation="", ai=""):
    if options is None:
        options = [{"label": "a", "text": "one"}, {"label": "b", "text": "two"},
                   {"label": "c", "text": "three"}, {"label": "d", "text": "four"}]
    return {"number": number, "section_level": level, "type": qtype, "text": text, "options": options,
            "marks": marks, "explanation": explanation, "ai_answer": ai}


def raw(questions, table=None, lst=None, found=None, header=None):
    table, lst = table or [], lst or []
    return {"header": header or {"institute": "", "title": "Class XII — Economics", "class_grade": "XII",
                                 "subject": "Economics", "topic": "", "duration_text": "1 Hour",
                                 "max_marks": len(questions)},
            "questions": questions,
            "answer_key": {"found": bool(table or lst) if found is None else found, "table": table, "list": lst}}


def row(n, letter, text=""):
    return {"number": n, "letter": letter, "text": text}


def codes(result, code=None):
    return [(w["question"], w["code"]) for w in result["warnings"] if code is None or w["code"] == code]


# ── answer key → bare label ───────────────────────────────────────────

@pytest.mark.parametrize("printed,label", [("b", "B"), ("(b)", "B"), ("B)", "B"), ("b.", "B"), ("Ans: B", "B"),
                                           ("Ans. (c)", "C"), ("Option D", "D"), ("2", "B"), (" (A) ", "A"),
                                           ("1-b", "B"), ("12. (c)", "C")])
def test_key_letters(printed, label):
    assert ci.norm_letter(printed) == label


@pytest.mark.parametrize("printed", ["", "none", "b and c", "(ab)"])
def test_not_a_letter(printed):
    assert ci.norm_letter(printed) is None


def test_table_wins_and_grid_cross_checks():
    r = ci.build_result(raw([q(1), q(2)], table=[row(1, "(b)", "two"), row(2, "(c)", "three")],
                            lst=[row(1, "b"), row(2, "d")]))
    assert [x["correct_answer"] for x in r["questions"]] == ["B", "C"]
    assert all(x["answer_source"] == "key" for x in r["questions"])
    assert codes(r) == [(2, "KEY_GRID_MISMATCH")]
    assert r["answer_key_found"] and r["answer_key_count"] == 2


def test_list_only_key_formats():
    r = ci.build_result(raw([q(1), q(2), q(3)], lst=[row(1, "1-b"), row(2, "2. (a)"), row(3, "Ans: D")]))
    assert [x["correct_answer"] for x in r["questions"]] == ["B", "A", "D"]
    assert codes(r) == []


def test_key_letter_outside_options():
    three = [{"label": "a", "text": "x"}, {"label": "b", "text": "y"}, {"label": "c", "text": "z"}]
    r = ci.build_result(raw([q(1, options=three)], table=[row(1, "(d)")]))
    assert r["questions"][0]["correct_answer"] is None
    assert r["questions"][0]["answer_source"] == "missing"
    assert codes(r) == [(1, "KEY_LETTER_NOT_IN_OPTIONS")]


def test_key_text_matching_another_option_is_flagged_not_fixed():
    r = ci.build_result(raw([q(1)], table=[row(1, "(b)", "three")]))
    assert r["questions"][0]["correct_answer"] == "B"
    assert codes(r) == [(1, "KEY_TEXT_MISMATCH")]


def test_key_text_paraphrase_is_fine():
    r = ci.build_result(raw([q(1, options=[{"label": "a", "text": "It requires large reserves"},
                                           {"label": "b", "text": "Other"}])],
                            table=[row(1, "(a)", "It requires large reserves.")]))
    assert codes(r) == []


def test_missing_key_page_uses_ai_answer_with_warning_per_question():
    r = ci.build_result(raw([q(1, ai="c"), q(2, ai="")]))
    assert not r["answer_key_found"] and r["answer_key_count"] == 0
    assert [(x["correct_answer"], x["answer_source"]) for x in r["questions"]] == [("C", "ai"), (None, "missing")]
    assert codes(r) == [(1, "AI_ANSWER"), (2, "ANSWER_MISSING")]


def test_key_count_and_missing_entries():
    r = ci.build_result(raw([q(1), q(2), q(3)], table=[row(1, "(a)"), row(2, "(b)"), row(7, "(c)")]))
    assert codes(r) == [(3, "ANSWER_MISSING"), (7, "KEY_WITHOUT_QUESTION")]
    r = ci.build_result(raw([q(1), q(2), q(3)], table=[row(1, "(a)"), row(2, "(b)")]))
    assert codes(r) == [(3, "ANSWER_MISSING"), (None, "KEY_COUNT_MISMATCH")]


# ── options ───────────────────────────────────────────────────────────

def test_two_column_options_as_returned_by_the_model():
    """A paper printing '(a) ₹40  (b) ₹42,000 (up from ₹40,000)' / '(c) 10%  (d) ½' two per line: Gemini returns
    them in label order with the prefix sometimes left in — labels uppercase, prefixes gone, symbols intact."""
    opts = [{"label": "(a)", "text": "(a) ₹40"}, {"label": "(b)", "text": "₹42,000 (up from ₹40,000)"},
            {"label": "c", "text": "c) 10%"}, {"label": "D", "text": "½ of $30 to $25"}]
    r = ci.build_result(raw([q(1, text="1. Price rose from ₹40,000 to ₹42,000. The rise is:", options=opts)],
                            table=[row(1, "(b)")]))
    x = r["questions"][0]
    assert x["question_text"] == "Price rose from ₹40,000 to ₹42,000. The rise is:"
    assert x["options"] == [{"label": "A", "text": "₹40"}, {"label": "B", "text": "₹42,000 (up from ₹40,000)"},
                            {"label": "C", "text": "10%"}, {"label": "D", "text": "½ of $30 to $25"}]
    assert x["correct_answer"] == "B" and codes(r) == []


def test_assertion_reason_keeps_full_stem():
    stem = ("Assertion (A): The supply curve of foreign exchange slopes upward.\n"
            "Reason (R): A rise in the exchange rate makes imports cheaper for domestic buyers.")
    r = ci.build_result(raw([q(46, text=stem, options=[
        {"label": "a", "text": "Both A and R are true, and R is the correct explanation of A."},
        {"label": "b", "text": "Both A and R are true, but R is not the correct explanation of A."},
        {"label": "c", "text": "A is true, but R is false."}, {"label": "d", "text": "A is false, but R is true."}])],
        table=[row(46, "(c)", "A is true, but R is false")], header={"max_marks": 0}))
    x = r["questions"][0]
    assert x["question_text"] == stem and x["correct_answer"] == "C"
    assert [w for w in codes(r) if w[1] != "MISSING_QUESTION_NUMBER"] == []


def test_too_few_and_too_many_options():
    six = [{"label": c, "text": c * 2} for c in "abcdef"]
    seven = [{"label": c, "text": c * 2} for c in "abcdefg"]
    r = ci.build_result(raw([q(1, options=[{"label": "a", "text": "only"}]), q(2, options=six),
                             q(3, options=seven)], table=[row(1, "a"), row(2, "e"), row(3, "a")]))
    usable = [x["usable"] for x in r["questions"]]
    assert usable == [False, True, False]
    assert codes(r) == [(1, "TOO_FEW_OPTIONS"), (2, "TOO_MANY_OPTIONS"), (3, "TOO_MANY_OPTIONS")]
    assert r["questions"][1]["correct_answer"] == "E"


def test_unclear_labels_are_renumbered_with_warning():
    opts = [{"label": "i", "text": "x"}, {"label": "ii", "text": "y"}]
    r = ci.build_result(raw([q(1, options=opts)], table=[row(1, "b")]))
    assert [o["label"] for o in r["questions"][0]["options"]] == ["A", "B"]
    assert codes(r) == [(1, "OPTION_LABELS_RENUMBERED")]


# ── paper-level checks ────────────────────────────────────────────────

def test_numbering_problems_empty_text_marks_and_difficulty():
    r = ci.build_result(raw([q(1, level="Easy"), q(2, text="", level="Moderate"), q(2, level="Difficult"),
                             q(5, level="Unknown", marks=2)],
                            table=[row(1, "a"), row(2, "b"), row(5, "c")],
                            header={"title": "", "subject": "Physics", "topic": "Optics", "max_marks": 4,
                                    "duration_text": "", "class_grade": "10th", "institute": ""}))
    assert [x["difficulty"] for x in r["questions"]] == ["Easy", "Medium", "Hard", "Medium"]
    assert r["questions"][1]["usable"] is False
    got = codes(r)
    for expected in [(2, "EMPTY_QUESTION_TEXT"), (2, "DUPLICATE_QUESTION_NUMBER"), (3, "MISSING_QUESTION_NUMBER"),
                     (4, "MISSING_QUESTION_NUMBER"), (None, "MARKS_TOTAL_MISMATCH")]:
        assert expected in got
    assert r["total_marks"] == 5
    assert r["title"] == "Physics — Optics" and r["class_grade"] == "10" and r["duration_minutes"] is None
    assert all(x["chapter"] == "Optics" for x in r["questions"])


@pytest.mark.parametrize("text,minutes", [("1 Hour", 60), ("Duration: 1½ hours", 90), ("1.5 hrs", 90),
                                          ("2 hours 30 minutes", 150), ("90 min", 90), ("Time: 3 Hours", 180),
                                          ("1:30", 90), ("two hours", 120), ("", None)])
def test_duration(text, minutes):
    assert ci.duration_minutes(text) == minutes


def test_duration_clamped_with_warning():
    r = ci.build_result(raw([q(1)], table=[row(1, "a")], header={"duration_text": "4 hours", "max_marks": 1}))
    assert r["duration_minutes"] == 180 and codes(r) == [(None, "DURATION_CLAMPED")]


def test_no_questions_is_rejected():
    with pytest.raises(ci.ImportRejected):
        ci.build_result(raw([]))


def test_result_is_a_valid_contest_payload():
    from app.schemas.contest import CreateContestRequest
    from app.schemas.contest_import import ImportResult
    r = ci.build_result(raw([q(1), q(2, qtype="Short", options=[])], table=[row(1, "b"), row(2, "", "Because…")]))
    ImportResult(**r)
    body = CreateContestRequest(title=r["title"], subject="Economics", class_grade="12", questions=r["questions"])
    assert body.questions[0].correct_answer == "B" and body.questions[1].correct_answer == "Because…"


# ── chunk merge ───────────────────────────────────────────────────────

def test_merge_joins_a_question_cut_by_the_chunk_boundary():
    part1 = raw([q(1), q(2, options=[{"label": "a", "text": "x"}, {"label": "b", "text": "y"}])],
                header={"title": "T", "max_marks": 3})
    part2 = raw([q(-1, text="", options=[{"label": "c", "text": "z"}, {"label": "d", "text": "w"}]), q(3)],
                table=[row(1, "a"), row(2, "d"), row(3, "a")], header={})
    merged = ci.merge_parts([part1, part2])
    assert [x["number"] for x in merged["questions"]] == [1, 2, 3]
    assert [o["label"] for o in merged["questions"][1]["options"]] == ["a", "b", "c", "d"]
    r = ci.build_result(merged)
    assert [x["correct_answer"] for x in r["questions"]] == ["A", "D", "A"] and codes(r) == []


# ── full flow on recordings (GEMINI_MOCK) ─────────────────────────────

@pytest.fixture
def recordings(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "GEMINI_MOCK_DIR", str(tmp_path))
    monkeypatch.setattr(svc, "_mock_seen", {})
    return tmp_path


def _pdf(tmp_path, pages=2):
    import pymupdf
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    p = tmp_path / "paper.pdf"
    doc.save(str(p))
    return str(p)


def test_import_paper_flags_key_mismatch_suspect(recordings):
    (recordings / "IMPORT-PARSE.json").write_text(json.dumps(
        raw([q(1), q(2)], table=[row(1, "(b)"), row(2, "(c)")])), encoding="utf-8")
    (recordings / "IMPORT-SOLVE.json").write_text(json.dumps(
        {"answers": [{"number": 1, "label": "B"}, {"number": 2, "label": "A"}]}), encoding="utf-8")
    path = _pdf(recordings)
    r = ci.import_paper(path, ci.check_pdf(path), time.monotonic() + 270)
    assert [x["correct_answer"] for x in r["questions"]] == ["B", "C"]          # key kept
    assert codes(r) == [(2, "KEY_MISMATCH_SUSPECT")]


def test_key_check_skipped_when_time_is_short(recordings):
    (recordings / "IMPORT-PARSE.json").write_text(json.dumps(raw([q(1)], table=[row(1, "b")])), encoding="utf-8")
    path = _pdf(recordings)
    r = ci.import_paper(path, 2, time.monotonic() + 85)       # parse fits, then < 90 s left; no SOLVE recording
    assert codes(r) == [(None, "KEY_CHECK_SKIPPED")]


def test_key_check_failure_never_fails_the_import(recordings):
    (recordings / "IMPORT-PARSE.json").write_text(json.dumps(raw([q(1)], table=[row(1, "b")])), encoding="utf-8")
    path = _pdf(recordings)
    r = ci.import_paper(path, 2, time.monotonic() + 270)                         # SOLVE recording missing → error
    assert codes(r) == [(None, "KEY_CHECK_SKIPPED")] and r["questions"][0]["correct_answer"] == "B"


def test_long_paper_is_parsed_in_chunks_merged_in_page_order(tmp_path, monkeypatch):
    parts = {1: raw([q(1)], header={"max_marks": 2}),
             14: raw([q(2)], table=[row(1, "a"), row(2, "b")], header={})}
    calls = []

    def fake_chunk(path, a, b, n, deadline):
        calls.append((a, b, n))
        if a == 1:
            time.sleep(0.2)                                  # first chunk finishes last
        return parts[a]
    monkeypatch.setattr(ci, "_parse_chunk", fake_chunk)
    monkeypatch.setenv("CONTEST_IMPORT_KEY_CHECK", "0")
    path = _pdf(tmp_path, pages=25)
    r = ci.import_paper(path, ci.check_pdf(path), time.monotonic() + 270)
    assert sorted(calls) == [(1, 13, 25), (14, 25, 25)]
    assert [x["correct_answer"] for x in r["questions"]] == ["A", "B"]
    assert codes(r) == [(None, "KEY_CHECK_SKIPPED")]


def test_no_time_to_parse_fails_cleanly(recordings):
    path = _pdf(recordings)
    with pytest.raises(TimeoutError):
        ci.import_paper(path, 2, time.monotonic() + 60)


def test_parse_retries_once_after_429_with_retry_after(recordings, monkeypatch):
    calls, sleeps = [], []

    def flaky(parts, tag, schema, **kw):
        calls.append(tag)
        if len(calls) == 1:
            raise svc.GeminiBusy(429, retry_after=7)
        return raw([q(1)], table=[row(1, "b")])
    monkeypatch.setattr(ci.gemini, "generate_json", flaky)
    monkeypatch.setattr(ci.time, "sleep", sleeps.append)
    monkeypatch.setenv("CONTEST_IMPORT_KEY_CHECK", "0")
    path = _pdf(recordings)
    r = ci.import_paper(path, 2, time.monotonic() + 270)
    assert calls == ["IMPORT-PARSE", "IMPORT-PARSE"] and sleeps == [7]
    assert r["questions"][0]["correct_answer"] == "B"


def test_parse_gives_up_after_the_one_retry(recordings, monkeypatch):
    def busy(*a, **k):
        raise svc.GeminiBusy(429)
    monkeypatch.setattr(ci.gemini, "generate_json", busy)
    monkeypatch.setattr(ci.time, "sleep", lambda s: None)
    with pytest.raises(svc.GeminiBusy):
        ci.import_paper(_pdf(recordings), 2, time.monotonic() + 270)


def test_check_pdf_rejections(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    with pytest.raises(ci.ImportRejected, match="not a PDF"):
        ci.check_pdf(str(p))
    with pytest.raises(ci.ImportRejected, match="31 pages"):
        ci.check_pdf(_pdf(tmp_path, pages=31))


# ── options copied into the stem (Q30 / Q31 of the golden paper) ──────

def test_options_repeated_at_the_end_of_the_stem_are_cut():
    opts = [{"label": "a", "text": "stop all imports"}, {"label": "b", "text": "sell foreign currency"},
            {"label": "c", "text": "fix the rate"}, {"label": "d", "text": "buy foreign currency"}]
    stem = ("the central bank will usually: (a) stop all imports (b) sell foreign currency\n(c) fix the rate "
            "(d) buy foreign currency")
    r = ci.build_result(raw([q(30, text=stem, options=opts)], table=[row(30, "b")], header={"max_marks": 0}))
    assert r["questions"][0]["question_text"] == "the central bank will usually:"
    assert [o["label"] for o in r["questions"][0]["options"]] == list("ABCD")


@pytest.mark.parametrize("stem", [
    "Which is right? (a) stop all imports (b) sell foreign currency",            # only some options: untouched
    "Assertion (A): X.\nReason (R): Y.",                                          # A / R lines are not options
    "(a) stop all imports (b) sell foreign currency (c) fix the rate (d) buy foreign currency",  # nothing left
])
def test_stem_left_alone_unless_it_ends_with_all_options(stem):
    opts = [{"label": "A", "text": "stop all imports"}, {"label": "B", "text": "sell foreign currency"},
            {"label": "C", "text": "fix the rate"}, {"label": "D", "text": "buy foreign currency"}]
    assert ci._strip_inline_options(stem, opts) == stem
