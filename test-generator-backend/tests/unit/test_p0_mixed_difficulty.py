"""A "Mixed" row is split ~40/40/20 into easy / medium / hard sub-rows (fix/p0-quality, task 4)."""
from app.api.v1.endpoints import test_generator as ep


def _chapters(quantity, difficulty="Mixed", cbse=False, **row):
    req = ep.FrontendGenerateRequest(
        subject="Science", cbsePattern=cbse,
        simpleData=[ep.FrontendChapterRow(topic="Light", quantity=quantity, difficulty=difficulty, **row)])
    return ep._transform_frontend_to_backend(req).chapters


def _levels(chapters):
    return [(c.difficulty.value, c.quantity) for c in chapters]


def test_mixed_10_is_4_4_2():
    assert _levels(_chapters(10)) == [("easy", 4), ("medium", 4), ("hard", 2)]


def test_mixed_1_is_one_medium():
    assert _levels(_chapters(1)) == [("medium", 1)]


def test_mixed_2_is_one_easy_one_medium():
    assert _levels(_chapters(2)) == [("easy", 1), ("medium", 1)]


def test_split_always_adds_up_and_has_no_empty_rows():
    for n in range(1, 51):
        parts = ep._split_mixed(n)
        assert sum(c for _, c in parts) == n and all(c >= 1 for _, c in parts), n
    assert ep._split_mixed(3) == [("easy", 1), ("medium", 1), ("hard", 1)]
    assert ep._split_mixed(5) == [("easy", 2), ("medium", 2), ("hard", 1)]


def test_sub_rows_keep_format_marks_topic():
    chapters = _chapters(10, format="Short", marks=3, subtopic="Mirrors")
    assert len(chapters) == 3
    for c in chapters:
        assert (c.chapter, c.topic, c.subtopics, c.format.value, c.marks_per_question) == \
               ("Light", "Mirrors", ["Mirrors"], "short_answer", 3)


def test_other_difficulties_and_cbse_mode_are_untouched():
    assert _levels(_chapters(10, difficulty="Hard")) == [("hard", 10)]
    assert _levels(_chapters(10, difficulty="mixed")) == [("easy", 4), ("medium", 4), ("hard", 2)]
    assert _levels(_chapters(10, cbse=True)) == [("medium", 10)]      # one row per chapter, as before
