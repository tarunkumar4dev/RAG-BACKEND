"""Response meta carries requested vs delivered question counts (fix/p0-quality, task 3)."""
from app.api.v1.endpoints import test_generator as ep
from app.models.test_generator import GeneratedQuestion


def _q(i, is_or=False):
    q = GeneratedQuestion(id=f"q{i}", text=f"Question {i} text long enough", correct_answer="x", explanation="y",
                          marks=1, difficulty="medium", chapter="Light", format="short_answer")
    q._is_or = is_or
    return q


def _req(**kw):
    rows = kw.pop("rows", [("Light", 6), ("Electricity", 4)])
    return ep.FrontendGenerateRequest(
        subject=kw.pop("subject", "Science"),
        simpleData=[ep.FrontendChapterRow(topic=t, quantity=n) for t, n in rows], **kw)


def test_custom_paper_requested_is_the_sum_of_rows():
    resp = ep._transform_backend_to_frontend([_q(i) for i in range(7)] + [_q(99, is_or=True)], _req(cbsePattern=False))
    assert resp.meta["requestedQuestions"] == 10
    assert resp.meta["deliveredQuestions"] == 7 == resp.totalQuestions     # the OR alternative is not counted


def test_cbse_paper_requested_is_the_pattern_size():
    resp = ep._transform_backend_to_frontend([_q(i) for i in range(31)], _req(cbsePattern=True))
    assert (resp.meta["requestedQuestions"], resp.meta["deliveredQuestions"]) == (38, 31)


def test_cbse_accountancy_requested_is_34():
    for subject in ("Accountancy", "Accounts"):
        resp = ep._transform_backend_to_frontend([_q(1)], _req(cbsePattern=True, subject=subject))
        assert resp.meta["requestedQuestions"] == 34
