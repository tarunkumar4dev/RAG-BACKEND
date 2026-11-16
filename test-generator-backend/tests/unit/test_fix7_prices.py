"""Gate 2 #7: checker price defaults = gemini-3.8-flash list prices (aistudio.google.com/docs/pricing, read
12 Nov 2026), switching on 1 Jan 2027; env vars still override."""
from app.services import job_worker as jw


def test_checker_price_2026_and_2027(monkeypatch):
    monkeypatch.delenv("CHECKER_PRICE_IN_PER_M_INR", raising=False)
    monkeypatch.delenv("CHECKER_PRICE_OUT_PER_M_INR", raising=False)
    assert jw.price_in_per_m_inr("2026-11-12") == 0.75 * 84 == 63.0
    assert jw.price_out_per_m_inr("2026-12-31") == 3.75 * 84 == 315.0
    assert jw.price_in_per_m_inr("2027-01-01") == 126.0
    assert jw.price_out_per_m_inr("2027-01-01") == 630.0


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("CHECKER_PRICE_IN_PER_M_INR", "73")
    monkeypatch.setenv("CHECKER_PRICE_OUT_PER_M_INR", "362")
    assert (jw.price_in_per_m_inr(), jw.price_out_per_m_inr()) == (73.0, 362.0)


def test_measured_copy_reprices(monkeypatch):
    """A measured October copy (6,245 in / 3,490 out + 1,876 thinking) at 2026 list price."""
    monkeypatch.delenv("CHECKER_PRICE_IN_PER_M_INR", raising=False)
    monkeypatch.delenv("CHECKER_PRICE_OUT_PER_M_INR", raising=False)
    monkeypatch.setattr(jw.time, "strftime", lambda f: "2026-11-12")
    cost = jw.est_cost_inr({"input_tokens": 6245, "output_tokens": 3490, "thinking_tokens": 1876})
    assert cost == round((6245 * 63 + 5366 * 315) / 1e6, 4) == 2.0837
