"""Gate 2 #7: Gemini STT is priced at the AUDIO input rate (aistudio.google.com/docs/pricing, read 12 Nov 2026)."""
from agent import config as agent_config
from agent.voice import config as vc


def test_gemini_stt_uses_audio_price(monkeypatch):
    monkeypatch.setattr(agent_config, "GEMINI_MODEL", "gemini-2.5-flash")
    # 10 s x 32 tokens/s x $1.00/M x Rs 84 = Rs 0.02688 (the text price $0.30 gave Rs 0.00806)
    assert vc.stt_cost_inr(10, "gemini") == round(10 * 32 * 1.00 / 1e6 * 84, 5) == 0.02688


def test_unknown_gemini_model_priced_conservatively(monkeypatch):
    monkeypatch.setattr(agent_config, "GEMINI_MODEL", "gemini-9-unknown")
    assert vc.stt_cost_inr(10, "gemini") == 0.02688


def test_sarvam_unchanged():
    assert vc.stt_cost_inr(60, "sarvam") == 0.49
