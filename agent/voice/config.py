"""Voice config: env var NAMES, limits, allowlists and INR prices. Key VALUES are never stored here.

Everything is read at call time so tests (monkeypatch.setenv) and a redeploy change behaviour without code changes.

Sarvam facts below were read from docs.sarvam.ai on 10 Nov 2026 (docs/VOICE_PLAN.md §3). Re-check them when the
catalog changes: speaker ids are case-sensitive and must match the model.
"""
from __future__ import annotations

import os
from typing import Optional

# ---- switches -------------------------------------------------------------------------------------------------
VOICE_ENABLED = "VOICE_ENABLED"                       # "1" = on. Unset / anything else = off (kill switch).
VOICE_INSTITUTE_ALLOWLIST = "VOICE_INSTITUTE_ALLOWLIST"  # comma list of institute ids; unset/empty = every institute

# ---- providers -----------------------------------------------------------------------------------------------
SARVAM_KEY_ENV = os.getenv("VOICE_SARVAM_KEY_ENV", "SARVAM_API_KEY")
SARVAM_BASE = os.getenv("VOICE_SARVAM_BASE", "https://api.sarvam.ai")
STT_MODEL_DEFAULT = "saaras:v3"      # has mode=translit (Roman script); saaras:v4 has keyterms but no translit
STT_MODE_DEFAULT = "translit"
TTS_MODEL_DEFAULT = "bulbul:v4-flash"
STT_TIMEOUT_S = 8.0
TTS_TIMEOUT_S = 10.0
GEMINI_STT_TIMEOUT_S = 10.0
BREAKER_FAILURES = 3        # consecutive failures that open the breaker
BREAKER_OPEN_S = 60.0       # how long a provider is skipped once the breaker is open

# ---- limits ---------------------------------------------------------------------------------------------------
MAX_SECONDS_DEFAULT = 30    # Sarvam REST accepts audio up to 30 s
MAX_BYTES_DEFAULT = 1_048_576
USER_PER_10MIN_DEFAULT = 30
USER_DAILY_DEFAULT = 100
IDEMPOTENCY_TTL_S = 600
SPEAK_MAX_CHARS = 300
ENTITY_TTL_S = 300          # entity list per (institute, user)
CONFIDENCE_REPEAT = 0.45    # below: "Dobara boliye" (nothing is sent)
CONFIDENCE_AUTO = 0.75      # below: "Maine suna: … Sahi hai?" (user confirms); at/above: auto-send after the pause

# ---- TTS catalog allowlist (bulbul:v4-flash, `enhi` = Hindi-English code-mixed personas) ----------------------
# Gender is inferred from the name (the catalog has no gender field) — confirm by listening (docs/VOICE_DATA_NOTE).
SPEAKERS: dict[str, dict] = {
    "ishita_enhi_companion": {"gender": "female", "label": "Ishita"},
    "simran_enhi_companion": {"gender": "female", "label": "Simran"},
    "shalini_enhi_companion": {"gender": "female", "label": "Shalini"},
    "shubh_enhi_companion": {"gender": "male", "label": "Shubh"},
    "sunny_enhi_customer": {"gender": "male", "label": "Sunny"},
}
DEFAULT_SPEAKER = "ishita_enhi_companion"
PACES = (0.9, 1.0, 1.1)
TTS_LANGUAGE = {"hi": "hi-IN", "hi-en": "hi-IN", "en": "en-IN"}

# ---- prices (INR, pay-as-you-go, excl. GST; docs.sarvam.ai/api/getting-started/pricing, 10 Nov 2026) ----------
STT_INR_PER_MIN = 0.49               # billed per second
TTS_INR_PER_1000_CHARS = {"bulbul:v4-flash": 1.00, "bulbul:v3": 3.00, "bulbul:v2": 3.00}
GEMINI_AUDIO_TOKENS_PER_S = 32       # aistudio.google.com/docs/audio
INR_PER_USD = 84.0
# Gemini AUDIO input is priced separately from text (aistudio.google.com/docs/pricing, read 12 Nov 2026, Gate 2 #7):
# gemini-2.5-flash $1.00/M audio (text $0.30), gemini-3-flash-preview $1.00, gemini-2.5-flash-lite $0.30.
# The old code used the text price, undercounting Gemini STT ~3.3x. Unknown model -> $1.00 (conservative).
GEMINI_AUDIO_USD_PER_M = {"gemini-2.5-flash": 1.00, "gemini-3-flash-preview": 1.00, "gemini-2.5-flash-lite": 0.30}

# ---- static phrases (no names, no numbers): the only TTS audio that is cached -----------------------------------
STATIC_PHRASES: dict[str, dict[str, str]] = {
    "repeat": {"hi": "Dobara boliye.", "en": "Please say that again."},
    "approve_blocked": {"hi": "Voice se approve nahi hota. Screen par Confirm dabaiye.",
                        "en": "Approval can't be done by voice. Please tap Confirm on the screen."},
    "typed_fallback": {"hi": "Awaaz abhi samajh nahi aayi. Likh kar poochhiye.",
                       "en": "I couldn't hear that. Please type your question."},
    "which_one": {"hi": "Kaun sa? Screen par chuniye.", "en": "Which one? Please pick on the screen."},
}


def _env(name: str, env=os.environ) -> str:
    return (env.get(name) or "").strip()


def _int(name: str, default: int, env=os.environ) -> int:
    try:
        return max(int(_env(name, env) or default), 0)
    except ValueError:
        return default


def enabled(env=os.environ) -> bool:
    return _env(VOICE_ENABLED, env) == "1"


def allowlist(env=os.environ) -> Optional[set[str]]:
    raw = _env(VOICE_INSTITUTE_ALLOWLIST, env)
    ids = {x.strip().lower() for x in raw.split(",") if x.strip()}
    return ids or None


def institute_allowed(institute_id, env=os.environ) -> bool:
    a = allowlist(env)
    return a is None or str(institute_id).lower() in a


def sarvam_key(env=os.environ) -> Optional[str]:
    return env.get(SARVAM_KEY_ENV) or None


def gemini_key(env=os.environ) -> Optional[str]:
    from .. import config as agent_config
    return env.get(agent_config.GEMINI_KEY_ENV) or None


def stt_model(env=os.environ) -> str:
    return _env("VOICE_STT_MODEL", env) or STT_MODEL_DEFAULT


def stt_mode(env=os.environ) -> Optional[str]:
    """`mode` is a saaras:v3 parameter; with v4 nothing is sent (v4 transcribes in native script)."""
    if stt_model(env).startswith("saaras:v4"):
        return None
    return _env("VOICE_STT_MODE", env) or STT_MODE_DEFAULT


def tts_model(env=os.environ) -> str:
    return _env("VOICE_TTS_MODEL", env) or TTS_MODEL_DEFAULT


def max_seconds(env=os.environ) -> int:
    return min(_int("VOICE_MAX_SECONDS", MAX_SECONDS_DEFAULT, env) or MAX_SECONDS_DEFAULT, MAX_SECONDS_DEFAULT)


def max_bytes(env=os.environ) -> int:
    return _int("VOICE_MAX_BYTES", MAX_BYTES_DEFAULT, env) or MAX_BYTES_DEFAULT


def user_per_10min(env=os.environ) -> int:
    return _int("VOICE_USER_PER_10MIN", USER_PER_10MIN_DEFAULT, env)


def user_daily(env=os.environ) -> int:
    return _int("VOICE_USER_DAILY", USER_DAILY_DEFAULT, env)


def stt_cost_inr(seconds: float, provider: str) -> float:
    if provider == "gemini":
        from .. import config as agent_config
        pin = GEMINI_AUDIO_USD_PER_M.get(agent_config.GEMINI_MODEL, 1.00)
        return round(seconds * GEMINI_AUDIO_TOKENS_PER_S * pin / 1e6 * INR_PER_USD, 5)
    if provider == "sarvam":
        return round(seconds / 60.0 * STT_INR_PER_MIN, 5)
    return 0.0


def tts_cost_inr(chars: int, model: str, cached: bool = False) -> float:
    if cached:
        return 0.0
    return round(chars / 1000.0 * TTS_INR_PER_1000_CHARS.get(model, 1.0), 5)
