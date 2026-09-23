"""
a4ai — NCERT Question Extraction (v5 — GOD-TIER PRODUCTION)
============================================================
Async parallel extraction: ncert_chunks → ncert_questions.

CRITICAL FIXES IN v5:
  ✓ FIXED: "unmatched '{' in format spec" — .format() breaks on { } in content.
           Now uses .replace() so NCERT math/physics/accountancy braces survive.
  ✓ FIXED: Empty-result retry — 0 questions no longer silently ignored.
           Retries with stricter prompt + temp bump (WAVES bug fix).
  ✓ JSON Engine: 7-layer repair — invalid escapes, truncation, markdown, control chars.
  ✓ Rate Limits: token bucket + per-key cooldown + dead-key tracking.
  ✓ Data Quality: hash dedup, whitelists, length guards, type coercion.
  ✓ Batching: chunk dedup, oversize truncation, dynamic sizing.
  ✓ DB: async parallel insert, batch+fallback, structured error logs.
  ✓ Observability: extraction_debug.txt + per-key telemetry.

USAGE:
  pip install supabase aiohttp google-genai python-dotenv

  .env:
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_SERVICE_KEY=eyJ...
    GEMINI_API_KEY=key1
    GEMINI_API_KEY_2=key2      # optional, for rotation
    GEMINI_MODEL=gemini-3.6-flash
    WORKERS=20                 # free tier; use 200 for paid
    RPM_PER_KEY=15             # free tier; use 900 for paid
    MAX_CHARS_PER_BATCH=40000
    MAX_OUTPUT_TOKENS=32768
    TEMPERATURE=0.35

  Run:
    python 02_extract_ncert_questions.py --dry-run --chapter WAVES
    python 02_extract_ncert_questions.py --chapter WAVES --class_grade 11
    python 02_extract_ncert_questions.py --resume
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import asyncio
import logging
import argparse
import hashlib
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# ── UTF-8 stdout for Windows / Linux ──
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = (
    os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    or os.environ.get("SUPABASE_KEY", "").strip()
)

_raw_keys = [
    os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_KEY", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
    os.environ.get("GEMINI_API_KEY_3", ""),
    os.environ.get("GEMINI_API_KEY_4", ""),
    os.environ.get("GEMINI_API_KEY_5", ""),
    os.environ.get("GEMINI_API_KEY_6", ""),
    os.environ.get("GEMINI_API_KEY_7", ""),
    os.environ.get("GEMINI_API_KEY_8", ""),
    os.environ.get("GEMINI_API_KEY_9", ""),
    os.environ.get("GEMINI_API_KEY_10", ""),
]
API_KEYS: List[str] = [k.strip() for k in _raw_keys if k and k.strip()]

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip()

DEFAULT_WORKERS = int(os.environ.get("WORKERS", "20"))
DEFAULT_RPM_PER_KEY = int(os.environ.get("RPM_PER_KEY", "15"))
MAX_CHARS_PER_BATCH = int(os.environ.get("MAX_CHARS_PER_BATCH", "40000"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32768"))
BATCH_INSERT_SIZE = int(os.environ.get("BATCH_INSERT_SIZE", "500"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "4"))

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.35"))
RETRY_TEMPERATURE = float(os.environ.get("RETRY_TEMPERATURE", "0.55"))

KEY_DEAD_THRESHOLD = int(os.environ.get("KEY_DEAD_THRESHOLD", "5"))
KEY_COOLDOWN_SECONDS = int(os.environ.get("KEY_COOLDOWN_SECONDS", "60"))

MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", str(MAX_CHARS_PER_BATCH)))
MAX_QUESTION_CHARS = int(os.environ.get("MAX_QUESTION_CHARS", "8000"))
MIN_QUESTION_CHARS = int(os.environ.get("MIN_QUESTION_CHARS", "5"))

EMPTY_RETRY_ENABLED = os.environ.get("EMPTY_RETRY_ENABLED", "1") == "1"
EMPTY_RETRY_MAX = int(os.environ.get("EMPTY_RETRY_MAX", "2"))


# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

logger = logging.getLogger("extract")


def setup_logging(level: int = logging.INFO) -> None:
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        fh = logging.FileHandler("extraction_log.txt", mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Warning: could not open extraction_log.txt: {e}", file=sys.stderr)

    try:
        dh = logging.FileHandler("extraction_debug.txt", mode="a", encoding="utf-8")
        dh.setFormatter(fmt)
        dh.setLevel(logging.DEBUG)
        logger.addHandler(dh)
    except Exception:
        pass


def _dump_debug_json_failure(stage: str, raw: str, extra: str = "") -> None:
    try:
        with open("extraction_debug.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 72 + "\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] JSON_FAIL stage={stage}\n")
            f.write(f"Raw length: {len(raw)} chars\n")
            if extra:
                f.write(f"Extra: {extra}\n")
            f.write(f"First 800 chars:\n{raw[:800]!r}\n")
            f.write(f"Last 800 chars:\n{raw[-800:]!r}\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are an expert NCERT textbook analyst for Indian school education.

Extract EVERY question from this Class {class_grade} {subject} — "{chapter}" content.

CONTENT:
---
{content}
---

EXTRACTION RULES (STRICT — follow exactly):

1. Extract EVERY question. NCERT chapters typically have 20-100+ questions per batch.
   Do NOT skip any. Do NOT summarize. Do NOT paraphrase.

2. Sources to scan:
   • Numbered exercise questions (Q1, Q2, 1., 2., i., ii.)
   • In-text questions mid-chapter (marked "?" or "Intext")
   • Solved examples (Example 3.1, Example 3.2)
   • Activities (Activity 3.1)
   • HOTS / Higher Order Thinking Skills
   • Fill in the blanks
   • Multiple choice questions
   • "Do you know?" prompts if phrased as question
   • Assertion-Reason items

3. For each question output:
   • question_number: as it appears (Q.1, Example 5.3, Activity 9.2)
   • question_text: EXACT verbatim text. Preserve Unicode, math symbols, Hindi.
   • question_type: "exercise" | "example" | "intext" | "activity" | "hots" | "mcq" | "fillblank"
   • section: section heading if visible, else null
   • answer: verbatim answer if present in content, else null
   • options: array of strings for MCQs; empty array [] otherwise
   • difficulty: "easy" | "medium" | "hard"
   • marks: 1 (MCQ/fill), 2 (short), 3 (reasoning), 5 (long/numerical)

4. If content is purely explanatory with ZERO questions, return empty questions array.

5. NEVER invent questions. Only extract what is ACTUALLY in the text.

6. Preserve original language (English, Hindi, or mixed as written).

JSON ESCAPE RULES (CRITICAL — follow exactly):
- Escape double quotes inside strings as \\"
- Escape backslashes as \\\\
- Do NOT use invalid escapes like \\s, \\d, \\w, \\k, \\(, \\{
- Use \\n only for line breaks inside strings, or just use a space
- Return ONLY valid JSON, no markdown fences, no commentary

RESPONSE FORMAT (JSON only):
{
  "questions": [
    {
      "question_number": "Q.1",
      "question_text": "Exact question text here",
      "question_type": "exercise",
      "section": "Exercise 3.1",
      "answer": "Answer if available, else null",
      "options": [],
      "difficulty": "medium",
      "marks": 2
    }
  ]
}

If no questions found, return: {"questions": []}
"""


EXTRACTION_PROMPT_RETRY = """You are a meticulous NCERT question extractor.

PREVIOUS extraction found 0 questions. This is almost certainly WRONG.
NCERT chapters contain many questions. Re-analyze this content CAREFULLY.

Class {class_grade} {subject} — "{chapter}":

---
{content}
---

SEARCH HARDER for ANY of these:
- Sentences ending with "?"
- Numbered items: 1., 2., (i), (ii), (a), (b)
- Imperative verbs: "Find", "Calculate", "Explain", "Define", "State", "Why", "How", "Derive"
- Examples with "Solution:" following
- "Activity" blocks
- Exercise section markers
- Anything that looks like a question or problem

If you still truly find NO questions, return empty questions array.

Return ONLY valid JSON (no markdown, no commentary):
{"questions": [{"question_number": "...", "question_text": "...", "question_type": "exercise", "section": null, "answer": null, "options": [], "difficulty": "medium", "marks": 2}]}
"""


# ═══════════════════════════════════════════════════════════════════
# ⭐ SAFE FORMATTER — THE CRITICAL FIX
# ═══════════════════════════════════════════════════════════════════

def safe_format(template: str, **kwargs) -> str:
    """
    Safe string formatter that survives { } in user content.

    WHY: Python's .format() treats {x} in the content as a placeholder,
    causing "KeyError: unmatched '{' in format spec" for NCERT math/physics
    content that contains set notation, equations, or JSON-like braces.

    SOLUTION: Use .replace() for each named placeholder instead.
    Content braces are preserved as-is.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


# ═══════════════════════════════════════════════════════════════════
# SUPABASE CLIENT
# ═══════════════════════════════════════════════════════════════════

_supabase_client = None


def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        if not SUPABASE_URL:
            raise ValueError("SUPABASE_URL not set in environment")
        if not SUPABASE_KEY:
            raise ValueError("SUPABASE_SERVICE_KEY (or SUPABASE_KEY) not set")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ═══════════════════════════════════════════════════════════════════
# JSON REPAIR ENGINE — 7 LAYERS
# ═══════════════════════════════════════════════════════════════════

_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _strip_fences(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    if t.lstrip().lower().startswith("json"):
        idx = t.find("json")
        t = t[idx + 4:]
    return t.strip()


def _escape_control_chars_in_strings(text: str) -> str:
    result: List[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            code = ord(ch)
            if ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
            elif code < 0x20:
                result.append(f"\\u{code:04x}")
            else:
                result.append(ch)
        else:
            result.append(ch)
    return "".join(result)


def _fix_invalid_escapes(text: str) -> str:
    return _INVALID_ESCAPE_RE.sub(r'\\\\', text)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _balanced_extract(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _regex_salvage_question(candidate: str) -> Optional[dict]:
    try:
        def get_field(name: str) -> Optional[str]:
            pattern = rf'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)"'
            m = re.search(pattern, candidate, re.DOTALL)
            if not m:
                return None
            val = m.group(1)
            val = val.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
            val = val.replace("\\r", " ").replace("\\\\", "\\")
            val = re.sub(r'\\(.)', r'\1', val)
            return val.strip()

        text_field = get_field("question_text") or get_field("text")
        if not text_field or len(text_field) < 5:
            return None

        def get_num(name: str, default: Any) -> Any:
            m = re.search(rf'"{name}"\s*:\s*(\d+)', candidate)
            if not m:
                return default
            try:
                return int(m.group(1))
            except Exception:
                return default

        opts: List[str] = []
        opts_m = re.search(r'"options"\s*:\s*\[(.*?)\]', candidate, re.DOTALL)
        if opts_m:
            for opt_m in re.finditer(r'"((?:[^"\\]|\\.)*)"', opts_m.group(1)):
                ov = opt_m.group(1)
                ov = re.sub(r'\\(.)', r'\1', ov).strip()
                if ov:
                    opts.append(ov)

        qtype_m = re.search(r'"question_type"\s*:\s*"([^"]+)"', candidate)
        qtype = qtype_m.group(1) if qtype_m else "exercise"

        return {
            "question_text": text_field,
            "question_number": get_field("question_number"),
            "question_type": qtype,
            "section": get_field("section"),
            "answer": get_field("answer"),
            "options": opts,
            "difficulty": get_field("difficulty") or "medium",
            "marks": get_num("marks", 2),
        }
    except Exception:
        return None


def _parse_question_object(candidate: str) -> Optional[dict]:
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = json.loads(_escape_control_chars_in_strings(candidate))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = json.loads(_fix_invalid_escapes(candidate))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        fixed = _escape_control_chars_in_strings(candidate)
        fixed = _fix_invalid_escapes(fixed)
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        fixed = _escape_control_chars_in_strings(candidate)
        fixed = _fix_invalid_escapes(fixed)
        fixed = _remove_trailing_commas(fixed)
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return _regex_salvage_question(candidate)


def _extract_questions_individually(text: str) -> List[dict]:
    questions: List[dict] = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    m = re.search(r'"questions"\s*:\s*\[', text)
    if not m:
        return questions

    i = m.end()
    while i < len(text):
        ch = text[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if ch == "\\":
            escape_next = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
        if not in_string:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start:i + 1]
                    obj = _parse_question_object(candidate)
                    if obj and obj.get("question_text"):
                        questions.append(obj)
                    start = -1
        i += 1
    return questions


def smart_json_parse(raw: str, debug_stage: str = "unknown") -> Optional[dict]:
    if not raw:
        return None

    text = _strip_fences(raw)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    balanced = _balanced_extract(text) or text
    try:
        obj = json.loads(balanced)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    try:
        obj = json.loads(_escape_control_chars_in_strings(balanced))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    try:
        fixed = _fix_invalid_escapes(balanced)
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    try:
        fixed = _escape_control_chars_in_strings(balanced)
        fixed = _fix_invalid_escapes(fixed)
        fixed = _remove_trailing_commas(fixed)
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    salvaged = _extract_questions_individually(balanced)
    if salvaged:
        logger.warning(f"   ⚠ Salvaged {len(salvaged)} questions via per-object regex")
        return {"questions": salvaged}

    _dump_debug_json_failure(debug_stage, raw)
    return None


# ═══════════════════════════════════════════════════════════════════
# TOKEN BUCKET RATE LIMITER
# ═══════════════════════════════════════════════════════════════════

class TokenBucket:
    __slots__ = ("rate_per_min", "capacity", "tokens", "last_refill", "_lock")

    def __init__(self, rate_per_min: int, capacity: Optional[int] = None):
        self.rate_per_min = max(1, rate_per_min)
        self.capacity = max(1, capacity if capacity is not None else rate_per_min)
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * (self.rate_per_min / 60.0),
            )
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            wait_time = (1.0 - self.tokens) * 60.0 / self.rate_per_min
            await asyncio.sleep(wait_time)
            self.tokens = 0.0
            self.last_refill = time.monotonic()


# ═══════════════════════════════════════════════════════════════════
# KEY MANAGER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class KeyState:
    key: str
    bucket: TokenBucket
    failures: int = 0
    exhausted_until: float = 0.0
    requests_sent: int = 0
    success_count: int = 0
    json_failures: int = 0
    rate_limited: int = 0


class KeyManager:
    def __init__(self, keys: List[str], rpm_per_key: int):
        if not keys:
            raise ValueError("KeyManager requires at least one API key")
        self.keys: List[KeyState] = [
            KeyState(k, TokenBucket(rpm_per_key)) for k in keys
        ]
        self._lock = asyncio.Lock()
        self._rr_index = 0

    def _is_available(self, ks: KeyState) -> bool:
        return ks.failures < KEY_DEAD_THRESHOLD and time.time() >= ks.exhausted_until

    async def acquire_key(self) -> KeyState:
        async with self._lock:
            n = len(self.keys)
            chosen: Optional[KeyState] = None
            for offset in range(n):
                idx = (self._rr_index + offset) % n
                ks = self.keys[idx]
                if self._is_available(ks):
                    self._rr_index = (idx + 1) % n
                    chosen = ks
                    break
            if chosen is None:
                chosen = min(self.keys, key=lambda k: k.exhausted_until)

        await chosen.bucket.acquire()
        chosen.requests_sent += 1
        return chosen

    def mark_rate_limited(self, ks: KeyState, cooldown: int = KEY_COOLDOWN_SECONDS) -> None:
        ks.exhausted_until = time.time() + cooldown
        ks.failures += 1
        ks.rate_limited += 1
        if ks.failures >= KEY_DEAD_THRESHOLD:
            logger.warning(f"🔴 Key ...{ks.key[-6:]} marked DEAD after {ks.failures} failures")

    def mark_success(self, ks: KeyState) -> None:
        ks.failures = 0
        ks.success_count += 1

    def mark_json_failure(self, ks: KeyState) -> None:
        ks.json_failures += 1

    def mark_bad_key(self, ks: KeyState) -> None:
        ks.failures = KEY_DEAD_THRESHOLD
        ks.exhausted_until = time.time() + 86400
        logger.error(f"⛔ Key ...{ks.key[-6:]} marked BAD (auth failure)")

    def stats(self) -> str:
        parts = []
        for i, ks in enumerate(self.keys, 1):
            if ks.failures >= KEY_DEAD_THRESHOLD:
                state = "DEAD"
            elif time.time() < ks.exhausted_until:
                state = "COOLDOWN"
            else:
                state = "OK"
            parts.append(
                f"#{i}:{state}(req={ks.requests_sent},ok={ks.success_count},"
                f"json_fail={ks.json_failures},429={ks.rate_limited})"
            )
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# GEMINI HTTP CALL
# ═══════════════════════════════════════════════════════════════════

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)


async def _raw_gemini_call(
    session: aiohttp.ClientSession,
    key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: Optional[int] = None,
) -> Tuple[Optional[str], int, Optional[str]]:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": max_output_tokens or MAX_OUTPUT_TOKENS,
            "temperature": temperature,
            "topP": 0.95,
            "topK": 40,
        },
    }
    url = GEMINI_ENDPOINT.format(model=model, key=key)

    try:
        timeout = aiohttp.ClientTimeout(total=240)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            status = resp.status

            if status == 200:
                data = await resp.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    return None, status, "empty_candidates"

                cand = candidates[0]
                finish = cand.get("finishReason", "")

                if finish in ("SAFETY", "RECITATION", "OTHER"):
                    return None, status, f"blocked:{finish}"

                parts = cand.get("content", {}).get("parts", [])
                raw = "".join(p.get("text", "") for p in parts)

                if not raw.strip():
                    return None, status, "empty_text"

                if finish == "MAX_TOKENS":
                    logger.warning("   ⚠ Gemini hit MAX_TOKENS — response truncated")

                return raw, status, None

            try:
                body = (await resp.text())[:300]
            except Exception:
                body = "<unreadable>"
            return None, status, body

    except asyncio.TimeoutError:
        return None, 0, "timeout"
    except aiohttp.ClientConnectorError as e:
        return None, 0, f"connector:{e}"
    except aiohttp.ClientError as e:
        return None, 0, f"client:{e}"
    except Exception as e:
        return None, 0, f"unexpected:{e}"


# ═══════════════════════════════════════════════════════════════════
# CALL GEMINI WITH RETRY + EMPTY-RESULT RETRY
# ═══════════════════════════════════════════════════════════════════

async def call_gemini_async(
    session: aiohttp.ClientSession,
    key_manager: KeyManager,
    model: str,
    prompt: str,
    retry_prompt: Optional[str] = None,
    debug_stage: str = "unknown",
) -> Optional[dict]:
    last_error: Optional[str] = None

    # ── Phase 1: Normal attempts ──
    for attempt in range(MAX_RETRIES):
        ks = await key_manager.acquire_key()
        temp = TEMPERATURE if attempt == 0 else RETRY_TEMPERATURE

        raw, status, err = await _raw_gemini_call(
            session, ks.key, model, prompt, temp
        )

        if status == 429:
            logger.warning(f"   429 on key ...{ks.key[-6:]} — cooling down {KEY_COOLDOWN_SECONDS}s")
            key_manager.mark_rate_limited(ks)
            last_error = "429"
            await asyncio.sleep(min(2 ** attempt, 20))
            continue

        if status in (500, 502, 503, 504):
            logger.warning(f"   HTTP {status} on key ...{ks.key[-6:]}")
            last_error = f"http_{status}"
            await asyncio.sleep(min(2 ** attempt, 15))
            continue

        if status in (401, 403):
            logger.error(f"   HTTP {status} — auth failure on key ...{ks.key[-6:]}")
            key_manager.mark_bad_key(ks)
            last_error = f"http_{status}"
            continue

        if status == 400:
            logger.error(f"   HTTP 400 — bad request: {err}")
            return None

        if status == 413:
            logger.error("   HTTP 413 — payload too large. Reduce MAX_CHARS_PER_BATCH")
            return None

        if status == 0:
            logger.warning(f"   Network/timeout: {err}")
            last_error = err
            await asyncio.sleep(min(2 ** attempt, 15))
            continue

        if status != 200 or raw is None:
            logger.warning(f"   Attempt {attempt + 1}: status={status} err={err}")
            last_error = err or f"status_{status}"
            await asyncio.sleep(2 ** attempt)
            continue

        parsed = smart_json_parse(raw, debug_stage=f"{debug_stage}/attempt{attempt+1}")

        if parsed is None:
            key_manager.mark_json_failure(ks)
            logger.warning(f"   ⚠ JSON unrecoverable (attempt {attempt + 1})")
            last_error = "json_unrecoverable"
            await asyncio.sleep(2 ** attempt)
            continue

        key_manager.mark_success(ks)
        questions = parsed.get("questions") or []

        if questions:
            return parsed

        logger.warning(
            f"   ⚠ 0 questions extracted (attempt {attempt + 1}) "
            f"— content was {len(prompt)} chars"
        )
        last_error = "empty_questions"
        break

    # ── Phase 2: Empty-result retry with stricter prompt ──
    if retry_prompt and EMPTY_RETRY_ENABLED:
        logger.info(f"   🔄 Retrying with stricter prompt (up to {EMPTY_RETRY_MAX} attempts)")

        for retry_i in range(EMPTY_RETRY_MAX):
            ks = await key_manager.acquire_key()
            temp = min(0.7, RETRY_TEMPERATURE + retry_i * 0.1)

            raw, status, err = await _raw_gemini_call(
                session, ks.key, model, retry_prompt, temp
            )

            if status == 429:
                key_manager.mark_rate_limited(ks)
                await asyncio.sleep(min(2 ** retry_i, 10))
                continue

            if status in (401, 403):
                key_manager.mark_bad_key(ks)
                continue

            if status != 200 or raw is None:
                await asyncio.sleep(min(2 ** retry_i, 10))
                continue

            parsed = smart_json_parse(
                raw,
                debug_stage=f"{debug_stage}/empty_retry{retry_i+1}",
            )

            if parsed is None:
                key_manager.mark_json_failure(ks)
                continue

            key_manager.mark_success(ks)
            questions = parsed.get("questions") or []

            if questions:
                logger.info(
                    f"   ✅ Retry succeeded: {len(questions)} questions "
                    f"(retry #{retry_i + 1})"
                )
                return parsed

            logger.warning(f"   ⚠ Retry {retry_i + 1}: still 0 questions")

        _dump_debug_json_failure(
            stage=f"{debug_stage}/empty_after_retries",
            raw=f"PROMPT_CHARS={len(prompt)}",
            extra="Returned 0 questions across all attempts",
        )

    logger.error(f"   ❌ All attempts exhausted. Last error: {last_error}")
    return None


# ═══════════════════════════════════════════════════════════════════
# FETCH + GROUP + DEDUP + BATCH
# ═══════════════════════════════════════════════════════════════════

def fetch_all_chunks(
    subject_filter: Optional[str] = None,
    class_filter: Optional[str] = None,
    chapter_filter: Optional[str] = None,
) -> List[Dict]:
    supabase = get_supabase()
    query = supabase.table("ncert_chunks").select(
        "id, class_grade, subject, chapter, content"
    )
    if subject_filter:
        query = query.ilike("subject", f"%{subject_filter}%")
    if class_filter:
        query = query.eq("class_grade", class_filter)
    if chapter_filter:
        query = query.ilike("chapter", f"%{chapter_filter}%")

    all_rows: List[Dict] = []
    page_size = 1000
    offset = 0
    while True:
        result = query.range(offset, offset + page_size - 1).execute()
        rows = result.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    logger.info(f"Fetched {len(all_rows)} chunks from ncert_chunks")
    return all_rows


def group_chunks(chunks: List[Dict]) -> Dict[Tuple[str, str, str], List[Dict]]:
    groups: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for chunk in chunks:
        key = (
            str(chunk.get("class_grade", "")),
            str(chunk.get("subject", "")),
            str(chunk.get("chapter", "")),
        )
        groups[key].append(chunk)
    logger.info(f"Grouped into {len(groups)} (class, subject, chapter) combos")
    return groups


def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    out: List[Dict] = []
    for ch in chunks:
        content = (ch.get("content") or "")
        fp = hashlib.md5(content[:300].encode("utf-8", errors="ignore")).hexdigest()
        if fp in seen:
            continue
        seen.add(fp)
        out.append(ch)
    return out


def batch_chunks(
    chunks: List[Dict], max_chars: int = MAX_CHARS_PER_BATCH
) -> List[List[Dict]]:
    batches: List[List[Dict]] = []
    current: List[Dict] = []
    current_chars = 0
    for chunk in chunks:
        content = chunk.get("content") or ""
        if len(content) > MAX_CHUNK_CHARS:
            logger.warning(
                f"Chunk id={chunk.get('id')} has {len(content)} chars > "
                f"{MAX_CHUNK_CHARS} — truncating"
            )
            content = content[:MAX_CHUNK_CHARS]
            chunk = {**chunk, "content": content}
        clen = len(content)
        if current_chars + clen > max_chars and current:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += clen
    if current:
        batches.append(current)
    return batches


# ═══════════════════════════════════════════════════════════════════
# DEDUP — question level
# ═══════════════════════════════════════════════════════════════════

_seen_question_hashes: Set[str] = set()


def _question_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode("utf-8", errors="ignore")).hexdigest()


def load_seen_hashes_from_db() -> None:
    try:
        supabase = get_supabase()
        result = supabase.table("ncert_questions").select("question_text").execute()
        count = 0
        for row in (result.data or []):
            txt = (row.get("question_text") or "").strip()
            if txt:
                _seen_question_hashes.add(_question_hash(txt))
                count += 1
        logger.info(f"Preloaded {count} question hashes from DB")
    except Exception as e:
        logger.warning(f"Could not preload hashes (continuing): {e}")


def get_existing_combos() -> Set[Tuple[str, str, str]]:
    try:
        supabase = get_supabase()
        result = (
            supabase.table("ncert_questions")
            .select("class_grade, subject, chapter")
            .execute()
        )
        combos: Set[Tuple[str, str, str]] = set()
        for row in (result.data or []):
            combos.add((
                str(row["class_grade"]),
                row["subject"],
                row["chapter"],
            ))
        return combos
    except Exception as e:
        logger.warning(f"Could not fetch existing combos: {e}")
        return set()


# ═══════════════════════════════════════════════════════════════════
# DB INSERT
# ═══════════════════════════════════════════════════════════════════

async def insert_questions_async(
    questions: List[Dict],
    class_grade: str,
    subject: str,
    chapter: str,
    chunk_ids: List[int],
) -> int:
    if not questions:
        return 0

    supabase = get_supabase()

    rows: List[Dict] = []
    for q in questions:
        if not isinstance(q, dict):
            continue

        q_text = (q.get("question_text") or q.get("text") or "").strip()
        if not q_text or len(q_text) < 5 or len(q_text) > MAX_QUESTION_CHARS:
            continue

        h = _question_hash(q_text)
        if h in _seen_question_hashes:
            continue
        _seen_question_hashes.add(h)

        try:
            marks = int(q.get("marks") or 2)
        except (TypeError, ValueError):
            marks = 2
        marks = max(1, min(marks, 10))

        diff = str(q.get("difficulty") or "medium").lower().strip()
        if diff not in ("easy", "medium", "hard"):
            diff = "medium"

        qtype = str(q.get("question_type") or "exercise").lower().strip()
        if qtype not in ("exercise", "example", "intext", "activity", "hots", "mcq", "fillblank"):
            qtype = "exercise"

        opts = q.get("options")
        if not isinstance(opts, list):
            opts = []

        rows.append({
            "class_grade": str(class_grade),
            "subject": subject,
            "chapter": chapter,
            "section": q.get("section"),
            "question_number": str(q.get("question_number") or "")[:50],
            "question_text": q_text,
            "question_type": qtype,
            "answer": q.get("answer"),
            "options": json.dumps(opts, ensure_ascii=False),
            "marks": marks,
            "difficulty": diff,
            "source_chunk_id": chunk_ids[0] if chunk_ids else None,
        })

    if not rows:
        return 0

    batches = [
        rows[i:i + BATCH_INSERT_SIZE]
        for i in range(0, len(rows), BATCH_INSERT_SIZE)
    ]

    async def do_insert(batch: List[Dict]) -> int:
        try:
            def _sync():
                return supabase.table("ncert_questions").insert(batch).execute()
            await _run_sync(_sync)
            return len(batch)
        except Exception as e:
            logger.warning(f"Bulk insert failed ({len(batch)} rows): {e}")
            inserted = 0
            for row in batch:
                try:
                    def _one(r=row):
                        return supabase.table("ncert_questions").insert(r).execute()
                    await _run_sync(_one)
                    inserted += 1
                except Exception as e2:
                    logger.error(
                        f"Row insert failed: {e2} — "
                        f"{row['question_text'][:80]!r}"
                    )
            return inserted

    results = await asyncio.gather(*[do_insert(b) for b in batches])
    return sum(results)


# ═══════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Stats:
    questions: int = 0
    batches: int = 0
    errors: int = 0
    skipped: int = 0
    empty_batches: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, q: int = 0, b: int = 0, e: int = 0, s: int = 0, eb: int = 0):
        async with self._lock:
            self.questions += q
            self.batches += b
            self.errors += e
            self.skipped += s
            self.empty_batches += eb


# ═══════════════════════════════════════════════════════════════════
# CHAPTER WORKER — THE CRITICAL FIX IS HERE
# ═══════════════════════════════════════════════════════════════════

async def process_chapter(
    session: aiohttp.ClientSession,
    key_manager: KeyManager,
    model: str,
    class_grade: str,
    subject: str,
    chapter: str,
    chapter_chunks: List[Dict],
    global_sem: asyncio.Semaphore,
    stats: Stats,
    dry_run: bool = False,
) -> int:
    chapter_chunks = dedupe_chunks(chapter_chunks)
    batches = batch_chunks(chapter_chunks)

    logger.info(
        f"📖 {subject} Class {class_grade} — {chapter}: "
        f"{len(chapter_chunks)} chunks → {len(batches)} batch(es)"
    )

    async def process_one(batch_idx: int, batch: List[Dict]) -> int:
        async with global_sem:
            combined = "\n\n---\n\n".join(c.get("content") or "" for c in batch)
            chunk_ids = [c["id"] for c in batch if c.get("id") is not None]
            content_text = combined.strip()

            # ═══════════════════════════════════════════════════════
            # ⭐ CRITICAL FIX: Use safe_format (not .format())
            # This prevents "unmatched '{' in format spec" when content
            # contains NCERT math/physics/accountancy braces.
            # ═══════════════════════════════════════════════════════
            prompt = safe_format(
                EXTRACTION_PROMPT,
                class_grade=class_grade,
                subject=subject,
                chapter=chapter,
                content=content_text,
            )
            retry_prompt = safe_format(
                EXTRACTION_PROMPT_RETRY,
                class_grade=class_grade,
                subject=subject,
                chapter=chapter,
                content=content_text,
            )

            if dry_run:
                logger.info(
                    f"   [DRY] {chapter} batch {batch_idx + 1}/{len(batches)}: "
                    f"{len(combined)} chars"
                )
                await stats.add(b=1)
                return 0

            logger.info(
                f"   🤖 {chapter} batch {batch_idx + 1}/{len(batches)}: "
                f"{len(combined)} chars → Gemini"
            )

            result = await call_gemini_async(
                session, key_manager, model,
                prompt, retry_prompt=retry_prompt,
                debug_stage=f"{chapter}/batch{batch_idx+1}",
            )
            await stats.add(b=1)

            if result is None:
                logger.error(f"   ❌ {chapter} batch {batch_idx + 1} FAILED")
                await stats.add(e=1)
                return 0

            questions = result.get("questions") or []
            logger.info(
                f"   ✅ {chapter} batch {batch_idx + 1}: {len(questions)} questions"
            )

            if not questions:
                await stats.add(eb=1)
                return 0

            inserted = await insert_questions_async(
                questions, class_grade, subject, chapter, chunk_ids
            )
            if inserted:
                logger.info(
                    f"   💾 {chapter} batch {batch_idx + 1}: {inserted} inserted"
                )
            await stats.add(q=inserted)
            return inserted

    tasks = [process_one(i, b) for i, b in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total = 0
    for r in results:
        if isinstance(r, int):
            total += r
        else:
            logger.error(f"Batch exception in {chapter}: {r}")
    return total


# ═══════════════════════════════════════════════════════════════════
# PREFLIGHT
# ═══════════════════════════════════════════════════════════════════

async def preflight_checks(model: str) -> bool:
    logger.info("Running pre-flight checks...")

    try:
        supabase = get_supabase()
        supabase.table("ncert_chunks").select("id").limit(1).execute()
        logger.info("   ✓ Supabase reachable")
    except Exception as e:
        logger.error(f"   ✗ Supabase check FAILED: {e}")
        return False

    try:
        supabase.table("ncert_questions").select("id").limit(1).execute()
        logger.info("   ✓ ncert_questions table exists")
    except Exception as e:
        logger.error(f"   ✗ ncert_questions table check FAILED: {e}")
        return False

    try:
        async with aiohttp.ClientSession() as session:
            url = GEMINI_ENDPOINT.format(model=model, key=API_KEYS[0])
            payload = {
                "contents": [{"parts": [{"text": "Reply with: OK"}]}],
                "generationConfig": {"maxOutputTokens": 10},
            }
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    logger.info(f"   ✓ Gemini API reachable (model: {model})")
                elif resp.status in (400, 404):
                    body = (await resp.text())[:200]
                    logger.error(f"   ✗ Gemini model '{model}' not available: {body}")
                    logger.error("   → Try: gemini-3.6-flash, gemini-2.5-pro, gemini-2.0-flash")
                    return False
                elif resp.status in (401, 403):
                    logger.error(f"   ✗ Gemini API key invalid (HTTP {resp.status})")
                    return False
                else:
                    logger.warning(f"   ⚠ Gemini returned HTTP {resp.status}")
    except Exception as e:
        logger.error(f"   ✗ Gemini check FAILED: {e}")
        return False

    logger.info("Pre-flight checks passed ✓")
    return True


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

async def run_async(args) -> None:
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    logger.info("=" * 72)
    logger.info("a4ai NCERT Question Extraction (v5 — GOD-TIER)")
    logger.info("=" * 72)
    logger.info(f"Model:               {args.model}")
    logger.info(f"Filters:             subject={args.subject!r} class={args.class_grade!r} chapter={args.chapter!r}")
    logger.info(f"Dry run:             {args.dry_run}")
    logger.info(f"Resume:              {args.resume}")
    logger.info(f"API keys loaded:     {len(API_KEYS)}")
    logger.info(f"RPM per key:         {args.rpm_per_key}")
    logger.info(f"Concurrent workers:  {args.workers}")
    logger.info(f"Max chars per batch: {MAX_CHARS_PER_BATCH}")
    logger.info(f"Max output tokens:   {MAX_OUTPUT_TOKENS}")
    logger.info(f"Temperature:         {TEMPERATURE} (retry: {RETRY_TEMPERATURE})")
    logger.info("=" * 72)

    if not args.dry_run and not API_KEYS:
        logger.error("No GEMINI_API_KEY* found. Aborting.")
        sys.exit(1)

    if not args.dry_run:
        ok = await preflight_checks(args.model)
        if not ok:
            logger.error("Pre-flight failed. Aborting.")
            sys.exit(1)

    logger.info("Fetching chunks...")
    chunks = await _run_sync(
        fetch_all_chunks, args.subject, args.class_grade, args.chapter
    )
    if not chunks:
        logger.error("No chunks found with given filters!")
        return

    groups = group_chunks(chunks)

    skip_combos: Set[Tuple[str, str, str]] = set()
    if args.resume:
        skip_combos = await _run_sync(get_existing_combos)
        logger.info(f"Resume: {len(skip_combos)} combos already in DB")

    if not args.dry_run:
        await _run_sync(load_seen_hashes_from_db)

    key_manager = KeyManager(API_KEYS, rpm_per_key=args.rpm_per_key)
    global_sem = asyncio.Semaphore(args.workers)
    stats = Stats()

    jobs = []
    for (cg, sub, ch), chapter_chunks in sorted(groups.items()):
        if (str(cg), sub, ch) in skip_combos:
            logger.info(f"⏭️  SKIP: {sub} {cg} — {ch}")
            await stats.add(s=1)
            continue
        jobs.append((cg, sub, ch, chapter_chunks))

    logger.info(
        f"\nProcessing {len(jobs)} chapters with {args.workers} concurrent workers\n"
    )

    t0 = time.time()
    connector = aiohttp.TCPConnector(
        limit=args.workers * 2, limit_per_host=args.workers * 2
    )
    timeout = aiohttp.ClientTimeout(total=300)

    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                process_chapter(
                    session, key_manager, args.model,
                    cg, sub, ch, chunks_list,
                    global_sem, stats, dry_run=args.dry_run,
                )
                for (cg, sub, ch, chunks_list) in jobs
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")

    elapsed = time.time() - t0

    logger.info("\n" + "=" * 72)
    logger.info("✅ EXTRACTION COMPLETE")
    logger.info(f"   Questions extracted:  {stats.questions}")
    logger.info(f"   Batches processed:    {stats.batches}")
    logger.info(f"   Empty batches:        {stats.empty_batches}")
    logger.info(f"   Errors:               {stats.errors}")
    logger.info(f"   Skipped (resume):     {stats.skipped}")
    logger.info(f"   Elapsed:              {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if elapsed > 0 and stats.batches > 0:
        logger.info(f"   Avg batches/minute:   {stats.batches / (elapsed / 60):.1f}")
    if stats.batches > 0:
        logger.info(f"   Avg questions/batch:  {stats.questions / stats.batches:.1f}")
    logger.info(f"   Key state:            {key_manager.stats()}")
    logger.info("=" * 72)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract NCERT questions using Gemini (async, production-ready)"
    )
    parser.add_argument("--subject", help="Filter by subject (e.g., 'Science')")
    parser.add_argument("--class_grade", help="Filter by class (e.g., '10')")
    parser.add_argument("--chapter", help="Filter by chapter (partial match)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts only, no API calls or DB writes")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (class, subject, chapter) combos already in DB")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--rpm-per-key", type=int, default=DEFAULT_RPM_PER_KEY,
                        help=f"Requests per minute per key (default: {DEFAULT_RPM_PER_KEY})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Debug logging")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(run_async(args))    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()