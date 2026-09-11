# app/api/v1/endpoints/chat.py
"""
Chat endpoint — Groq LLM proxy (Security Hardened)

FIXES:
  - Added extra="forbid" on all Pydantic models
  - Sanitized message content (length limit, role validation)
  - Removed str(e) from error responses
  - Added message count limit (prevent context stuffing)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import List
import os
import re
import httpx

router = APIRouter()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODELS_PREFERRED = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

SKIP_PATTERNS = (
    "whisper", "tts", "guard", "orpheus", "canopy",
    "distil", "playai", "vision", "audio",
)

# ── Security limits ─────────────────────────────────────────
MAX_MESSAGES = 20           # Max conversation history
MAX_MESSAGE_LENGTH = 4000   # Per message
VALID_ROLES = {"user", "assistant", "system"}


def clean_response(text: str) -> str:
    """Strip thinking blocks, reasoning, and XML tags from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"(?:Here'?s\s+(?:a\s+|my\s+)?thinking\s+process:?|Thinking\s+Process:?).*?(?=\n\n[A-Z]|\n\n[a-z]|\Z)",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"^\s*\d+\.\s+\*?\*?(?:Analyze|Check|Action|Constraint|Refine|Draft|Verify|Output|Self-Correct|Final).*?$",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(r"^.*?(?:Let'?s re-?read|Wait,|Actually,|I(?:'ll| will) (?:stick|go) with).*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def get_available_models(api_key: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            data = res.json()
            if "data" not in data:
                return GROQ_MODELS_PREFERRED
            available_ids = {m["id"] for m in data["data"]}
            safe = [
                m for m in GROQ_MODELS_PREFERRED
                if m in available_ids
                and not any(p in m.lower() for p in SKIP_PATTERNS)
            ]
            if not safe:
                safe = [
                    m["id"] for m in data["data"]
                    if any(ok in m["id"].lower() for ok in ("llama", "qwen", "deepseek", "mistral"))
                    and not any(p in m["id"].lower() for p in SKIP_PATTERNS)
                ]
            return safe or GROQ_MODELS_PREFERRED
    except Exception:
        return GROQ_MODELS_PREFERRED


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str = Field(max_length=MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[ChatMessage] = Field(max_length=MAX_MESSAGES)


class ChatResponse(BaseModel):
    content: str
    model: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Chat service unavailable")

    # SECURITY: Validate roles
    for msg in request.messages:
        if msg.role not in VALID_ROLES:
            raise HTTPException(status_code=422, detail=f"Invalid message role: {msg.role}")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n\nCRITICAL: Never show your thinking, reasoning, analysis, or decision process. Output ONLY the final answer. No 'Here is my thinking' or numbered reasoning steps."

    models = await get_available_models(api_key)

    SKIP_ERRORS = (
        "does not exist", "not have access", "decommissioned",
        "no longer supported", "deprecated", "requires terms",
        "terms acceptance", "not_active", "model_not",
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in models:
            try:
                res = await client.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 1024,
                    },
                )
                data = res.json()

                if "error" in data:
                    err = data["error"].get("message", "") + data["error"].get("code", "")
                    if any(s in err for s in SKIP_ERRORS):
                        continue
                    raise HTTPException(status_code=400, detail="Chat request failed. Try again.")

                if data.get("choices"):
                    raw = data["choices"][0]["message"]["content"]
                    cleaned = clean_response(raw)
                    return ChatResponse(
                        content=cleaned if cleaned else raw,
                        model=model,
                    )

            except httpx.TimeoutException:
                continue
            except HTTPException:
                raise
            except Exception:
                continue

    raise HTTPException(
        status_code=503,
        detail="Chat service temporarily unavailable. Please try again."
    )