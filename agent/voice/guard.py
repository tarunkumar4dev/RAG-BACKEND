"""Deterministic guards that run BEFORE the model sees a spoken message.

approve-by-voice: "haan kar do", "approve kar do", "confirm", "finalize kar do", "assign kar do" … never reach the
agent loop (which could turn them into a finalize/assign approval request). The turn answers with a fixed sentence
and a `ui_action highlight_confirm` so the user taps the on-screen Confirm. Nothing is written. Phase 2 fills the
patterns in; the hook exists from Phase 1 so the voice endpoint has one shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

APPROVE_BY_VOICE_REPLY = "Voice se approve nahi hota. Screen par Confirm dabaiye."
APPROVE_BY_VOICE_SPEECH = "Voice se approve नहीं होता। Screen पर Confirm दबाइए।"

_APPROVE = re.compile(
    r"(?i)\b("
    r"approve|approved|confirm|finali[sz]e|finalise|assign\s+kar|assign\s+it|"
    r"(?:haan|ha|haa|han|ok|okay|theek\s*hai|thik\s*hai|yes)[\s,]*(?:kar\s*do|kar\s*dijiye|karo|karen|karein|do\s+it|go\s+ahead|proceed)|"
    r"(?:kar|karo|kar\s*do)\s+(?:final|approve|assign|confirm)|"
    r"(?:final|approve|assign|confirm)\s+(?:kar|karo|kar\s*do|kar\s*dijiye|karna)|"
    r"हाँ\s*कर\s*दो|अप्रूव|कन्फर्म|फाइनल\s*कर"
    r")\b")


@dataclass(frozen=True)
class Canned:
    kind: str          # "approve_by_voice"
    text: str          # shown
    speech: str        # spoken
    action: str        # ui_action.action


def is_approve_by_voice(text: str) -> bool:
    return bool(_APPROVE.search(text or ""))


def guard_turn(text: str) -> Optional[Canned]:
    if is_approve_by_voice(text):
        return Canned("approve_by_voice", APPROVE_BY_VOICE_REPLY, APPROVE_BY_VOICE_SPEECH, "highlight_confirm")
    return None
