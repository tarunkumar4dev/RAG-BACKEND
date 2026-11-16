"""Deterministic guards that run BEFORE the model sees a spoken message (Phase 2: noise + scope).

approve-by-voice: "haan kar do", "approve kar do", "confirm", "finalize kar do", "assign kar do" … never reach the
agent loop (which could turn them into a finalize/assign approval request). The turn answers with a fixed sentence
and a `ui_action highlight_confirm` so the user taps the on-screen Confirm. Nothing is written.

The patterns are deliberately narrow: a QUESTION that merely contains the word ("is mahine kitne tests finalize
hue?") is not an approval and goes to the model as usual. A refusal is always safe (no write), a false positive only
costs the user one retry, so short imperative utterances lean towards refusing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

APPROVE_BY_VOICE_REPLY = "Voice se approve nahi hota. Screen par Confirm dabaiye."
APPROVE_BY_VOICE_SPEECH = "Voice se approve नहीं होता। Screen पर Confirm दबाइए।"

_VERB = r"(?:approve|confirm|finali[sz]e|final|assign|publish|lock)"
_DO = r"(?:kar\s*do|kar\s*dijiye|kar\s*dena|kardo|karo|karen|karein|kijiye|kar\s*de|do\s+it|go\s+ahead|proceed)"
_YES = r"(?:haan|haa|ha|han|hanji|ok|okay|yes|yeah|theek\s*hai|thik\s*hai|sahi\s*hai|bilkul|zaroor|ji)"
_PATTERNS = [
    re.compile(rf"(?i)\b{_VERB}\s+{_DO}\b"),                 # approve kar do / finalize kar do / assign karo
    re.compile(rf"(?i)\b{_DO}\s+{_VERB}\b"),                 # kar do approve
    re.compile(rf"(?i)^\W*{_YES}[\s,!.]*{_DO}\b"),           # haan kar do / ok go ahead / theek hai karo
    re.compile(rf"(?i)^\W*{_YES}[\s,!.]*{_VERB}\b"),         # haan approve / yes confirm
    re.compile(rf"(?i)\b(?:approve|confirm|finali[sz]e)\s+(?:it|this|the\s+(?:test|paper)|kar|karo|test|paper)\b"),
    re.compile(r"(?i)(?:हाँ|हां|ठीक\s*है)\s*(?:कर\s*दो|करो|कर\s*दीजिए)|अप्रूव\s*कर|कन्फर्म\s*कर|फाइनल\s*कर|असाइन\s*कर"),
]
# a whole utterance that is just the verb (one to three words): "approve", "confirm karo", "finalize"
_SHORT = re.compile(rf"(?i)^\W*(?:{_YES}[\s,]*)?{_VERB}(?:\s+(?:it|karo|kar|please))?\W*$")
_QUESTION = re.compile(r"(?i)\b(kitn[aei]|how\s+many|kaun|which|kya\s+hua|hue|hui|were|was|status)\b|\?\s*$")


@dataclass(frozen=True)
class Canned:
    kind: str          # "approve_by_voice"
    text: str          # shown
    speech: str        # spoken
    action: str        # ui_action.action


def is_approve_by_voice(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _SHORT.match(t):
        return True
    if _QUESTION.search(t):
        return False  # a question about approvals / finalised tests is data, not an instruction
    return any(p.search(t) for p in _PATTERNS)


def guard_turn(text: str) -> Optional[Canned]:
    if is_approve_by_voice(text):
        return Canned("approve_by_voice", APPROVE_BY_VOICE_REPLY, APPROVE_BY_VOICE_SPEECH, "highlight_confirm")
    return None
