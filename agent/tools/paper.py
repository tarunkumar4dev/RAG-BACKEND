"""Test-paper tools. Draft state lives server-side (agent_drafts); the model only passes ids/enums."""
from __future__ import annotations

import json
import uuid
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .. import config
from ..llm import LLMProvider, complete_json
from ..verify import AR_OPTIONS, verify_paper

Subject = Literal["Mathematics", "Science", "Physics", "Chemistry", "Biology"]


class _A(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlueprintArgs(_A):
    class_grade: int = Field(ge=6, le=12)
    subject: Subject
    chapter: str = Field(max_length=60, pattern=r"^[A-Za-z0-9 ,&\-()']+$")
    total_marks: Literal[20, 25, 30, 40, 50, 80]
    assertion_reason_count: int = Field(default=2, ge=0, le=4)
    case_study_count: int = Field(default=1, ge=0, le=3)


class DraftArgs(_A):
    draft_id: UUID


class RegenArgs(_A):
    draft_id: UUID
    question_ids: list[str] = Field(max_length=40, json_schema_extra={"items": {"pattern": r"^q\d{1,3}$"}})


class AssignArgs(_A):
    draft_id: UUID
    section_id: UUID


def blueprint_defaults(ctx, a: BlueprintArgs) -> dict:
    """CBSE-style split. Sum is enforced exactly to total_marks."""
    ar, cs = a.assertion_reason_count, a.case_study_count
    if a.total_marks == 40:
        secs = [("A", "mcq", 8 - ar, 1), ("A", "assertion_reason", ar, 1), ("B", "very_short", 4, 2),
                ("C", "short_answer", 3, 3), ("D", "long_answer", 1, 5), ("E", "case_study", cs, 4)]
        # 8 + 8 + 9 + 5 + 4cs = 30 + 4cs -> pad long answers to hit total
        rest = 40 - (8 + 8 + 9 + 4 * cs)
        secs[4] = ("D", "long_answer", rest // 5, 5)
        left = rest % 5
        if left:
            secs[3] = ("C", "short_answer", 3 + left // 3, 3)
    else:
        n_case = cs * 4
        mcq = max(a.total_marks // 5, ar + 1)
        rem = a.total_marks - mcq - n_case
        secs = [("A", "mcq", mcq - ar, 1), ("A", "assertion_reason", ar, 1), ("B", "very_short", rem // 4, 2)]
        rem -= (rem // 4) * 2
        secs += [("C", "short_answer", rem // 3, 3)]
        rem -= (rem // 3) * 3
        if rem:
            secs.append(("B", "very_short", 0, 2)) if False else secs.__setitem__(0, ("A", "mcq", mcq - ar + rem, 1))
        secs.append(("E", "case_study", cs, 4))
    diff = a.total_marks - sum(c * m for _, _, c, m in secs)
    secs[0] = (secs[0][0], secs[0][1], secs[0][2] + diff, 1)  # 1-mark MCQs absorb the exact remainder
    sections = [{"section": s, "qtype": t, "count": c, "marks_each": m} for s, t, c, m in secs if c > 0]
    total = sum(s["count"] * s["marks_each"] for s in sections)
    return {"status": "ok", "class_grade": a.class_grade, "subject": a.subject, "chapters": [a.chapter],
            "total_marks": float(a.total_marks), "duration_min": 90 if a.total_marks <= 40 else 180,
            "sections": sections, "sum_ok": total == a.total_marks}


def ncert_retrieve(class_grade: int, subject: str, chapter: str) -> dict:
    """Reuse existing RAG (rag_service.retrieve_context). Needs Supabase env; offline -> honest fallback."""
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test-generator-backend"))
        from app.services.rag_service import retrieve_context  # type: ignore
        chunks = retrieve_context([chapter], [], subject, str(class_grade), max_chunks=12)
        return {"source": "rag_service.retrieve_context", "chunks": [str(c)[:600] for c in chunks][:12]}
    except Exception as e:
        return {"source": "offline_outline", "error": type(e).__name__, "chunks": []}


GEN_SYSTEM = (
    "You write CBSE board-style questions strictly from the NCERT syllabus. Output ONLY JSON. "
    "Use $...$ LaTeX for math. Every numeric/MCQ maths question MUST include check_expr: a sympy-parseable "
    "expression (use sin(30*deg), sqrt(3), pi) whose value is the correct answer. MCQ options: 4 distinct, "
    "exactly one correct. Assertion-Reason: text must contain 'Assertion (A):' and 'Reason (R):' and the "
    "options MUST be exactly: " + json.dumps(AR_OPTIONS) + " (capitalise A/R normally). "
    "Rubric questions (short_answer, long_answer) need marking_scheme: [{step, marks}] summing to the "
    "question marks. case_study: a passage in text + sub_questions [{text, marks, answer}] summing to marks."
)


class QuestionGenerator:
    """Adapter. LLM path (default, emits check_expr for sympy); existing Gemini service path is
    app.services.test_generator_service.generate_questions (needs Supabase + TestGenerationRequest)."""

    def __init__(self, provider: LLMProvider):
        self.p = provider
        self.usage = [0, 0]

    def generate(self, bp: dict, slots: list[dict], context: list[str], avoid: list[str]) -> list[dict]:
        prompt = json.dumps({
            "class": bp["class_grade"], "subject": bp["subject"], "chapter": bp["chapters"][0],
            "ncert_context": context[:6], "avoid_duplicates_of": avoid[:40],
            "slots": slots,
            "output": {"questions": [{"id": "<slot id>", "qtype": "...", "marks": 0, "text": "...",
                                      "options": ["..."], "answer": "letter or value", "check_expr": "...",
                                      "solution": "...", "marking_scheme": [{"step": "...", "marks": 0}],
                                      "sub_questions": [{"text": "...", "marks": 0, "answer": "..."}]}]},
        })
        data, r = complete_json(self.p, GEN_SYSTEM, "Generate one question per slot. " + prompt, max_tokens=12000)
        self.usage[0] += r.tokens_in
        self.usage[1] += r.tokens_out
        qs = data["questions"] if isinstance(data, dict) else data
        by_id = {s["id"]: s for s in slots}
        out = []
        for q in qs:
            s = by_id.get(q.get("id"))
            if not s:
                continue
            q.update({"qtype": s["qtype"], "marks": s["marks"], "section": s["section"],
                      "chapter": bp["chapters"][0]})
            out.append(q)
        return out


SOLVER_SYSTEM = ("You are an independent examiner. Solve each question yourself. No answer key is given. "
                 "Return ONLY JSON {\"answers\": {\"<id>\": \"<letter A-D for MCQ/assertion-reason, else final "
                 "value in sympy syntax e.g. sqrt(3)/2>\"}}.")


def make_solver(provider: LLMProvider, usage: list):
    def solve(qs: list[dict]) -> dict:
        data, r = complete_json(provider, SOLVER_SYSTEM, json.dumps({"questions": qs}), max_tokens=6000)
        usage[0] += r.tokens_in
        usage[1] += r.tokens_out
        return data.get("answers", {})
    return solve


def slots_from_blueprint(bp: dict) -> list[dict]:
    slots, i = [], 1
    for s in bp["sections"]:
        for _ in range(s["count"]):
            slots.append({"id": f"q{i}", "section": s["section"], "qtype": s["qtype"], "marks": s["marks_each"]})
            i += 1
    return slots


def build_and_verify(provider: LLMProvider, bp: dict, on_status=None) -> dict:
    """generate (batch) -> verify -> regenerate only failed ids, max 3 rounds."""
    gen = QuestionGenerator(provider)
    rag = ncert_retrieve(bp["class_grade"], bp["subject"], bp["chapters"][0])
    slots = slots_from_blueprint(bp)
    if on_status: on_status("status.generating", {"n": len(slots)})
    qs = {q["id"]: q for q in gen.generate(bp, slots, rag["chunks"], [])}
    missing = [s for s in slots if s["id"] not in qs]
    if missing:
        qs.update({q["id"]: q for q in gen.generate(bp, missing, rag["chunks"], [])})
    paper = {"total_marks": bp["total_marks"], "questions": [qs[s["id"]] for s in slots if s["id"] in qs]}
    solver = make_solver(provider, gen.usage)
    rounds, summary = 0, None
    while True:
        if on_status: on_status("status.verifying", {"round": rounds + 1})
        summary = verify_paper(paper, solver, bp["chapters"])
        rounds += 1
        bad = summary["failed"] + summary["duplicates"]
        if not bad or rounds >= config.MAX_VERIFY_ROUNDS:
            break
        if on_status: on_status("status.regenerating", {"ids": bad})
        fix_slots = [s for s in slots if s["id"] in set(bad)]
        new = {q["id"]: q for q in gen.generate(bp, fix_slots, rag["chunks"],
                                                [q["text"] for q in paper["questions"] if q["id"] not in bad])}
        paper["questions"] = [new.get(q["id"], q) for q in paper["questions"]]
    summary["rounds"] = rounds
    return {"paper": paper, "summary": summary, "rag_source": rag["source"], "usage": gen.usage}
