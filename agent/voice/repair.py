"""Entity repair: a transcript's misheard names are matched against the caller's OWN teachers / sections / students.

1. Candidate spans in the transcript (1–2 word windows, honorifics stripped) are compared with the names in the
   caller's scope (same `scope_entities` the suggestions use; cached per (institute, user) for 5 minutes).
2. A single clear winner is confirmed with the existing `resolve_entity` tool (scope-checked, read-only) and the
   transcript span is replaced by the canonical spelling.
3. Several close names -> `clarify` chips (the same shape as the agent's clarify event); nothing is auto-picked.
Nothing here logs transcript text. The list of names is the scope the caller can already see.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..schemas import SessionCtx
from ..suggest import EntityCache
from ..text import normalize_section, strip_honorifics
from . import config as vc

CACHE = EntityCache(ttl_s=vc.ENTITY_TTL_S)
MIN_SCORE = 0.84       # difflib ratio of the normalised span vs the normalised name
AMBIGUOUS_GAP = 0.05   # two names this close -> ask
_STOP = {"ki", "ka", "ke", "ko", "ne", "se", "me", "mein", "kya", "hai", "hain", "aur", "the", "of", "in", "for",
         "attendance", "marks", "class", "section", "batch", "student", "students", "teacher", "test", "tests",
         "aaj", "kal", "is", "mahine", "hafte", "month", "week", "today", "batao", "dikhao", "show", "open",
         "kholo", "karo", "do", "please", "what", "how", "many", "kitne", "kitni", "kaun", "sabse", "kam", "zyada"}


@dataclass
class Repair:
    text: str
    changed: bool = False
    entity: Optional[dict] = None          # {kind, name, id} when a unique name was confirmed
    clarify: Optional[dict] = None         # {question, chips:[{label, value, hint}]} when ambiguous
    db_calls: int = 0
    candidates: list[str] = field(default_factory=list)  # kinds considered (trace only)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


MAX_STUDENTS = 2000


def student_entities(ctx: SessionCtx) -> list[dict]:
    """Students the caller may see (same visibility rule as resolve_entity), names + ids only."""
    if ctx.data_source == "live":
        from ..real.schema_adapter import STUDENT_VISIBLE, connect_real, load_scope
        with connect_real() as c:
            sc = load_scope(c, ctx)
            rows = c.execute(f"select st.id, st.name from students st where {STUDENT_VISIBLE} order by st.name, st.id "
                             f"limit {MAX_STUDENTS}", sc.p()).fetchall()
    else:
        from ..db import connect
        from ..tools.insights import _SCOPE_BATCHES, _params
        with connect(readonly=True) as c:
            rows = c.execute(f"with {_SCOPE_BATCHES} select s.id, s.name from agent_demo.students s "
                             "join scope_batches sb on sb.id = s.batch_id where s.institute_id = %(inst)s and s.is_active "
                             f"order by s.name, s.id limit {MAX_STUDENTS}", _params(ctx)).fetchall()
    return [{"kind": "student", "id": str(r["id"]), "name": r["name"]} for r in rows if r["name"]]


def voice_loader(ctx: SessionCtx) -> list[dict]:
    """Teachers + batches (+ subjects) from the suggestions loader, plus the caller's students."""
    from ..suggest import default_loader
    return default_loader(ctx) + student_entities(ctx)


def _entities(ctx: SessionCtx, loader: Optional[Callable] = None) -> list[dict]:
    try:
        return CACHE.get(ctx, loader or voice_loader)
    except Exception:  # noqa: BLE001  DB down: no repair, the agent loop reports db_unavailable itself
        return []


def _spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, span) over 1- and 2-word windows of the raw transcript, skipping function words."""
    out = []
    for m in re.finditer(r"[A-Za-z][A-Za-z'’.\-]*", text):
        out.append((m.start(), m.end(), m.group()))
    spans: list[tuple[int, int, str]] = []
    for i, (a, b, w) in enumerate(out):
        if w.lower() in _STOP or len(w) < 3:
            continue
        spans.append((a, b, w))
        if i + 1 < len(out):
            a2, b2, w2 = out[i + 1]
            if w2.lower() not in _STOP and b2 - a <= 40:
                spans.append((a, b2, text[a:b2]))
    return spans


def repair(ctx: SessionCtx, text: str, loader: Optional[Callable] = None, resolve: Optional[Callable] = None) -> Repair:
    ents = _entities(ctx, loader)
    if not ents or not text.strip():
        return Repair(text=text)
    names = []
    for e in ents:
        if e.get("kind") not in ("teacher", "batch", "student"):
            continue
        key = _norm(normalize_section(e["name"]) if e["kind"] == "batch" else strip_honorifics(e["name"]))
        if len(key) >= 3:
            names.append((key, e))
    best: list[tuple[float, tuple[int, int, str], dict]] = []
    for a, b, span in _spans(text):
        sk = _norm(normalize_section(span) if re.fullmatch(r"(?i)(class\s*)?\d{1,2}\s*[A-Za-z]?", span) else strip_honorifics(span))
        if len(sk) < 3:
            continue
        for key, e in names:
            if sk == key:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, sk, key).ratio()
            if score >= MIN_SCORE:
                best.append((score, (a, b, span), e))
    if not best:
        return Repair(text=text)
    best.sort(key=lambda x: (-x[0], x[2]["name"]))
    top = best[0]
    close = [x for x in best if top[0] - x[0] <= AMBIGUOUS_GAP and x[2]["id"] != top[2]["id"]
             and _norm(x[2]["name"]) != _norm(top[2]["name"])]
    kind = {"batch": "section"}.get(top[2]["kind"], top[2]["kind"])
    if close:
        seen, chips = set(), []
        for _, _, e in [top] + close:
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            hint = {"teacher": "Teacher", "batch": "Section", "student": "Student"}.get(e["kind"])
            chips.append({"label": e["name"], "value": _replace(text, top[1], e["name"]), "hint": hint})
        return Repair(text=text, clarify={"question": "Kaun sa? / Which one?", "chips": chips[:5]},
                      candidates=[kind])
    # exact spelling already in the transcript: nothing to change, no DB call
    if _norm(top[1][2]) == _norm(top[2]["name"]) or top[0] >= 0.999:
        return Repair(text=text, entity={"kind": kind, "name": top[2]["name"], "id": top[2]["id"]}, candidates=[kind])
    # confirm through resolve_entity (scope-checked) before rewriting the user's words
    res = _resolve(ctx, kind, top[2]["name"], resolve)
    if res.get("status") == "ok":
        return Repair(text=_replace(text, top[1], top[2]["name"]), changed=True, db_calls=1,
                      entity={"kind": kind, "name": top[2]["name"], "id": res["matches"][0]["id"]}, candidates=[kind])
    if res.get("status") == "ambiguous":
        chips = [{"label": m["label"], "value": _replace(text, top[1], m["label"]), "hint": m.get("hint")}
                 for m in res["matches"][:5]]
        return Repair(text=text, clarify={"question": "Kaun sa? / Which one?", "chips": chips}, db_calls=1,
                      candidates=[kind])
    return Repair(text=text, db_calls=1, candidates=[kind])


def _replace(text: str, span: tuple[int, int, str], name: str) -> str:
    a, b, _ = span
    return text[:a] + name + text[b:]


def _resolve(ctx: SessionCtx, kind: str, name: str, resolve: Optional[Callable]) -> dict:
    try:
        if resolve is not None:
            return resolve(ctx, kind, name)
        from ..tools.insights import ResolveEntityArgs
        from ..tools.registry import active_tools
        fn = active_tools(ctx.data_source == "live")["resolve_entity"].fn
        return fn(ctx, ResolveEntityArgs(kind=kind, name=re.sub(r"[^\w\s.'\-]", "", name)[:80] or name[:80]))
    except Exception:  # noqa: BLE001
        return {"status": "error"}


def keyterms(ctx: SessionCtx, loader: Optional[Callable] = None, limit: int = 50) -> list[str]:
    """Names from the caller's scope to bias saaras:v4 recognition (ids never leave the process)."""
    out = []
    for e in _entities(ctx, loader):
        n = (e.get("name") or "").strip()
        if n and len(n) <= 64 and n not in out:
            out.append(n)
        if len(out) >= limit:
            break
    return out
