"""GET /agent/suggestions: recent queries + scope entities as ready questions + role starters (docs/AGENT_API_V2.md).

Entities come only from the caller's scope (seed: agent/tools/insights.scope_entities, real: schema_adapter) and are
cached per user for ENTITY_TTL_S so a keystroke never hits the database. Nothing here logs the query text.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable, Optional

from .schemas import SessionCtx
from .store import Owner, SQLiteStore, normalize_query

ENTITY_TTL_S = 300
HALF_LIFE_DAYS = 14.0

TEMPLATES = {
    "admin": ["Is mahine sabse kam attendance kis class ki hai?", "Is hafte 75% se kam attendance wale students",
              "Classes ke marks compare karo", "Is mahine kitne tests bane?", "Saari batches dikhao"],
    "hod": ["Mere department mein sabse kam attendance kis class ki hai?", "Is hafte 75% se kam attendance wale students",
            "Classes ke marks compare karo", "Is mahine department mein kitne tests bane?", "Meri batches dikhao"],
    "teacher": ["Meri classes dikhao", "Meri classes mein sabse kam attendance kis class ki hai?",
                "Mere students mein 75% se kam attendance kiska hai?", "Is mahine maine kitne tests banaye?",
                "Class 10 Maths ka 20 marks ka test banao"],
}
_KIND_ORDER = {"batch": 0, "teacher": 1, "subject": 2}


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class EntityCache:
    def __init__(self, ttl_s: float = ENTITY_TTL_S, clock: Callable[[], float] = time.monotonic):
        self.ttl, self.clock = ttl_s, clock
        self._lock = threading.Lock()
        self._d: dict[tuple, tuple[float, list[dict]]] = {}

    def get(self, ctx: SessionCtx, loader: Callable[[SessionCtx], list[dict]]) -> list[dict]:
        k = (str(ctx.user_id), str(ctx.institute_id), ctx.role, ctx.data_source)
        now = self.clock()
        with self._lock:
            hit = self._d.get(k)
            if hit and now - hit[0] < self.ttl:
                return hit[1]
        rows = loader(ctx)
        with self._lock:
            self._d[k] = (now, rows)
        return rows

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


ENTITIES = EntityCache()


def default_loader(ctx: SessionCtx) -> list[dict]:
    if ctx.data_source == "live":
        from .real.schema_adapter import scope_entities
    else:
        from .tools.insights import scope_entities
    return scope_entities(ctx)


def _entity_text(e: dict) -> tuple[str, str]:
    if e["kind"] == "batch":
        n = int(e.get("student_count") or 0)
        return f"{e['name']} ki is mahine ki attendance", f"Batch · {n} student" + ("" if n == 1 else "s")
    if e["kind"] == "teacher":
        return f"{e['name']} ki is hafte ki attendance", "Teacher"
    return f"{e['name']} ke tests is mahine", "Subject"


def recent(store: SQLiteStore, owner: Owner, q: str, now: float) -> list[dict]:
    nq = normalize_query(q)
    rows = store.list_queries(owner, None, limit=500)

    def score(r):
        age_days = max(now - r["_ts"], 0) / 86400
        return r["use_count"] * 0.5 ** (age_days / HALF_LIFE_DAYS)

    groups: list[list[dict]] = [[], []]
    for r in rows:
        if not nq or r["norm"].startswith(nq):
            groups[0].append(r)
        elif nq in r["norm"]:
            groups[1].append(r)
    out = []
    for g in groups:
        g.sort(key=lambda r: (-score(r), -r["_ts"], r["id"]))
        out += [{"type": "recent", "text": r["text"],
                 "hint": "Recent" if r["use_count"] < 2 else f"Recent · {r['use_count']}×"} for r in g]
    return out


def entities(ctx: SessionCtx, q: str, loader: Optional[Callable] = None, cache: EntityCache = ENTITIES) -> list[dict]:
    kq = _key(q)
    if not kq:
        return []
    ents = cache.get(ctx, loader or default_loader)
    hits = []
    for e in ents:
        kn = _key(e["name"])
        if not kn:
            continue
        if kn.startswith(kq):
            rank = 0
        elif kq in kn or (len(kn) >= 2 and kq.startswith(kn)):  # typed part of the name, or the name + more
            rank = 1
        else:
            continue
        hits.append((rank, _KIND_ORDER.get(e["kind"], 9), e["name"].lower(), e))
    hits.sort(key=lambda h: h[:3])
    out = []
    for *_, e in hits:
        text, hint = _entity_text(e)
        out.append({"type": "entity", "text": text, "hint": hint})
    return out


def templates(ctx: SessionCtx, q: str) -> list[dict]:
    nq = normalize_query(q)
    return [{"type": "template", "text": t, "hint": "Starter"}
            for t in TEMPLATES.get(ctx.role, TEMPLATES["teacher"]) if not nq or nq in normalize_query(t)]


def suggest(store: SQLiteStore, ctx: SessionCtx, q: Optional[str], limit: int = 8, now: Optional[float] = None,
            loader: Optional[Callable] = None, cache: EntityCache = ENTITIES) -> list[dict]:
    q = q or ""
    now = store.clock() if now is None else now
    owner = Owner.of(ctx)
    rec = recent(store, owner, q, now)
    if not normalize_query(q):
        cands = rec[:4] + templates(ctx, q)
    else:
        cands = rec + entities(ctx, q, loader, cache) + templates(ctx, q)
    seen, out = set(), []
    for c in cands:
        k = normalize_query(c["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= limit:
            break
    return out
