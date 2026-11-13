"""PostgresStore: production-grade agent store backed by Postgres (psycopg + psycopg_pool).

Same interface as SQLiteStore (agent/store.py). Uses the `agent` schema (agent/sql/agent_store_supabase.sql on
Supabase; agent/sql/agent_store_local.sql for dev/test with embedded pgserver). The DDL is NOT applied from here:
migrations are applied externally, the store just connects. If the tables don't exist, ping() fails and /readyz
returns 503.

Wire it up:  AGENT_STORE=postgres AGENT_STORE_URL=postgres://... python -m agent.dev_server
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

import psycopg
from psycopg.rows import dict_row

from .store import (
    PREVIEW_MAX, RETENTION_DAYS, TITLE_MAX, Owner,
    auto_title, iso, normalize_query, valid_id, _enc, _dec,
    SQLiteStore,  # for resolve_expired_approvals
)

try:
    from psycopg_pool import ConnectionPool
except ImportError:
    ConnectionPool = None  # type: ignore[misc,assignment]


_LIKE_ESC = str.maketrans({"%": "\\%", "_": "\\_", "\\": "\\\\"})


def _like(q: str) -> str:
    return "%" + q.translate(_LIKE_ESC) + "%"


class PostgresStore:
    """Postgres-backed agent store. Requires the `agent` schema tables to exist already."""

    def __init__(self, conninfo: str, pool_min: int = 1, pool_max: int = 5,
                 clock: Callable[[], float] = time.time):
        self.clock = clock
        if ConnectionPool is None:
            raise ImportError("psycopg_pool is required for PostgresStore")
        self._pool = ConnectionPool(
            conninfo, min_size=pool_min, max_size=pool_max,
            check=ConnectionPool.check_connection,
            kwargs={"row_factory": dict_row, "autocommit": True},
            configure=self._configure,
            timeout=5.0, max_idle=300, max_lifetime=1800, name="agent_store",
            open=False,
        )
        self._pool.open(wait=False)

    @staticmethod
    def _configure(conn: psycopg.Connection) -> None:
        conn.execute("SET search_path = agent, public")

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection]:
        with self._pool.connection() as c:
            yield c

    @contextmanager
    def _tx(self) -> Iterator[psycopg.Connection]:
        with self._pool.connection() as c:
            with c.transaction():
                yield c

    def close(self) -> None:
        self._pool.close(timeout=2)

    def ping(self) -> bool:
        with self._conn() as c:
            c.execute("SELECT 1 FROM agent.sessions LIMIT 0")
        return True

    # -- sessions ----------------------------------------------------------

    _SESSION_SELECT = f"""
        SELECT s.id, s.title, s.title_source, s.pinned, s.role,
               extract(epoch from s.created_at) AS created_at,
               extract(epoch from s.last_message_at) AS last_message_at,
               (SELECT count(*) FROM messages m WHERE m.session_id = s.id
                AND m.user_id = s.user_id AND m.institute_id = s.institute_id) AS message_count,
               coalesce(
                 (SELECT left(m.text, {PREVIEW_MAX}) FROM messages m WHERE m.session_id = s.id
                  AND m.user_id = s.user_id AND m.institute_id = s.institute_id
                  AND m.role = 'assistant' AND m.text IS NOT NULL ORDER BY m.seq DESC LIMIT 1),
                 (SELECT left(m.text, {PREVIEW_MAX}) FROM messages m WHERE m.session_id = s.id
                  AND m.user_id = s.user_id AND m.institute_id = s.institute_id
                  AND m.role = 'user' ORDER BY m.seq DESC LIMIT 1)
               ) AS preview
        FROM sessions s
    """

    @staticmethod
    def _session_out(r: dict) -> dict:
        return {"id": str(r["id"]), "title": r["title"], "created_at": iso(r["created_at"]),
                "last_message_at": iso(r["last_message_at"]),
                "message_count": int(r["message_count"]),
                "pinned": bool(r["pinned"]), "preview": r["preview"], "role": r["role"],
                "title_source": r["title_source"], "_ts": r["last_message_at"]}

    def create_session(self, o: Owner, sid: str, role: str, title: str) -> dict:
        with self._conn() as c:
            c.execute("INSERT INTO sessions (id, user_id, institute_id, role, title) VALUES (%s,%s,%s,%s,%s)",
                      (sid, o.user_id, o.institute_id, role, title))
        return self.get_session(o, sid)

    def get_session(self, o: Owner, sid: str) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute(self._SESSION_SELECT + " WHERE s.id = %s AND s.user_id = %s AND s.institute_id = %s",
                          (sid, o.user_id, o.institute_id)).fetchone()
        return self._session_out(r) if r else None

    def list_sessions(self, o: Owner, cursor: Optional[str] = None, limit: int = 20,
                      q: Optional[str] = None, pinned: Optional[bool] = None) -> tuple[list[dict], Optional[str]]:
        where = ["s.user_id = %s", "s.institute_id = %s"]
        args: list[Any] = [o.user_id, o.institute_id]
        if pinned is not None:
            where.append("s.pinned = %s")
            args.append(pinned)
        if q and q.strip():
            pat = _like(normalize_query(q))
            where.append("(lower(s.title) LIKE %s ESCAPE '\\' OR EXISTS ("
                         "SELECT 1 FROM messages m WHERE m.session_id = s.id AND m.user_id = s.user_id "
                         "AND m.institute_id = s.institute_id AND m.role = 'user' AND lower(m.text) LIKE %s ESCAPE '\\'))")
            args += [pat, pat]
        if cursor:
            c = _dec(cursor)
            if not (isinstance(c, list) and len(c) == 2):
                raise ValueError("invalid cursor")
            where.append("(s.last_message_at < to_timestamp(%s) OR "
                         "(s.last_message_at = to_timestamp(%s) AND s.id::text < %s))")
            args += [float(c[0]), float(c[0]), str(c[1])]
        sql = (self._SESSION_SELECT + " WHERE " + " AND ".join(where) +
               " ORDER BY s.last_message_at DESC, s.id DESC LIMIT %s")
        args.append(limit + 1)
        with self._conn() as cn:
            rows = cn.execute(sql, args).fetchall()
        items = [self._session_out(r) for r in rows[:limit]]
        nxt = _enc([items[-1]["_ts"], items[-1]["id"]]) if len(rows) > limit else None
        return items, nxt

    def patch_session(self, o: Owner, sid: str, title: Optional[str] = None,
                      pinned: Optional[bool] = None) -> Optional[dict]:
        sets, args = [], []
        if title is not None:
            sets += ["title = %s", "title_source = 'user'"]
            args.append(" ".join(title.split()))
        if pinned is not None:
            sets.append("pinned = %s")
            args.append(pinned)
        if sets:
            with self._conn() as c:
                c.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = %s AND user_id = %s AND institute_id = %s",
                          args + [sid, o.user_id, o.institute_id])
        return self.get_session(o, sid)

    def set_auto_title(self, o: Owner, sid: str, title: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE sessions SET title = %s WHERE id = %s AND user_id = %s AND institute_id = %s "
                      "AND title_source = 'auto'", (title, sid, o.user_id, o.institute_id))

    def delete_session(self, o: Owner, sid: str) -> bool:
        with self._tx() as c:
            n = c.execute("DELETE FROM sessions WHERE id = %s AND user_id = %s AND institute_id = %s",
                          (sid, o.user_id, o.institute_id)).rowcount
            c.execute("DELETE FROM messages WHERE session_id = %s AND user_id = %s AND institute_id = %s",
                      (sid, o.user_id, o.institute_id))
            c.execute("DELETE FROM feedback WHERE session_id = %s AND user_id = %s AND institute_id = %s",
                      (sid, o.user_id, o.institute_id))
        return n > 0

    # -- messages ----------------------------------------------------------

    def add_message(self, o: Owner, sid: str, role: str, text: Optional[str], events: Optional[list] = None,
                    llm: Optional[list] = None) -> Optional[int]:
        now = self.clock()

        def run(c):
            if not c.execute("SELECT 1 FROM sessions WHERE id = %s AND user_id = %s AND institute_id = %s",
                             (sid, o.user_id, o.institute_id)).fetchone():
                return None
            seq = c.execute("SELECT coalesce(max(seq), 0) + 1 FROM messages WHERE session_id = %s",
                            (sid,)).fetchone()["coalesce"]
            ev_json = None if events is None else json.dumps(events, default=str, ensure_ascii=False)
            llm_json = None if llm is None else json.dumps(llm, default=str, ensure_ascii=False)
            c.execute("INSERT INTO messages (session_id, seq, user_id, institute_id, role, text, events, llm, "
                      "created_at) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,to_timestamp(%s))",
                      (sid, seq, o.user_id, o.institute_id, role, text, ev_json, llm_json, now))
            c.execute("UPDATE sessions SET last_message_at = to_timestamp(%s) WHERE id = %s", (now, sid))
            return seq

        with self._tx() as c:
            return run(c)

    @staticmethod
    def _msg_out(r: dict, with_llm: bool = False) -> dict:
        events = r["events"] if r["events"] is not None else None
        # Postgres jsonb is already parsed; resolve expired approvals
        SQLiteStore.resolve_expired_approvals(events)
        d = {"seq": r["seq"], "role": r["role"], "text": r["text"],
             "events": events,
             "created_at": iso(r["created_at"])}
        if with_llm:
            d["llm"] = r["llm"] if r["llm"] is not None else None
        return d

    def list_messages(self, o: Owner, sid: str, cursor: Optional[str] = None,
                      limit: int = 100) -> tuple[list[dict], Optional[str]]:
        after = 0
        if cursor:
            c = _dec(cursor)
            if not isinstance(c, int):
                raise ValueError("invalid cursor")
            after = c
        with self._conn() as cn:
            rows = cn.execute(
                "SELECT seq, role, text, events, llm, extract(epoch from created_at) AS created_at "
                "FROM messages WHERE session_id = %s AND user_id = %s AND institute_id = %s AND seq > %s "
                "ORDER BY seq LIMIT %s",
                (sid, o.user_id, o.institute_id, after, limit + 1)).fetchall()
        items = [self._msg_out(r) for r in rows[:limit]]
        return items, (_enc(items[-1]["seq"]) if len(rows) > limit else None)

    def all_messages(self, o: Owner, sid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT seq, role, text, events, llm, extract(epoch from created_at) AS created_at "
                "FROM messages WHERE session_id = %s AND user_id = %s AND institute_id = %s ORDER BY seq",
                (sid, o.user_id, o.institute_id)).fetchall()
        return [self._msg_out(r, with_llm=True) for r in rows]

    def get_message(self, o: Owner, sid: str, seq: int) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute(
                "SELECT seq, role, text, events, llm, extract(epoch from created_at) AS created_at "
                "FROM messages WHERE session_id = %s AND user_id = %s AND institute_id = %s AND seq = %s",
                (sid, o.user_id, o.institute_id, seq)).fetchone()
        return self._msg_out(r, with_llm=True) if r else None

    def truncate(self, o: Owner, sid: str, from_seq: int) -> int:
        with self._tx() as c:
            n = c.execute("DELETE FROM messages WHERE session_id = %s AND user_id = %s AND institute_id = %s AND seq >= %s",
                          (sid, o.user_id, o.institute_id, from_seq)).rowcount
            c.execute("DELETE FROM feedback WHERE session_id = %s AND user_id = %s AND institute_id = %s "
                      "AND message_seq >= %s", (sid, o.user_id, o.institute_id, from_seq))
        return n

    # -- approval persistence -----------------------------------------------

    def update_approval_in_events(self, o: Owner, sid: str, approval_id: str, status: str,
                                  result_ref: Optional[str] = None) -> bool:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._tx() as c:
            rows = c.execute(
                "SELECT seq, events FROM messages WHERE session_id = %s AND user_id = %s AND institute_id = %s "
                "AND role = 'assistant' AND events IS NOT NULL ORDER BY seq DESC",
                (sid, o.user_id, o.institute_id)).fetchall()
            for r in rows:
                events = r["events"]  # already parsed (jsonb)
                changed = False
                for e in events:
                    if e.get("type") == "pending_approval" and e.get("approval_id") == approval_id:
                        e["status"] = status
                        e["decided_at"] = now
                        if result_ref is not None:
                            e["result_ref"] = result_ref
                        changed = True
                if changed:
                    c.execute("UPDATE messages SET events = %s::jsonb WHERE session_id = %s AND seq = %s "
                              "AND user_id = %s AND institute_id = %s",
                              (json.dumps(events, default=str, ensure_ascii=False), sid, r["seq"],
                               o.user_id, o.institute_id))
                    return True
        return False

    # -- query history -----------------------------------------------------

    def record_query(self, o: Owner, text: str) -> None:
        norm = normalize_query(text)
        if not norm:
            return
        with self._conn() as c:
            c.execute("INSERT INTO queries (id, user_id, institute_id, norm, text, last_used_at, use_count) "
                      "VALUES (%s,%s,%s,%s,%s,to_timestamp(%s),1) ON CONFLICT (user_id, institute_id, norm) DO UPDATE SET "
                      "text = EXCLUDED.text, last_used_at = EXCLUDED.last_used_at, use_count = queries.use_count + 1",
                      (uuid.uuid4().hex, o.user_id, o.institute_id, norm, " ".join(text.split()), self.clock()))

    def list_queries(self, o: Owner, q: Optional[str] = None, limit: int = 10) -> list[dict]:
        where = "user_id = %s AND institute_id = %s"
        args: list = [o.user_id, o.institute_id]
        if q and normalize_query(q):
            where += " AND norm LIKE %s ESCAPE '\\'"
            args.append(_like(normalize_query(q)))
        with self._conn() as c:
            rows = c.execute(f"SELECT id, text, norm, extract(epoch from last_used_at) AS last_used_at, use_count "
                             f"FROM queries WHERE {where} ORDER BY last_used_at DESC, id LIMIT %s",
                             args + [limit]).fetchall()
        return [{"id": str(r["id"]), "text": r["text"], "norm": r["norm"],
                 "last_used_at": iso(r["last_used_at"]), "use_count": int(r["use_count"]),
                 "_ts": r["last_used_at"]} for r in rows]

    def delete_query(self, o: Owner, qid: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM queries WHERE id = %s AND user_id = %s AND institute_id = %s",
                             (qid, o.user_id, o.institute_id)).rowcount > 0

    def clear_queries(self, o: Owner) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM queries WHERE user_id = %s AND institute_id = %s",
                             (o.user_id, o.institute_id)).rowcount

    # -- feedback ----------------------------------------------------------

    def set_feedback(self, o: Owner, sid: str, seq: int, rating: str, note: Optional[str]) -> bool:
        with self._tx() as c:
            ok = c.execute("SELECT 1 FROM messages WHERE session_id = %s AND seq = %s AND user_id = %s "
                           "AND institute_id = %s AND role = 'assistant'",
                           (sid, seq, o.user_id, o.institute_id)).fetchone()
            if not ok:
                return False
            c.execute("INSERT INTO feedback (user_id, institute_id, session_id, message_seq, rating, note, created_at) "
                      "VALUES (%s,%s,%s,%s,%s,%s,to_timestamp(%s)) "
                      "ON CONFLICT (user_id, institute_id, session_id, message_seq) DO UPDATE "
                      "SET rating = EXCLUDED.rating, note = EXCLUDED.note, created_at = EXCLUDED.created_at",
                      (o.user_id, o.institute_id, sid, seq, rating, note, self.clock()))
        return True

    def get_feedback(self, o: Owner, sid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT message_seq, rating, note FROM feedback WHERE session_id = %s AND user_id = %s "
                             "AND institute_id = %s ORDER BY message_seq",
                             (sid, o.user_id, o.institute_id)).fetchall()
        return [dict(r) for r in rows]

    # -- retention ---------------------------------------------------------

    def purge(self, days: int = RETENTION_DAYS) -> dict:
        with self._tx() as c:
            old = [r["id"] for r in c.execute("SELECT id FROM sessions WHERE last_message_at < now() - make_interval(days => %s)",
                                               (days,)).fetchall()]
            for sid in old:
                c.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
                c.execute("DELETE FROM feedback WHERE session_id = %s", (sid,))
            c.execute("DELETE FROM sessions WHERE last_message_at < now() - make_interval(days => %s)", (days,))
            nq = c.execute("DELETE FROM queries WHERE last_used_at < now() - make_interval(days => %s)", (days,)).rowcount
        return {"sessions": len(old), "queries": nq}
