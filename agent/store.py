"""Agent store: the agent's OWN writable state (sessions, messages, query history, feedback). Never the a4ai tables.

- SQLiteStore(path): dev + real mode, file agent/.devdata/agent_store.db (gitignored), WAL mode.
- MemoryStore(): the same SQL on SQLite ':memory:' (tests). One code path, so both behave identically.
- Every row carries user_id + institute_id and every read/write filters by both (Owner). A row of another user or
  institute is indistinguishable from a missing row (callers turn None/False into 404).
- Retention: purge() deletes sessions (+ messages, feedback) and queries not used for RETENTION_DAYS.
- Privacy: nothing here logs. Message / query text never leaves this module except to its owner.
- Future production deploy: agent/sql/agent_store_supabase.sql (same schema, `agent` schema, RLS).

Paper drafts in real mode are previews only: they stay in process memory (DRAFTS), never on disk.
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

RETENTION_DAYS = 90
DEFAULT_PATH = Path(__file__).resolve().parent / ".devdata" / "agent_store.db"
TITLE_MAX = 48
PREVIEW_MAX = 120

SCHEMA = """
create table if not exists sessions (
  id text primary key,
  user_id text not null,
  institute_id text not null,
  role text not null,
  title text not null,
  title_source text not null default 'auto',      -- auto | user (PATCH): auto titles never overwrite user titles
  pinned integer not null default 0,
  created_at real not null,
  last_message_at real not null
);
create index if not exists sessions_owner on sessions (user_id, institute_id, last_message_at desc, id desc);
create table if not exists messages (
  session_id text not null,
  seq integer not null,
  user_id text not null,
  institute_id text not null,
  role text not null check (role in ('user', 'assistant')),
  text text,
  events text,                                      -- JSON list of AgentEvent (assistant)
  llm text,                                         -- JSON list of provider-neutral messages of the turn (assistant)
  created_at real not null,
  primary key (session_id, seq)
);
create index if not exists messages_owner on messages (user_id, institute_id, session_id, seq);
create table if not exists queries (
  id text primary key,
  user_id text not null,
  institute_id text not null,
  norm text not null,
  text text not null,
  last_used_at real not null,
  use_count integer not null default 1,
  unique (user_id, institute_id, norm)
);
create table if not exists feedback (
  user_id text not null,
  institute_id text not null,
  session_id text not null,
  message_seq integer not null,
  rating text not null check (rating in ('up', 'down')),
  note text,
  created_at real not null,
  primary key (user_id, institute_id, session_id, message_seq)
);
"""


@dataclass(frozen=True)
class Owner:
    user_id: str
    institute_id: str

    @classmethod
    def of(cls, ctx) -> "Owner":
        return cls(str(ctx.user_id), str(ctx.institute_id))


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def normalize_query(text: str) -> str:
    """lowercase, trimmed, inner whitespace collapsed"""
    return " ".join((text or "").lower().split())


def auto_title(text: str, limit: int = TITLE_MAX) -> str:
    """Deterministic session title from the first user message: collapse whitespace; if longer than `limit`, cut at
    the last word boundary in the second half and append '…' (result is at most `limit` chars). No LLM call."""
    t = " ".join((text or "").split())
    if not t:
        return "New chat"
    if len(t) <= limit:
        return t
    cut = t[: limit - 1]
    sp = cut.rfind(" ")
    if sp >= limit // 2:
        cut = cut[:sp]
    return cut.rstrip(" ,.;:-–—?!") + "…"


def _like(q: str) -> str:
    return "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _enc(obj: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode().rstrip("=")


def _dec(s: str) -> Any:
    try:
        return json.loads(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)))
    except Exception as e:  # noqa: BLE001
        raise ValueError("invalid cursor") from e


class SQLiteStore:
    def __init__(self, path: str | os.PathLike = DEFAULT_PATH, clock: Callable[[], float] = time.time):
        self.path = str(path)
        self.clock = clock
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._c.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._c.execute("pragma journal_mode=wal")
            self._c.execute("pragma busy_timeout=5000")
            self._c.executescript(SCHEMA)

    # -- plumbing ----------------------------------------------------------
    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._c.execute(sql, args).fetchall()

    def _tx(self, fn):
        with self._lock:
            self._c.execute("begin immediate")
            try:
                out = fn(self._c)
                self._c.execute("commit")
                return out
            except BaseException:
                self._c.execute("rollback")
                raise

    def close(self) -> None:
        with self._lock:
            self._c.close()

    # -- sessions ----------------------------------------------------------
    _SESSION_COLS = (
        "s.id, s.title, s.title_source, s.pinned, s.role, s.created_at, s.last_message_at, "
        "(select count(*) from messages m where m.session_id = s.id and m.user_id = s.user_id "
        " and m.institute_id = s.institute_id) as message_count, "
        "coalesce((select substr(m.text, 1, %d) from messages m where m.session_id = s.id and m.user_id = s.user_id "
        " and m.institute_id = s.institute_id and m.role = 'assistant' and m.text is not null order by m.seq desc limit 1),"
        " (select substr(m.text, 1, %d) from messages m where m.session_id = s.id and m.user_id = s.user_id "
        " and m.institute_id = s.institute_id and m.role = 'user' order by m.seq desc limit 1)) as preview"
        % (PREVIEW_MAX, PREVIEW_MAX))

    @staticmethod
    def _session_out(r: sqlite3.Row) -> dict:
        return {"id": r["id"], "title": r["title"], "created_at": iso(r["created_at"]),
                "last_message_at": iso(r["last_message_at"]), "message_count": int(r["message_count"]),
                "pinned": bool(r["pinned"]), "preview": r["preview"], "role": r["role"],
                "title_source": r["title_source"], "_ts": r["last_message_at"]}

    def create_session(self, o: Owner, sid: str, role: str, title: str) -> dict:
        now = self.clock()
        self._q("insert into sessions (id, user_id, institute_id, role, title, created_at, last_message_at) "
                "values (?,?,?,?,?,?,?)", (sid, o.user_id, o.institute_id, role, title, now, now))
        return self.get_session(o, sid)

    def get_session(self, o: Owner, sid: str) -> Optional[dict]:
        rows = self._q(f"select {self._SESSION_COLS} from sessions s where s.id = ? and s.user_id = ? "
                       "and s.institute_id = ?", (sid, o.user_id, o.institute_id))
        return self._session_out(rows[0]) if rows else None

    def list_sessions(self, o: Owner, cursor: Optional[str] = None, limit: int = 20, q: Optional[str] = None,
                      pinned: Optional[bool] = None) -> tuple[list[dict], Optional[str]]:
        where, args = ["s.user_id = ?", "s.institute_id = ?"], [o.user_id, o.institute_id]
        if pinned is not None:
            where.append("s.pinned = ?")
            args.append(1 if pinned else 0)
        if q and q.strip():
            pat = _like(normalize_query(q))
            where.append("(lower(s.title) like ? escape '\\' or exists (select 1 from messages m where "
                         "m.session_id = s.id and m.user_id = s.user_id and m.institute_id = s.institute_id and "
                         "m.role = 'user' and lower(m.text) like ? escape '\\'))")
            args += [pat, pat]
        if cursor:
            c = _dec(cursor)
            if not (isinstance(c, list) and len(c) == 2):
                raise ValueError("invalid cursor")
            where.append("(s.last_message_at < ? or (s.last_message_at = ? and s.id < ?))")
            args += [float(c[0]), float(c[0]), str(c[1])]
        rows = self._q(f"select {self._SESSION_COLS} from sessions s where {' and '.join(where)} "
                       "order by s.last_message_at desc, s.id desc limit ?", tuple(args) + (limit + 1,))
        items = [self._session_out(r) for r in rows[:limit]]
        nxt = _enc([items[-1]["_ts"], items[-1]["id"]]) if len(rows) > limit else None
        return items, nxt

    def patch_session(self, o: Owner, sid: str, title: Optional[str] = None,
                      pinned: Optional[bool] = None) -> Optional[dict]:
        sets, args = [], []
        if title is not None:
            sets += ["title = ?", "title_source = 'user'"]
            args.append(" ".join(title.split()))
        if pinned is not None:
            sets.append("pinned = ?")
            args.append(1 if pinned else 0)
        if sets:
            self._q(f"update sessions set {', '.join(sets)} where id = ? and user_id = ? and institute_id = ?",
                    tuple(args) + (sid, o.user_id, o.institute_id))
        return self.get_session(o, sid)

    def set_auto_title(self, o: Owner, sid: str, title: str) -> None:
        self._q("update sessions set title = ? where id = ? and user_id = ? and institute_id = ? "
                "and title_source = 'auto'", (title, sid, o.user_id, o.institute_id))

    def delete_session(self, o: Owner, sid: str) -> bool:
        def run(c):
            n = c.execute("delete from sessions where id = ? and user_id = ? and institute_id = ?",
                          (sid, o.user_id, o.institute_id)).rowcount
            c.execute("delete from messages where session_id = ? and user_id = ? and institute_id = ?",
                      (sid, o.user_id, o.institute_id))
            c.execute("delete from feedback where session_id = ? and user_id = ? and institute_id = ?",
                      (sid, o.user_id, o.institute_id))
            return n > 0
        return self._tx(run)

    # -- messages ----------------------------------------------------------
    def add_message(self, o: Owner, sid: str, role: str, text: Optional[str], events: Optional[list] = None,
                    llm: Optional[list] = None) -> Optional[int]:
        now = self.clock()

        def run(c):
            if not c.execute("select 1 from sessions where id = ? and user_id = ? and institute_id = ?",
                             (sid, o.user_id, o.institute_id)).fetchone():
                return None
            seq = c.execute("select coalesce(max(seq), 0) + 1 from messages where session_id = ?", (sid,)).fetchone()[0]
            c.execute("insert into messages (session_id, seq, user_id, institute_id, role, text, events, llm, "
                      "created_at) values (?,?,?,?,?,?,?,?,?)",
                      (sid, seq, o.user_id, o.institute_id, role, text,
                       None if events is None else json.dumps(events, default=str, ensure_ascii=False),
                       None if llm is None else json.dumps(llm, default=str, ensure_ascii=False), now))
            c.execute("update sessions set last_message_at = ? where id = ?", (now, sid))
            return seq
        return self._tx(run)

    @staticmethod
    def _msg_out(r: sqlite3.Row, with_llm: bool = False) -> dict:
        d = {"seq": r["seq"], "role": r["role"], "text": r["text"],
             "events": json.loads(r["events"]) if r["events"] is not None else None,
             "created_at": iso(r["created_at"])}
        if with_llm:
            d["llm"] = json.loads(r["llm"]) if r["llm"] is not None else None
        return d

    def list_messages(self, o: Owner, sid: str, cursor: Optional[str] = None,
                      limit: int = 100) -> tuple[list[dict], Optional[str]]:
        after = 0
        if cursor:
            c = _dec(cursor)
            if not isinstance(c, int):
                raise ValueError("invalid cursor")
            after = c
        rows = self._q("select * from messages where session_id = ? and user_id = ? and institute_id = ? and seq > ? "
                       "order by seq limit ?", (sid, o.user_id, o.institute_id, after, limit + 1))
        items = [self._msg_out(r) for r in rows[:limit]]
        return items, (_enc(items[-1]["seq"]) if len(rows) > limit else None)

    def all_messages(self, o: Owner, sid: str) -> list[dict]:
        rows = self._q("select * from messages where session_id = ? and user_id = ? and institute_id = ? order by seq",
                       (sid, o.user_id, o.institute_id))
        return [self._msg_out(r, with_llm=True) for r in rows]

    def get_message(self, o: Owner, sid: str, seq: int) -> Optional[dict]:
        rows = self._q("select * from messages where session_id = ? and user_id = ? and institute_id = ? and seq = ?",
                       (sid, o.user_id, o.institute_id, seq))
        return self._msg_out(rows[0], with_llm=True) if rows else None

    def truncate(self, o: Owner, sid: str, from_seq: int) -> int:
        """Delete messages with seq >= from_seq (and their feedback). Used by regenerate / edit."""
        def run(c):
            n = c.execute("delete from messages where session_id = ? and user_id = ? and institute_id = ? and seq >= ?",
                          (sid, o.user_id, o.institute_id, from_seq)).rowcount
            c.execute("delete from feedback where session_id = ? and user_id = ? and institute_id = ? "
                      "and message_seq >= ?", (sid, o.user_id, o.institute_id, from_seq))
            return n
        return self._tx(run)

    # -- query history -----------------------------------------------------
    def record_query(self, o: Owner, text: str) -> None:
        norm = normalize_query(text)
        if not norm:
            return
        self._q("insert into queries (id, user_id, institute_id, norm, text, last_used_at, use_count) "
                "values (?,?,?,?,?,?,1) on conflict (user_id, institute_id, norm) do update set "
                "text = excluded.text, last_used_at = excluded.last_used_at, use_count = queries.use_count + 1",
                (uuid.uuid4().hex, o.user_id, o.institute_id, norm, " ".join(text.split()), self.clock()))

    def list_queries(self, o: Owner, q: Optional[str] = None, limit: int = 10) -> list[dict]:
        where, args = "user_id = ? and institute_id = ?", [o.user_id, o.institute_id]
        if q and normalize_query(q):
            where += " and norm like ? escape '\\'"
            args.append(_like(normalize_query(q)))
        rows = self._q(f"select * from queries where {where} order by last_used_at desc, id limit ?",
                       tuple(args) + (limit,))
        return [{"id": r["id"], "text": r["text"], "norm": r["norm"], "last_used_at": iso(r["last_used_at"]),
                 "use_count": int(r["use_count"]), "_ts": r["last_used_at"]} for r in rows]

    def delete_query(self, o: Owner, qid: str) -> bool:
        return self._tx(lambda c: c.execute("delete from queries where id = ? and user_id = ? and institute_id = ?",
                                            (qid, o.user_id, o.institute_id)).rowcount > 0)

    def clear_queries(self, o: Owner) -> int:
        return self._tx(lambda c: c.execute("delete from queries where user_id = ? and institute_id = ?",
                                            (o.user_id, o.institute_id)).rowcount)

    # -- feedback ----------------------------------------------------------
    def set_feedback(self, o: Owner, sid: str, seq: int, rating: str, note: Optional[str]) -> bool:
        """Only on an assistant message of the caller's own session; one rating per message (replace)."""
        def run(c):
            ok = c.execute("select 1 from messages where session_id = ? and seq = ? and user_id = ? "
                           "and institute_id = ? and role = 'assistant'",
                           (sid, seq, o.user_id, o.institute_id)).fetchone()
            if not ok:
                return False
            c.execute("insert into feedback (user_id, institute_id, session_id, message_seq, rating, note, created_at) "
                      "values (?,?,?,?,?,?,?) on conflict (user_id, institute_id, session_id, message_seq) do update "
                      "set rating = excluded.rating, note = excluded.note, created_at = excluded.created_at",
                      (o.user_id, o.institute_id, sid, seq, rating, note, self.clock()))
            return True
        return self._tx(run)

    def get_feedback(self, o: Owner, sid: str) -> list[dict]:
        rows = self._q("select message_seq, rating, note from feedback where session_id = ? and user_id = ? "
                       "and institute_id = ? order by message_seq", (sid, o.user_id, o.institute_id))
        return [dict(r) for r in rows]

    # -- retention ---------------------------------------------------------
    def purge(self, days: int = RETENTION_DAYS) -> dict:
        cutoff = self.clock() - days * 86400

        def run(c):
            old = [r[0] for r in c.execute("select id from sessions where last_message_at < ?", (cutoff,)).fetchall()]
            for sid in old:
                c.execute("delete from messages where session_id = ?", (sid,))
                c.execute("delete from feedback where session_id = ?", (sid,))
            c.execute("delete from sessions where last_message_at < ?", (cutoff,))
            c.execute("delete from messages where session_id not in (select id from sessions)")
            c.execute("delete from feedback where session_id not in (select id from sessions)")
            nq = c.execute("delete from queries where last_used_at < ?", (cutoff,)).rowcount
            return {"sessions": len(old), "queries": nq}
        return self._tx(run)


class MemoryStore(SQLiteStore):
    """Tests: identical behaviour, nothing on disk."""

    def __init__(self, clock: Callable[[], float] = time.time):
        super().__init__(":memory:", clock=clock)


AgentStore = SQLiteStore  # the interface (both implementations share it)

_STORE: Optional[SQLiteStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> SQLiteStore:
    """AGENT_STORE=memory -> MemoryStore, else SQLite at AGENT_STORE_PATH (default agent/.devdata/agent_store.db)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            if os.getenv("AGENT_STORE", "sqlite").lower() == "memory":
                _STORE = MemoryStore()
            else:
                _STORE = SQLiteStore(os.getenv("AGENT_STORE_PATH") or DEFAULT_PATH)
        return _STORE


def set_store(store: Optional[SQLiteStore]) -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = store


def valid_id(s: str) -> bool:
    try:
        uuid.UUID(s or "")
        return True
    except ValueError:
        return False


# ---- real-mode paper drafts: previews only, process memory (never Supabase, never disk) ----------------
class DraftCache:
    MAX_SESSIONS = 200

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._d: dict[str, dict] = {}

    def load_drafts(self, sid: str) -> dict:
        with self._lock:
            return dict(self._d.get(sid, {}))

    def save_draft(self, sid: str, did: str, paper: dict) -> None:
        with self._lock:
            self._d.setdefault(sid, {})[did] = paper
            while len(self._d) > self.MAX_SESSIONS:
                self._d.pop(next(iter(self._d)))

    def drop(self, sid: str) -> None:
        with self._lock:
            self._d.pop(sid, None)


DRAFTS = DraftCache()
