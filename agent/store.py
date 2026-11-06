"""Session store for REAL mode: process memory only (nothing is written to Supabase). Lost on restart.
Seed/dev mode keeps using the local Postgres tables (agent_sessions / agent_messages / agent_drafts)."""
from __future__ import annotations

import threading
import time
from typing import Optional

MAX_SESSIONS_PER_USER = 50


class MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sessions: dict[str, dict] = {}   # id -> {user_id, institute_id, role, title, created_at, updated_at}
        self.history: dict[str, list] = {}
        self.events: dict[str, list] = {}
        self.drafts: dict[str, dict] = {}     # session_id -> {draft_id: paper}

    def create_session(self, sid: str, user_id: str, institute_id: str, role: str, title: str) -> None:
        now = time.time()
        with self._lock:
            self.sessions[sid] = {"user_id": user_id, "institute_id": institute_id, "role": role,
                                  "title": title[:60], "created_at": now, "updated_at": now}
            mine = sorted((s for s in self.sessions.items() if s[1]["user_id"] == user_id),
                          key=lambda kv: kv[1]["updated_at"])
            for old, _ in mine[:-MAX_SESSIONS_PER_USER]:
                for d in (self.sessions, self.history, self.events, self.drafts):
                    d.pop(old, None)

    def get(self, sid: str) -> Optional[dict]:
        return self.sessions.get(sid)

    def list_for(self, user_id: str) -> list[dict]:
        rows = [{"session_id": k, "title": v["title"], "created_at": v["created_at"], "updated_at": v["updated_at"]}
                for k, v in self.sessions.items() if v["user_id"] == user_id]
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)

    def touch(self, sid: str) -> None:
        if sid in self.sessions:
            self.sessions[sid]["updated_at"] = time.time()

    # drafts (paper previews) — keyed by session, checked against the owner by the caller
    def load_drafts(self, sid: str) -> dict:
        return dict(self.drafts.get(sid, {}))

    def save_draft(self, sid: str, did: str, paper: dict) -> None:
        with self._lock:
            self.drafts.setdefault(sid, {})[did] = paper


MEMORY = MemoryStore()
