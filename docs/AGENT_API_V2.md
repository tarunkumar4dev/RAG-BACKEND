# a4ai Agent API v2 (contract 1.2)

Status: **stable** from commit "agent: API v2 contract". Additive to contract 1.1 (`agent/schemas.py`): nothing in 1.1
was renamed or removed. Pydantic models: `agent/schemas.py` (events, `ChatRequest`) and `agent/api_models.py`
(everything else). OpenAPI: `docs/agent_openapi_v2.json` (regenerate with `python -m agent.openapi_dump`).

## Rules for every endpoint

- Auth: `Authorization: Bearer <Supabase access token>` (real mode) or `X-Dev-Role: admin|teacher` (dev mode only).
  Identity, role and institute come from the verified token and DB rows (`SessionCtx`), never from the body.
- Everything is scoped to the caller: `user_id` **and** `institute_id`. An id that belongs to another user or another
  institute returns **404** (never 403), the same as an id that does not exist. Malformed ids also return 404.
- Timestamps: ISO-8601 UTC (`2026-11-09T10:15:00+00:00`). Cursors are opaque: send back `next_cursor` unchanged;
  `next_cursor: null` = last page.
- Errors: FastAPI default `{"detail": ...}`. 401 = no/invalid token, 403 = not staff / read-only action, 404 = not
  found or not yours, 422 = invalid body / query.
- Privacy: query text, message text and feedback notes are never written to logs or traces (traces keep ids, tool
  names, arg keys, latency, tokens). Access-log lines for `/agent/*` have the query string removed.
- Storage: sessions, messages, queries and feedback live in the **agent store** (SQLite file
  `agent/.devdata/agent_store.db` today; `agent/sql/agent_store_supabase.sql` later). Never in the a4ai tables.
  Retention: sessions and queries not used for 90 days are deleted at startup.

## Streaming chat

### `POST /agent/chat` → `text/event-stream`

```json
{ "message": "ECE 1 ki is mahine ki attendance",   // required unless regenerate_from_seq is set (1..4000 chars)
  "session_id": "uuid | null",                      // omit / null = new session
  "regenerate_from_seq": 4,                          // optional: re-answer; needs session_id
  "edit_of_seq": 3 }                                 // optional: replace a user message; needs session_id
```

- New session: the title is the first user message, whitespace collapsed, trimmed to **48 chars** (cut at a word
  boundary, `…` appended when cut). No LLM call. A title set with PATCH is never overwritten.
- `regenerate_from_seq = N`: N is an assistant message (or the user message it answers). The server deletes that
  assistant message and everything after it, then re-runs the user message before it. `message` is ignored.
- `edit_of_seq = N`: N must be a user message. The server deletes message N and everything after it, then runs
  `message` as the new user message. If N is the first message and the title is automatic, the title is recomputed.
- Deleted messages lose their feedback. New messages always get `seq = max(remaining seq) + 1`, so after a
  regenerate the new answer usually reuses the old seq. Reload `/messages` after `final` if you need exact seqs.
- Response header `X-Agent-Request-Id` = the `request_id` carried by every event (needed for cancel).
- Unknown / foreign `session_id`, or a seq that is not in the session → 404 before the stream starts.
  `edit_of_seq` pointing at an assistant message → 422.

Each SSE frame is `data: <AgentEvent JSON>\n\n`. Every event has `seq` (0, 1, 2… per request) and `request_id`.
Event types (`type`): `session`, `text_delta`, `thinking_status`, `stat_card`, `chart`, `table`, `clarify`,
`blueprint`, `paper_preview`, `verification_summary`, `pending_approval`, `final`, `error`.

New in 1.2:

```json
{"type": "session", "seq": 0, "request_id": "9f2c…", "session_id": "uuid", "title": "ECE 1 ki is mahine ki attendance"}
{"type": "final", "seq": 17, "request_id": "9f2c…", "session_id": "uuid", "usage": {…}, "cancelled": false}
{"type": "clarify", …, "chips": [{"label": "ECE 1", "value": "ECE 1 (id 6b0e…)", "hint": "Electronics · 2 students"},
                                 {"label": "ECE1",  "value": "ECE1 (id 77c1…)",  "hint": "Electronics · 0 students"}]}
```

- `session` is always the first event (seq 0). `final` is always the last event, also after an `error`.
- `clarify.chips[].hint` (optional) tells near-duplicates apart: student count, department, roll number.
  Send `chips[].value` back as the next `message` when the user taps a chip.

### `POST /agent/chat/{request_id}/cancel` → `202 {"request_id": "…", "cancelled": true}`

Stops a running stream of **your own** request. The stream then ends with `final {"cancelled": true}` (no `error`).
Unknown, finished or foreign `request_id` → 404. The partial assistant turn (events so far) is stored.
A running LLM call is abandoned, not interrupted: the stream ends within ~0.2 s of the cancel.

## Sessions

### `GET /agent/sessions?cursor=&limit=20&q=&pinned=`

- `limit` 1..50 (default 20). `q` (max 80): case-insensitive substring of the title or of any user message.
  `pinned=true|false` filters; omitted = all. Order: `last_message_at` desc, then id.

```json
{ "items": [ { "id": "uuid", "title": "ECE 1 ki is mahine ki attendance", "created_at": "…", "last_message_at": "…",
               "message_count": 2, "pinned": false, "preview": "ECE 1 ki attendance is mahine theek rahi." } ],
  "next_cursor": "opaque | null",
  "sessions": [ { "session_id": "uuid", "title": "…", "created_at": "…", "updated_at": "…" } ] }
```

`sessions` is the 1.1 shape, **deprecated**, same rows as `items`.

### `GET /agent/sessions/{id}/messages?cursor=&limit=100`

Ascending by `seq`. `limit` 1..200 (default 100).

```json
{ "items": [ { "seq": 1, "role": "user", "text": "ECE 1 ki attendance", "events": null, "created_at": "…" },
             { "seq": 2, "role": "assistant", "text": "ECE 1 ki attendance theek hai.", "events": [AgentEvent, …],
               "created_at": "…" } ],
  "next_cursor": null }
```

Assistant `events` is the exact list streamed for that turn (including `session` and `final`), so the UI can replay
cards, tables, charts, chips and paper previews. `text` = the turn's `text_delta` texts joined with a blank line
(null when there were none). A cancelled turn has `final.cancelled = true` in its events.

### `PATCH /agent/sessions/{id}` `{"title"?: str 1..80, "pinned"?: bool}` → `SessionItem`

At least one field. A PATCHed title is never replaced by the automatic title.

### `DELETE /agent/sessions/{id}` → `204`

Deletes the session, its messages and feedback immediately.

### `GET /agent/sessions/{id}` (1.1, unchanged shape)

`{"session_id", "role", "created_at", "turns": [[AgentEvent…], …]}`. Owner only. Prefer `/messages`.

## Query history

### `GET /agent/history/queries?q=&limit=10` → `[{"id", "text", "last_used_at", "use_count"}]`

One row per **normalized** text (lowercase, trimmed, inner whitespace collapsed); `text` is the latest original
spelling. Every user message sent to `/agent/chat` (new or edited, not regenerated) is recorded. Order:
`last_used_at` desc. `q`: substring of the normalized text. `limit` 1..50.

### `DELETE /agent/history/queries/{id}` → `204`   ·   `DELETE /agent/history/queries` → `204` (clear all)

## Suggestions (composer autocomplete)

### `GET /agent/suggestions?q=&limit=8` → `[{"type": "recent|template|entity", "text", "hint"}]`

- `recent`: your past queries. Prefix matches first, then substring matches; inside each group by
  `use_count × 0.5^(age_days / 14)` (recency-weighted use count), then most recent. Hint `"Recent"` or `"Recent · 3×"`.
- `entity`: batch / section / teacher / subject names **in your scope** matching `q` (punctuation and spaces
  ignored, so `ece1` matches `ECE 1` and `ECE-1`), returned as ready questions:
  batch → `"<name> ki is mahine ki attendance"` hint `"Batch · 22 students"`;
  teacher → `"<name> ki is hafte ki attendance"` hint `"Teacher"`;
  subject → `"<name> ke tests is mahine"` hint `"Subject"`. Requires `q` (≥ 1 char).
- `template`: role-based starters (admin/HOD vs teacher), hint `"Starter"`. With `q`, only templates containing it.
- Order: recent, then entity, then template; duplicates (normalized text) removed; at most `limit` (1..20, default 8).
  Empty `q`: up to 4 recent, then templates.
- Latency: scope entities are cached per user for 5 minutes; p95 < 150 ms on a warm cache.

## Feedback

### `POST /agent/feedback` `{"session_id", "message_seq", "rating": "up|down", "note"?: str ≤ 1000}` → `{"ok": true}`

`message_seq` must be an assistant message of your session (else 404). One rating per message: sending again
replaces it.

## New tools (visible to the UI only through the usual events)

- `my_scope` → `table` "Meri classes" (Batch, Students, Role, Subjects). Empty scope → no table, honest text.
- `section_comparison(date_from, date_to, metric: attendance|marks)` → `table` (rank, class, %, classes held /
  tests, marked / graded) + `chart` kind `bar`.
