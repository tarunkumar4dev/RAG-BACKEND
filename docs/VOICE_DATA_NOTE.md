# Voice data note — **NEEDS LEGAL REVIEW**

Status: draft written by engineering on 10 Nov 2026. Not reviewed by a lawyer. Do not publish to users as-is.
Vendor terms below are what their public docs said on that date; confirm them in the signed contracts.

## What goes where

| data | goes to | why | kept by a4ai? |
|---|---|---|---|
| the recorded audio (≤ 30 s, WebM/Opus or MP4/AAC) | **Sarvam AI** (`api.sarvam.ai/speech-to-text`); if Sarvam fails, **Google Gemini** API | turn speech into text | **No.** Read into memory, sent, dropped. Never written to disk, database, logs or traces. |
| the transcript text | the a4ai Agent server, then the **LLM** (Anthropic Claude by default, Gemini if configured) — exactly like a typed question | answer the question | **Yes, as a normal chat message**: same session history, same 90-day retention, same "clear" buttons as typed chat. Never in logs or traces. |
| names of the caller's own teachers / sections / students | stays on the a4ai server (5-minute in-memory cache); with `VOICE_STT_MODEL=saaras:v4` up to 50 of those names are sent to Sarvam as "keyterms" to help recognition | fix misheard names | in memory only, 5 minutes |
| the one-sentence spoken answer | **Sarvam AI** (`/text-to-speech`) — contains names and numbers from the answer | read it aloud | **No.** Streamed to the phone, not stored. Only fixed phrases with no names or numbers ("Dobara boliye") are cached in server memory. |
| on a slow network | nothing extra: the phone's own speech engine reads the answer | | — |
| trace / metrics | a4ai server log files | operations | ids, sizes, durations, provider name, estimated cost, outcome. **No audio, no text.** |

Identity is the logged-in a4ai session. There is no speaker recognition and no voiceprint.

## Retention and deletion
- Audio: not retained by a4ai. Vendor retention: **check Sarvam's and Google's API data terms** (whether they keep or
  train on API audio, and for how long; whether a zero-retention option exists for our account). Record the answer here.
- Transcripts: in the agent store with chat history. Deleted when the user deletes the chat, clears history, or
  after 90 days unused (agent store purge at startup).
- Voice settings + consent: in the browser's local storage for that user on that device; removed on sign-out.

## The consent record
- First mic use shows one line in Hindi and English: the audio is sent for transcription and not stored, with a
  "Theek hai" button. The mic stays disabled until it is tapped.
- What is kept: the acceptance timestamp and the note's version (`2026-11-10`), per user, **on that device only**
  (`a4ai.agent.v3.<user id>.pref.voice`). A new version of this note asks again.
- Gap for legal to decide: a device-local record is lost on sign-out or a new phone, and the server has no copy. If a
  server-side consent log is required (DPDP Act 2023 notice/consent), it needs a small table (not built).

## Questions for legal
1. Is the one-line notice enough, or does DPDP need a fuller notice with purpose, vendors and a grievance contact?
2. Is audio of a staff member's voice "personal data" sent abroad (Gemini fallback, Anthropic)? Sarvam is India-based.
3. Do we need a server-side consent log?
4. Can transcripts that mention student names be sent to a non-Indian LLM under the current customer agreements?
