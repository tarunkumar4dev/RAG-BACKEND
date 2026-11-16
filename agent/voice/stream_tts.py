"""Streaming TTS for voice turns: the model's SPEECH line is spoken while it is still being written.

Flow (one voice turn):
  LLM streams text -> feed() finds the line that starts with "SPEECH:" -> the line is cut at sentence ends
  (। . ? !) or at ~80 characters -> every chunk goes to the TTS provider in parallel (a small thread pool, one
  shared keep-alive HTTP client in agent/voice/providers.py) -> finished chunks become `audio` payloads that the
  agent (agent/core.py, generator thread) turns into SSE events, tagged with their index. The browser plays them
  in index order (src/lib/voice/streamPlayer.ts).

Only the SPEECH line is ever spoken; the prose stays on screen. If a step ends with tool calls, whatever was
captured in that step is discarded (it was not the answer). If the provider does not stream text (tests' fakes,
canned guard answers), ensure(text) speaks the whole SPEECH text after the fact.

Time-to-first-audio (ttfa_ms) is measured from the start of the LLM step that produced the SPEECH line to the
moment the first chunk's audio is ready; it rides on chunk 0 and goes to the trace.
"""
from __future__ import annotations

import base64
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

from .speech import classify, tts_language_code

MAX_CHUNK_CHARS = 80              # cut a long sentence at the last space before this
MIN_CHUNK_CHARS = 2
_SENT_END = re.compile(r"[।.?!](?=\s|$)")
_SPEECH_START = re.compile(r"(?:^|\n)\s*SPEECH\s*:\s*", re.IGNORECASE)
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tts")


def split_ready(buf: str, final: bool = False) -> tuple[list[str], str]:
    """Pure: cut `buf` into finished chunks and the unfinished rest. A chunk ends at a sentence end, or at the last
    space before MAX_CHUNK_CHARS when a sentence runs long. final=True also returns the rest as a chunk."""
    chunks: list[str] = []
    while True:
        m = _SENT_END.search(buf)
        if m and m.end() <= MAX_CHUNK_CHARS + 20:
            piece, buf = buf[:m.end()], buf[m.end():]
        elif len(buf) >= MAX_CHUNK_CHARS:
            cut = buf.rfind(" ", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > 20 else MAX_CHUNK_CHARS
            piece, buf = buf[:cut], buf[cut:]
        else:
            break
        piece = " ".join(piece.split())
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)
    if final:
        rest = " ".join(buf.split())
        if len(rest) >= MIN_CHUNK_CHARS:
            chunks.append(rest)
        buf = ""
    return chunks, buf


class SpeechStream:
    def __init__(self, tts, speaker: str, pace: float, language: Optional[str] = None,
                 on_first_audio: Optional[Callable[[], None]] = None, clock: Callable[[], float] = time.monotonic,
                 pool: ThreadPoolExecutor = _POOL):
        self.tts, self.speaker, self.pace, self.language = tts, speaker, pace, language
        self.on_first_audio = on_first_audio
        self.clock, self.pool = clock, pool
        self._lock = threading.Lock()
        self._raw = ""            # text of the current step so far
        self._line: Optional[str] = None   # SPEECH text captured in this step (None = not seen yet)
        self._line_done = False
        self._buf = ""            # captured SPEECH text not yet cut into chunks
        self._step_t0: Optional[float] = None
        self._futures: list[tuple[int, Future]] = []
        self._ready: dict[int, dict] = {}
        self._emitted = 0         # next index to hand out (in order)
        self._next_index = 0
        self.spoken_text = ""     # everything sent to TTS this turn
        self.cancelled = False
        self.ttfa_ms: Optional[int] = None
        self.chars = 0
        self.errors = 0
        self._first_done = False

    # ---- input --------------------------------------------------------------------------------------------------
    def begin_step(self) -> None:
        with self._lock:
            self._raw, self._line, self._line_done, self._buf = "", None, False, ""
            self._step_t0 = self.clock()

    def feed(self, delta: str) -> None:
        """Model text delta (LLM worker thread)."""
        if self.cancelled or not delta:
            return
        with self._lock:
            if self._line_done:
                return
            self._raw += delta
            if self._line is None:
                m = _SPEECH_START.search(self._raw)
                if not m:
                    return
                self._line = ""
                tail = self._raw[m.end():]
            else:
                tail = delta
            nl = tail.find("\n")
            if nl >= 0:
                tail, self._line_done = tail[:nl], True
            self._line += tail
            self._buf += tail
            chunks, self._buf = split_ready(self._buf, final=self._line_done)
            for c in chunks:
                self._submit(c)

    def discard_step(self) -> None:
        """The step ended with tool calls: what it said was not the answer."""
        with self._lock:
            self._raw, self._line, self._line_done, self._buf = "", None, False, ""

    def end_step(self) -> None:
        """The final step ended: speak whatever is left of the SPEECH line."""
        with self._lock:
            if self._line is not None and not self._line_done:
                chunks, self._buf = split_ready(self._buf, final=True)
                self._line_done = True
                for c in chunks:
                    self._submit(c)

    def ensure(self, speech: Optional[str]) -> None:
        """After the answer is complete: if nothing was spoken while streaming (provider without text streaming,
        canned answer), speak the whole SPEECH text now."""
        if not speech or self.cancelled:
            return
        with self._lock:
            if self._next_index > 0:
                return
            if self._step_t0 is None:
                self._step_t0 = self.clock()
            chunks, _ = split_ready(speech, final=True)
            for c in chunks:
                self._submit(c)

    def cancel(self) -> None:
        self.cancelled = True
        with self._lock:
            for _, f in self._futures:
                f.cancel()

    # ---- TTS ----------------------------------------------------------------------------------------------------
    def _submit(self, text: str) -> None:   # caller holds the lock
        idx = self._next_index
        self._next_index += 1
        self.spoken_text = (self.spoken_text + " " + text).strip()
        self.chars += len(text)
        self._futures.append((idx, self.pool.submit(self._synth, idx, text)))

    def _synth(self, idx: int, text: str) -> Optional[dict]:
        if self.cancelled:
            return None
        lang = self.language or classify(text)
        audio = b"".join(self.tts.synthesize(text, tts_language_code(text, lang), self.speaker, self.pace))
        if self.cancelled:
            return None
        if not audio:  # nothing to play: skip this index so later chunks are not held back
            with self._lock:
                self._ready[idx] = {"index": idx, "error": True}
                self.errors += 1
            return None
        payload = {"index": idx, "mime": getattr(self.tts, "mime", "audio/mpeg"),
                   "data": base64.b64encode(audio).decode("ascii"), "chars": len(text)}
        with self._lock:
            if idx == 0 and self.ttfa_ms is None and self._step_t0 is not None:
                self.ttfa_ms = int((self.clock() - self._step_t0) * 1000)
                payload["ttfa_ms"] = self.ttfa_ms
            self._ready[idx] = payload
        if not self._first_done:
            self._first_done = True
            if self.on_first_audio:
                try:
                    self.on_first_audio()
                except Exception:  # noqa: BLE001  accounting must never break playback
                    pass
        return payload

    # ---- output (agent generator thread) ------------------------------------------------------------------------
    def _harvest_errors(self) -> None:
        with self._lock:
            futs = list(self._futures)
        for idx, f in futs:
            if idx < self._emitted or idx in self._ready:
                continue
            if f.done() and not f.cancelled() and f.exception() is not None:
                with self._lock:
                    self._ready[idx] = {"index": idx, "error": True}
                    self.errors += 1

    def ready(self) -> list[dict]:
        """Payloads that can go out now, in index order (a failed chunk is skipped, never blocks the rest)."""
        if self.cancelled:
            return []
        self._harvest_errors()
        out = []
        with self._lock:
            while self._emitted in self._ready:
                p = self._ready.pop(self._emitted)
                self._emitted += 1
                if not p.get("error"):
                    out.append(p)
        return out

    def pending(self) -> bool:
        with self._lock:
            return self._emitted < self._next_index and not self.cancelled

    @property
    def produced(self) -> int:
        return self._next_index
