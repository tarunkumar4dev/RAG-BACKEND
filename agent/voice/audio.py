"""Container-level decoding of what the browser records, so the 30 s / 1 MB limits are enforced on what is IN the
file, not on headers the client sent. No codec decoding (no ffmpeg): the containers are walked for their timing.

- WebM / Matroska (Chrome Android: audio/webm;codecs=opus). MediaRecorder writes an unknown-size Segment and
  Clusters and usually no Info/Duration, so the duration is the last block's absolute timecode.
- MP4 / M4A (iOS Safari: audio/mp4, AAC): mvhd, then mdhd, then fragmented trun sums.
- Ogg (Firefox: audio/ogg;codecs=opus): last page granule position at 48 kHz.
- WAV: data size / byte rate.

Anything else (or a file that does not parse) -> AudioError("unsupported" | "corrupt").
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

ACCEPTED_MIMES = ("audio/webm", "video/webm", "audio/mp4", "audio/x-m4a", "audio/m4a", "audio/aac", "audio/ogg",
                  "audio/wav", "audio/x-wav", "audio/wave")


class AudioError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code, self.detail = code, detail


@dataclass(frozen=True)
class AudioInfo:
    container: str   # webm | mp4 | ogg | wav
    mime: str        # what the STT provider is told
    duration_ms: int


# ---- WebM / EBML -----------------------------------------------------------------------------------------------
_EBML = 0x1A45DFA3
_SEGMENT, _INFO, _CLUSTER, _BLOCKGROUP = 0x18538067, 0x1549A966, 0x1F43B675, 0xA0
_TIMECODE_SCALE, _DURATION, _TIMECODE, _SIMPLEBLOCK, _BLOCK = 0x2AD7B1, 0x4489, 0xE7, 0xA3, 0xA1
_MASTERS = {_SEGMENT, _INFO, _CLUSTER, _BLOCKGROUP}


def _vint(buf: bytes, pos: int, keep_marker: bool) -> tuple[int, int, bool]:
    """-> (value, new_pos, unknown_size). EBML variable-size integer; unknown size = all value bits set."""
    if pos >= len(buf):
        raise AudioError("corrupt", "ebml truncated")
    first = buf[pos]
    length = 1
    mask = 0x80
    while length <= 8 and not (first & mask):
        mask >>= 1
        length += 1
    if length > 8 or pos + length > len(buf):
        raise AudioError("corrupt", "ebml vint")
    raw = buf[pos:pos + length]
    value = int.from_bytes(raw, "big")
    if not keep_marker:
        value &= (1 << (7 * length)) - 1
    unknown = (not keep_marker) and value == (1 << (7 * length)) - 1
    return value, pos + length, unknown


def _webm_duration_ms(buf: bytes) -> int:
    if len(buf) < 4 or int.from_bytes(buf[:4], "big") != _EBML:
        raise AudioError("corrupt", "not ebml")
    pos, end = 0, len(buf)
    scale_ns = 1_000_000
    info_duration = None
    cluster_tc = 0
    last_ts = 0
    blocks = 0
    while pos < end:
        try:
            eid, pos, _ = _vint(buf, pos, keep_marker=True)
            size, pos, unknown = _vint(buf, pos, keep_marker=False)
        except AudioError:
            break  # trailing partial element (MediaRecorder stop mid-write): keep what was parsed
        if eid in _MASTERS:
            continue  # descend: children follow immediately
        if unknown:
            break
        data = buf[pos:pos + size]
        if eid == _TIMECODE_SCALE and data:
            scale_ns = int.from_bytes(data, "big") or scale_ns
        elif eid == _DURATION and data:
            info_duration = struct.unpack(">f", data)[0] if len(data) == 4 else struct.unpack(">d", data)[0]
        elif eid == _TIMECODE and data:
            cluster_tc = int.from_bytes(data, "big")
        elif eid in (_SIMPLEBLOCK, _BLOCK) and len(data) >= 4:
            # track number (vint) + int16 relative timecode
            _, p2, _ = _vint(data, 0, keep_marker=False)
            if p2 + 2 <= len(data):
                rel = struct.unpack(">h", data[p2:p2 + 2])[0]
                last_ts = max(last_ts, cluster_tc + rel)
                blocks += 1
        pos += size
    if info_duration is not None and info_duration > 0:
        return int(info_duration * scale_ns / 1_000_000)
    if blocks == 0:
        raise AudioError("corrupt", "no audio blocks")
    return int((last_ts + 20) * scale_ns / 1_000_000)  # + one 20 ms opus frame for the last block


# ---- MP4 -------------------------------------------------------------------------------------------------------
def _boxes(buf: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(buf[pos:pos + 4], "big")
        typ = buf[pos + 4:pos + 8]
        hdr = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = int.from_bytes(buf[pos + 8:pos + 16], "big")
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            return
        yield typ, pos + hdr, min(pos + size, end)
        pos += size


def _mp4_duration_ms(buf: bytes) -> int:
    if len(buf) < 12 or buf[4:8] != b"ftyp":
        raise AudioError("corrupt", "not mp4")
    mvhd = None
    mdhd: list[tuple[int, int]] = []
    frag_ms = 0.0
    default_dur = 0
    timescale = 0
    for typ, a, b in _boxes(buf, 0, len(buf)):
        if typ == b"moov":
            for t2, c, d in _boxes(buf, a, b):
                if t2 == b"mvhd":
                    v = buf[c]
                    ts, dur = (int.from_bytes(buf[c + 12:c + 16], "big"), int.from_bytes(buf[c + 16:c + 20], "big")) \
                        if v == 0 else (int.from_bytes(buf[c + 20:c + 24], "big"), int.from_bytes(buf[c + 24:c + 32], "big"))
                    mvhd = (ts, dur)
                elif t2 == b"trak":
                    for t3, e, f in _boxes(buf, c, d):
                        if t3 == b"mdia":
                            for t4, g, h in _boxes(buf, e, f):
                                if t4 == b"mdhd":
                                    v = buf[g]
                                    if v == 0:
                                        mdhd.append((int.from_bytes(buf[g + 12:g + 16], "big"), int.from_bytes(buf[g + 16:g + 20], "big")))
                                    else:
                                        mdhd.append((int.from_bytes(buf[g + 20:g + 24], "big"), int.from_bytes(buf[g + 24:g + 32], "big")))
                elif t2 == b"mvex":
                    for t3, e, f in _boxes(buf, c, d):
                        if t3 == b"trex":
                            default_dur = int.from_bytes(buf[e + 16:e + 20], "big")
        elif typ == b"moof":
            for t2, c, d in _boxes(buf, a, b):
                if t2 != b"traf":
                    continue
                tf_default = default_dur
                for t3, e, f in _boxes(buf, c, d):
                    if t3 == b"tfhd":
                        flags = int.from_bytes(buf[e + 1:e + 4], "big")
                        p = e + 8
                        if flags & 0x1:
                            p += 8
                        if flags & 0x2:
                            p += 4
                        if flags & 0x8:
                            tf_default = int.from_bytes(buf[p:p + 4], "big")
                    elif t3 == b"trun":
                        flags = int.from_bytes(buf[e + 1:e + 4], "big")
                        n = int.from_bytes(buf[e + 4:e + 8], "big")
                        p = e + 8
                        if flags & 0x1:
                            p += 4
                        if flags & 0x4:
                            p += 4
                        per = (4 if flags & 0x100 else 0) + (4 if flags & 0x200 else 0) + (4 if flags & 0x400 else 0) + (4 if flags & 0x800 else 0)
                        total = 0
                        for _ in range(n):
                            if flags & 0x100:
                                total += int.from_bytes(buf[p:p + 4], "big")
                            else:
                                total += tf_default
                            p += per
                        ts = mdhd[0][0] if mdhd else (mvhd[0] if mvhd else 0)
                        timescale = ts or timescale
                        if ts:
                            frag_ms += total * 1000.0 / ts
    if mvhd and mvhd[0] and mvhd[1]:
        return int(mvhd[1] * 1000 / mvhd[0])
    for ts, dur in mdhd:
        if ts and dur:
            return int(dur * 1000 / ts)
    if frag_ms > 0:
        return int(frag_ms)
    raise AudioError("corrupt", "no duration in mp4")


# ---- Ogg / WAV -------------------------------------------------------------------------------------------------
def _ogg_duration_ms(buf: bytes) -> int:
    if buf[:4] != b"OggS":
        raise AudioError("corrupt", "not ogg")
    last = buf.rfind(b"OggS")
    granule = int.from_bytes(buf[last + 6:last + 14], "little", signed=True) if last >= 0 and last + 14 <= len(buf) else 0
    if granule <= 0:
        raise AudioError("corrupt", "ogg granule")
    return int(granule * 1000 / 48000)  # opus always runs at 48 kHz


def _wav_duration_ms(buf: bytes) -> int:
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise AudioError("corrupt", "not wav")
    pos, byte_rate, data_size = 12, 0, 0
    while pos + 8 <= len(buf):
        cid, size = buf[pos:pos + 4], int.from_bytes(buf[pos + 4:pos + 8], "little")
        if cid == b"fmt " and pos + 16 <= len(buf):
            byte_rate = int.from_bytes(buf[pos + 16:pos + 20], "little")
        elif cid == b"data":
            data_size = min(size, len(buf) - pos - 8) if size else len(buf) - pos - 8
            break
        pos += 8 + size + (size & 1)
    if not byte_rate or not data_size:
        raise AudioError("corrupt", "wav header")
    return int(data_size * 1000 / byte_rate)


# ---- entry -----------------------------------------------------------------------------------------------------
def sniff(buf: bytes) -> str:
    if buf[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if len(buf) >= 12 and buf[4:8] == b"ftyp":
        return "mp4"
    if buf[:4] == b"OggS":
        return "ogg"
    if buf[:4] == b"RIFF" and buf[8:12] == b"WAVE":
        return "wav"
    raise AudioError("unsupported", "unknown container")


_MIME = {"webm": "audio/webm", "mp4": "audio/mp4", "ogg": "audio/ogg", "wav": "audio/wav"}
_DECODE = {"webm": _webm_duration_ms, "mp4": _mp4_duration_ms, "ogg": _ogg_duration_ms, "wav": _wav_duration_ms}


def inspect(buf: bytes) -> AudioInfo:
    """Container + duration from the bytes themselves (the client's Content-Type and duration hint are ignored)."""
    if not buf:
        raise AudioError("empty")
    c = sniff(buf)
    return AudioInfo(container=c, mime=_MIME[c], duration_ms=_DECODE[c](buf))
