"""Synthetic audio CONTAINERS for the voice tests (no real audio): enough structure for agent/voice/audio.py to
measure duration the way it does on a real MediaRecorder file. Shared by test_voice_*.py and the audio eval."""
from __future__ import annotations

import struct


def _vint_size(n: int) -> bytes:
    """EBML size vint (1..8 bytes)."""
    for length in range(1, 9):
        if n < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | n).to_bytes(length, "big")
    raise ValueError("too big")


def _el(eid: int, payload: bytes, unknown: bool = False) -> bytes:
    idb = eid.to_bytes((eid.bit_length() + 7) // 8, "big")
    size = b"\x01\xff\xff\xff\xff\xff\xff\xff" if unknown else _vint_size(len(payload))
    return idb + size + payload


def webm(seconds: float, with_duration: bool = False, cluster_ms: int = 1000) -> bytes:
    """MediaRecorder-like WebM: unknown-size Segment + Clusters, 20 ms SimpleBlocks, no Info/Duration by default."""
    header = _el(0x1A45DFA3, _el(0x4282, b"webm") + _el(0x4287, b"\x02"))
    info = _el(0x2AD7B1, (1_000_000).to_bytes(3, "big"))
    if with_duration:
        info += _el(0x4489, struct.pack(">d", seconds * 1000))
    body = _el(0x1549A966, info)
    total_ms = int(seconds * 1000)
    tc = 0
    while tc < total_ms:
        blocks = b""
        rel = 0
        while rel < cluster_ms and tc + rel < total_ms:
            blocks += _el(0xA3, b"\x81" + struct.pack(">h", rel) + b"\x80" + b"\x00" * 10)  # track 1, keyframe flag
            rel += 20
        body += _el(0x1F43B675, _el(0xE7, tc.to_bytes(2, "big") if tc else b"\x00") + blocks, unknown=True)
        tc += cluster_ms
    return header + _el(0x18538067, body, unknown=True)


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def mp4(seconds: float, timescale: int = 1000) -> bytes:
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")
    mvhd = _box(b"mvhd", b"\x00\x00\x00\x00" + struct.pack(">II", 0, 0) + struct.pack(">II", timescale, int(seconds * timescale)) + b"\x00" * 80)
    mdhd = _box(b"mdhd", b"\x00\x00\x00\x00" + struct.pack(">II", 0, 0) + struct.pack(">II", timescale, int(seconds * timescale)) + b"\x00\x00\x00\x00")
    trak = _box(b"trak", _box(b"mdia", mdhd))
    return ftyp + _box(b"moov", mvhd + trak) + _box(b"mdat", b"\x00" * 64)


def mp4_fragmented(seconds: float, sample_ms: int = 20) -> bytes:
    """iOS-style: empty mvhd duration, trex default duration, moof/traf/trun with per-sample durations."""
    ts = 1000
    ftyp = _box(b"ftyp", b"iso5" + struct.pack(">I", 1) + b"iso5dash")
    mvhd = _box(b"mvhd", b"\x00\x00\x00\x00" + struct.pack(">II", 0, 0) + struct.pack(">II", ts, 0) + b"\x00" * 80)
    mdhd = _box(b"mdhd", b"\x00\x00\x00\x00" + struct.pack(">II", 0, 0) + struct.pack(">II", ts, 0) + b"\x00\x00\x00\x00")
    trex = _box(b"trex", b"\x00\x00\x00\x00" + struct.pack(">IIIII", 1, 1, sample_ms, 0, 0))
    moov = _box(b"moov", mvhd + _box(b"trak", _box(b"mdia", mdhd)) + _box(b"mvex", trex))
    n = int(seconds * 1000 / sample_ms)
    out = ftyp + moov
    per = 50
    for i in range(0, n, per):
        cnt = min(per, n - i)
        tfhd = _box(b"tfhd", b"\x00\x00\x00\x00" + struct.pack(">I", 1))
        trun = _box(b"trun", b"\x00\x00\x01\x00" + struct.pack(">I", cnt) + b"".join(struct.pack(">I", sample_ms) for _ in range(cnt)))
        out += _box(b"moof", _box(b"mfhd", b"\x00\x00\x00\x00" + struct.pack(">I", i)) + _box(b"traf", tfhd + trun))
        out += _box(b"mdat", b"\x00" * 8)
    return out


def ogg(seconds: float) -> bytes:
    granule = int(seconds * 48000)
    page = b"OggS" + b"\x00" + b"\x04" + struct.pack("<q", granule) + b"\x00" * 4 + b"\x00" * 4 + b"\x00" * 4 + b"\x01" + b"\x08" + b"\x00" * 8
    return (b"OggS" + b"\x00" + b"\x02" + struct.pack("<q", 0) + b"\x00" * 13 + b"\x01" + b"\x13" + b"OpusHead" + b"\x00" * 11) + page


def wav(seconds: float, rate: int = 16000) -> bytes:
    n = int(seconds * rate) * 2
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
            + b"data" + struct.pack("<I", n) + b"\x00" * n)
