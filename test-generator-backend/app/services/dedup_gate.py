"""
Semantic dedup GATE (Phase 2A). Serverless-safe: no torch, no sentence-transformers.

Two signals, both deterministic:
  1. STRUCTURAL pattern key  — the stem with every number -> '#', every proper
     name -> 'N', ratios -> 'RATIO'. Two questions that differ only in their
     numbers/names collapse to the same key. This catches the D3 failure class
     "same stem, changed numbers and names" that lexical prefix-matching missed.
  2. LEXICAL near-identity    — SequenceMatcher on the normalised stem, with a
     HIGH threshold (default 0.90), to catch reworded near-duplicates whose
     structural key differs by a token or two.

A genuinely different cognitive task on the same concept has a different pattern
key (different given/asked structure) and low lexical overlap, so it is kept —
we flag repeated STRUCTURES, not all topical overlap.

An optional `embed_fn` hook (e.g. a Gemini embeddings call) can add a semantic
cosine signal; it is off by default so the gate stays deterministic and free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Sequence, Tuple

_RATIO = re.compile(r"\b\d+\s*:\s*\d+(?:\s*:\s*\d+)*\b")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CAPWORD = re.compile(r"\b[A-Z][a-z]{2,}\b")

DEFAULT_LEXICAL_THRESHOLD = 0.90


def pattern_key(text: str) -> str:
    """STRUCTURE key: the stem with numbers -> '#', ratios -> 'RATIO', names -> 'N'.
    Two stems that differ only in numbers/names share a key (same cognitive task)."""
    t = text or ""
    t = _RATIO.sub(" RATIO ", t)
    t = _CAPWORD.sub(" N ", t)
    t = t.lower()
    t = _NUM.sub(" # ", t)
    t = re.sub(r"[^a-z#\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def exact_key(text: str) -> str:
    """HARD-duplicate key: names -> 'N' but NUMBERS KEPT. Two stems with the same
    structure AND the same numbers share a key — the literal "same question
    repeated" (D3's near-verbatim case), even when the names differ."""
    t = text or ""
    t = _CAPWORD.sub(" N ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9:\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


@dataclass
class DedupResult:
    kept: List                      # the questions that survive (same objects)
    removed: List[dict] = field(default_factory=list)   # {"index", "reason", "dup_of"}

    @property
    def removed_count(self) -> int:
        return len(self.removed)


def _text_of(q) -> str:
    if hasattr(q, "text"):
        return q.text or ""
    if isinstance(q, dict):
        return q.get("text") or ""
    return str(q)


def dedup(questions: Sequence,
          *,
          lexical_threshold: float = DEFAULT_LEXICAL_THRESHOLD,
          max_per_structure: Optional[int] = None,
          embed_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
          embed_threshold: float = 0.90,
          prior_exact: Optional[set] = None,
          prior_structure_counts: Optional[dict] = None,
          prior_texts: Optional[Sequence[str]] = None) -> DedupResult:
    """Keep the first of each duplicate group.

    A later question is a HARD duplicate (always removed) of an earlier kept one if:
        - same exact key (same structure AND same numbers), OR
        - lexical similarity >= lexical_threshold, OR
        - (embed_fn given) cosine >= embed_threshold.

    Additionally, if `max_per_structure` is set, a question is removed once its
    STRUCTURE key has already been kept that many times (controlled repetition:
    e.g. allow 2 interest-on-capital questions with different numbers, not 8).
    `max_per_structure=None` allows unlimited same-structure-different-number
    questions (hard duplicates are still removed).

    `prior_exact` / `prior_structure_counts` / `prior_texts` carry state ACROSS
    batches; they are updated in place with what survives.
    """
    kept: List = []
    removed: List[dict] = []
    seen_exact = set(prior_exact) if prior_exact else set()
    struct_counts = dict(prior_structure_counts) if prior_structure_counts else {}
    kept_texts: List[str] = list(prior_texts) if prior_texts else []

    embeds = None
    if embed_fn is not None:
        try:
            embeds = list(embed_fn([_text_of(q) for q in questions]))
        except Exception:
            embeds = None

    def _cosine(u, v) -> float:
        du = sum(x * x for x in u) ** 0.5
        dv = sum(x * x for x in v) ** 0.5
        if du == 0 or dv == 0:
            return 0.0
        return sum(a * b for a, b in zip(u, v)) / (du * dv)

    kept_embeds: List = []
    for i, q in enumerate(questions):
        text = _text_of(q)
        ek = exact_key(text)
        sk = pattern_key(text)

        reason = None
        dup_of = None
        if ek and ek in seen_exact:
            reason, dup_of = "exact", ek
        else:
            for j, kt in enumerate(kept_texts):
                if lexical_similarity(text, kt) >= lexical_threshold:
                    reason, dup_of = "lexical", j
                    break
            if reason is None and embeds is not None:
                for j, ke in enumerate(kept_embeds):
                    if _cosine(embeds[i], ke) >= embed_threshold:
                        reason, dup_of = "embedding", j
                        break
            if reason is None and max_per_structure is not None \
                    and sk and struct_counts.get(sk, 0) >= max_per_structure:
                reason, dup_of = "structure_cap", sk

        if reason:
            removed.append({"index": i, "reason": reason, "dup_of": dup_of,
                            "text": text[:80]})
            continue

        kept.append(q)
        kept_texts.append(text)
        if ek:
            seen_exact.add(ek)
        if sk:
            struct_counts[sk] = struct_counts.get(sk, 0) + 1
        if embeds is not None:
            kept_embeds.append(embeds[i])

    if prior_exact is not None:
        prior_exact.update(seen_exact)
    if prior_structure_counts is not None:
        prior_structure_counts.clear()
        prior_structure_counts.update(struct_counts)

    return DedupResult(kept=kept, removed=removed)


def distinct_structures(questions: Sequence) -> int:
    """How many distinct structural pattern keys a set of questions has."""
    return len({pattern_key(_text_of(q)) for q in questions})
