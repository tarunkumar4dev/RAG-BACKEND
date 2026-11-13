"""Phase 2A: semantic dedup gate. Deterministic, no network."""
from app.services import dedup_gate as dg


def _q(text):
    return {"text": text}


# ── pattern/exact keys ───────────────────────────────────────────────────

def test_same_structure_diff_numbers_share_structure_key_not_exact():
    a = "Ram's opening capital is Rs. 5,00,000 at 10% p.a. Interest on capital is:"
    b = "Sita's opening capital is Rs. 2,00,000 at 6% p.a. Interest on capital is:"
    assert dg.pattern_key(a) == dg.pattern_key(b)          # same cognitive task
    assert dg.exact_key(a) != dg.exact_key(b)              # different numbers


def test_same_numbers_diff_names_share_exact_key():
    a = "Ram's capital is Rs. 5,00,000 at 10%. IOC?"
    b = "Shyam's capital is Rs. 5,00,000 at 10%. IOC?"
    assert dg.exact_key(a) == dg.exact_key(b)              # D3 near-verbatim


# ── hard duplicate removal ────────────────────────────────────────────────

def test_removes_exact_repeat_even_with_different_names():
    qs = [_q("Ram's capital is Rs. 5,00,000 at 10%. Interest on capital is:"),
          _q("Shyam's capital is Rs. 5,00,000 at 10%. Interest on capital is:")]
    r = dg.dedup(qs)
    assert len(r.kept) == 1
    assert r.removed[0]["reason"] == "exact"


def test_removes_near_verbatim_lexical():
    qs = [_q("A firm's net profit is Rs. 3,60,000. The deed provides interest on capital Rs. 20,000 and salary Rs. 18,000. Divisible profit is:"),
          _q("A firm's net profit is Rs. 3,80,000. The deed provides interest on capital Rs. 20,000 and salary Rs. 18,000. Divisible profit is:")]
    r = dg.dedup(qs, lexical_threshold=0.90)
    assert len(r.kept) == 1
    assert r.removed[0]["reason"] == "lexical"


def test_keeps_genuinely_different_cognitive_tasks():
    qs = [_q("Ram's opening capital is Rs. 5,00,000 at 10% p.a. Interest on capital is:"),
          _q("The manager gets 10% commission on net profit of Rs. 2,20,000 after charging it. Commission is:"),
          _q("Partners share 3:2:1. Divisible profit Rs. 1,80,000. A's share is:")]
    r = dg.dedup(qs)
    assert len(r.kept) == 3
    assert r.removed_count == 0


# ── structure cap ─────────────────────────────────────────────────────────

_CAP_QS = [
    "Arun's opening capital is Rs. 5,00,000 at 10% p.a. Interest on capital is:",
    "Bela's opening capital is Rs. 6,50,000 at 11% p.a. Interest on capital is:",
    "Charu's opening capital is Rs. 7,20,000 at 9% p.a. Interest on capital is:",
    "Deepa's opening capital is Rs. 8,40,000 at 12% p.a. Interest on capital is:",
]


def test_structure_cap_limits_same_structure_repeats():
    # Same structure, different names + numbers -> one structure key; cap at 2.
    r = dg.dedup([_q(t) for t in _CAP_QS], lexical_threshold=0.99, max_per_structure=2)
    assert len(r.kept) == 2
    assert all(x["reason"] == "structure_cap" for x in r.removed)


def test_no_cap_keeps_all_distinct_number_variants():
    r = dg.dedup([_q(t) for t in _CAP_QS], lexical_threshold=0.99, max_per_structure=None)
    assert len(r.kept) == 4


# ── cross-batch state ─────────────────────────────────────────────────────

def test_cross_batch_state_drops_repeat_from_later_batch():
    exact: set = set()
    counts: dict = {}
    texts: list = []
    b1 = dg.dedup([_q("Ram's capital Rs. 5,00,000 at 10%. IOC?")],
                  prior_exact=exact, prior_structure_counts=counts, prior_texts=texts)
    texts = [q["text"] for q in b1.kept]
    b2 = dg.dedup([_q("Shyam's capital Rs. 5,00,000 at 10%. IOC?")],   # exact repeat vs batch 1
                  prior_exact=exact, prior_structure_counts=counts, prior_texts=texts)
    assert b2.kept == []
    assert b2.removed[0]["reason"] == "exact"


# ── embedding hook ─────────────────────────────────────────────────────────

def test_embedding_hook_flags_semantic_duplicate():
    # Two texts the lexical check would NOT catch, but identical embeddings do.
    qs = [_q("What converges parallel light rays?"),
          _q("Which lens brings distant rays to a focus?")]
    def embed(texts):
        return [[1.0, 0.0], [1.0, 0.0]]     # identical vectors
    r = dg.dedup(qs, embed_fn=embed, embed_threshold=0.95)
    assert len(r.kept) == 1
    assert r.removed[0]["reason"] == "embedding"


def test_embedding_failure_falls_back_to_lexical():
    qs = [_q("alpha one"), _q("beta two")]
    def embed(texts):
        raise RuntimeError("embed service down")
    r = dg.dedup(qs, embed_fn=embed)       # must not raise
    assert len(r.kept) == 2


# ── Phase 2B: a realistic rotation paper keeps (almost) its full count ────

def test_fundamentals_50_paper_mostly_unique_after_variety():
    """After Phase 2B variety, a 50-MCQ Fundamentals paper should survive the
    dedup gate near-intact. We allow a small residue (<=2) of genuine collisions
    which the generation top-up loop refills in the live path."""
    from app.services import accountancy_scenarios as sc
    worst = 0
    for seed in range(20):
        qs = sc.generate_numeric_mcqs(50, seed=seed, difficulty="hard")
        kept = len(dg.dedup(qs).kept)
        worst = max(worst, 50 - kept)
    assert worst <= 2, f"a seed lost {worst} questions to dedup (variety regressed)"
