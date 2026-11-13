import pytest

from agent.text import normalize_section, strip_honorifics
from agent.tools.insights import ResolveEntityArgs, resolve_entity
from _agent_ids import B10A, SHARMA, VERMA


@pytest.mark.parametrize("name", ["Sharma ma'am", "sharma mam", "Priya Sharma madam", "Sharma ma'am ki"])
def test_sharma_variants(admin_ctx, name):
    r = resolve_entity(admin_ctx, ResolveEntityArgs(kind="teacher", name=name))
    assert r["status"] == "ok" and r["matches"] == [{"id": str(SHARMA), "label": "Priya Sharma"}]


def test_verma_sir(admin_ctx):
    r = resolve_entity(admin_ctx, ResolveEntityArgs(kind="teacher", name="Verma sir"))
    assert r["status"] == "ok" and r["matches"][0]["id"] == str(VERMA)


@pytest.mark.parametrize("name", ["10th A", "10-A", "X-A", "10 a", "Xa", "class 10A"])
def test_section_variants(admin_ctx, name):
    r = resolve_entity(admin_ctx, ResolveEntityArgs(kind="section", name=name))
    assert r["status"] == "ok" and r["matches"] == [{"id": str(B10A), "label": "10A"}]


def test_rahul_kumar_ambiguous(admin_ctx):
    r = resolve_entity(admin_ctx, ResolveEntityArgs(kind="student", name="Rahul Kumar"))
    assert r["status"] == "ambiguous"
    assert sorted(m["label"] for m in r["matches"]) == ["Rahul Kumar (10A)", "Rahul Kumar (10B)"]


def test_unknown_and_decoy(admin_ctx):
    assert resolve_entity(admin_ctx, ResolveEntityArgs(kind="student", name="Nobody Xyz"))["status"] == "not_found"
    assert resolve_entity(admin_ctx, ResolveEntityArgs(kind="student", name="Gaurav Sood"))["status"] == "not_found"


def test_text_helpers():
    assert strip_honorifics("Sharma ma'am ki") == "Sharma"
    assert strip_honorifics("Mr. Verma ji") == "Verma"
    assert strip_honorifics("Mrs Kiran") == "Kiran"  # 'ki' inside a word is kept
    assert normalize_section("10th") == "10"
    assert normalize_section("XII-B") == "12B"
    assert normalize_section("IX") == "9"
