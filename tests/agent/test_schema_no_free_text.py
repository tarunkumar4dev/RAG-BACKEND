import importlib

import pytest
from pydantic import BaseModel

from agent.tools import insights

BANNED = {"sql", "query", "filter", "where", "raw"}


def _models():
    models = [m for m in vars(insights).values()
              if isinstance(m, type) and issubclass(m, BaseModel) and m.__name__.endswith("Args")
              and not m.__name__.startswith("_")]
    try:
        reg = importlib.import_module("agent.tools.registry")
    except ImportError:
        reg = None
    if reg is not None:
        for v in vars(reg).values():
            if isinstance(v, type) and issubclass(v, BaseModel) and v.__name__.endswith("Args") and v not in models:
                models.append(v)
            if isinstance(v, dict):
                models += [m for m in v.values() if isinstance(m, type) and issubclass(m, BaseModel) and m not in models]
    return models


def _string_props(schema, defs):
    for name, prop in schema.get("properties", {}).items():
        yield name, prop, defs


def _check_prop(name, prop, defs):
    if "$ref" in prop:
        return _check_schema(defs[prop["$ref"].split("/")[-1]], defs)
    for alt in prop.get("anyOf", []) + prop.get("oneOf", []):
        if alt.get("type") == "null":
            continue
        _check_prop(name, alt, defs)
    if prop.get("type") == "string":
        assert "enum" in prop or "const" in prop or prop.get("format") in {"uuid", "date", "date-time"} \
            or "maxLength" in prop, f"free-text string field {name!r}: {prop}"
    if prop.get("type") == "array":
        _check_prop(name, prop.get("items", {}), defs)
    if prop.get("type") == "object" and "properties" in prop:
        _check_schema(prop, defs)


def _check_schema(schema, defs):
    for name, prop in schema.get("properties", {}).items():
        assert name.lower() not in BANNED, f"banned field {name!r}"
        _check_prop(name, prop, defs)


@pytest.mark.parametrize("model", _models(), ids=lambda m: m.__name__)
def test_no_free_text(model):
    schema = model.model_json_schema()
    assert schema.get("additionalProperties") is False
    _check_schema(schema, schema.get("$defs", {}))


def test_models_found():
    assert len(_models()) >= 4
