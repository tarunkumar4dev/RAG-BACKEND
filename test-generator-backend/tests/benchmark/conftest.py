"""Benchmark suite config.

Auto-marks every test under tests/benchmark/ with `benchmark`, so CI can run
the quality gate on its own:

    pytest -m benchmark          # just the gate
    pytest -m "not benchmark"    # everything else

Marking here rather than in each module means new benchmark test files are
picked up without having to remember the marker.
"""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/benchmark" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.benchmark)
