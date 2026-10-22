"""Inline request budget: downscale 2000 → 1600 → 1280, then a clear error — never a Gemini call when over."""

import os

import pymupdf
import pytest

from app.services import smart_assignment_service as svc
from app.services.answer_checker import copy_check, preprocess as pp
from tests.conftest import FIXTURES


def _no_gemini(*a, **k):
    raise AssertionError("Gemini must not be called")


def _photo_sheet(tmp_path, n: int, noisy: bool) -> list:
    """n phone-photo pages (JPEG). noisy=True: random pixels, the worst case for JPEG size."""
    paths = []
    src = pymupdf.open(FIXTURES / "answer_sheet_messy.pdf")
    for i in range(n):
        if noisy:
            w, h = 1000, 1400
            pix = pymupdf.Pixmap(pymupdf.csRGB, w, h, os.urandom(w * h * 3), False)
        else:   # a real handwritten page at phone-photo resolution (3000 px long edge)
            page = src[i % src.page_count]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(3000 / page.rect.height, 3000 / page.rect.height))
        path = tmp_path / f"p{i:02d}.jpg"
        path.write_bytes(pix.tobytes("jpeg", jpg_quality=90))
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def real20(tmp_path_factory):
    return _photo_sheet(tmp_path_factory.mktemp("real20"), 20, noisy=False)


def test_hard_cap_is_30_pages_without_gemini(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "grade_sheet", _no_gemini)
    assert pp.MAX_PAGES == 30
    paths = []
    for i in range(31):                       # small pages: this test is about the count, not the size
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 560), False)
        pix.clear_with(240)
        path = tmp_path / f"p{i:02d}.jpg"
        path.write_bytes(pix.tobytes("jpeg"))
        paths.append(str(path))
    with pytest.raises(pp.BadFile, match="Too many pages in one copy, max 30"):
        copy_check.grade_copy(paths, rubric={})


def test_20_real_photo_pages_fit_the_budget_at_full_size(real20):
    norm = copy_check.fit_pages(real20)
    size = sum(svc.b64_size(len(p.jpeg)) for p in norm)
    assert len(norm) == 20 and max(max(p.w, p.h) for p in norm) == 2000
    assert size <= svc.INLINE_FILES_BUDGET
    print(f"20 real photo pages at 2000 px: {size / 1e6:.1f} MB encoded")


def test_downscales_before_giving_up(real20, monkeypatch):
    paths = real20
    full = pp.normalize_sources(paths, 20, 2000)
    at = {e: sum(svc.b64_size(len(p.jpeg)) for p in pp.downscale(full, e)) for e in (2000, 1600)}
    monkeypatch.setattr(svc, "INLINE_FILES_BUDGET", (at[2000] + at[1600]) // 2)     # 2000 too big, 1600 fits
    norm = copy_check.fit_pages(paths)
    assert max(max(p.w, p.h) for p in norm) == 1600


def test_20_noise_pages_over_budget_clear_error_no_gemini(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "grade_sheet", _no_gemini)
    with pytest.raises(pp.BadFile, match=r"Too many pages in one copy, max \d+") as e:
        copy_check.grade_copy(_photo_sheet(tmp_path, 20, noisy=True), rubric={})
    fits = int(str(e.value).split("max ")[1].split(".")[0])
    assert 1 <= fits < 20


def test_base64_size():
    assert svc.b64_size(0) == 0 and svc.b64_size(1) == 4 and svc.b64_size(3) == 4 and svc.b64_size(4) == 8
