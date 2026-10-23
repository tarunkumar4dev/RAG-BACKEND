"""Stage A: page images, blank detection, bad files, rotation, reorder, position remapping."""

import pymupdf
import pytest

from app.services.answer_checker import preprocess as pp
from tests.conftest import FIXTURES


def _ruled_pdf(path, pages=1, writing=False):
    doc = pymupdf.open()
    for _ in range(pages):
        pg = doc.new_page(width=595, height=842)
        pg.draw_rect(pg.rect, color=None, fill=(0.96, 0.95, 0.92))
        for x in range(0, 120, 4):          # shadow along the left edge
            pg.draw_rect(pymupdf.Rect(x, 0, x + 4, 842), color=None, fill=(0, 0, 0), fill_opacity=0.35 * (1 - x / 120))
        for y in range(80, 820, 20):
            pg.draw_line((20, y), (575, y), color=(0.7, 0.78, 0.9), width=0.6)
        pg.draw_line((60, 0), (60, 842), color=(0.85, 0.3, 0.3), width=0.8)
        pg.draw_circle((400, 600), 60, color=(0.75, 0.65, 0.5), width=4, stroke_opacity=0.5)   # stain ring
        if writing:
            pg.insert_text((90, 120), "(c)", fontsize=14, color=(0.1, 0.15, 0.45))
    doc.save(path)
    doc.close()
    return str(path)


def test_fixture_pages():
    pages = pp.normalize_sources([str(FIXTURES / "answer_sheet_messy.pdf")])
    assert len(pages) == 7
    for i, p in enumerate(pages):
        assert p.source_index == i
        assert max(p.w, p.h) <= pp.MAX_LONG_EDGE
        assert (p.w, p.h) == (1240, 1754)            # A4 at 150 dpi
        assert p.jpeg[:3] == b"\xff\xd8\xff"
        assert not p.blank


def test_blank_detection_ignores_lines_shadow_and_stain(tmp_path):
    blank = pp.normalize_sources([_ruled_pdf(tmp_path / "b.pdf")])[0]
    assert blank.blank
    # a single short answer like "(c)" must never be treated as blank (blank pages are not graded)
    tiny = pp.normalize_sources([_ruled_pdf(tmp_path / "t.pdf", writing=True)])[0]
    assert not tiny.blank


def test_photo_is_not_upscaled(tmp_path):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 800, 600), False)
    pix.clear_with(240)
    path = tmp_path / "small.png"
    pix.save(str(path))
    p = pp.normalize_sources([str(path)])[0]
    assert (p.w, p.h) == (800, 600)


@pytest.mark.parametrize("name,data,ext", [
    ("empty", b"", ".pdf"),
    ("text", b"hello", ".txt"),
    ("corrupt", b"%PDF-1.4 not really a pdf", ".pdf"),
])
def test_bad_files_raise_badfile(tmp_path, name, data, ext):
    path = tmp_path / f"{name}{ext}"
    path.write_bytes(data)
    with pytest.raises(pp.BadFile):
        pp.normalize_sources([str(path)])


def test_password_pdf(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "locked.pdf"
    doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    with pytest.raises(pp.BadFile, match="password"):
        pp.normalize_sources([str(path)])


def test_page_limit(tmp_path):
    with pytest.raises(pp.BadFile, match="Too many pages"):
        pp.normalize_sources([_ruled_pdf(tmp_path / "many.pdf", pages=4)], max_pages=3)


def test_rotate_clockwise():
    doc = pymupdf.open()
    pg = doc.new_page(width=200, height=100)
    pg.draw_rect(pymupdf.Rect(0, 0, 40, 40), color=None, fill=(1, 0, 0))     # red block top-left
    jpeg = pg.get_pixmap().tobytes("jpeg")
    j, w, h = pp.rotate_jpeg(jpeg, 200, 100, 90)
    pix = pymupdf.Pixmap(j)
    assert (w, h) == (100, 200)
    assert pix.pixel(w - 5, 5)[0] > 200 and pix.pixel(w - 5, 5)[1] < 60       # now top-right
    j, w, h = pp.rotate_jpeg(jpeg, 200, 100, 180)
    pix = pymupdf.Pixmap(j)
    assert pix.pixel(w - 5, h - 5)[0] > 200 and pix.pixel(5, 5)[1] > 200      # bottom-right, top-left white


def _pages(n):
    return [pp.NormPage(source_index=i, jpeg=b"", w=10, h=20, blank=False) for i in range(n)]


def _meta(numbers, rotations=None):
    rotations = rotations or [0] * len(numbers)
    return [{"page_index": i, "student_page_number": n, "rotation_needed": 0, "blank": False}
            if not rotations[i] else
            {"page_index": i, "student_page_number": n, "rotation_needed": rotations[i], "blank": False}
            for i, n in enumerate(numbers)]


def test_reorder_by_written_page_numbers(monkeypatch):
    monkeypatch.setattr(pp, "rotate_jpeg", lambda j, w, h, r: (j, h, w) if r in (90, 270) else (j, w, h))
    pages = _pages(4)
    ordered, old_to_new, flags = pp.finalize_pages(pages, [0, 1, 2, 3], _meta(["1", "2", "4", "3"], [0, 0, 180, 0]))
    assert [p.source_index for p in ordered] == [0, 1, 3, 2]
    assert old_to_new == [0, 1, 3, 2]
    codes = [f["code"] for f in flags]
    assert codes == ["PAGES_REORDERED", "PAGE_ROTATED"]
    assert flags[1]["page"] == 3 and "Page 4" in flags[1]["message"] and "upside down" in flags[1]["message"]


@pytest.mark.parametrize("numbers", [["1", "1", "2"], ["1", None, "3"], [None, None, None]])
def test_no_reorder_when_numbers_are_missing_or_duplicated(numbers):
    ordered, old_to_new, flags = pp.finalize_pages(_pages(3), [0, 1, 2], _meta(numbers))
    assert old_to_new == [0, 1, 2]
    assert not [f for f in flags if f["code"] == "PAGES_REORDERED"]


def test_blank_page_keeps_its_place_and_is_flagged():
    pages = _pages(4)
    pages[2].blank = True                     # uploaded right after the sheet marked "1"; not sent to the grader
    ordered, old_to_new, flags = pp.finalize_pages(pages, [0, 1, 3], _meta(["2", "1", "3"]))
    assert [p.source_index for p in ordered] == [1, 2, 0, 3]    # blank stays right after the page before it
    assert old_to_new == [2, 0, 3]
    assert {"code": "BLANK_PAGE", "page": 1, "message": "Page 2 is blank."} in flags


def test_remap_positions_uses_earliest_subpart_on_final_pages():
    result = {"graded_questions": [
        {"q_no": "7", "page_index": 0, "y_start": 522, "subpart_grades": [
            {"sub_id": "7i", "page_index": 0, "y_start": 522}, {"sub_id": "7ii", "page_index": 0, "y_start": 493}]},
        {"q_no": "12", "page_index": 2, "y_start": 132, "subpart_grades": []},
        {"q_no": "X", "page_index": 2, "y_start": 900, "subpart_grades": [
            {"sub_id": "Xa", "page_index": 2, "y_start": 900}, {"sub_id": "Xb", "page_index": 3, "y_start": 100}]},
        {"q_no": "bad", "page_index": 9, "y_start": 10, "subpart_grades": []},
    ]}
    pp.remap_positions(result, [0, 1, 3, 2])
    q7, q12, qx, bad = result["graded_questions"]
    assert (q7["page_index"], q7["y_start"]) == (0, 493)
    assert (q12["page_index"], q12["y_start"]) == (3, 132)
    assert (qx["page_index"], qx["y_start"]) == (2, 100)          # Xb is on the earlier FINAL page
    assert (bad["page_index"], bad["y_start"]) == (None, None)
