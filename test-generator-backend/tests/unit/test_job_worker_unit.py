"""Job worker pieces that need no database: the kick route's secret, file validation, review reasons, chunk merge,
log masking. The queue itself is tested on real SQL in tests/integration/test_job_worker_pg.py."""

import io
import logging
import re
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.log_mask import PIIMaskFilter, mask_email, mask_name, mask_phone
from app.services.answer_checker import portal_result, preprocess, upload_validation as uv

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "answer_checker"
SHEET = (FIXTURES / "answer_sheet_messy.pdf").read_bytes()


# ── POST /internal/jobs/run ──────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.routers import internal_jobs_router

    runs = []

    class W:
        def run(self):
            runs.append(1)
            return {"jobs": 0}

    monkeypatch.setattr(internal_jobs_router, "_worker", lambda: W())
    c = TestClient(app)
    c.runs = runs
    return c


@pytest.mark.parametrize("headers", [{}, {"X-Job-Secret": ""}, {"X-Job-Secret": "wrong"},
                                     {"X-Job-Secret": "s3cret "}, {"Authorization": "Bearer s3cret"}])
def test_kick_without_the_right_secret_is_401(client, monkeypatch, headers):
    monkeypatch.setenv("JOB_SECRET", "s3cret")
    r = client.post("/internal/jobs/run", headers=headers, json={})
    assert r.status_code == 401 and client.runs == []


def test_kick_is_refused_when_job_secret_is_not_configured(client, monkeypatch):
    monkeypatch.delenv("JOB_SECRET", raising=False)
    assert client.post("/internal/jobs/run", headers={"X-Job-Secret": ""}).status_code == 401
    assert client.runs == []


def test_kick_with_secret_is_202_and_runs_the_worker(client, monkeypatch):
    monkeypatch.setenv("JOB_SECRET", "s3cret")
    r = client.post("/internal/jobs/run", headers={"X-Job-Secret": "s3cret", "Content-Type": "application/json"},
                    content=b"{}")
    assert r.status_code == 202 and r.json() == {"accepted": True} and client.runs == [1]


def test_only_the_exact_path_is_mounted(client, monkeypatch):
    monkeypatch.setenv("JOB_SECRET", "s3cret")
    assert client.post("/api/v1/internal/jobs/run", headers={"X-Job-Secret": "s3cret"}).status_code == 404


def test_regrade_needs_a_teacher_token(client):
    r = client.post("/internal/regrade", json={"assignment_id": "00000000-0000-0000-0000-000000000001",
                                               "submission_ids": ["00000000-0000-0000-0000-000000000002"]})
    assert r.status_code == 401


# ── validation ───────────────────────────────────────────────────────

def _entry(order, ext, mime, path=None):
    return {"storage_path": path or f"i/a/s/{order}.{ext}", "mime": mime, "page_order": order, "sha256": "x" * 64}


def _jpeg(exif=None) -> bytes:
    from PIL import Image
    im = Image.new("RGB", (300, 400), "white")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", **({"exif": exif} if exif else {}))
    return buf.getvalue()


def test_sniff_reads_magic_bytes_not_names():
    assert uv.sniff(SHEET) == "pdf"
    assert uv.sniff(_jpeg()) == "jpeg"
    assert uv.sniff(b"\x89PNG\r\n\x1a\n....") == "png"
    assert uv.sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert uv.sniff(b"\x00\x00\x00\x18ftypheic\x00\x00") == "heic"
    assert uv.sniff(b"MZ\x90\x00 not a page") is None


def test_pdf_named_jpg_is_rejected_per_file():
    e = [_entry(1, "jpg", "image/jpeg"), _entry(2, "jpg", "image/jpeg")]
    v = uv.validate(e, {e[0]["storage_path"]: _jpeg(), e[1]["storage_path"]: SHEET}, ["image/jpeg"], 30)
    assert [p.page_order for p in v.problems] == [2]
    assert "Page 2" in v.problems[0].message and ".jpg" in v.problems[0].message and "PDF" in v.problems[0].message


def test_password_pdf_and_owner_only_pdf():
    doc = pymupdf.open(stream=SHEET, filetype="pdf")
    locked = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="1", owner_pw="2")
    owner_only = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="2", permissions=0)
    e = [_entry(1, "pdf", "application/pdf")]
    v = uv.validate(e, {e[0]["storage_path"]: locked}, ["application/pdf"], 30)
    assert v.problems[0].code == "bad_pdf" and "password" in v.problems[0].message
    v = uv.validate(e, {e[0]["storage_path"]: owner_only}, ["application/pdf"], 30)
    assert v.ok and v.total_pages == 7            # restrictions only: opens fine, graded


def test_page_cap_is_the_assignment_setting_and_never_above_30():
    doc = pymupdf.open()
    for _ in range(31):
        doc.new_page()
    e = [_entry(1, "pdf", "application/pdf")]
    blobs = {e[0]["storage_path"]: doc.tobytes()}
    assert "max 30 allowed" in uv.validate(e, blobs, ["application/pdf"], 50).summary()
    assert "max 10 allowed" in uv.validate(e, blobs, ["application/pdf"], 10).summary()


def test_empty_missing_corrupt_and_not_allowed_files():
    e = [_entry(1, "jpg", "image/jpeg"), _entry(2, "jpg", "image/jpeg"), _entry(3, "pdf", "application/pdf"),
         _entry(4, "png", "image/png")]
    png = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (10, 10)).save(png, format="PNG")
    blobs = {e[0]["storage_path"]: b"", e[2]["storage_path"]: b"%PDF-1.7 broken",
             e[3]["storage_path"]: png.getvalue()}
    v = uv.validate(e, blobs, ["image/jpeg", "application/pdf"], 30)
    assert [(p.page_order, p.code) for p in v.problems] == [(1, "empty_file"), (2, "not_uploaded"),
                                                            (3, "bad_pdf"), (4, "bad_format")]


def test_exif_and_gps_are_removed_and_photo_turned_upright():
    from PIL import Image
    exif = Image.Exif()
    exif[0x0112] = 6                                   # orientation: rotate 90
    exif[0x8825] = {1: "N", 2: (28.0, 36.0, 0.0)}
    e = [_entry(1, "jpg", "image/jpeg")]
    v = uv.validate(e, {e[0]["storage_path"]: _jpeg(exif)}, ["image/jpeg"], 30)
    f = v.files[0]
    assert f.exif_removed
    with Image.open(io.BytesIO(f.data)) as im:
        assert not im.getexif() and im.size == (400, 300)
    clean = uv.validate(e, {e[0]["storage_path"]: _jpeg()}, ["image/jpeg"], 30).files[0]
    assert not clean.exif_removed                      # nothing to strip → bytes untouched


def test_heic_is_read_by_stage_a(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    from PIL import Image
    pillow_heif.register_heif_opener()
    page = pymupdf.open(stream=SHEET, filetype="pdf")[0].get_pixmap(dpi=100).tobytes("png")
    path = tmp_path / "page.heic"
    Image.open(io.BytesIO(page)).save(path, format="HEIF")
    pages = preprocess.normalize_sources([str(path)])
    assert len(pages) == 1 and not pages[0].blank and pages[0].jpeg[:3] == b"\xff\xd8\xff"
    e = [_entry(1, "heic", "image/heic")]
    assert uv.validate(e, {e[0]["storage_path"]: path.read_bytes()}, ["image/heic"], 30).ok


# ── review reasons / chunk merge ─────────────────────────────────────

def _q(q_no, marks, of, conf="high", **kw):
    return {"q_no": q_no, "marks_awarded": marks, "max_marks": of, "confidence": conf, "counted": True,
            "not_attempted": False, "legibility_issue": False, "review_reasons": [], "subpart_grades": [], **kw}


def test_review_reasons_per_question():
    rubric = {"questions": [{"q_no": "3", "review_min_confidence": "high"}]}
    result = {"graded_questions": [
        _q("1", 2, 2), _q("2", 0, 2, not_attempted=True), _q("3", 3, 5, conf="medium"),
        _q("4", 1, 2, legibility_issue=True), _q("5", 0, 3, not_attempted=True, counted=False),
        _q("6", 1, 3, review_reasons=["Marking split was estimated (not in the answer key)"])],
        "sheet_flags": [{"code": "BLANK_PAGE"}], "readability_score": "poor"}
    reasons, conf, by_q = portal_result.review(result, rubric)
    assert reasons == ["empty_answer", "illegible", "low_confidence", "key_missing", "blank_pages", "bad_scan"]
    assert by_q == {"2": ["empty_answer"], "3": ["low_confidence"], "4": ["illegible"], "6": ["key_missing"]}
    assert "5" not in by_q and 0 < conf < 1        # unused OR alternative is ignored


def test_clean_copy_has_no_reasons():
    reasons, conf, _ = portal_result.review({"graded_questions": [_q("1", 2, 2), _q("2", 1, 2, conf="medium")]}, {})
    assert reasons == [] and conf == round((0.95 + 0.75) / 2, 3)


def test_merge_chunks_by_question_with_global_pages():
    rubric = {"total_marks": 10, "questions": [{"q_no": "1"}, {"q_no": "2"}, {"q_no": "3"}]}
    c1 = {"graded_questions": [_q("1", 3, 4, page_index=0), _q("2", 1, 3, page_index=9),
                               _q("3", 0, 3, not_attempted=True, page_index=None)], "max_marks": 10,
          "sheet_flags": [{"code": "BLANK_PAGE", "page": 2}], "readability_score": "good"}
    c2 = {"graded_questions": [_q("1", 0, 4, not_attempted=True, page_index=None), _q("2", 2, 3, page_index=0),
                               _q("3", 2, 3, page_index=4)], "max_marks": 10, "sheet_flags": [],
          "readability_score": "average"}
    m = portal_result.merge_chunks([{"result": c1, "pages": 10}, {"result": c2, "pages": 10}], rubric)
    by = {g["q_no"]: g for g in m["graded_questions"]}
    assert by["1"]["marks_awarded"] == 3 and by["1"]["page_index"] == 0
    assert by["2"]["marks_awarded"] == 2 and by["2"]["page_index"] == 10 and by["2"]["needs_review"]   # spans
    assert by["3"]["marks_awarded"] == 2 and by["3"]["page_index"] == 14
    assert m["total_marks_awarded"] == 7 and m["percentage"] == 70.0
    assert m["sheet_flags"][0]["page"] == 2 and m["readability_score"] == "average"
    assert "low_confidence" in portal_result.review(m, rubric)[0]


# ── logs ─────────────────────────────────────────────────────────────

def test_masking_helpers():
    assert mask_name("Asha Verma") == "A*** V***"
    assert mask_email("asha.verma@gmail.com") == "a***@gmail.com"
    assert mask_phone("+91 98765 43210") == "********3210"


def test_pii_filter_masks_emails_and_phones_in_log_lines(caplog):
    log = logging.getLogger("test.pii")
    log.addFilter(PIIMaskFilter())
    with caplog.at_level(logging.INFO, logger="test.pii"):
        log.info("student %s phone %s", "asha.verma@gmail.com", "9876543210")
        log.info("parent +91-98765-43210, id 12345678901234")
    text = caplog.text
    assert "asha.verma@" not in text and "a***@gmail.com" in text
    assert "9876543210" not in text and "98765-43210" not in text and "******3210" in text
    assert "12345678901234" in text                   # long ids are not phone numbers


# ── every string the worker writes for people is short plain English ──
_HINGLISH = re.compile(r"\b(hai|hain|nahi|karo|karein|baar|apne|aap|mein|ki|ka|ke|dobara|khaali|pehle|hogi|gayi|"
                       r"raha|abhi|kal|aaj|lag|saaf|badli|wali|chuno)\b", re.I)


def _user_strings() -> list:
    from app.services import job_worker, regrade
    doc = pymupdf.open()
    for _ in range(31):
        doc.new_page()
    locked = pymupdf.open(stream=SHEET, filetype="pdf").tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                                                 user_pw="1", owner_pw="2")
    kinds = [("jpg", "image/jpeg"), ("jpg", "image/jpeg"), ("pdf", "application/pdf"), ("pdf", "application/pdf"),
             ("png", "image/png"), ("jpg", "image/jpeg"), ("jpg", "image/jpeg")]
    e = [_entry(i, ext, mime) for i, (ext, mime) in enumerate(kinds, 1)]
    blobs = {e[0]["storage_path"]: b"", e[1]["storage_path"]: SHEET, e[2]["storage_path"]: b"%PDF-1.7 broken",
             e[3]["storage_path"]: locked, e[4]["storage_path"]: b"\x89PNG\r\n\x1a\n", e[5]["storage_path"]: b"MZ??"}
    probs = [p.message for p in uv.validate(e, blobs, ["image/jpeg", "application/pdf"], 30).problems]
    probs += [p.message for p in uv.validate([_entry(1, "pdf", "application/pdf")], {"i/a/s/1.pdf": doc.tobytes()},
                                             ["application/pdf"], 30).problems]
    bad = [uv.bad_file_message(preprocess.BadFile(m)) for m in
           ("Every page of this copy is blank.", "Too many pages in one copy, max 9.", "password", "other")]
    return [*job_worker.MSG.values(), *probs, *bad, regrade.UNPUBLISH_REASON]


def test_user_facing_strings_are_short_plain_english():
    strings = _user_strings()
    assert len(strings) > 20
    for s in strings:
        assert not _HINGLISH.search(s), s
        assert s.isascii(), s
        assert len(s) <= 110, s
