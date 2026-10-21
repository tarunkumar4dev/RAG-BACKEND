"""Real student copies: originals are never committed, the anonymiser blacks out the header and strips identifiers."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pymupdf

from tests.conftest import FIXTURES, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import anonymise_copies as anon  # noqa: E402


def test_only_readme_and_anonymised_are_tracked():
    tracked = subprocess.run(["git", "ls-files", "tests/fixtures/real"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    bad = [p for p in tracked if p != "tests/fixtures/real/README.md"
           and not p.startswith("tests/fixtures/real/anonymised/")]
    assert not bad, f"real student files tracked outside anonymised/: {bad}"


def test_raw_folder_is_ignored():
    r = subprocess.run(["git", "check-ignore", "-q", "tests/fixtures/real/raw/anything.pdf"], cwd=ROOT)
    assert r.returncode == 0


def test_anonymiser(tmp_path):
    src = tmp_path / "raw" / "Rohit Kumar 10B"
    (src / "pages").mkdir(parents=True)
    shutil.copy(FIXTURES / "answer_sheet_messy.pdf", src / "pages" / "copy.pdf")
    (src / "test.json").write_text(json.dumps({"test_id": "t1", "teacher_name": "Mrs X"}))
    (src / "expected.json").write_text(json.dumps({"max_marks": 40, "total": 35, "student_name": "Rohit Kumar",
                                                   "questions": [{"q": "1", "max": 1, "marks": 1, "roll_no": "23"}]}))
    (src / "redact.json").write_text(json.dumps([{"page": 3, "x0": 0.0, "y0": 0.8, "x1": 1.0, "y1": 1.0}]))

    sys.argv = ["anonymise_copies.py", "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "anon")]
    anon.main()

    out = tmp_path / "anon" / "copy_01"
    pages = sorted((out / "pages").glob("*.jpg"))
    assert [p.name for p in pages][:2] == ["p01.jpg", "p02.jpg"] and len(pages) == 7
    expected = json.loads((out / "expected.json").read_text())
    assert "student_name" not in expected and "roll_no" not in expected["questions"][0]
    assert "teacher_name" not in json.loads((out / "test.json").read_text())
    # the original folder name (a student's name) appears only in the local mapping
    assert json.loads((tmp_path / "raw" / "_mapping.json").read_text()) == {"copy_01": "Rohit Kumar 10B"}
    assert "Rohit" not in "".join(p.name for p in out.rglob("*"))

    # page 1 header band is solid grey; page 2's isn't touched
    p1 = pymupdf.Pixmap(str(pages[0]))
    assert p1.pixel(p1.width // 2, int(p1.height * 0.10))[:3] == p1.pixel(p1.width // 4, int(p1.height * 0.05))[:3]
    assert abs(p1.pixel(p1.width // 2, int(p1.height * 0.10))[0] - 89) <= 6       # 0.35 grey
    p3 = pymupdf.Pixmap(str(pages[2]))
    assert abs(p3.pixel(p3.width // 2, int(p3.height * 0.9))[0] - 89) <= 6        # redact.json box
    log = json.loads((out / "anonymise.json").read_text())
    assert log["blacked_out"][1]["boxes"] == []
