"""
ncert_cleanup.py — read-time cleanup + quality checks for rows of `ncert_questions`.

The NCERT extraction left four kinds of damage in the table:
  1. Option labels inside the options ("(a) atomic number"), so the UI/PDF printed "A (a) …".
  2. The same options again at the end of question_text, so they were printed twice.
  3. PDF font glyph names instead of characters ("/square6Take …", "AlCl3/unif02b/unif0ae …").
  4. Content that was a drawing or a special font (reactions, structures, statement lists) is
     simply missing: "… as a result of the following reaction?" with nothing after it.

`clean_question()` fixes 1–3 on the fly (used by /ncert-questions, so the builder preview and
every export get the same text). `quality_issue()` detects 4 (and garbled text that can't be
repaired); scripts/ncert_quality_pass.py stores the result in ncert_questions.quality_flag so
the library can hide those rows.
"""

import re
from typing import Optional

# ── 3. Glyph junk ────────────────────────────────────────────────────────────
# "/square6" and a lone "n" are the bullet glyphs of NCERT's activity boxes.
_BULLET_GLYPH = re.compile(r"/square\d\s*")
_N_BULLET = re.compile(r"(?:(?<=^)|(?<=\s))n(?=[A-Z][a-z]{2,})")
# "/unif028" = "(" — the font's own glyph names. The characters also come out of order
# ("AlCl Cl AlCl34/unif02b/unif0ae…"), so a row that has them is unreadable, not repairable.
_UNI_GLYPH = re.compile(r"/uni[0-9a-fA-F]{3,4}")


def clean_glyphs(text: str) -> str:
    if not text:
        return text
    text = _BULLET_GLYPH.sub("", text)
    text = _N_BULLET.sub("", text)
    return text


# ── 1. Option labels ─────────────────────────────────────────────────────────
_ROMAN = ["i", "ii", "iii", "iv", "v", "vi"]
# "(a) x", "(a)x", "a) x", "a. x", "(A) x", "A) x", "(c ) x", "(b )x"
_OPT_LABEL = re.compile(r"^\s*\(?\s*([a-hA-H])\s*\)\s*|^\s*([a-hA-H])[.)]\s+")


def _letter_label(opt: str) -> Optional[str]:
    m = _OPT_LABEL.match(opt or "")
    if not m:
        return None
    return (m.group(1) or m.group(2)).lower()


_ROMAN_LABEL = re.compile(r"^\s*\(\s*(i{1,3}|iv|v|vi)\s*\)\s*", re.IGNORECASE)
_ROMAN_REF = re.compile(r"\(\s*(?:i{1,3}|iv|v|vi)\s*\)", re.IGNORECASE)


def _roman_labels_strippable(options: list) -> bool:
    """"(i) … (iv) …" in order, and no option points at another one ("(iv) (i) & (iii)")."""
    labels = []
    for o in options:
        m = _ROMAN_LABEL.match(o)
        if not m:
            return False
        labels.append(m.group(1).lower())
        if _ROMAN_REF.search(o[m.end():]):
            return False
    return labels == _ROMAN[: len(options)]


def strip_option_labels(options: list) -> list:
    """Drop the option's own label ("(a)", "a.", "A)", "(i)") — the UI and PDFs print A–D.

    Only when every option carries the next label in order. Roman labels stay when an option
    refers to the others ("(iv) (i) & (iii)"), since renaming them would break that reference.
    """
    if not options or not all(isinstance(o, str) for o in options):
        return options
    letters = [_letter_label(o) for o in options]
    if letters == [chr(ord("a") + i) for i in range(len(options))]:
        return [_OPT_LABEL.sub("", o, count=1).strip() for o in options]
    if _roman_labels_strippable(options):
        return [_ROMAN_LABEL.sub("", o, count=1).strip() for o in options]
    return options


def _answer_label_to_letter(answer: str, n_options: int) -> str:
    """"(ii) Sorting" → "(b) Sorting" once the options are shown as A–D."""
    m = _ROMAN_LABEL.match(answer or "")
    if not m:
        return answer
    idx = _ROMAN.index(m.group(1).lower())
    if idx >= n_options:
        return answer
    return f"({chr(ord('a') + idx)}) " + answer[m.end():]


# ── 2. Options repeated inside question_text ─────────────────────────────────
def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _label_positions(text: str, label: str) -> list:
    """Start offsets of "(a)" / "a)" / "a." / "(i)" written as a list label in text."""
    lab = re.escape(label)
    pat = re.compile(rf"(?:(?<=^)|(?<=[\s:?.]))\(?\s*{lab}\s*[).](?=\s|\S)", re.IGNORECASE)
    return [m.start() for m in pat.finditer(text)]


def strip_inline_options(text: str, options: list) -> str:
    """Cut the options off the end of question_text when they are the same as `options`.

    Only cut when every option body appears, in order, after the first label — a statement
    list ("(i) … (ii) …") that the options merely refer to is never cut.
    """
    if not text or not options or len(options) < 2:
        return text
    bodies = [_alnum(_OPT_LABEL.sub("", o, count=1) if isinstance(o, str) else "") for o in options]
    if not all(bodies):
        return text
    first_labels = ["a", "i"]
    norm_text = text
    for lab in first_labels:
        for pos in _label_positions(norm_text, lab):
            tail = _alnum(norm_text[pos:])
            # every body in order, and the tail is (almost) nothing but the options
            idx = 0
            ok = True
            for b in bodies:
                found = tail.find(b[:40], idx)
                if found < 0:
                    ok = False
                    break
                idx = found + len(b[:40])
            if not ok:
                continue
            labels_len = 4 * len(options)
            if len(tail) > sum(len(b) for b in bodies) + labels_len + 30:
                continue  # something else follows the options (e.g. "Give reasons…")
            stem = norm_text[:pos].rstrip()
            if len(_alnum(stem)) >= 10:
                return stem
    return text


# ── 4. Missing / unreadable content ──────────────────────────────────────────
_PAIR = re.compile(r"\(([a-h]|i{1,3}|iv|v|vi)\)\s*\(([a-h]|i{1,3}|iv|v|vi)\)")
_PLACEHOLDER = re.compile(r"\bimage of (the )?(structure|figure|diagram|compound)", re.IGNORECASE)
# "… the following reaction?" / "… categorise the following molecules/ions as X or Y:" at the end.
# Only words may follow — "following (i) 52 moles of Ar …" has its items.
_ENDS_WITH_FOLLOWING = re.compile(
    r"\b(?:the|of|in|for|from|these)\s+following(?:\s+[A-Za-z/\-’']+){0,8}\s*[:?.]?\s*$", re.IGNORECASE
)
_GIVEN_BELOW_END = re.compile(r"\b(?:given|shown)\s+below(?:\s+[A-Za-z/\-’']+){0,6}\s*[:?.]?\s*$", re.IGNORECASE)
# An option that only points at items the stem must print: "(i) and (ii) only", "Only I and III".
_REF_LEAD = r"^(?:(?:only|both|all|all of|none of|neither|either)\s+)?"
_REF_TAIL = r"(?:\s+(?:only|are correct|is correct|are true|is true|are false|are incorrect))?\s*\.?\s*$"
_ONLY_ROMAN_REFS = re.compile(
    _REF_LEAD + r"\(?(?:i{1,3}|iv|v|vi)\)?(?:\s*(?:,|and|&|or|nor)\s*\(?(?:i{1,3}|iv|v|vi)\)?)*" + _REF_TAIL, re.IGNORECASE)
_ONLY_NUMBER_REFS = re.compile(
    _REF_LEAD + r"\(?[1-6]\)?(?:\s*(?:,|and|&|or|nor)\s*\(?[1-6]\)?)*" + _REF_TAIL, re.IGNORECASE)
# Match-the-following answers: "A-2, B-1", "(a)-(iii)", "P → 1".
_OPTS_REFER_MATCH = re.compile(r"\b[A-D]\s*[-–→:]\s*(?:\d|\(?[ivx]+\)?|[p-s])\b")
_STEM_HAS_ROMAN_ITEMS = re.compile(r"(?:^|[\s:])\(?(?:i|I)\s*[).]\s*\S.*\(?(?:ii|II)\s*[).]\s*\S", re.DOTALL)


def has_empty_labels(text: str) -> bool:
    """"(a) (b) (c)" with nothing between — the items were drawings."""
    if _PLACEHOLDER.search(text or ""):
        return True
    for a, b in _PAIR.findall(text or ""):
        if a in _ROMAN and b in _ROMAN and _ROMAN.index(b) == _ROMAN.index(a) + 1:
            return True
        # "NO (g) (b) …" is a state symbol followed by the next label, not an empty item.
        if len(a) == len(b) == 1 and ord(b) == ord(a) + 1 and a not in ("g", "i"):
            return True
    return False


_STEM_HAS_NUMBERED_ITEMS = re.compile(r"(?:^|\n|[\s:])(?:\(?1[).]|statement\s*(?:1|I)\b).*(?:\(?2[).]|statement\s*(?:2|II)\b)", re.IGNORECASE | re.DOTALL)
_COLUMNS = re.compile(r"(?:column|list)\s*[-–]?\s*(?:i|I|1|A)\b.*\S.*(?:column|list)\s*[-–]?\s*(?:ii|II|2|B)\b", re.DOTALL | re.IGNORECASE)
# A copied template instead of content: "(I)... (II)...", "Column A... Column B...".
_PLACEHOLDER_ITEMS = re.compile(r"(?:\(\s*[IVXivx]{1,4}\s*\)|\bcolumn\s+[AB12I]{1,2}|\bstatement\s+[1-4IV]{1,3})\s*:?\s*(?:\.{3}|…)", re.IGNORECASE)


def missing_items_reason(text: str, options: Optional[list] = None, has_table: bool = False) -> Optional[str]:
    """Why a question that points at a list ("the following …", statements, match columns)
    can't be answered from what is printed, or None. Used for NCERT rows and AI questions."""
    text = text or ""
    if has_empty_labels(text):
        return "list items are empty"
    if len(_PLACEHOLDER_ITEMS.findall(text)) >= 2:  # one "(ii) …" can be a quoted extract
        return "list items are placeholders"
    opts = [o for o in (options or []) if isinstance(o, str) and o.strip()]
    if has_table:
        return None
    if not opts:
        if _ENDS_WITH_FOLLOWING.search(text) or _GIVEN_BELOW_END.search(text):
            return "'the following' has nothing after it"
        return None
    joined = " ".join(opts)
    bodies = [_OPT_LABEL.sub("", o, count=1).strip() for o in opts]
    # Options that are nothing but references: "(i) and (ii) only", "I, II and III", "1 and 3".
    # When the options are themselves the labelled items ("(i) Trial balance", "(iv) (i) & (iii)")
    # the references point at other options, not at a missing list.
    roman_refs = sum(bool(_ONLY_ROMAN_REFS.match(b)) for b in bodies)
    if roman_refs >= 2 and not _STEM_HAS_ROMAN_ITEMS.search(text) \
            and not any(_ROMAN_LABEL.match(b) and not _ONLY_ROMAN_REFS.match(b) for b in bodies):
        return "options refer to statements (i), (ii)… that are not in the question"
    number_refs = sum(bool(_ONLY_NUMBER_REFS.match(b)) for b in bodies)
    if number_refs >= 2 and not _STEM_HAS_NUMBERED_ITEMS.search(text) \
            and re.search(r"\b(?:statements?|following)\b", text, re.IGNORECASE):
        return "options refer to statements 1, 2… that are not in the question"
    if re.search(r"\bmatch\b", text, re.IGNORECASE) and _OPTS_REFER_MATCH.search(joined):
        if not _COLUMNS.search(text) and len(re.findall(r"\n", text)) < 3:
            return "match-the-following without the columns"
    return None


def quality_issue(
    text: str,
    options: Optional[list] = None,
    has_image: bool = False,
    has_table: bool = False,
) -> Optional[str]:
    """'garbled' / 'incomplete' when an NCERT row can't be printed as stored, else None.

    A rendered image of the question block (image_url) carries the missing drawing,
    so a row with an image is never incomplete.
    """
    text = text or ""
    if has_image:
        return None
    if _UNI_GLYPH.search(text):
        return "garbled"
    if missing_items_reason(text, options, has_table):
        return "incomplete"
    return None


# ── Entry point used by the API ──────────────────────────────────────────────
def clean_question(q: dict) -> dict:
    """Clean one ncert_questions row in place (text, options) and return it."""
    opts = q.get("options")
    if isinstance(opts, list):
        opts = [clean_glyphs(o) if isinstance(o, str) else o for o in opts]
    text = clean_glyphs(q.get("question_text") or "")
    roman_relabelled = False
    if isinstance(opts, list) and opts:
        text = strip_inline_options(text, opts)
        roman_relabelled = all(isinstance(o, str) for o in opts) and _roman_labels_strippable(opts)
        opts = strip_option_labels(opts)
        q["options"] = opts
    q["question_text"] = text
    if isinstance(q.get("answer"), str):
        q["answer"] = clean_glyphs(q["answer"])
        if roman_relabelled:
            q["answer"] = _answer_label_to_letter(q["answer"], len(opts))
    return q
