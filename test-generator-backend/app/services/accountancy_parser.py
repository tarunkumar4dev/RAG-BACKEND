"""
app/services/accountancy_parser.py — Reconstructs flattened CBSE/NCERT Accountancy tables.

NCERT PDF extraction left most financial questions as one flat string:
    "... particulars for the year ended March 31, 2017: ` Opening stock 37,500 Purchases 1, 05,000 ..."
    "... Liabilities Amount Assets Amount ` ` Owners Funds Non-Current Assets Capital 12,000 Furniture 15,000 ..."
or as newline pipe tables that never had a |---| separator line.

parse_and_structure_accountancy_text(text) turns that into ordered segments:
    [{"type": "text",  "content": "Prepare a trading account ..."},
     {"type": "table", "table": {"headers": [...], "rows": [[...]], "kind": "...",
                                 "amount_cols": [...], "total_rows": [...], "section_rows": [...],
                                 "caption": "..."}}]

Pure Python (no ReportLab) so the exporter, the DB remediation script and tests share one
implementation. The TypeScript port lives in the frontend at src/utils/accountancyTableParser.ts —
keep the two in sync.

Design rule: a false table is worse than a flat paragraph. Every detector requires an explicit
cue (column header, pipes, currency-marked stream, dates, "Label: Rs. N" lines) plus at least
three well-formed items, and otherwise leaves the text alone.
"""
from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# Normalisation
# ═══════════════════════════════════════════════════════════════════════

RUPEE = "₹"

_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

_WS_RE = re.compile(r"[ \t   ]+")


def normalize_accountancy_text(text: str) -> str:
    """Repairs extraction artefacts without changing meaning."""
    if not text:
        return ""
    t = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Soft hyphens: used as word separators in some chapters, as real hyphens in others.
    if t.count("\xad") > 4:
        t = t.replace("\xad", " ")
    else:
        t = re.sub(r"(?<=\w)\xad(?=\w)", "-", t).replace("\xad", "")

    # NCERT's rupee font glyph was extracted as a backtick.
    t = t.replace("`", RUPEE)
    t = re.sub(r"\bRs\.?\s*(?=\d)", "Rs. ", t)

    # Split Indian-format numbers: "1, 05,000" / "1,25 ,000" / "3, 00, 000".
    t = re.sub(r"(?<![\d,])(\d{1,2}),\s+(\d{2},\d{3})(?!\d)", r"\1,\2", t)
    t = re.sub(r"(?<![\d,])(\d{1,2}),(\d{2})\s+,(\d{3})(?!\d)", r"\1,\2,\3", t)
    t = re.sub(r"(?<![\d,])(\d{1,3}),\s(\d{2,3}),\s?(\d{3})(?!\d)", r"\1,\2,\3", t)
    # "(1,500 )" -> "(1,500)"
    t = re.sub(r"\(\s*(\d[\d,]*(?:\.\d+)?)\s*\)", r"(\1)", t)
    # Broken header words.
    t = re.sub(r"\bAmou\s+nt\b", "Amount", t)
    t = re.sub(r"\bParti\s+culars\b", "Particulars", t)
    t = re.sub(r"\bBalanc\s+e\b", "Balance", t)
    # Figures glued to the next label: "50,000Plant & Machinery".
    t = re.sub(r"(\d)([A-Z][a-z])", r"\1 \2", t)
    # Nil placeholder in "Debit / Credit" pairs: "Bank overdraft — / 24,660".
    t = re.sub(r"\s[—–]\s*/\s*(?=\d)", " ", t)
    # Split plural endings: "Purchase s 1,64,000", "Carriage outward s".
    t = re.sub(r"\b([A-Za-z]{3,}[a-rt-z]) s\b(?=\s+(?:\d|[A-Z(₹]|$)|$)", r"\1s", t)
    # Broken years: "March 31, 201 7".
    t = re.sub(r"(\d{1,2},\s*)(20[0-3]|19\d)\s(\d)\b", r"\1\2\3", t)
    # Running page headers: "Reprint 2026-27 133".
    t = re.sub(r"\s*Reprint\s+20\d{2}-\d{2}\s+\d{1,3}\s*", " ", t)

    lines = [_WS_RE.sub(" ", ln).strip() for ln in t.split("\n")]
    return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════════════════════
# Amount tokenisation
# ═══════════════════════════════════════════════════════════════════════

_NUM_RE = re.compile(
    r"(?<![\w/.%])"
    r"(?P<neg>\()?"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"(?(neg)\))"
    r"(?![\w%/]|\.\d)"
)
_NOT_AMOUNT_AFTER = re.compile(
    r"^\s*(?:%|per\b|each\b|p\.?a\b|days?\b|months?\b|years?\b|yrs?\b|(?:equity\s+|preference\s+)?shares?\b|"
    r"debentures?\s+of\b|units?\b|kg\b|pieces?\b|@|:\d|\bto\s+\d|times\b|hours?\b|metres?\b|meters?\b|"
    r"shirts?\b|pants?\b|trousers?\b|jackets?\b|items?\b|articles?\b|dozens?\b|nos?\.?\b|[A-Z][a-z]+\s+@)",
    re.IGNORECASE,
)
_NOT_AMOUNT_BEFORE = re.compile(
    r"(?:\bNo\.?\s*|\bNos?\.\s*|@\s*(?:Rs\.\s*|₹\s*)?|\bRatio\s*|\bClass\s*|\bQ\.?\s*|\bChapter\s*|\bpage\s*|"
    + _MONTH + r"\.?\s*|\bnote\s*|\bto\s+|\bdated\s*)$",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


class _Amt:
    __slots__ = ("start", "end", "text")

    def __init__(self, start: int, end: int, text: str):
        self.start, self.end, self.text = start, end, text


def _find_amounts(s: str) -> List[_Amt]:
    out: List[_Amt] = []
    for m in _NUM_RE.finditer(s):
        num = m.group("num")
        digits = num.replace(",", "")
        has_comma = "," in num
        before = s[max(0, m.start() - 24):m.start()]
        after = s[m.end():m.end() + 24]

        if _NOT_AMOUNT_AFTER.match(after):
            continue
        if re.search(_NOT_AMOUNT_BEFORE, before) and not re.search(r"(?:₹|Rs\.)\s*$", before):
            continue
        # "(... bill of ₹1,600)" — a figure closing a parenthetical note is part of the label.
        if not m.group("neg") and after.startswith(")"):
            continue
        if not has_comma:
            int_part = digits.split(".")[0]
            if len(int_part) == 1 and "." not in digits and not m.group("neg"):
                continue
            # Day numbers "01", "05".
            if int_part.startswith("0") and len(int_part) <= 2:
                continue
            if _YEAR_RE.match(digits):
                continue
            # "31, 2017" style day numbers.
            if re.match(r"^,\s*(?:19|20)\d{2}", after):
                continue
            # Enumerators "(1)", "(12)".
            if m.group("neg") and len(int_part) <= 2:
                continue
        # "March 31, 2017" — comma-grouped year after a day.
        if has_comma and re.match(r"^\d{1,2},\d{3}$", num) and re.search(_MONTH + r"\.?\s*$", before):
            continue
        out.append(_Amt(m.start(), m.end(), m.group(0)))
    return out


def _is_amount_token(tok: str) -> bool:
    tok = tok.strip().replace(RUPEE, "").replace("Rs.", "").strip()
    return bool(re.fullmatch(r"\(?-?\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?\)?|\(?-?\d+(?:\.\d{1,2})?\)?", tok))


def _clean_amount(tok: str) -> str:
    tok = tok.replace(RUPEE, "").replace("Rs.", "").strip()
    return re.sub(r"\s+", "", tok)


def _clean_label(label: str) -> str:
    label = label.replace("|", " ")
    label = re.sub(r"(?:(?<=\s)|^)(?:₹|Rs\.)(?=\s|$)", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    label = label.strip(" ,;:-–—")
    label = re.sub(r"^\.\s+", "", label)
    label = re.sub(r"^\((?:₹|Rs\.?)\)\s*:?\s*", "", label)
    label = re.sub(r"\s*:\s*$", "", label)
    return label


# ═══════════════════════════════════════════════════════════════════════
# Item stream
# ═══════════════════════════════════════════════════════════════════════

class _Item:
    __slots__ = ("label", "amounts")

    def __init__(self, label: str, amounts: List[str]):
        self.label = label
        self.amounts = amounts

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Item({self.label!r}, {self.amounts})"


_INSTRUCTION_START = re.compile(
    r"^(?:Prepare|Calculate|Compute|Show|Find|Pass|Record|Give|Journalise|Journalize|Ascertain|Determine|Draw|"
    r"Make|State|You are|Also|Post|Enter|Balance the|Close|Explain|Identify|What|How|Why|Which|Rectify|Classify|"
    r"Indicate|Present|Redraft|Value|Analyse|Analyze|Additional [Ii]nformation|Adjustments?|Notes?\b|Other [Ii]nformation|"
    r"On the basis|From the above|Using the|With the help|The following|Assume|Assuming|Consider|Required|Hint)\b"
)


def _is_instruction(fragment: str) -> bool:
    f = fragment.strip()
    if not f:
        return False
    if _INSTRUCTION_START.match(f):
        return True
    words = f.split()
    if len(words) >= 7 and re.search(r"[.?]\s*$", f):
        return True
    return bool(re.search(r"(?:^|\s)(?:i{1,3}|iv|v|vi{0,3})\)\s*\S", f) and len(words) > 4)


def _tokenize_items(body: str) -> Tuple[List[_Item], str]:
    """Splits "Label 1,000 Label 2 2,000 3,000 ..." into items.

    Returns (items, trailing_text). The trailing text is whatever follows the last amount if it
    reads like an instruction/sentence rather than a dangling label.
    """
    amts = _find_amounts(body)
    items: List[_Item] = []
    cursor = 0
    i = 0
    while i < len(amts):
        label_raw = body[cursor:amts[i].start]
        group = [amts[i]]
        j = i + 1
        while j < len(amts):
            gap = body[group[-1].end:amts[j].start]
            if re.fullmatch(r"[\s|₹/]*(?:Rs\.)?[\s|₹]*", gap):
                group.append(amts[j])
                j += 1
            else:
                break
        items.append(_Item(_clean_label(label_raw), [_clean_amount(a.text) for a in group]))
        cursor = group[-1].end
        i = j

    tail = body[cursor:].strip()
    trailing = ""
    if tail:
        tail_clean = _clean_label(tail)
        if not tail_clean or re.fullmatch(r"[.;,]*", tail_clean):
            pass
        elif _is_instruction(tail_clean) or len(tail_clean.split()) > 8:
            trailing = tail.strip(" |")
        else:
            items.append(_Item(tail_clean.rstrip("."), []))
    _reattach_orphans(items)
    return items, trailing


_INCOMPLETE_END = re.compile(
    r"\b(?:for|of|to|on|and|in|the|by|from|with|accrued|outstanding|prepaid|further|provision|less|add|at)$",
    re.IGNORECASE,
)


def _reattach_orphans(items: List[_Item]) -> None:
    """Moves wrapped lowercase fragments ("doubtful debts", "capital account)") back to their owner."""
    for idx in range(1, len(items)):
        label = items[idx].label
        if re.match(r"^\(?[a-z0-9]{1,4}\)", label):  # "(a)Cheque ..." / "b) Trade payables" enumerators
            continue
        # Fragment tokens start lowercase / "₹" / "&" / ")" — never a digit ("10% Debentures" is a label).
        m = re.match(r"^((?:[a-z₹&)][^\s]*|\()(?:\s+(?:[a-z₹&)][^\s]*|\)))*)\s*(.*)$", label)
        if not m or not m.group(1):
            continue
        frag, rest = m.group(1).strip(), m.group(2).strip()
        if not re.search(r"[a-z0-9]", frag) or (rest and not rest[:1].isupper() and not rest[:1] in "(["):
            continue
        # Find an owner among the previous three items.
        owner = None
        for k in range(idx - 1, max(-1, idx - 4), -1):
            lbl = items[k].label
            if lbl.count("(") > lbl.count(")") or _INCOMPLETE_END.search(lbl):
                owner = k
                break
        if owner is None:
            if not rest:
                continue
            owner = idx - 1
        if not items[owner].label:
            continue
        items[owner].label = f"{items[owner].label} {frag}".strip()
        items[idx].label = rest


# ═══════════════════════════════════════════════════════════════════════
# Header detection
# ═══════════════════════════════════════════════════════════════════════

_HDR_WORD = (
    r"(?:Particulars|Accounts?\s+Titles?|Titles?\s+of\s+(?:the\s+)?Accounts?|Name\s+of\s+(?:the\s+)?Accounts?|Heads?\s+of\s+Accounts?|"
    r"Items?|Details|Description|Transactions?|Date|Amount|Amounts|Debit|Credit|Dr\.?|Cr\.?|Balances?|J\.\s?F\.?|L\.\s?F\.?|"
    r"Note\s+No\.?|Liabilities|Assets|Capital\s+and\s+Liabilities|Property\s+and\s+Assets|"
    r"Expenses\s*/\s*Losses|Revenues?\s*/\s*Gains|Expenses|Incomes?|Revenues?|Losses|Gains|Receipts|Payments|"
    r"Cash|Bank|Discount|V\.?\s?No\.?|Voucher\s+No\.?|R\.?\s?No\.?|"
    r"\((?:Rs\.?|₹|in\s+Rs\.?|in\s+₹)\)|Rs\.|₹|\||/|(?:31st\s+)?(?:March\s+31,?\s+)?(?:19|20)\d{2}(?:-\d{2})?(?:\s*\((?:Rs\.?|₹)\))?)"
)
_HDR_ANCHOR = re.compile(
    r"Particulars|Accounts?\s+Titles?|Titles?\s+of|Name\s+of|Heads?\s+of|Details|Description|Liabilities|Assets|"
    r"Expenses|Revenues?|Receipts|Payments|Debit|Items?\b|Date\b",
    re.IGNORECASE,
)
_HDR_RUN_RE = re.compile(r"(?:(?<=\s)|^)(?:" + _HDR_WORD + r")(?:\s+(?:" + _HDR_WORD + r"))*(?=\s|$)", re.IGNORECASE)


class _Header:
    __slots__ = ("start", "end", "text", "kind")

    def __init__(self, start, end, text, kind):
        self.start, self.end, self.text, self.kind = start, end, text, kind


def _classify_header(h: str) -> Optional[str]:
    hl = h.lower()
    words = re.findall(r"[a-z]+", hl)
    if "liabilities" in hl and "assets" in hl:
        return "balance_sheet"
    if ("expenses" in hl or "losses" in hl) and ("revenue" in hl or "gains" in hl or "incomes" in hl or "income" in hl):
        return "trading_pl"
    if "receipts" in hl and "payments" in hl:
        return "receipts_payments"
    if "date" in words and ("j" in words or "l" in words) and hl.count("particulars") >= 2:
        return "ledger"
    if "date" in words:
        return "dated"
    if "debit" in hl and "credit" in hl:
        return "trial_balance"
    if re.search(r"(?:19|20)\d{2}\D+(?:19|20)\d{2}", hl):
        return "multi_year"
    return "list"


_HDR_TOKEN_RE = re.compile(_HDR_WORD, re.IGNORECASE)
# Header words that can equally start the first data row ("... (Rs.) Revenue from operations 16,00,000").
_HDR_DATA_WORD = re.compile(
    r"^(?:Revenues?|Incomes?|Cash|Bank|Discount|Items?|Transactions?|Losses|Gains|Balances?|Expenses|Capital)$", re.IGNORECASE
)


def _find_header(s: str, start: int = 0) -> Optional[_Header]:
    for m in _HDR_RUN_RE.finditer(s, start):
        toks = [(t.start() + m.start(), t.end() + m.start(), t.group(0)) for t in _HDR_TOKEN_RE.finditer(m.group(0))]
        if not toks:
            continue
        # Drop leading dates/years/currency swallowed into the run ("March 31, 2017 Liabilities ...").
        first_anchor = next((k for k, t in enumerate(toks) if _HDR_ANCHOR.fullmatch(t[2]) or _HDR_ANCHOR.match(t[2])), None)
        if first_anchor is None:
            continue
        lead = toks[:first_anchor]
        if lead and not all(re.fullmatch(r"(?:Dr|Cr)\.?", t[2]) for t in lead):
            toks = toks[first_anchor:]
        while len(toks) > 2 and _HDR_DATA_WORD.match(toks[-1][2]):
            toks.pop()
        h_start, h_end = toks[0][0], toks[-1][1]
        h = s[h_start:h_end].strip()
        # A header needs at least two tokens (e.g. "Particulars Amount", "Particulars ₹", "Date Details").
        if len(toks) < 2:
            continue
        # "Trade expenses ₹ 2,000" is data, not a header: two-token headers need a column-title anchor.
        if len(toks) == 2 and not re.match(
            r"Particulars|Accounts?\s+Titles?|Details|Items?|Description|Date|Name\s+of|Heads?\s+of", toks[0][2], re.I
        ):
            continue
        # Accept "Particulars ₹"/"Particulars (Rs.)", reject "Details of ..." used in prose.
        if not re.search(r"Amount|Debit|Credit|₹|Rs\.|\(Rs|Dr\.|Cr\.|Assets|Liabilities|Gains|Payments|Details|Particulars\s+(?:J|L)\.|\d{4}", h, re.IGNORECASE):
            continue
        kind = _classify_header(h)
        # "Date Details Amount" must be followed by something that looks like a date/year.
        if kind == "dated" and not re.match(r"\s*(?:₹\s*)*(?:(?:19|20)\d{2}\s+)?(?:" + _MONTH + r"|\d{1,2}\s)", s[h_end:]):
            kind = "list"
        return _Header(h_start, h_end, h, kind)
    return None


def _is_two_sided(h: str, kind: str) -> bool:
    """Side-by-side layouts: Liabilities|Assets, Debit balances|Credit balances, Particulars Amount Particulars Amount."""
    if kind in ("balance_sheet", "trading_pl", "receipts_payments"):
        return True
    hl = h.lower()
    if re.search(r"debit\s+balances?.*credit\s+balances?", hl):
        return True
    if len(re.findall(r"particulars|accounts?\s+titles?|name\s+of", hl)) >= 2:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Table builders
# ═══════════════════════════════════════════════════════════════════════

def _table(headers, rows, kind, amount_cols, total_rows=None, section_rows=None, caption=None) -> dict:
    t = {
        "headers": [_WS_RE.sub(" ", str(h)).strip() for h in headers],
        "rows": [[_WS_RE.sub(" ", str(c)).strip() for c in r] for r in rows],
        "kind": kind,
        "amount_cols": sorted(set(amount_cols)),
        "total_rows": sorted(set(total_rows or [])),
        "section_rows": sorted(set(section_rows or [])),
    }
    if caption:
        t["caption"] = caption
    return t


def _is_total_label(label: str) -> bool:
    return not label or bool(re.match(r"^(?:total|grand total|balance total)\b", label, re.IGNORECASE))


_BAD_LABEL = re.compile(
    r"^[.;,)=/]|=|@|\s(?:of|at|by|amounted\s+to|amounts\s+to|being|to|for|in|on|with|from|and|costing|than|as)$|"
    r"\b(?:is|are|was|were|has|have|had)\b",
    re.IGNORECASE,
)


def _label_is_bad(label: str) -> bool:
    return bool(_BAD_LABEL.search(label)) or label.count("(") != label.count(")") or len(label.split()) > 16


def _valid_stream(items: List[_Item], max_words: int = 10, min_items: int = 3) -> bool:
    labelled = [it for it in items if it.label and it.amounts]
    if len(labelled) < min_items:
        return False
    # Prose ("Operating expenses are Rs. 60,000. Revenue ... is Rs. 2,50,000.") is not a table.
    bad = sum(1 for it in labelled if _label_is_bad(it.label))
    if bad > len(labelled) // 7:
        return False
    long_labels = sum(1 for it in labelled if len(it.label.split()) > max_words)
    if long_labels > max(1, len(labelled) // 4):
        return False
    avg = sum(len(it.label.split()) for it in labelled) / len(labelled)
    return avg <= max(6, max_words * 0.6)


def _amount_header(h_text: str) -> str:
    return f"Amount ({RUPEE})"


def _build_list_table(items: List[_Item], header_text: str) -> dict:
    """Particulars | Amount — or Particulars | <year> | <year> for comparative streams."""
    # Split "Closing stock 15,000 56,350 56,350" into the item and its totals row.
    split: List[_Item] = []
    for it in items:
        if it.label and len(it.amounts) >= 3 and it.amounts[-1] == it.amounts[-2]:
            split += [_Item(it.label, it.amounts[:-2]), _Item("", it.amounts[-2:])]
        else:
            split.append(it)
    items = split
    max_amts = max((len(it.amounts) for it in items if it.label), default=1)
    years = re.findall(r"(?:19|20)\d{2}(?:-\d{2})?", header_text or "")
    first = "Account Title" if re.search(r"Account", header_text or "", re.I) else "Particulars"
    if max_amts >= 2 and len(years) >= max_amts:
        headers = [first] + [f"{y} ({RUPEE})" for y in years[:max_amts]]
        n_amt, inner = max_amts, False
    elif max_amts >= 2 and re.search(r"debit|credit", header_text or "", re.I):
        return _build_trial_balance(items, header_text)
    elif max_amts >= 2:
        # "Add Net profit 20,850 32,850": inner working column + amount column (header spans both).
        headers = [first, "", f"Amount ({RUPEE})"]
        n_amt, inner = 2, True
    else:
        headers = [first, f"Amount ({RUPEE})"]
        n_amt, inner = 1, False
    rows, totals = [], []
    for it in items:
        if not it.label and not it.amounts:
            continue
        if not it.label:
            totals.append(len(rows))
            if inner or n_amt == 1:
                rows.append(["Total"] + [""] * (n_amt - 1) + [it.amounts[-1]])
            else:
                rows.append(["Total"] + (it.amounts + [""] * n_amt)[:n_amt])
            continue
        if inner:
            vals = [it.amounts[0], it.amounts[-1]] if len(it.amounts) >= 2 else ["", it.amounts[0] if it.amounts else ""]
        else:
            vals = (it.amounts + [""] * n_amt)[:n_amt]
        if _is_total_label(it.label):
            totals.append(len(rows))
        rows.append([it.label] + vals)
    return _table(headers, rows, "list", range(1, 1 + n_amt), totals)


# Normal balances for trial-balance side inference.
_DR_WORDS = re.compile(
    r"purchase(?!s?\s+returns?)|returns?\s+inward|sales?\s+returns?|wages|salar|rent(?!\s+received)|carriage|freight|expense|"
    r"insurance|drawings|debtors|receivable|stock|inventor|cash|machinery|plant|furniture|building|land|premises|goodwill|"
    r"vehicle|motor|scooter|car\b|equipment|computer|investment|bad debts|discount allowed|interest\s+(?:paid|on\s+(?:loan|overdraft|bank))|"
    r"commission\s+paid|advertis|repairs|depreciation|power|fuel|lighting|electricity|postage|telephone|stationery|printing|"
    r"prepaid|accrued|loss|tools|patent|trade\s*mark|fixtures|bills receivable|taxes|travelling|conveyance|office|general|"
    r"sundry expenses|audit fee|legal|charity|donation|octroi|bank(?!\s+(?:overdraft|loan))",
    re.IGNORECASE,
)
_CR_WORDS = re.compile(
    r"capital|sales(?!\s+returns?)|revenue from operations|returns?\s+outward|purchases?\s+returns?|creditors|payable|"
    r"loan|overdraft|reserve|surplus|premium|provision|outstanding|received|income|commission(?!\s+paid)|"
    r"discount received|interest received|rent received|profit|gain|debentures|share capital|bills payable|dividend received|"
    r"apprentice premium|advance",
    re.IGNORECASE,
)


def _side_for_tb(label: str) -> Optional[str]:
    l = label.lower()
    # Order matters: the more specific phrase wins.
    if re.search(r"^interest\s+(?:on|paid)\b", l) and "capital" not in l:
        return "dr"
    if re.search(r"returns?\s+inward|sales?\s+returns?", l):
        return "dr"
    if re.search(r"returns?\s+outward|purchases?\s+returns?", l):
        return "cr"
    if re.search(r"bank\s+(?:overdraft|loan)|received|payable|creditors|capital|reserve|provision\s+for\s+(?:doubtful|bad)|"
                 r"outstanding|surplus|premium|debentures|loan", l):
        # "Loan to X" / "Loans and advances" are assets.
        if re.search(r"loans?\s+(?:to|and\s+advances)|advance to|prepaid", l):
            return "dr"
        return "cr"
    if _DR_WORDS.search(l):
        return "dr"
    if _CR_WORDS.search(l):
        return "cr"
    return None


def _build_trial_balance(items: List[_Item], header_text: str) -> dict:
    headers = ["Account Title", f"Debit ({RUPEE})", f"Credit ({RUPEE})"]
    rows, totals = [], []
    unknown = 0
    staged = []
    for it in items:
        if not it.label and not it.amounts:
            continue
        if not it.label or _is_total_label(it.label):
            a = it.amounts + [""] * 2
            staged.append(("total", it.label or "Total", a[0], a[1] if len(it.amounts) > 1 else a[0]))
            continue
        if len(it.amounts) >= 2:
            staged.append(("row", it.label, it.amounts[0], it.amounts[1]))
        elif len(it.amounts) == 1:
            side = _side_for_tb(it.label)
            if side is None:
                unknown += 1
            staged.append(("row", it.label, it.amounts[0] if side != "cr" else "", it.amounts[0] if side == "cr" else ""))
        else:
            staged.append(("row", it.label, "", ""))
    if unknown:
        # Can't place every single balance honestly — show amounts in reading order instead.
        rows = []
        for kind, label, d, c in staged:
            if kind == "total":
                totals.append(len(rows))
            both = [x for x in (d, c) if x]
            rows.append([label, " / ".join(both) if len(both) == 2 and d != c else (both[0] if both else "")])
        return _table(["Account Title", f"Amount ({RUPEE})"], rows, "list", [1], totals)
    for kind, label, d, c in staged:
        if kind == "total":
            totals.append(len(rows))
        rows.append([label, d, c])
    return _table(headers, rows, "trial_balance", [1, 2], totals)


_DATE_TOKEN = re.compile(r"(?:(?<=\s)|^)(?:(?:19|20)\d{2}\s+)?" + _MONTH + r"\.?\s*\d{1,2}(?:st|nd|rd|th)?(?![\d,])(?!,\s*(?:19|20)\d{2})")


_BARE_DAY = re.compile(r"(?:(?<=\s)|^)(?:0[1-9]|[12]\d|3[01])(?=\s+[A-Z])")


def _build_dated_table(body: str, header_text: str, month: str = "") -> Optional[Tuple[dict, str]]:
    """Cash book / journal / transaction lists: Date | Details | Amount.

    `month` ("Dec.") enables bare day numbers ("01 Started business ... 03 Cash paid into bank ...")
    when the intro names the month."""
    marks = list(_DATE_TOKEN.finditer(body))
    if month and len(marks) < 3:
        marks = [m for m in _BARE_DAY.finditer(body)
                 if not re.search(r"(?:₹|Rs\.)\s*$", body[max(0, m.start() - 5):m.start()])]
    if len(marks) < 3:
        return None
    lead_year = re.search(r"((?:19|20)\d{2})\s*$", body[:marks[0].start()].strip())
    rows, max_amts = [], 0
    trailing = ""
    for k, m in enumerate(marks):
        seg_end = marks[k + 1].start() if k + 1 < len(marks) else len(body)
        seg = body[m.end():seg_end].strip()
        # Year printed between entries ("... 60,000 2015 Jan. 01") belongs to the next date.
        date = m.group(0).strip()
        if k + 1 < len(marks):
            ym = re.search(r"\s((?:19|20)\d{2})\s*$", " " + seg)
            if ym:
                seg = seg[: len(seg) - len(ym.group(1))].strip()
        if k == len(marks) - 1:
            sub_items, tr = _tokenize_items(seg)
            trailing = tr
            if tr:
                seg = seg[: seg.rfind(tr)].strip() if tr in seg else seg
        # "01 Cash in hand 17,500 Cash at bank 5,000" — two entries under one date.
        sub, sub_tr = _tokenize_items(seg)
        if (len(sub) >= 2 and not sub_tr and "@" not in seg
                and all(it.label and it.amounts and it.label[0].isupper() and not _label_is_bad(it.label) for it in sub)):
            if month and re.fullmatch(r"\d{1,2}", date):
                date = f"{month} {date}"
            if k == 0 and lead_year and not date[:4].isdigit():
                date = f"{lead_year.group(1)} {date}"
            for n, it in enumerate(sub):
                rows.append([date if n == 0 else "", it.label, it.amounts[:3]])
                max_amts = max(max_amts, len(it.amounts[:3]))
            continue
        amts = _find_amounts(seg)
        # Only amounts at the end of the segment are column amounts; inline "@ ₹300" stays in details.
        tail_amts: List[_Amt] = []
        cut = len(seg)
        for a in reversed(amts):
            if re.fullmatch(r"[\s|₹]*(?:Rs\.)?[\s|₹]*", seg[a.end:cut]):
                tail_amts.insert(0, a)
                cut = a.start
            else:
                break
        details = _clean_label(seg[:cut])
        vals = [_clean_amount(a.text) for a in tail_amts]
        max_amts = max(max_amts, len(vals))
        if month and re.fullmatch(r"\d{1,2}", date):
            date = f"{month} {date}"
        if k == 0 and lead_year and not date[:4].isdigit():
            date = f"{lead_year.group(1)} {date}"
        rows.append([date, details, vals])

    if sum(1 for r in rows if r[1]) < 3:
        return None
    # Dated scenarios written as sentences ("Jan. 1 The directors decide to allot ...") stay text.
    with_amt = sum(1 for r in rows if r[2])
    avg_words = sum(len(r[1].split()) for r in rows) / len(rows)
    if with_amt < len(rows) / 2 and avg_words > 10:
        return None
    hl = (header_text or "").lower()
    n_amt = max(1, min(max_amts, 3))
    if n_amt == 2 and "debit" in hl and "credit" in hl:
        amt_headers = [f"Debit ({RUPEE})", f"Credit ({RUPEE})"]
    elif n_amt == 2 and "cash" in hl and "bank" in hl:
        amt_headers = [f"Cash ({RUPEE})", f"Bank ({RUPEE})"]
    else:
        amt_headers = [f"Amount ({RUPEE})"] * n_amt
    second = "Particulars" if "particular" in hl else "Details"
    out_rows = []
    for date, details, vals in rows:
        # Right-align a lone amount to the last column (running total style) only when every row has one.
        vals = (vals + [""] * n_amt)[:n_amt]
        out_rows.append([date, details] + vals)
    return _table(["Date", second] + amt_headers, out_rows, "dated", range(2, 2 + n_amt)), trailing


# ── T-shape accounts (Balance Sheet, Trading & P&L) ─────────────────────

_BS_HEADINGS = [
    (r"Owners?[’']?s?\s+Funds?", "L"), (r"Owners?[’']?s?\s+Equity", "L"), (r"Shareholders?[’']?\s+Funds", "L"),
    (r"Non[-\s]?Current\s+Liabilities", "L"), (r"Long[-\s]term\s+Liabilities", "L"),
    (r"Current\s+Liabilities(?:\s+(?:and|&)\s+Provisions)?", "L"),
    (r"Non[-\s]?Current\s+Assets", "R"), (r"Fixed\s+Assets", "R"), (r"Current\s+Assets", "R"),
]
_BS_L = re.compile(
    r"capital|net\s+profit|interest\s+on\s+capital|drawings|net\s+loss|loan(?!s?\s+(?:to|and\s+advances))|creditors|payable|overdraft|"
    r"outstanding|received\s+in\s+advance|unearned|reserve|surplus|premium|debentures|provision\s+for\s+(?:tax|taxation)|"
    r"borrowings|advance\s+from",
    re.IGNORECASE,
)
_BS_R = re.compile(
    r"furniture|machinery|plant|building|land|premises|goodwill|investment|stock|inventor|debtors|receivable|bank(?!\s+(?:overdraft|loan))|"
    r"cash|prepaid|paid\s+in\s+advance|accrued|vehicle|motor|scooter|car\b|equipment|computer|tools|patent|trade\s*mark|fixtures|"
    r"depreciation|bad\s+debts|provision\s+for\s+(?:doubtful|discount|bad)|loans?\s+(?:to|and\s+advances)|advance\s+to|fittings",
    re.IGNORECASE,
)
_PL_L = re.compile(
    r"purchase|opening\s+stock|wages|carriage\s+inward|freight|salar|rent(?!\s+received)|depreciation|bad\s+debts|provision|"
    r"insurance|expense|discount\s+allowed|interest\s+(?:paid|on)|commission\s+paid|advertis|repairs|power|fuel|lighting|"
    r"electricity|postage|telephone|stationery|printing|gross\s+profit\s+c/d|net\s+profit|loss\s+by|outstanding|prepaid|"
    r"taxes|travelling|conveyance|audit|legal|charity|donation|sales\s+returns?|returns?\s+inward|trade\s+expenses",
    re.IGNORECASE,
)
_PL_R = re.compile(
    r"sales(?!\s+returns?)|closing\s+stock|gross\s+profit\s+b/d|commission\s+received|discount\s+received|interest\s+received|"
    r"rent\s+received|income|gain|net\s+loss|gross\s+loss\s+c/d|accrued|received\s+in\s+advance|dividend\s+received|"
    r"purchases?\s+returns?|returns?\s+outward|apprentice\s+premium|bad\s+debts\s+recovered",
    re.IGNORECASE,
)


def _split_headings(label: str, kind: str) -> Tuple[List[Tuple[str, str]], str]:
    """Pulls known section headings out of a label. Returns ([(heading, side)], remaining_label)."""
    if kind != "balance_sheet":
        return [], label
    found = []
    changed = True
    while changed and label:
        changed = False
        for pat, side in _BS_HEADINGS:
            m = re.match(r"^\s*(" + pat + r")\b\s*", label, re.IGNORECASE)
            if m:
                found.append((m.group(1).strip(), side))
                label = label[m.end():]
                changed = True
                break
    return found, label.strip()


def _t_side(label: str, kind: str, last_side: Optional[str]) -> Optional[str]:
    l = label.lower()
    adj = re.match(r"^(add|less)\b[:\s]*(.*)$", l)
    core = adj.group(2) if adj else l
    if kind == "balance_sheet":
        # Specific phrases first.
        if re.search(r"received\s+in\s+advance|unearned|outstanding|provision\s+for\s+tax", core):
            return "L"
        if re.search(r"prepaid|paid\s+in\s+advance|accrued|provision\s+for\s+(?:doubtful|discount|bad)", core):
            return "R"
        if _BS_L.search(core) and not _BS_R.search(core):
            return "L"
        if _BS_R.search(core) and not _BS_L.search(core):
            return "R"
        if _BS_L.search(core) and _BS_R.search(core):
            return "L" if _BS_L.search(core).start() < _BS_R.search(core).start() else "R"
        return None
    if kind == "trading_pl":
        if re.search(r"gross\s+profit\s+c/d|net\s+profit", core):
            return "L"
        if re.search(r"gross\s+profit\s+b/d|net\s+loss", core):
            return "R"
        if adj and re.search(r"outstanding|prepaid", core):
            return "L"
        if adj and re.search(r"accrued|received\s+in\s+advance", core):
            return "R"
        r_hit, l_hit = _PL_R.search(core), _PL_L.search(core)
        if r_hit and not l_hit:
            return "R"
        if l_hit and not r_hit:
            return "L"
        if l_hit and r_hit:
            return "L" if l_hit.start() < r_hit.start() else "R"
        return None
    return None


def _two_sided_titles(kind: str, header_text: str) -> Tuple[str, str]:
    hl = (header_text or "").lower()
    if kind == "balance_sheet":
        return "Liabilities", "Assets"
    if kind == "trading_pl":
        return "Expenses/Losses", "Revenues/Gains"
    if kind == "receipts_payments":
        return "Receipts", "Payments"
    if "debit" in hl and "credit" in hl:
        return "Debit Balances", "Credit Balances"
    return "Particulars", "Particulars"


def _side_of(label: str, kind: str, header_text: str) -> Optional[str]:
    if kind in ("balance_sheet", "trading_pl"):
        return _t_side(label, kind, None)
    hl = (header_text or "").lower()
    if "debit" in hl and "credit" in hl:
        s = _side_for_tb(label)
        return {"dr": "L", "cr": "R"}.get(s or "")
    return None


def _build_t_account(item_lines: List[List[_Item]], kind: str, header_text: str = "", loose_headings: bool = False) -> Optional[dict]:
    """Side-by-side statements. `item_lines` holds the items of each printed line; with real line
    breaks, a line with two entries is split positionally (left, right). Single-line input (fully
    flattened text) falls back to accounting vocabulary to decide each entry's side."""
    lh, rh = _two_sided_titles(kind, header_text)
    line_mode = len(item_lines) > 1

    left: List[list] = []   # [label, inner, outer, is_heading]
    right: List[list] = []
    total_marks: List[int] = []  # row index where a total row sits
    last_side: Optional[str] = None
    unknown = 0
    pending_totals: Dict[str, Optional[str]] = {"L": None, "R": None}

    def flush_equal():
        n = max(len(left), len(right))
        while len(left) < n:
            left.append(["", "", "", False])
        while len(right) < n:
            right.append(["", "", "", False])

    def add_total(a: str, b: str):
        flush_equal()
        total_marks.append(len(left))
        left.append(["", "", a, False])
        right.append(["", "", b, False])

    def place(label: str, amounts: List[str], side: str):
        inner, outer = ("", "")
        if len(amounts) >= 2:
            inner, outer = amounts[0], amounts[-1]
        elif len(amounts) == 1:
            outer = amounts[0]
        (left if side == "L" else right).append([label, inner, outer, False])

    # Flattened input: walk entry by entry so headings interleave in printed order.
    groups = item_lines if line_mode else [[it] for it in (item_lines[0] if item_lines else [])]
    for items in groups:
        entries = []  # (label, amounts) after splitting headings/totals
        pending_total = None
        for it in items:
            # "Closing stock 15,000 56,350 56,350" — a trailing equal pair is the totals row.
            if it.label and len(it.amounts) >= 3 and it.amounts[-1] == it.amounts[-2]:
                pending_total = it.amounts[-2:]
                it = _Item(it.label, it.amounts[:-2])
            headings, label = _split_headings(it.label, kind)
            for h, side in headings:
                (left if side == "L" else right).append([f"**{h}**", "", "", True])
            if not label and not it.amounts:
                continue
            if not label:
                if len(it.amounts) >= 2:
                    pending_total = it.amounts[:2]
                elif it.amounts:
                    # A lone number continuing the previous line (outer amount of an adjustment chain).
                    tgt = left if last_side == "L" else right
                    if tgt and not tgt[-1][2]:
                        tgt[-1][2] = it.amounts[0]
                continue
            entries.append((label, it.amounts))

        # Explicit "Total" entries are collected per side and emitted as one aligned totals row.
        kept = []
        for pos, (label, amounts) in enumerate(entries):
            if _is_total_label(label) and amounts:
                if line_mode and len(entries) == 2:
                    side = "L" if pos == 0 else "R"
                else:
                    side = "L" if pending_totals["L"] is None else "R"
                pending_totals[side] = amounts[-1]
                continue
            kept.append((label, amounts))
        entries = kept
        # "Capitals: Debtors 12,000" — a left-column heading printed on the same line as a right entry.
        if line_mode and entries:
            hm = re.match(r"^([A-Z][^:]{1,40}):\s+(\S.*)$", entries[0][0])
            if not hm and loose_headings:
                hm = re.match(r"^((?:Partners?[’']?s?\s+)?Capitals?(?:\s+Accounts?)?|Reserves?\s+and\s+Surplus)\s+([A-Z]\S*.*)$", entries[0][0])
            if hm and not _find_amounts(hm.group(1)):
                left.append([f"**{hm.group(1).strip()}**", "", "", True])
                if len(entries) == 1:
                    place(hm.group(2), entries[0][1], "R")
                    last_side = "R"
                    entries = []
                else:
                    entries[0] = (hm.group(2), entries[0][1])
        if line_mode and len(entries) == 2:
            place(entries[0][0], entries[0][1], "L")
            place(entries[1][0], entries[1][1], "R")
            last_side = "R"
        else:
            for label, amounts in entries:
                side = _side_of(label, kind, header_text)
                if side is None:
                    if line_mode and len(entries) == 1:
                        # One entry on a line with no vocabulary hint: the left column continues.
                        side = "L"
                    else:
                        unknown += 1
                        side = "R" if last_side == "L" else "L"
                place(label, amounts, side)
                last_side = side
        if pending_total:
            add_total(pending_total[0], pending_total[1])
            last_side = None
        if pending_totals["L"] is not None and pending_totals["R"] is not None:
            add_total(pending_totals["L"], pending_totals["R"])
            pending_totals["L"] = pending_totals["R"] = None

    if pending_totals["L"] is not None or pending_totals["R"] is not None:
        add_total(pending_totals["L"] or "", pending_totals["R"] or "")

    labelled = sum(1 for r in left + right if r[0] and not r[3])
    if labelled < 3 or unknown > max(2, labelled // 3):
        return None
    if any(_label_is_bad(r[0]) for r in left + right if r[0] and not r[3]) and not line_mode:
        return None

    # Base item before an "Add/Less" adjustment carries an inner amount ("Capital 12,000 / Add Net profit 20,850 32,850").
    for side_rows in (left, right):
        for k in range(len(side_rows) - 1):
            cur, nxt = side_rows[k], side_rows[k + 1]
            if re.match(r"^(?:add|less)\b", nxt[0], re.IGNORECASE) and cur[2] and not cur[1] and nxt[1] and nxt[2]:
                cur[1], cur[2] = cur[2], ""
        # Grouped balances: "A 30,000 / B 20,000 50,000" — earlier members are inner amounts too.
        for k, row in enumerate(side_rows):
            if not (row[1] and row[2]) or re.match(r"^(?:add|less)\b", row[0], re.IGNORECASE):
                continue
            target, acc, members = _to_num(row[2]), _to_num(row[1]), []
            j = k - 1
            while j >= 0 and side_rows[j][2] and not side_rows[j][1] and not side_rows[j][3] and side_rows[j][0]:
                acc += _to_num(side_rows[j][2])
                members.append(j)
                if target is not None and abs(acc - target) < 0.5:
                    for m in members:
                        side_rows[m][1], side_rows[m][2] = side_rows[m][2], ""
                    break
                j -= 1

    flush_equal()
    verified = _verify_t_totals(left, right, total_marks)
    # Vocabulary-placed entries that don't add up to the printed totals are a guess; don't show it.
    if verified is False and not line_mode:
        return None
    has_inner = any(r[1] for r in left + right)
    rows, section_rows = [], []
    for k, (l, r) in enumerate(zip(left, right)):
        if has_inner:
            rows.append([l[0], l[1], l[2], r[0], r[1], r[2]])
        else:
            rows.append([l[0], l[2], r[0], r[2]])
    if has_inner:
        headers = [lh, "", f"Amount ({RUPEE})", rh, "", f"Amount ({RUPEE})"]
        amount_cols = [1, 2, 4, 5]
    else:
        headers = [lh, f"Amount ({RUPEE})", rh, f"Amount ({RUPEE})"]
        amount_cols = [1, 3]
    tbl = _table(headers, rows, "t_account", amount_cols, total_marks, section_rows)
    if verified is not None:
        tbl["totals_verified"] = verified
    return tbl


def _to_num(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return 0.0
    neg = s.startswith("(") and s.endswith(")")
    try:
        v = float(s.strip("()").replace(",", ""))
    except ValueError:
        return None
    return -v if neg else v


def _verify_t_totals(left: List[list], right: List[list], total_rows: List[int]) -> Optional[bool]:
    """Checks each side's outer amounts against every printed totals row.

    True  — every total matches on both sides (layout is right).
    False — some total disagrees (entries probably sit on the wrong side).
    None  — nothing to check."""
    if not total_rows:
        return None
    ok = True
    checked = False
    for side_rows in (left, right):
        start = 0
        for t in total_rows:
            if t >= len(side_rows):
                continue
            expected = _to_num(side_rows[t][2])
            if not side_rows[t][2] or expected is None:
                start = t + 1
                continue
            total = 0.0
            for k in range(start, t):
                label, _inner, outer, is_heading = side_rows[k]
                if is_heading or not outer:
                    continue
                nxt = side_rows[k + 1] if k + 1 < t else None
                # In an adjustment chain only the last running figure counts.
                if nxt and re.match(r"^(?:add|less)\b", nxt[0], re.IGNORECASE):
                    continue
                v = _to_num(outer)
                if v is None:
                    return None
                total += abs(v)
            checked = True
            if abs(total - expected) > 0.5:
                ok = False
            start = t + 1
    return ok if checked else None


# ── Ledger accounts ─────────────────────────────────────────────────────

_LEDGER_HDR = re.compile(
    r"(?P<name>(?:[A-Z][\w’'&.-]*\s+){0,5}(?:Account|A/c))\s*:?\s+Dr\.?\s+Cr\.?\s+Date\s+Particulars\s+[JL]\.\s?F\.?\s+Amount\s+"
    r"Date\s+Particulars\s+[JL]\.\s?F\.?\s+Amount(?:\s*₹)*",
)


def _build_ledgers(text: str) -> Optional[List[dict]]:
    """T-ledgers whose Dr/Cr sides were interleaved by extraction.

    The side of each posting cannot be recovered reliably from the flattened stream, so the entries
    are laid out in their printed order per account (Date | Particulars | Amount) — faithful to the
    source rather than guessing Dr/Cr.
    """
    heads = list(_LEDGER_HDR.finditer(text))
    if not heads:
        return None
    segs: List[dict] = []
    intro = text[:heads[0].start()].strip()
    if intro:
        segs.append({"type": "text", "content": intro})
    for k, h in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        body = text[h.end():end]
        items, trailing = _tokenize_items(body)
        rows, totals = [], []
        year = ""
        for it in items:
            label = it.label
            ym = re.findall(r"(?:19|20)\d{2}", label)
            label = re.sub(r"(?:^|\s)(?:19|20)\d{2}(?=\s|$)", " ", label).strip()
            if ym:
                year = ym[-1]
            dm = re.match(r"^(" + _MONTH + r"\.?\s*\d{1,2})\s+(.*)$", label)
            date, part = (dm.group(1), dm.group(2)) if dm else ("", label)
            if date and year:
                date, year = f"{year} {date}", ""
            if not it.amounts and not part:
                continue
            amts = it.amounts or [""]
            if part:
                rows.append([date, part, amts[0]])
                amts = amts[1:]
            for a in amts:
                # Dr. and Cr. totals print as an equal pair — show it once.
                if rows and rows[-1][1] == "Total" and rows[-1][2] == a:
                    continue
                totals.append(len(rows))
                rows.append(["", "Total", a])
        if len(rows) >= 2:
            segs.append({"type": "table", "table": _table(
                ["Date", "Particulars", f"Amount ({RUPEE})"], rows, "ledger", [2], totals,
                caption=h.group("name").strip(),
            )})
            # Extraction interleaved the Dr./Cr. sides; entries are kept in printed order.
            segs[-1]["table"]["sides_unknown"] = True
        if trailing:
            segs.append({"type": "text", "content": trailing})
    return segs if any(s["type"] == "table" for s in segs) else None


# ── Pipe tables ─────────────────────────────────────────────────────────

def _split_pipe_cells(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


_SEP_LINE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _norm_header_cell(h: str) -> str:
    h = h.strip().strip("*")
    h = re.sub(r"\((?:Rs\.?|₹|in\s+Rs\.?)\)", f"({RUPEE})", h)
    if h in (RUPEE, "Rs.", "Rs", f"({RUPEE})"):
        return f"Amount ({RUPEE})"
    return h


def _looks_section_line(ln: str) -> bool:
    s = ln.strip()
    if not s or len(s) > 90 or s.endswith(":"):
        return False
    if re.search(r"[.?]$", s) and not re.match(r"^(?:[IVX]+\.|\d+\.|[a-z]\))", s):
        return False
    if _INSTRUCTION_START.match(s):
        return False
    return len(s.split()) <= 10


_CAPTION_WORDS = re.compile(r"\b(?:Account|A/c|Statement|Balance\s+Sheet|Book|Balances?|Schedule|Notes?)\b", re.IGNORECASE)


def _parse_multiline_pipes(text: str) -> Optional[List[dict]]:
    lines = text.split("\n")
    pipe_idx = [i for i, ln in enumerate(lines) if ln.count("|") >= 1 and not _SEP_LINE.match(ln)]
    if len(pipe_idx) < 2:
        return None

    segs: List[dict] = []
    buf: List[str] = []
    i = 0
    n = len(lines)

    def is_pipe(j: int) -> bool:
        return lines[j].count("|") >= 1 and not _SEP_LINE.match(lines[j])

    def starts_table(j: int) -> bool:
        # A header row followed by a |---| separator opens a new table.
        return is_pipe(j) and j + 1 < n and bool(_SEP_LINE.match(lines[j + 1]))

    def flush_text():
        if buf:
            content = "\n".join(buf).strip()
            if content:
                segs.extend(_parse_block(content))
            buf.clear()

    while i < n:
        ln = lines[i]
        if not is_pipe(i):
            buf.append(ln)
            i += 1
            continue
        # Need a second pipe line within the next 3 lines to call it a table.
        if not any(lines[j].count("|") >= 1 for j in range(i + 1, min(n, i + 4))):
            buf.append(ln)
            i += 1
            continue
        # "Rawat's Capital Account" printed directly above the table is its caption.
        caption = None
        if buf and buf[-1].strip():
            cand = buf[-1].strip()
            if (len(cand) < 100 and len(cand.split()) <= 12 and not cand.endswith((":", ".", "?", ")")) and not _is_instruction(cand)
                    and (_CAPTION_RE.search(cand) or _CAPTION_WORDS.search(cand))):
                caption = cand
                buf.pop()
        flush_text()
        header = [_norm_header_cell(c) for c in _split_pipe_cells(ln)]
        ncol = len(header)
        rows, section_rows, total_rows = [], [], []
        if any(_is_amount_token(c) for c in header[1:]) or header[0].lower().startswith(("total", "a)", "i)", "(a)", "(i)")):
            # No header line (e.g. note schedules) — the first pipe line is already data.
            rows.append(_split_pipe_cells(ln))
            header = ["Particulars"] + [f"Amount ({RUPEE})"] * (ncol - 1)
        i += 1
        while i < n:
            cur = lines[i]
            if _SEP_LINE.match(cur):
                i += 1
                continue
            if is_pipe(i):
                if rows and starts_table(i):
                    break
                cells = _split_pipe_cells(cur)
                cells = (cells + [""] * ncol)[:ncol] if len(cells) <= ncol else cells[:ncol - 1] + [" ".join(cells[ncol - 1:])]
                if _is_total_label(cells[0]) and any(_is_amount_token(c) for c in cells[1:]):
                    total_rows.append(len(rows))
                rows.append(cells)
                i += 1
                continue
            if not cur.strip():
                # Blank line: continue only if the same table resumes shortly (section break inside a statement).
                look = [j for j in range(i + 1, min(n, i + 6)) if lines[j].strip()]
                first_pipe = next((x for x, j in enumerate(look) if is_pipe(j)), None)
                if (first_pipe is not None and not starts_table(look[first_pipe])
                        and all(_looks_section_line(lines[j]) for j in look[:first_pipe])):
                    i += 1
                    continue
                break
            # Non-pipe line inside the block — a section heading only if more rows of this table follow.
            ahead = [j for j in range(i + 1, min(n, i + 4)) if lines[j].strip()]
            nxt_pipe = next((j for j in ahead if is_pipe(j)), None)
            if _looks_section_line(cur) and nxt_pipe is not None and not starts_table(nxt_pipe):
                section_rows.append(len(rows))
                rows.append([cur.strip()] + [""] * (ncol - 1))
                i += 1
                continue
            break
        data_rows = [r for k, r in enumerate(rows) if k not in section_rows]
        if len(data_rows) >= 1 and ncol >= 2:
            segs.append({"type": "table", "table": _finalize_generic(header, rows, "pipe", total_rows, section_rows, caption)})
        elif caption:
            buf.append(caption)
    flush_text()
    return segs if any(s["type"] == "table" for s in segs) else None


def _finalize_generic(headers, rows, kind, total_rows, section_rows, caption=None) -> dict:
    ncol = len(headers)
    amount_cols = []
    for c in range(1, ncol):
        vals = [r[c] for k, r in enumerate(rows) if k not in section_rows and c < len(r) and r[c].strip()]
        hc = headers[c].lower()
        if (vals and sum(_is_amount_token(v) for v in vals) >= 0.6 * len(vals)) or re.search(r"amount|debit|credit|₹|rs|\d{4}", hc):
            if not re.search(r"note|j\.?\s?f|l\.?\s?f|date|ratio", hc):
                amount_cols.append(c)
    for k, r in enumerate(rows):
        if k in section_rows or k in total_rows:
            continue
        if _is_total_label(r[0]) and r[0] and any(r[c] for c in amount_cols):
            total_rows.append(k)
        # "| I. Equity and Liabilities | | | |" — a heading row written as a markdown row.
        elif ncol > 1 and r[0].strip() and all(not c.strip() for c in r[1:]) and _looks_section_line(r[0]) and k + 1 < len(rows):
            section_rows.append(k)
    return _table(headers, rows, kind, amount_cols, total_rows, section_rows, caption)


def _parse_inline_pipes(text: str) -> Optional[Tuple[str, dict, str]]:
    """Single-line pipe table: 'Particulars | ₹ Opening stock | 37,500 Purchases | 1,05,000 ...'."""
    if text.count("|") < 3 or "\n" in text:
        return None
    first = text.find("|")
    pre = text[:first]
    # The column title is the last header word before the first pipe, after any sentence end
    # ("Prepare ... from the following particulars for the year ...: Particulars | ...").
    heads = [h for h in re.finditer(r"\b(?:Particulars|Accounts?\s+Titles?|Items?|Details|Date|Description|Liabilities|Expenses)\b",
                                    pre, re.IGNORECASE)
             if not re.search(r"[:.]\s", pre[h.start():])]
    if not heads:
        return None
    start = heads[0].start()
    intro = pre[:start].strip()
    body = text[start:]
    raw_cells = [c.strip() for c in body.split("|")]

    # Re-split cells where one row's last amount ran into the next row's label.
    stream: List[Optional[str]] = []  # None = row break
    for c in raw_cells:
        m = re.match(r"^((?:₹|Rs\.)?\s*\(?\d[\d,]*(?:\.\d+)?\)?|₹|Rs\.)\s+(\D.*)$", c)
        if m and not re.match(r"^\d+\s*%", c):
            stream.extend([m.group(1).strip(), None, m.group(2).strip()])
        else:
            stream.append(c)
    rows: List[List[str]] = [[]]
    for tok in stream:
        if tok is None:
            rows.append([])
        else:
            rows[-1].append(tok)
    rows = [r for r in rows if any(x for x in r)]
    if len(rows) < 3:
        return None
    header = [_norm_header_cell(h) for h in rows[0]]
    body_rows = rows[1:]
    ncol = max(len(header), max(len(r) for r in body_rows))
    if ncol < 2:
        return None
    header = (header + [f"Amount ({RUPEE})"] * ncol)[:ncol]
    # The last cell may carry trailing instruction text.
    trailing = ""
    last = body_rows[-1]
    if last and not _is_amount_token(last[-1]):
        tail_items, tr = _tokenize_items(last[-1])
        if tr:
            trailing = tr
            last[-1] = last[-1][: last[-1].rfind(tr)].strip()
    body_rows = [(r + [""] * ncol)[:ncol] for r in body_rows]
    if sum(1 for r in body_rows if any(_is_amount_token(c) for c in r[1:])) < 2:
        return None
    return intro, _finalize_generic(header, body_rows, "pipe", [], []), trailing


# ── "Label: Rs. 2,40,000" listings ──────────────────────────────────────

_KV_LINE = re.compile(
    r"^(?P<label>[^:\n]{2,80}?)\s*[:\-–]\s*(?:(?:₹|Rs\.)\s*)?(?P<amt>\(?\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?\)?|\(?\d+(?:\.\d{1,2})?\)?)"
    r"(?:\s*\((?P<side>Dr|Cr)\.?\))?\s*[.;,]?$",
    re.IGNORECASE,
)
_ITEM_LINE = re.compile(
    r"^(?P<label>[A-Za-z(][^\n]{1,80}?)\s+(?:(?:₹|Rs\.)\s*)?(?P<amt>\(?\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?\)?|\(?\d{2,}(?:\.\d{1,2})?\)?)"
    r"(?:\s*\((?P<side>Dr|Cr)\.?\))?\s*[.;,]?$",
    re.IGNORECASE,
)


def _parse_line_listing(text: str) -> Optional[List[dict]]:
    """Consecutive 'Label: Rs. N' / 'Label ₹ N (Dr)' lines -> tables (with optional caption line)."""
    lines = text.split("\n")
    segs: List[dict] = []
    buf: List[str] = []
    i = 0
    found = False
    while i < len(lines):
        run = []
        j = i
        while j < len(lines):
            ln = lines[j].strip()
            m = _KV_LINE.match(ln) or (_ITEM_LINE.match(ln) if (RUPEE in ln or "Rs." in ln or "(Dr" in ln or "(Cr" in ln) else None)
            if not m or re.search(r"\b(?:19|20)\d{2}\s*$", m.group("label")) and not m.group("side"):
                break
            if len(m.group("label").split()) > 12:
                break
            run.append(m)
            j += 1
        under_notes = any(_STOP_LINE.match(b.strip()) for b in buf[-12:])
        bad = sum(1 for m in run if _label_is_bad(_clean_label(m.group("label"))))
        if under_notes or bad > len(run) // 7:
            run = []
        if len(run) >= 3 or (len(run) >= 2 and buf and buf[-1].rstrip().endswith(":") and len(buf[-1]) < 60):
            caption = None
            if buf and buf[-1].rstrip().endswith(":") and len(buf[-1].strip()) < 60 and not _is_instruction(buf[-1]):
                caption = buf.pop().strip().rstrip(":")
            if buf:
                segs.append({"type": "text", "content": "\n".join(buf).strip()})
                buf = []
            has_side = any(m.group("side") for m in run)
            if has_side:
                rows = []
                for m in run:
                    side = (m.group("side") or "").lower()
                    amt = _clean_amount(m.group("amt"))
                    rows.append([_clean_label(m.group("label")), amt if side != "cr" else "", amt if side == "cr" else ""])
                tbl = _table(["Particulars", f"Debit ({RUPEE})", f"Credit ({RUPEE})"], rows, "trial_balance", [1, 2], caption=caption)
            else:
                rows = [[_clean_label(m.group("label")), _clean_amount(m.group("amt"))] for m in run]
                tbl = _finalize_generic(["Particulars", f"Amount ({RUPEE})"], rows, "list", [], [], caption)
            segs.append({"type": "table", "table": tbl})
            found = True
            i = j
            continue
        buf.append(lines[i])
        i += 1
    if buf:
        segs.append({"type": "text", "content": "\n".join(buf).strip()})
    return segs if found else None


_INLINE_KV = re.compile(
    r"(?P<label>[A-Z][A-Za-z0-9%&’'()./ -]{1,60}?)\s*:\s*(?P<val>(?:(?:(?:19|20)\d{2}\s*)?(?:₹|Rs\.)\s*\(?\d[\d,]*(?:\.\d+)?\)?[,\s]*(?:and\s+)?){1,4})(?:[;.]\s*|\s+(?=[A-Z])|$)"
)


def _parse_inline_kv(text: str) -> Optional[Tuple[str, dict, str]]:
    """'Preliminary expenses: Rs. 2,40,000 Goodwill: Rs. 30,000 ...' and
    'Share Capital: 2017 Rs. 1,300, 2016 Rs. 1,400; Reserve ...: 2017 Rs. 4,700, 2016 Rs. 4,000'."""
    ms = list(_INLINE_KV.finditer(text))
    if len(ms) < 3:
        return None
    # Must be one contiguous run.
    run = [ms[0]]
    for m in ms[1:]:
        if m.start() - run[-1].end() <= 2:
            run.append(m)
        elif len(run) < 3:
            run = [m]
        else:
            break
    if len(run) < 3:
        return None
    parsed = []
    year_cols: List[str] = []
    for m in run:
        pairs = re.findall(r"(?:((?:19|20)\d{2})\s*)?(?:₹|Rs\.)\s*(\(?\d[\d,]*(?:\.\d+)?\)?)", m.group("val"))
        parsed.append((m.group("label").strip(), pairs))
        for y, _ in pairs:
            if y and y not in year_cols:
                year_cols.append(y)
    if year_cols:
        headers = ["Particulars"] + [f"{y} ({RUPEE})" for y in year_cols]
        rows = []
        for label, pairs in parsed:
            vals = {y: _clean_amount(a) for y, a in pairs if y}
            rows.append([label] + [vals.get(y, "") for y in year_cols])
        amount_cols = range(1, 1 + len(year_cols))
    else:
        headers = ["Particulars", f"Amount ({RUPEE})"]
        rows = [[label, _clean_amount(pairs[0][1])] for label, pairs in parsed if pairs]
        amount_cols = [1]
    intro = text[:run[0].start()].strip()
    trailing = text[run[-1].end():].strip()
    return intro, _table(headers, rows, "list", amount_cols), trailing


_YEAR_SERIES = re.compile(
    r"(?P<y>(?:19|20)\d{2}(?:-\d{2})?)\s*[:\-–]?\s*(?:(?:₹|Rs\.)\s*)?(?P<a>\(?\d{1,3}(?:,\d{2,3})+\)?(?:\s*\((?:loss|profit)\))?)"
)


def _parse_year_series(text: str) -> Optional[Tuple[str, dict, str]]:
    """'profits for the last five years were: 2011–Rs. 40,000; 2012-Rs. 50,000; ...' -> Year | Amount."""
    ms = list(_YEAR_SERIES.finditer(text))
    if len(ms) < 3:
        return None
    run = [ms[0]]
    for m in ms[1:]:
        gap = text[run[-1].end():m.start()]
        if re.fullmatch(r"\s*(?:[;,:.]|and)?\s*(?:and\s*)?", gap):
            run.append(m)
        elif len(run) >= 3:
            break
        else:
            run = [m]
    if len(run) < 3:
        return None
    label = "Profit/(Loss)" if re.search(r"profit|loss", text, re.I) else "Amount"
    rows = [[m.group("y"), _clean_amount(m.group("a"))] for m in run]
    intro = text[:run[0].start()].rstrip(" ;:,")
    intro = re.sub(r"\s*(?:₹|Rs\.)\s*$", "", intro).strip()
    trailing = text[run[-1].end():].lstrip(" ;,.").strip()
    return intro + (":" if intro and not intro.endswith((":", ".")) else ""), _table(
        ["Year", f"{label} ({RUPEE})"], rows, "list", [1]), trailing


# ═══════════════════════════════════════════════════════════════════════
# Flat-text orchestration
# ═══════════════════════════════════════════════════════════════════════

_INTRO_CUE = re.compile(
    r"(?:following|given|below|these)\s+(?:particulars|balances|information|details|items|figures|data|transactions|"
    r"accounts?|trial\s+balance|expenses|incomes|assets|liabilities)[^:]{0,120}:\s*(?:₹|Rs\.)?",
    re.IGNORECASE,
)
# Where a flattened table stops and the question's adjustments/notes begin.
_STOP_IN_STREAM = re.compile(
    r"(?:(?<=\s)|^)(?:Adjustments?|Additional\s+[Ii]nformation|Other\s+[Ii]nformation|Notes?\s+to\s+[Aa]ccounts|Information)\s*:?\s+"
    r"(?=\(?(?:\d{1,2}|[ivx]{1,4}|[a-h])[.)]|[A-Z])"
)
_ENUM_RE = re.compile(r"(?:(?<=\s)|^)\((?:[ivx]{1,4}|[a-h]|\d{1,2})\)")
# Abbreviations that never end a sentence ("Ltd." / "Co." can, so they are not listed).
_ABBREV_END = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Messrs|M/s|No|Nos|Rs|Pvt|Sr|Jr|St|viz|i\.e|e\.g)|\b[A-Z])$")


def _cut_body(body: str) -> Tuple[str, str]:
    """Splits a stream at "Adjustments:" / "Additional Information" so those notes stay text."""
    m = _STOP_IN_STREAM.search(body)
    if m and m.start() > 0:
        return body[:m.start()], body[m.start():].strip()
    return body, ""


def _join_trailing(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _month_from(intro: str) -> str:
    m = re.search(r"\b(" + _MONTH + r")\b\.?,?\s*(?:(?:19|20)\d{2})", intro)
    if not m:
        return ""
    name = m.group(1)
    return name[:4] + "." if name.lower().startswith("sep") else name[:3] + "."


_SEMI_PART = re.compile(
    r"^(?P<label>[^;]{1,90}?)\s*[:\-–]?\s*(?:(?:₹|Rs\.)\s*)?(?P<amt>\(?\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?\)?|\(?\d{2,}(?:\.\d{1,2})?\)?)"
    r"(?:\s*\((?P<side>Dr|Cr)\.?\))?\s*\.?$",
    re.IGNORECASE,
)


def _parse_semicolon_list(text: str) -> Optional[List[dict]]:
    """'Building Rs. 10,00,000; Investments ... Rs. 3,00,000; ...' and
    'Debit balances Amount (₹): Plant and Machinery 1,30,000; Debtors 50,000; ... Total: 8,70,430.'"""
    if text.count(";") < 2:
        return None
    colon = re.match(r"^(?P<cap>[^;:]{3,120}?):\s*(?=[A-Z(])", text)
    prefix = ""
    body = text
    if colon and not _find_amounts(colon.group("cap")):
        prefix, body = colon.group("cap").strip(), text[colon.end():]
    else:
        # Intro sentence ends before the first list item (skip "Rs." / "Ltd." style abbreviations).
        first_semi = body.find(";")
        for m in reversed(list(re.finditer(r"[.:]\s+", body[:first_semi]))):
            if body[m.start()] == "." and _ABBREV_END.search(body[:m.start()]):
                continue
            prefix, body = body[:m.start() + 1].strip(), body[m.end():]
            break
    body = body.strip()
    trailing = ""
    parts = re.split(r"\s*;\s*|\.\s+(?=Total\b)", body)
    # The last part may carry a closing sentence ("... Rs. 10,000. The company ...").
    last = parts[-1]
    lm = re.match(r"^(.*?\d[\d,]*\)?)\s*\.\s+([A-Z].*)$", last)
    if lm:
        parts[-1], trailing = lm.group(1), lm.group(2)
    rows, matched, totals = [], 0, []
    for p in parts:
        p = p.strip().rstrip(".")
        if not p:
            continue
        m = _SEMI_PART.match(p)
        if m and len(m.group("label").split()) <= 14 and not _label_is_bad(_clean_label(m.group("label"))):
            label = _clean_label(m.group("label"))
            if _is_total_label(label):
                totals.append(len(rows))
            rows.append([label, _clean_amount(m.group("amt"))])
            matched += 1
        else:
            rows.append([_clean_label(p), ""])
    if matched < 3 or matched < 0.75 * len(rows):
        return None
    caption = None
    intro = prefix
    if prefix and re.search(r"balances?|liabilities|assets|amount", prefix, re.I) and len(prefix) < 60:
        caption = re.sub(r"\s*Amount\s*\((?:₹|Rs\.?)\)\s*$", "", prefix).strip()
        intro = ""
    return _segments(intro, _table(["Particulars", f"Amount ({RUPEE})"], rows, "list", [1], totals, caption=caption), trailing)


def _parse_flat(text: str) -> Optional[List[dict]]:
    """Detects a single flattened table inside a paragraph with no newlines."""
    if len(_find_amounts(text)) < 3 and "|" not in text:
        return None

    ledgers = _build_ledgers(text)
    if ledgers:
        return ledgers

    piped = _parse_inline_pipes(text)
    if piped:
        intro, tbl, trailing = piped
        return _segments(intro, tbl, trailing)

    if text.count(";") >= 3:
        semi = _parse_semicolon_list(text)
        if semi:
            return semi

    hdr = _find_header(text)
    if hdr:
        intro = text[:hdr.start].strip()
        # Caption: "Balance Sheet of Ankit as at March 31, 2017" right before the header.
        caption = None
        cm = re.search(r"((?:Balance\s+Sheet|Trading(?:\s+and\s+Profit\s+and\s+Loss)?\s+Account|Profit\s+and\s+Loss\s+Account|"
                       r"Trial\s+Balance|Cash\s+Book|Receipts\s+and\s+Payments\s+Account)[^.:]{0,80}?)\s*(?:Dr\.?\s+Cr\.?)?\s*$", intro)
        if cm and cm.start() > 0 and not re.search(r"(?:the|a|an|following|of)\s*$", intro[:cm.start()], re.I):
            caption = cm.group(1).strip()
            intro = intro[:cm.start()].strip()
        elif cm and cm.start() == 0:
            caption, intro = cm.group(1).strip(), ""
        body, stop_tail = _cut_body(text[hdr.end:])
        tbl, trailing = None, ""
        if hdr.kind == "dated":
            ym = re.search(r"((?:19|20)\d{2})\s*$", hdr.text)
            res = _build_dated_table((ym.group(1) + " " + body) if ym else body, hdr.text, _month_from(intro))
            if res:
                tbl, trailing = res
        if tbl is None:
            items, trailing = _tokenize_items(body)
            if hdr.kind in ("balance_sheet", "trading_pl"):
                tbl = _build_t_account([items], hdr.kind, hdr.text)
            elif _is_two_sided(hdr.text, hdr.kind) and re.search(r"trial\s+balance|debit|credit", intro + " " + hdr.text, re.I):
                # "Particulars Amount Particulars Amount" of a trial balance: Dr. entries left, Cr. right.
                tbl = _build_t_account([items], "two_sided", "Debit Credit " + hdr.text)
            if tbl is None and _valid_stream(items, 12):
                tbl = _build_trial_balance(items, hdr.text) if hdr.kind == "trial_balance" else _build_list_table(items, hdr.text)
        if tbl is not None:
            if caption:
                tbl["caption"] = caption
            return _segments(intro, tbl, _join_trailing(trailing, stop_tail))
        if caption:
            intro = f"{intro} {caption}".strip()

    semi = _parse_semicolon_list(text)
    if semi:
        return semi

    # Date-driven transaction lists without a formal header.
    first_date = _DATE_TOKEN.search(text)
    cue = re.search(r"(?:transactions?|following|information|details)[^:]{0,160}:\s*(?:₹\s*)?", text, re.IGNORECASE)
    if cue:
        month = _month_from(text[:cue.end()])
        if first_date or month:
            body, stop_tail = _cut_body(text[cue.end():])
            res = _build_dated_table(body, "", month)
            if res and len(res[0]["rows"]) >= 3:
                return _segments(text[:cue.end()].strip().rstrip("₹").strip(), res[0], _join_trailing(res[1], stop_tail))

    kv = _parse_inline_kv(text)
    if kv:
        return _segments(*kv)

    # Headerless stream after "... following particulars/balances: ₹".
    cue = _INTRO_CUE.search(text)
    if cue:
        body, stop_tail = _cut_body(text[cue.end():])
        intro = text[:cue.end()]
        # "(i) Gross Profit Ratio (ii) Current Ratio Rs. Gross Profit 50,000 ..." — a bare currency
        # column header marks where the figures start.
        first_amt = _find_amounts(body)
        cur = None
        if first_amt:
            cur = list(re.finditer(r"(?:(?<=\s)|^)(?:₹|Rs\.)\s+(?=[A-Z(])", body[:first_amt[0].start]))
        if cur:
            intro, body = intro + body[:cur[-1].end()], body[cur[-1].end():]
        items, trailing = _tokenize_items(body)
        marker = re.search(r"(?:₹|Rs\.)\s*$", intro)
        if _valid_stream(items, 8 if marker else 6, 3 if marker else 4):
            intro = re.sub(r"\s*(?:₹|Rs\.)\s*$", "", intro.strip()).strip()
            return _segments(intro, _build_list_table(items, ""), _join_trailing(trailing, stop_tail))

    # Currency-marked stream ("Ram Kumar ₹ 1,000 Kishore Kumar ₹ 500 (b)Bank Charges ₹ 300").
    marked = list(re.finditer(r"(?:₹|Rs\.)\s*\(?\d", text))
    if len(marked) >= 3:
        start = _stream_start(text, marked[0].start())
        body, stop_tail = _cut_body(text[start:])
        items, trailing = _tokenize_items(body)
        if _valid_stream(items, 14):
            return _segments(text[:start].strip(), _build_list_table(items, ""), _join_trailing(trailing, stop_tail))

    ys = _parse_year_series(text)
    if ys:
        return _segments(*ys)
    return None


def _stream_start(text: str, first_marker: int) -> int:
    """Start of a currency-marked stream.

    Prefers the first "(i)"/"(a)" enumerator when the items are enumerated; otherwise the last
    sentence end or colon before the first amount (ignoring "Mr." / "Ltd." style abbreviations)."""
    enums = list(_ENUM_RE.finditer(text))
    if len(enums) >= 3:
        return enums[0].start()
    pre = text[:first_marker]
    for m in reversed(list(re.finditer(r"[:.?]\s+", pre))):
        if pre[m.start()] == "." and _ABBREV_END.search(pre[:m.start()]):
            continue
        return m.end()
    return 0


def _segments(intro: str, tbl: dict, trailing: str) -> List[dict]:
    out: List[dict] = []
    if intro and intro.strip(" :"):
        out.append({"type": "text", "content": intro.strip()})
    out.append({"type": "table", "table": tbl})
    if trailing and trailing.strip(" .;,"):
        # Trailing text may itself hold another inline table (e.g. two statements in one question).
        sub = _parse_flat(trailing.strip()) if len(trailing) > 40 else None
        out.extend(sub or [{"type": "text", "content": trailing.strip()}])
    return out


_CAPTION_RE = re.compile(
    r"((?:[A-Z][\w’'.&-]*\s+){0,6}(?:Balance\s+Sheet|Trading(?:\s+and\s+Profit\s+and\s+Loss)?\s+Account|Profit\s+and\s+Loss\s+Account|"
    r"Trial\s+Balance|Cash\s+Book|Receipts\s+and\s+Payments\s+Account|Income\s+and\s+Expenditure\s+Account)[^.:]{0,80}?)\s*(?:Dr\.?\s+Cr\.?)?\s*$"
)
_CURRENCY_ONLY = re.compile(r"^(?:[₹|\s:/]|\((?:Rs\.?|₹)\)|Rs\.|Dr\.?|Cr\.?)*$")
_STOP_LINE = re.compile(
    r"^(?:Adjustments?|Additional\s+[Ii]nformation|Other\s+[Ii]nformation|Notes?(?:\s+to\s+[Aa]ccounts)?|Information|Required|"
    r"Hint)\b\s*:?"
)


def _parse_lined(text: str) -> Optional[List[dict]]:
    """Tables whose rows kept their line breaks but lost column separators:

        Debit Balances Amount Credit balances Amount
        ₹ ₹
        Cash 20,000 Sales 3,61,000
        Debtors (Including a 60,000
        dishonoured bill of ₹1,600) Stock 81,600
        5,38,060 5,38,060
    """
    lines = text.split("\n")
    for idx, ln in enumerate(lines):
        hdr = _find_header(ln)
        if not hdr:
            continue
        prefix = ln[:hdr.start].strip()
        caption = None
        if prefix:
            cm = _CAPTION_RE.search(prefix)
            if cm and cm.start() == 0:
                caption, prefix = cm.group(1).strip(), ""
            else:
                continue
        caption_line_used = False
        if caption is None and idx > 0:
            cm = _CAPTION_RE.search(lines[idx - 1].strip())
            if cm and cm.start() == 0 and len(lines[idx - 1].strip()) < 110:
                caption, caption_line_used = cm.group(1).strip(), True

        body_lines = [ln[hdr.end:]] + lines[idx + 1:]
        rows_raw: List[str] = []
        j = 0
        while j < len(body_lines):
            b = body_lines[j].strip()
            if not b:
                if rows_raw:
                    break
                j += 1
                continue
            if _CURRENCY_ONLY.match(b):
                j += 1
                continue
            if _STOP_LINE.match(b) or (_is_instruction(b) and not re.search(r"\d{1,3},\d{2,3}\s*$", b)):
                break
            if re.match(r"^\d+\.\s+\S", b) and len(b.split()) > 5 and not re.search(r"\d\s*$", b):
                break
            if rows_raw and re.match(r"^(?:[a-z]|\((?![a-z0-9]{1,4}\))|&)", b):
                rows_raw[-1] += " " + b  # wrapped label
            else:
                rows_raw.append(b)
            j += 1
        if len(rows_raw) < 2:
            continue

        consumed_to = idx + max(j, 1)
        kind = hdr.kind
        tbl = None
        tail_text = ""
        if kind == "dated":
            res = _build_dated_table(" ".join(rows_raw), hdr.text)
            if res:
                tbl, tail_text = res
        else:
            item_lines, all_items = [], []
            for k, r in enumerate(rows_raw):
                its, tr = _tokenize_items(r)
                if tr:
                    if k == len(rows_raw) - 1:
                        tail_text = tr
                    else:
                        its.append(_Item(_clean_label(tr), []))
                item_lines.append(its)
                all_items.extend(its)
            if not _valid_stream(all_items, 14, 2):
                continue
            if _is_two_sided(hdr.text, kind):
                t_kind = kind if kind in ("balance_sheet", "trading_pl", "receipts_payments") else "two_sided"
                tbl = _build_t_account(item_lines, t_kind, hdr.text)
                if tbl is not None and tbl.get("totals_verified") is False:
                    # Try reading "Capital Investment 38,500" as heading "Capital" + right-side "Investment".
                    alt = _build_t_account(item_lines, t_kind, hdr.text, loose_headings=True)
                    if alt is not None and alt.get("totals_verified"):
                        tbl = alt
            if tbl is None:
                if kind == "trial_balance":
                    tbl = _build_trial_balance(all_items, hdr.text)
                else:
                    tbl = _build_list_table(all_items, hdr.text)
        if tbl is None:
            continue
        if caption:
            tbl["caption"] = caption

        out: List[dict] = []
        before = "\n".join(lines[:idx - 1 if caption_line_used else idx] + ([prefix] if prefix else [])).strip()
        if before:
            out.extend(_parse_block(before))
        out.append({"type": "table", "table": tbl})
        after = "\n".join(([tail_text] if tail_text else []) + lines[consumed_to:]).strip()
        if after:
            out.extend(_parse_block(after))
        return out
    return None


_DAY_LINE = re.compile(r"^(?P<date>(?:" + _MONTH + r"\.?\s*)?\d{1,2}(?:st|nd|rd|th)?)\s+(?P<rest>[A-Za-z(].*)$")


def _parse_dated_lines(text: str) -> Optional[List[dict]]:
    """'01 Cash in hand 17,500' / 'Sept. 05 Sold goods ...' lines (cash books, journals, purchase books)."""
    lines = text.split("\n")
    start = next((k for k, ln in enumerate(lines) if _DAY_LINE.match(ln.strip())), None)
    if start is None:
        return None
    rows, k = [], start
    while k < len(lines):
        ln = lines[k].strip()
        if not ln or _CURRENCY_ONLY.match(ln):
            if rows:
                break
            k += 1
            continue
        m = _DAY_LINE.match(ln)
        if m:
            date, rest = m.group("date"), m.group("rest")
        elif rows and (_find_amounts(ln) or re.match(r"^[a-z(]", ln)) and not _is_instruction(ln):
            date, rest = "", ln
        else:
            break
        amts = _find_amounts(rest)
        tail = []
        cut = len(rest)
        for a in reversed(amts):
            if re.fullmatch(r"[\s|₹]*(?:Rs\.)?[\s|₹]*", rest[a.end:cut]):
                tail.insert(0, _clean_amount(a.text))
                cut = a.start
            else:
                break
        if not m and not tail and rows:
            rows[-1][1] += " " + rest  # wrapped details
        else:
            rows.append([date, _clean_label(rest[:cut]), tail])
        k += 1
    if sum(1 for r in rows if r[0]) < 3:
        return None
    if sum(1 for r in rows if r[2]) < len(rows) / 2 and sum(len(r[1].split()) for r in rows) / len(rows) > 10:
        return None
    intro = "\n".join(lines[:start]).strip()
    month = re.search(r"\b(" + _MONTH + r")\b\.?\s*,?\s*((?:19|20)\d{2})?", intro)
    mon = ""
    if month:
        mon = month.group(1)[:3] + ("t." if month.group(1).lower().startswith("sep") else ".")
    n_amt = max(1, min(3, max(len(r[2]) for r in rows)))
    hl = intro.lower()
    if n_amt == 2 and "cash" in hl and ("bank" in hl or "double" in hl):
        amt_headers = [f"Cash ({RUPEE})", f"Bank ({RUPEE})"]
    else:
        amt_headers = [f"Amount ({RUPEE})"] * n_amt
    out_rows = []
    for date, details, vals in rows:
        if date and mon and re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", date):
            date = f"{mon} {date}"
        out_rows.append([date, details] + (vals + [""] * n_amt)[:n_amt])
    if month and month.group(2) and out_rows and out_rows[0][0]:
        out_rows[0][0] = f"{month.group(2)} {out_rows[0][0]}"
    segs: List[dict] = []
    if intro:
        segs.append({"type": "text", "content": intro.rstrip("₹").strip()})
    segs.append({"type": "table", "table": _table(["Date", "Details"] + amt_headers, out_rows, "dated", range(2, 2 + n_amt))})
    rest_text = "\n".join(lines[k:]).strip()
    if rest_text:
        segs.extend(_parse_block(rest_text))
    return segs


def _parse_block(text: str) -> List[dict]:
    """A block with no pipe table: paragraphs, line tables, listings or one flattened table."""
    text = text.strip()
    if not text:
        return []
    # Adjustments / notes stay as the teacher wrote them.
    if _STOP_LINE.match(text):
        return [{"type": "text", "content": text}]
    if "\n" not in text:
        return _parse_flat(text) or [{"type": "text", "content": text}]

    lined = _parse_lined(text)
    if lined:
        return _merge_text(lined)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) > 1:
        out: List[dict] = []
        for p in paras:
            out.extend(_parse_block(p))
        return _merge_text(out, "\n\n")
    dated = _parse_dated_lines(text)
    if dated:
        return _merge_text(dated)
    listing = _parse_line_listing(text)
    if listing:
        out = []
        for seg in listing:
            if seg["type"] == "text" and seg["content"].strip() != text:
                out.extend(_parse_block(seg["content"]))
            else:
                out.append(seg)
        return _merge_text(out)
    # A single paragraph whose items wrapped across lines: parse it flattened, but keep the
    # original line breaks when nothing is found.
    flat = _parse_flat(text.replace("\n", " "))
    return flat or [{"type": "text", "content": text}]


def _merge_text(segs: List[dict], sep: str = "\n") -> List[dict]:
    out: List[dict] = []
    for s in segs:
        if s["type"] == "text" and out and out[-1]["type"] == "text":
            out[-1] = {"type": "text", "content": out[-1]["content"] + sep + s["content"]}
        else:
            out.append(s)
    return out


_ENUM_BREAK = re.compile(r"\s+(?=\((?:[a-h]|[ivx]{1,4}|\d{1,2})\)\s?\S)")
_NUM_BREAK = re.compile(r"\s+(?=(\d{1,2})\.\s?[A-Z])")


def _break_enumerations(text: str) -> str:
    """Puts '(a) ... (b) ...' and 'Adjustments 1. ... 2. ...' items on their own lines."""
    out = []
    for line in text.split("\n"):
        if len(_ENUM_BREAK.findall(line)) >= 2 and len(_ENUM_RE.findall(line)) >= 3:
            line = _ENUM_BREAK.sub("\n", line)
        nums = [int(m.group(1)) for m in _NUM_BREAK.finditer(line)]
        if len(nums) >= 3 and nums[:3] == list(range(nums[0], nums[0] + 3)):
            line = _NUM_BREAK.sub("\n", line)
        out.append(line)
    return "\n".join(out)


@lru_cache(maxsize=4096)
def _parse_cached(text: str) -> Tuple[dict, ...]:
    norm = normalize_accountancy_text(text)
    if not norm:
        return ({"type": "text", "content": ""},)
    # Fast path: every table shape needs pipes or at least three figures.
    if "|" not in norm and len(_find_amounts(norm)) < 3:
        return ({"type": "text", "content": _break_enumerations(norm)},)
    try:
        segs = _parse_multiline_pipes(norm) if "|" in norm and "\n" in norm else None
        if segs is None:
            segs = _parse_block(norm)
    except Exception:  # never let a heuristic take down an export
        segs = [{"type": "text", "content": norm}]
    segs = [s for s in _merge_text(segs) if not (s["type"] == "text" and not s["content"].strip())]
    for s in segs:
        if s["type"] == "text":
            s["content"] = _break_enumerations(s["content"])
    return tuple(segs) or ({"type": "text", "content": norm},)


def parse_and_structure_accountancy_text(text: str) -> List[dict]:
    """Public entry point. Returns a fresh list of segments (safe to mutate)."""
    return copy.deepcopy(list(_parse_cached(text or "")))


def has_structured_table(segments: List[dict]) -> bool:
    return any(s.get("type") == "table" for s in segments)


def build_question_segments(text: str, question_table=None) -> List[dict]:
    """Ordered text/table segments for one question — shared by the PDF/DOCX exporter and the
    /ncert-questions preview so both show exactly the same tables.

    Tables are rebuilt from the question text (flattened NCERT text or canonical markdown). A stored
    question_table carries richer layout metadata, so it takes the place of the first table found in
    the text — or is appended when the text has none."""
    if isinstance(question_table, str):
        try:
            import json
            question_table = json.loads(question_table)
        except ValueError:
            question_table = None
    qt = question_table if isinstance(question_table, dict) and question_table.get("headers") and question_table.get("rows") else None
    segs = parse_and_structure_accountancy_text(text) if (text or "").strip() else []
    if qt:
        table = infer_table_meta(qt)
        idx = next((k for k, s in enumerate(segs) if s["type"] == "table"), None)
        if idx is None:
            segs.append({"type": "table", "table": table})
        else:
            segs[idx] = {"type": "table", "table": table}
            # The markdown copy carries the caption as a text line; don't print it twice.
            cap = str(table.get("caption") or "").strip()
            if cap and idx > 0 and segs[idx - 1]["type"] == "text" and segs[idx - 1]["content"].rstrip().endswith(cap):
                rest = segs[idx - 1]["content"].rstrip()[: -len(cap)].rstrip()
                if rest:
                    segs[idx - 1] = {"type": "text", "content": rest}
                else:
                    segs.pop(idx - 1)
    return segs


# ═══════════════════════════════════════════════════════════════════════
# Helpers shared by renderers
# ═══════════════════════════════════════════════════════════════════════

def infer_table_meta(table: dict) -> dict:
    """Fills amount_cols / total_rows / section_rows for tables that arrived without metadata
    (question_table rows from the DB, AI-generated answer tables, markdown tables)."""
    t = dict(table)
    headers = [str(h) for h in (t.get("headers") or [])]
    rows = [[str(c) if c is not None else "" for c in (r if isinstance(r, list) else [r])] for r in (t.get("rows") or [])]
    ncol = len(headers)
    rows = [(r + [""] * ncol)[:ncol] if ncol else r for r in rows]
    t["headers"], t["rows"] = headers, rows
    if "amount_cols" not in t:
        t["amount_cols"] = _finalize_generic(headers, rows, "generic", [], [])["amount_cols"] if ncol else []
    if "section_rows" not in t:
        t["section_rows"] = [
            k for k, r in enumerate(rows)
            if r and r[0].strip() and all(not c.strip() for c in r[1:]) and ncol > 1 and _looks_section_line(r[0])
            and k + 1 < len(rows)
        ]
    if "total_rows" not in t:
        amt = t["amount_cols"]
        text_cols = [c for c in range(ncol) if c not in amt]
        t["total_rows"] = [
            k for k, r in enumerate(rows)
            if any(r[c].strip() for c in amt if c < len(r))
            and ((r[0].strip() and _is_total_label(r[0]))
                 # T-account totals: no labels on either side, figures on both.
                 or (ncol >= 4 and all(not r[c].strip() for c in text_cols if c < len(r))
                     and sum(1 for c in amt if c < len(r) and r[c].strip()) >= 2))
        ]
    t.setdefault("kind", "generic")
    return t


def table_to_markdown(table: dict) -> str:
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    esc = lambda c: str(c).replace("|", "/").replace("\n", " ").strip()
    lines = ["| " + " | ".join(esc(h) or " " for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    out = "\n".join(lines)
    if table.get("caption"):
        out = f"{table['caption']}\n{out}"
    return out


def segments_to_markdown(segments: List[dict]) -> str:
    parts = []
    for s in segments:
        if s["type"] == "text":
            parts.append(s["content"].strip())
        else:
            parts.append(table_to_markdown(s["table"]))
    return "\n\n".join(p for p in parts if p)
