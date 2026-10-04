#!/usr/bin/env python3
"""
Regression checks for app/services/accountancy_parser.py (no pytest needed).

    python scripts/check_accountancy_parser.py

Covers the flattened NCERT shapes the exporter must rebuild: inline pipe lists, trial-balance
streams, cash books, T-shape balance sheets / P&L, "Label: Rs. N" listings, and prose that must
stay prose.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.accountancy_parser import (  # noqa: E402
    parse_and_structure_accountancy_text as parse,
    segments_to_markdown,
)

FAILURES = []


def tables(text):
    return [s["table"] for s in parse(text) if s["type"] == "table"]


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(f"{name}: {detail}")


# Format A — inline pipe list
t = tables("Prepare a trading account from the following particulars for the year ended March 31, 2017: "
           "Particulars | ` Opening stock | 37,500 Purchases | 1,05,000 Sales | 2,70,000 Wages | 30,000")
check("A: inline pipes -> 4 rows", len(t) == 1 and t[0]["rows"] == [
    ["Opening stock", "37,500"], ["Purchases", "1,05,000"], ["Sales", "2,70,000"], ["Wages", "30,000"]], t)
check("A: header is Particulars | Amount (₹)", t and t[0]["headers"] == ["Particulars", "Amount (₹)"], t and t[0]["headers"])

# Format B — trial balance stream (backtick rupee, split lakh figure)
t = tables("prepare a trial balance as at March 31, 2014 based on the following balances: Accounts Title Amount ` "
           "Capital 1,00,000 Drawings 16,000 Machinery 20,000 Sales 2,00,000 Purchases 2,10,000 Sales return 20,000 "
           "Purchases 1, 05,000")
check("B: trial balance stream", len(t) == 1 and len(t[0]["rows"]) == 7 and t[0]["rows"][0] == ["Capital", "1,00,000"], t)
check("B: '1, 05,000' repaired", t and t[0]["rows"][-1] == ["Purchases", "1,05,000"], t and t[0]["rows"][-1])

# Format C — cash book with dates
t = tables("The following transactions related to M/s Tools India : Date Details Amount ` 2017 Sept. 01 Bank balance 42,000 "
           "Sept. 01 Cash balance 15,000 Sept. 04 Purchased goods by cheque 12,000 Sept. 08 Sales of goods for cash 6,000")
check("C: dated rows", len(t) == 1 and t[0]["headers"][:2] == ["Date", "Details"] and len(t[0]["rows"]) == 4, t)
check("C: year on first date", t and t[0]["rows"][0][0] == "2017 Sept. 01", t and t[0]["rows"][0])

# Format D — T-shape balance sheet with Add/Less adjustments and totals
bs = ("Balance Sheet of Ankit as at March 31, 2017 Liabilities Amount Assets Amount ` ` Owners Funds Non-Current Assets "
      "Capital 12,000 Furniture 15,000 Add Net profit 20,850 32,850 Less Depreciation (1,500) 13,500 Non-Current Liabilities "
      "Current Assets Long-term loan 5,000 Debtors 15,500 Less Further bad debts 2,500 13,000 Less Provision for 650 12,350 "
      "doubtful debts Current Liabilities & Provisions Prepaid salary 5,000 Creditors 15,000 Accrued commission 1,500 "
      "Outstanding wages 500 Bank 5,000 Rent received in advance 3,000 Cash 4,000 Closing stock 15,000 56,350 56,350")
t = tables(bs)
check("D: balance sheet is a T-account", len(t) == 1 and t[0]["kind"] == "t_account", t and t[0].get("kind"))
check("D: totals verified (56,350 both sides)", t and t[0].get("totals_verified") is True, t and t[0].get("totals_verified"))
check("D: caption kept", t and t[0].get("caption") == "Balance Sheet of Ankit as at March 31, 2017", t and t[0].get("caption"))
flat = [c for r in (t[0]["rows"] if t else []) for c in r]
check("D: wrapped label re-joined", "Less Provision for doubtful debts" in flat, flat)

# Format E — "Label: Rs. N" lines
t = tables("Show the following items in the balance sheet:\nPreliminary Expenses: Rs. 2,40,000\nGoodwill: Rs. 30,000\n"
           "Discount on issue of shares: Rs. 20,000\nLoose tools: Rs. 12,000")
check("E: key-value lines", len(t) == 1 and t[0]["rows"][1] == ["Goodwill", "30,000"], t)

# Semicolon list whose parts end in a date: the trailing year is not the amount.
t = tables("Interest on drawings @ 12% p.a. Drawings during the year: Priya withdrew Rs. 10,000 on 1st July 2023; "
           "Riya withdrew Rs. 8,000 on 1st October 2023; Siya withdrew Rs. 6,000 on 1st January 2024.")
check("semicolon list: year is not the amount",
      len(t) == 1 and t[0]["rows"][0] == ["Priya withdrew on 1st July 2023", "10,000"]
      and [r[1] for r in t[0]["rows"]] == ["10,000", "8,000", "6,000"], t)

# Prose must stay prose
for prose in [
    "Cost of Revenue from Operations is Rs. 1,50,000. Operating expenses are Rs. 60,000. Revenue from Operations is Rs. 2,50,000. Calculate Operating Ratio.",
    "Leela, Meera and Neha are partners. Their fixed capitals were: Leela Rs. 80,000, Meera Rs. 60,000 and Neha Rs. 1,00,000. Record adjustment entry.",
]:
    check("prose untouched: " + prose[:40], not tables(prose), tables(prose))

# Stored markdown renders back to the same tables
md = segments_to_markdown(parse(bs))
t2 = tables(md)
check("markdown round-trip", t2 and t2[0]["rows"] == tables(bs)[0]["rows"], md[:200])

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for f in FAILURES:
        print("  -", f[:300])
    sys.exit(1)
print("All accountancy parser checks passed.")
