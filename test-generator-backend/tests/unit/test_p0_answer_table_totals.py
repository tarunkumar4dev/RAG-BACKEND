"""Task 8: an answer table's total row must equal the column sums, not just Dr total == Cr total."""
from app.models.test_generator import AnswerTable
from app.services.test_generator_service import _amount, _answer_table_issue

TB_HEADERS = ["S.No.", "Account Name", "L.F.", "Debit (Rs.)", "Credit (Rs.)"]
LEDGER_HEADERS = ["Date", "Particulars", "J.F.", "Amount (Rs.)", "Date", "Particulars", "J.F.", "Amount (Rs.)"]


def tb(rows, total):
    return AnswerTable(type="trial_balance", headers=TB_HEADERS, rows=rows, total_row=["", "Total", "", *total])


def test_indian_amounts_parse():
    assert _amount("4,50,000") == 450000
    assert _amount("Rs. 1,20,000.50") == 120000.5
    assert _amount("₹ 5,00,000/-") == 500000
    assert _amount("—") is None


def test_orion_trial_balance_is_rejected():
    # Dr 4,50,000 + 50,000 = 5,00,000 but Cr has only 4,00,000; the total row claims 5,00,000 on both sides
    table = tb([["1", "Machinery", "", "4,50,000", ""],
                ["2", "Cash", "", "50,000", ""],
                ["3", "Capital", "", "", "4,00,000"]],
               ["5,00,000", "5,00,000"])
    issue = _answer_table_issue(table)
    assert issue and "does not balance" in issue


def test_total_row_that_disagrees_with_its_column_is_rejected():
    # rows balance (5,00,000 each) but the stated totals do not match the rows
    table = tb([["1", "Machinery", "", "4,50,000", ""], ["2", "Cash", "", "50,000", ""],
                ["3", "Capital", "", "", "5,00,000"]],
               ["5,50,000", "5,50,000"])
    assert "add up to" in _answer_table_issue(table)


def test_balanced_trial_balance_passes():
    table = tb([["1", "Machinery", "", "4,50,000", ""], ["2", "Cash", "", "50,000", ""],
                ["3", "Capital", "", "", "4,00,000"], ["4", "Loan", "", "", "1,00,000"]],
               ["5,00,000", "5,00,000"])
    assert _answer_table_issue(table) is None


def test_rounding_and_a_total_line_inside_rows_are_tolerated():
    table = tb([["1", "Machinery", "", "3,33,333.33", ""], ["2", "Cash", "", "1,66,666.67", ""],
                ["3", "Capital", "", "", "5,00,000"], ["", "Total", "", "5,00,000", "5,00,000"]],
               ["5,00,000", "5,00,001"])
    assert _answer_table_issue(table) is None


def test_ledger_side_not_matching_its_total_is_rejected():
    table = AnswerTable(type="ledger", headers=LEDGER_HEADERS, rows=[
        ["Apr 1", "To Balance b/d", "", "4,50,000", "Mar 31", "By Cash", "", "4,00,000"],
        ["Apr 9", "To Bank", "", "50,000", "Mar 31", "By Balance c/d", "", ""],
    ], total_row=["", "", "", "5,00,000", "", "", "", "5,00,000"])
    assert "Cr side" in _answer_table_issue(table)


def test_balanced_ledger_with_next_period_balance_bd_passes():
    table = AnswerTable(type="ledger", headers=LEDGER_HEADERS, rows=[
        ["Apr 1", "To Balance b/d", "", "4,50,000", "Mar 31", "By Cash", "", "4,00,000"],
        ["Apr 9", "To Bank", "", "50,000", "Mar 31", "By Balance c/d", "", "1,00,000"],
        ["Apr 1", "To Balance b/d", "", "1,00,000", "", "", "", ""],
    ], total_row=["", "", "", "5,00,000", "", "", "", "5,00,000"])
    assert _answer_table_issue(table) is None


def test_journal_check_unchanged():
    jt = AnswerTable(type="journal_entry", headers=TB_HEADERS, rows=[
        ["1", "Cash A/c Dr.", "", "10,000", ""], ["", "  To Capital A/c", "", "", "10,000"]], total_row=None)
    assert _answer_table_issue(jt) is None
    jt.rows[1][4] = "9,000"
    assert "does not balance" in _answer_table_issue(jt)
