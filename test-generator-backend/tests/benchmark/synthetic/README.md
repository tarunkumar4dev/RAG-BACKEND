# Synthetic papers

Hand-written inputs the harness scores to verify each metric fires. Not generated
by the service; they exist only to exercise the scorer.

| file | intended metric outcomes |
|---|---|
| `clean.json` | all metrics pass |
| `wrong_numeric.json` | `numeric_correctness` fails (correct_answer is Rs 3,70,000 but the hand-verified answer is Rs 50,000; replicates D1) |
| `no_match_option.json` | `exactly_one_correct_option` fails (D1-style: correct_answer not among the four options) |
| `duplicate_pair.json` | `semantic_duplicate_count` fails (two near-verbatim stems + one numeric-pattern duplicate; replicates D3) |
| `count_wrong.json` | `count_marks_reconcile` fails (49 Qs where 50 were asked) |
