# Hand-computed expected values (fixture data.sql; % = (present + late) / marked, leave counts as marked;
# only class_sessions.status = 'conducted'; dates in IST)

| # | Caller | Question | Expected |
|---|---|---|---|
| R1 | admin | Priya Sharma attendance 2026-11-01..05 | sessions 3, marked 9, present 4, late 1, absent 3, leave 1, **55.6 %**; previous 5 days (Oct 27–31): 3/3 = **100.0** |
| R2 | admin | section ECE-2 (main batch) Nov 1–5 | **55.6 %**, marked 9 (the 5 Nov session is cancelled → excluded) |
| R3 | admin | Aarav Mehta Nov 1–5 | 4/4 = **100.0** (3 Maths + 1 Physics lab) |
| R4 | admin | below 75 % Nov 1–5 | Ishita Rao 0.0 (ECE-2), Rahul Kumar 0.0 (CSE-1), Rahul Kumar 50.0 (ECE-2) |
| R5 | HOD ECE / proctor ECE-2 | below 75 % Nov 1–5 | Ishita Rao 0.0, Rahul Kumar 50.0 (CSE-1 not visible) |
| R6 | Sharma (teacher) | Rahul Kumar (ECE-2) Nov 1–5 | 2/3 = **66.7** (lab session by Verma not visible) |
| R7 | Verma (teacher) | Aarav Mehta Nov 1–5 | 1/1 = **100.0** (only his Physics lab) ; Ishita Rao → not_found |
| R8 | admin | trend ECE-2 Oct 26–Nov 5, day | 10-28 100.0 · 11-02 66.7 · 11-03 33.3 · 11-04 66.7 ; week: 10-26 100.0 · 11-02 55.6 |
| R9 | admin / Sharma / HOD | marks section ECE-2 | Unit Test 1 (check copies, 30 Oct): avg 25.0, **62.5 %**, hi 30, lo 20, absent 1 ; Trig worksheet (1 Nov): avg 15.0, **75.0 %**, hi 18, lo 12, absent 1 ; overall (pooled 75, 50, 90, 60) **68.8** |
| R10 | admin | marks Aarav Mehta | Unit Test 1 30/40 = 75.0 ; Trig worksheet 18/20 = 90.0 ; overall **82.5** |
| R11 | Verma | marks section CSE-1 | DSA quiz avg 6.5, 65.0 %, hi 9, lo 4, absent 0 ; overall 65.0 |
| R12 | admin | tests by Sharma Nov 1–5 (IST) | **4** (incl. 01 Nov 00:10 IST and 05 Nov 23:50 IST; excl. Oct 15) ; institute-wide **5** (Sharma 4, Verma 1) |
| R13 | decoy admin | section "ECE-2" | only the decoy batch ba..04 ; Sharma → not_found |
| R14 | resolve "Rahul Kumar" | admin → 2 matches (not the decoy one) ; Verma → 2 (CSE-1 roll 12, ECE-2 roll 03 via lab) ; Sharma → 1 (ECE-2) |

## API v2 additions (data.sql "phase 3 additions": ECE 1 = 2 students, 2 HOD classes, 4/4 good; ECE1 = 0 students)

| # | Caller | Question | Expected |
|---|---|---|---|
| R15 | admin | my_scope | CSE-1 2 · ECE 1 2 · ECE-2 3 (Left Student inactive) · ECE-2 A 2 (lab: Aarav, Rahul) · ECE1 0 ; all role admin ; subjects ECE-2 [Mathematics], CSE-1 [DSA], ECE-2 A [Physics] ; 7 distinct students |
| R16 | Sharma / Verma / HOD / proctor / Sunita | my_scope | Sharma: ECE-2 (3, teacher, Mathematics), 3 students · Verma: CSE-1 (2, DSA) + ECE-2 A (2, Physics), 4 students · HOD: ECE 1, ECE-2, ECE-2 A, ECE1 all role hod, 5 students · proctor (ctx role teacher): ECE-2, ECE-2 A role proctor · Sunita: empty + message |
| R17 | admin | section_comparison attendance Nov 1–5 | ECE 1 **100.0** (2 classes, 4 marked) #1 · ECE-2 **55.6** (3, 9) #2 · CSE-1 **50.0** (2, 4) #3 · ECE-2 A **50.0** (1, 2) #3 · ECE1 — ; highest [ECE 1], lowest [CSE-1, ECE-2 A] (tie), no_data [ECE1] |
| R18 | HOD / Sharma / Verma / Sunita | section_comparison attendance Nov 1–5 | HOD: ECE 1 100.0, ECE-2 55.6, ECE-2 A 50.0, ECE1 — ; lowest [ECE-2 A] · Sharma: ECE-2 55.6 only · Verma: CSE-1 50.0 and ECE-2 A 50.0 both #1 · Sunita: no rows + message |
| R19 | admin | section_comparison marks Oct 26–Nov 5 / Nov 1–5 | ECE-2 **68.8** (2 tests, 4 graded: 75, 50, 90, 60) · CSE-1 **65.0** (1, 2) · others — ; Nov 1–5: ECE-2 **75.0** (1, 2: 90, 60) · CSE-1 65.0 |
| R20 | admin / HOD | resolve section "ECE 1" | ambiguous: ECE 1 hint "Electronics · 2 students", ECE1 hint "Electronics · 0 students (empty)" ; chip values carry the id ; Sharma → not_found |
