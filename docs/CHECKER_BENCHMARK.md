# Checker fallback benchmark (Gate 2 #2): `gemini-3.8-flash` (current) vs `gemini-3.6-flash` (proposed fallback)

**Status: INCOMPLETE.** Only 3 of the 10 copies have a usable side-by-side.

**Run:** 12 Nov 2026, 22:46–23:04 IST.

**Spend: ₹6.95, MEASURED.** That is the token counts Gemini returned in `usage_metadata`, priced at the list price below. Only successful calls are counted. Gemini does not bill 503 and 504 responses. For the 3 calls that hit our client timeout, whether Google billed them is unknown. If it did, it would add about ₹7 at most.

**Price used:** $0.75 per million input tokens and $3.75 per million output tokens (thinking included), the same for both models until 31 Dec 2026. From 1 Jan 2027 it is $1.50 / $7.50. Converted at ₹84/$. Source: aistudio.google.com/docs/pricing, read 12 Nov 2026.

**Method:**
- Production code path: `copy_check.grade_copy`, using the October rubric with strictness "medium".
- Fallbacks were off, so each column is one model only.
- Copies are 10 Economics copies from October, anonymised as Copy 01–10. No names, roll numbers or ids appear here.
- Runner: `test-generator-backend/scripts/bench_checker_fallback.py`.
- Raw output, which stays outside the repo because it contains page images: `C:\Users\login\Downloads\a4ai_checker_bench_2026-11-12\`.

## 1. What happened

| Copy | Pages | Oct prod model → marks | `gemini-3.8-flash` today | `gemini-3.6-flash` today |
|---|---|---|---|---|
| 01 | 6 | 3-flash-preview → 23/30 | ❌ 503 overloaded (2 tries) | ❌ client timeout at 60 s |
| 02 | 2 | **3.8-flash → 14/30** | ❌ 503 ×2; a retry at 23:04 got 429, then 503 | ✅ **16/30** |
| 03 | 3 | 3.8-flash → 16/30 | ❌ 503 ×2 | ❌ unreadable reply, then 503 |
| 04 | 1 | 3-flash-preview → 22/30 | ❌ 503 ×2 | ❌ client timeout at 60 s |
| 05 | 6 | 3-flash-preview → 19.5/30 | ❌ 503 ×2 | ❌ 503 ×2 |
| 06 | 6 | 3.8-flash → 18/30 | ❌ 503 ×2 | ❌ client timeout at 60 s |
| 07 | 1 | **3.8-flash → 6.5/30** | ❌ 504, then 503 | ✅ **7.5/30** |
| 08 | 3 | **3.8-flash → 18.5/30** | ❌ 504, then 503 | ✅ **18.5/30** |
| 09 | 4 | 3.8-flash → 19.5/30 | ❌ 504 ×2 | ❌ 503 |
| 10 | 4 | 3-flash-preview → 21/30 | ❌ 503 ×2 | ❌ 503 |

**Why it's incomplete.**
- **`gemini-3.8-flash` graded 0 of 10 copies tonight.** Every attempt was refused by Google: 503 "overloaded" or 504, and once 429. This is Google capacity, not our code. The same requests would have failed in production at this hour.
- **`gemini-3.6-flash` graded 3 of 10.**
  - Copies 01, 04 and 06 hit **our** 60 s client timeout (`CHECKER_GEMINI_TIMEOUT_S`, the production value).
  - Copies 03, 05, 09 and 10 got 503s.
  - Copy 03 also returned one unreadable reply before a 503.
- I stopped after one probe retry. I'm not retrying further tonight, because with the current model down a rerun would mostly buy more 503s.
- **The ₹100 cap leaves about ₹86 for a rerun** (₹93 if the timed-out calls were not billed). A full 10 × 2 pass costs about ₹45–50. I suggest running it off-peak, for example 7–9 am IST, with the timeout raised to 180 s for the benchmark only. That needs your go: `--resume` keeps the 3 finished copies.

**What I compared instead.** The 3 copies that 3.6-flash graded (02, 07, 08) were all graded by **`gemini-3.8-flash` in production on 24 Oct**. Their stored October per-question results are the "current model" column below. Caveats:
- It's a different day.
- The October run had the production fallbacks on, but all three show one call, so it was 3.8-flash alone.
- The grading code is unchanged since then, apart from the Gate 2 #4 retry cap, which does not affect a successful call.

## 2. Per question: the 3 copies with both models

`Δ` = 3.6 − 3.8. Question 11 has sub-parts. Both models write "See sub-part feedback" on the parent row, and the sub-part rows are not shown here.

### Copy 02: 3.8 = 14, 3.6 = 16, **Δ = +2.0**

| Q | Max | 3.8 (Oct) | 3.6 (today) | Δ | 3.8 feedback | 3.6 feedback |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 0 | Correct definition. | Correct definition. |
| 2 | 1 | 0 | 0 | 0 | You defined domestic currency instead of appreciation of domestic currency. | The question asks for appreciation of domestic currency, not the definition of domestic currency. |
| 3 | 1 | 1 | 1 | 0 | Correct. | Correct answer. |
| 4 | 1 | 0.5 | 1 | +0.5 | Correctly identified as supply of foreign exchange; mention that payment for exported services brings foreign currency into India. | Correct identification. |
| 5 | 3 | 3 | 3 | 0 | Well-structured tabular distinction covering all three bases accurately. | Excellent structured table covering all three bases of distinction clearly. |
| 6 | 3 | 1.5 | 1.5 | 0 | Points are correctly identified, but the question asked to state and explain them. | Points are correctly identified, but provide a brief explanation for each source as asked by the question. |
| 7 | 3 | 1.5 | 1.5 | 0 | Sources of supply are correctly stated; add brief explanations to secure full marks. | Sources are correctly listed, but explain briefly how each source generates supply of foreign exchange. |
| 8 | 3 | 0 (not attempted) | 0 (attempted, blank) | 0 | Unattempted. | Question was left blank. |
| 9 | 4 | 0.5 | 1 | +0.5 | Only stated that reserves are needed to maintain the fixed rate; explain how the central bank buys/sells foreign currency to counter fluctuations. | Good initial point, but complete the explanation detailing how the central bank buys/sells reserves to counteract market fluctuations. |
| 10 | 4 | 2 | 3 | **+1** | Good understanding of the price effect, but explicitly conclude that exports increase and imports decrease. | Points are relevant; explicitly conclude that exports increase and imports decrease as a result. |
| 11 | 6 | 3 | 3 | 0 | See sub-part feedback | See sub-part feedback |
| **Total** | **30** | **14** | **16** | **+2.0** | | |

### Copy 07: 3.8 = 6.5, 3.6 = 7.5, **Δ = +1.0**

| Q | Max | 3.8 (Oct) | 3.6 (today) | Δ | 3.8 feedback | 3.6 feedback |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.5 | 0.5 | 0 | Define specifically in terms of domestic price of one unit of foreign currency. | Correct basic concept. Specify that it is the price of foreign currency in terms of domestic currency. |
| 2 | 1 | 0.5 | 0.5 | 0 | Specify that it refers to a rise in the value of domestic currency under a flexible exchange rate regime. | Good definition. Mention that appreciation occurs under a flexible exchange rate system. |
| 3 | 1 | 0 | 0 | 0 | Not attempted. | Not attempted. |
| 4 | 1 | 0 | 0 | 0 | Not attempted. | Not attempted. |
| 5 | 3 | 1 | 1.5 | +0.5 | Complete both columns of the comparison table; devaluation is done deliberately by the government/monetary authority. | You correctly identified the exchange rate systems and the cause of depreciation, but complete the devaluation column as well. |
| 6 | 3 | 0 | 0 | 0 | Not attempted. | Not attempted. |
| 7 | 3 | 0 | 0 | 0 | Not attempted. | Not attempted. |
| 8 | 3 | 0.5 | 1 | +0.5 | Explain clearly how exchange rate affects quantity demanded (downward sloping demand curve) and quantity supplied via exports (upward sloping supply curve). | Explain supply curve behavior by showing how a higher exchange rate makes domestic goods cheaper for foreigners, boosting exports. |
| 9 | 4 | 0 | 0 | 0 | Not attempted. | Not attempted. |
| 10 | 4 | 4 | 4 | 0 | Very well explained with correct reasoning on both imports and exports. | Excellent answer for Question 10. Clear and accurate explanation of both export and import impacts. |
| 11 | 6 | 0 | 0 | 0 | See sub-part feedback | See sub-part feedback |
| **Total** | **30** | **6.5** | **7.5** | **+1.0** | | |

### Copy 08: 3.8 = 18.5, 3.6 = 18.5, **Δ = 0**

| Q | Max | 3.8 (Oct) | 3.6 (today) | Δ | 3.8 feedback | 3.6 feedback |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 0 | Correct definition of foreign exchange rate. | Correct definition. |
| 2 | 1 | 0.5 | 0.5 | 0 | Mention that currency appreciation occurs under a flexible exchange rate system. | Good definition. Mention that this occurs under a flexible exchange rate system. |
| 3 | 1 | 1 | 1 | 0 | Correct answer. | Correct. |
| 4 | 1 | 1 | 0.5 | −0.5 | Correct identification. | Correct identification as supply of foreign exchange, but provide a complete sentence/explanation. |
| 5 | 3 | 0 | 0 | 0 | Unattempted question. | Question was not attempted. |
| 6 | 3 | 3 | 3 | 0 | Well-explained sources of demand for foreign exchange. | Excellent answer covering all three sources with accurate explanations. |
| 7 | 3 | 3 | 2.5 | −0.5 | Good answers covering exports, foreign investment, and speculative inflows. | Good answer explaining three sources of supply. |
| 8 | 3 | 0 | 0 | 0 | Unattempted question. | Question was not attempted. |
| 9 | 4 | 2 | 3 | **+1** | Clarify how the central bank specifically buys or sells reserves during excess supply and excess demand situations. | Well written explanation of the central bank's role in maintaining fixed rates using reserves. |
| 10 | 4 | 4 | 4 | 0 | Clear and correct explanation of the impact of depreciation on imports and exports. | Excellent, well-structured answer clearly addressing both exports and imports. |
| 11 | 6 | 3 | 3 | 0 | See sub-part feedback | See sub-part feedback |
| **Total** | **30** | **18.5** | **18.5** | **0** | | |

## 3. Cost and time per copy

| Copy | 3.8 (Oct) ₹ | 3.6 (today) ₹ | 3.8 (Oct) s | 3.6 (today) s | 3.8 thinking tokens | 3.6 thinking tokens |
|---|---|---|---|---|---|---|
| 02 | 2.08 | 2.27 | 86.1 | 30.4 | 1,876 | 2,532 |
| 07 | 1.42 | 2.47 | 122.4 | 31.4 | 0 | 3,322 |
| 08 | 2.21 | 2.21 | 126.0 | 32.2 | 1,849 | 1,859 |
| **Average** | **₹1.90** | **₹2.32 (+22 %)** | **111.5 s** | **31.3 s** | 1,242 | 2,571 |

Both columns use October/today token counts as MEASURED, priced at today's list price. The 3.8 time is October's grading call (`ms_by_tag.GRADE`). The 3.6 time is today's whole `grade_copy`.

## 4. Summary flags

**a) Copies where totals differ by more than 2 marks: none.** The largest gap is Copy 02 at exactly +2.0, so it doesn't pass the threshold. It comes from three questions: Q4 +0.5, Q9 +0.5, Q10 +1.

**b) Marks for an answer the other model marked wrong: none in the 3 copies.** No question got 0 from one model and more than 0 from the other. These are the part-mark disagreements of 0.5 or more, listed for your judgement:
- **Copy 02 Q4** (1 mark): 3.8 gave 0.5, 3.6 gave 1.
- **Copy 02 Q9** (4 marks): 3.8 gave 0.5, 3.6 gave 1.
- **Copy 02 Q10** (4 marks): 3.8 gave 2, 3.6 gave 3. Both ask for the same missing conclusion, yet 3.6 gave 1 mark more.
- **Copy 07 Q5** (3 marks): 3.8 gave 1, 3.6 gave 1.5.
- **Copy 07 Q8** (3 marks): 3.8 gave 0.5, 3.6 gave 1.
- **Copy 08 Q4** (1 mark): 3.8 gave 1, 3.6 gave 0.5.
- **Copy 08 Q7** (3 marks): 3.8 gave 3, 3.6 gave 2.5.
- **Copy 08 Q9** (4 marks): 3.8 gave 2, 3.6 gave 3.

Overall 3.6 was slightly more generous: +3 marks over 33 questions, and +1.0 on average per copy.

**c) Feedback that is unclear or not useful.** All feedback is in English, the same as October. No Hindi was asked for, so none is in the wrong language.
- **3.6, Copy 08 Q7:** gave 2.5 of 3 with "Good answer explaining three sources of supply." It doesn't say what lost the half mark.
- **3.6, Copy 08 Q9:** gave 3 of 4 with "Well written explanation…". It doesn't say what lost the mark.
- **3.6, Copy 08 Q4:** "Correct identification…, but provide a complete sentence/explanation" on a 1-mark identify question. The half-mark cut looks inconsistent with "correct identification". Please judge.
- **3.6, Copy 02 Q10:** gave 3 of 4 with the same advice 3.8 used for 2 of 4, so the mark and the text disagree between the two models.
- **3.6, Copy 07 Q10:** "Excellent answer for Question 10." is fine but redundant.
- **3.8 has one similar gap: Copy 02 Q4** gave 0.5 with advice to add a point the key may not require. Please judge.
- **Both models:** Q11 shows "See sub-part feedback" on the parent row. That's fine in the UI, but not self-contained in exports.

**d) Average cost and time per copy, old vs new (3 copies).**
- **Cost:** ₹1.90 (3.8, October) vs ₹2.32 (3.6, today), **+22 %**. The difference is driven by 3.6's thinking tokens: 2,571 vs 1,242 on average.
- **Time:** 111.5 s vs 31.3 s.
- Three copies is far too few to call either difference real.

**Benchmark spend: ₹6.95 MEASURED** of the ₹100 cap. Add up to about ₹7 if the 3 timed-out calls were billed, which is unknown.

## 5. Operational finding (not part of the comparison)

Tonight, 22:46–23:04 IST, the production primary `gemini-3.8-flash` refused every request. That was 22 attempts: 17 got 503, 4 got 504 after about 59 s, and 1 got 429.

In production, those copies would have gone to today's fallbacks, `gemini-2.5-flash` and then `gemini-3-flash-preview`. Neither was tested tonight.

`gemini-3.6-flash` was not overloaded in the same way, but 3 of its 10 attempts ran past our 60 s client timeout. The 60 s timeout (`CHECKER_GEMINI_TIMEOUT_S`) is a candidate to revisit with fix #3. I have not changed it.
