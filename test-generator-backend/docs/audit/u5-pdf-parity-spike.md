# U5 — PDF parity spike: subscripts / superscripts (28 Oct 2026)

Question: can the PDF export show H₂O, CO₂, x² the way the builder does? Decide the PDF path before
`formatChemicalFormula` (Slice B) is written. No code changed by this spike.

## There are two PDF paths, not one
| Path | When | Engine |
|---|---|---|
| **Server** `POST /test-generator/export` → `export_service.generate_pdf` | always first | ReportLab `Paragraph` markup |
| **Browser fallback** `src/utils/testPdfExporter.ts` | only when the server call fails | jsPDF 3.0.1, built-in Helvetica |

Samples: live rows 8323 (O2, Na2O, H2O), 16803 (CO2), 8198 (H2SO4, HNO3, CH3COOH, [Ca(OH)2]), 45877 (x²),
1461 / 1469 (Maths AP: S13, a13, n2, S1) + one synthetic line (A4, Q2, 2H+ + 2e-, CuSO4·5H2O, H₂O, x² ≤ √2 ∞).

## Server path (ReportLab) — renders `<sub>` / `<super>` correctly
- Already subscripts via `_fix_chemical_formulas` (`([A-Z][a-z]?)(\d+)`, every subject): H₂SO₄, CO₂, Na₂O right.
- False positives: "A₄ size paper", "Q₂". Misses: `[Ca(OH)2]` (count after `)`), charges `2H+`, `2e-`.
- Maths: S₁₃ / S₁₀ / S₁ subscripted, but `a13`, `a12`, `n2` not (lower-case) → inconsistent within one question.
- Unicode H₂O and Latin-1 x² render fine.
- Multi-line questions lose their line breaks in the default template ("(i) … (ii) …" run together); in the
  markup the `__LINEBREAK__` placeholder is turned into `_<sub>L</sub>INEBREAK__` by the LaTeX `_x` rule.
  Separate existing bug, not investigated further.
- DOCX uses `_latex_to_plain`: no subscripts at all.
- Time: 4 questions in 91–426 ms (first call includes font loading).

## Browser fallback (jsPDF) — cannot
- `<sub>`/`<sup>` print as literal tags.
- Unicode subscripts (₂ ⁺) print as garbage — built-in fonts are WinAnsi (Latin-1) only.
- ≤ ≥ ≈ ≠ ∞ √ → also print as garbage — and `cleanTextForPdf` produces exactly these from LaTeX, so the fallback
  is already broken for maths symbols today.
- Works: plain text, Latin-1 ² ³ × ÷ ± °.

## html2canvas (rasterise each card, embed as image) — timing
5 cards (chemistry `<sub>` + KaTeX), Chromium, scale 2:
- CPU ×1: 157–389 ms per question (within the 500 ms budget).
- CPU ×4 (mid-range phone): 582–1,123 ms per question → a 20-question paper ≈ 15–20 s on a phone. Over budget.
- Output is an image: text not selectable/searchable, larger file.

## Options
- (a) **Server ReportLab as the parity path** (recommended): it already renders sub/sup. Slice B replaces
  `_fix_chemical_formulas` with the parser from `ncert_chem_formula.py` (no A4/Q2, handles `)2` and charges).
- (b) Fallback: map to Latin-1 where possible (², ³) and accept the gap for subscripts; or embed a Unicode TTF
  in jsPDF (Noto Sans, ~300 KB+) so ₂ ⁺ ≤ √ work — fixes the existing symbol garbage too.
- (c) html2canvas only for the fallback, desktop only — too slow on phones.

Spike scripts (not committed): session scratchpad `g45/spike/` (be_spike.py, jspdf_spike.mjs, h2c_spike.mjs).
