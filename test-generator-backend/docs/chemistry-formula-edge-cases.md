# Chemistry formula rendering — edge cases (Slice B input, not implemented)

Status: **written for review.** `formatChemicalFormula` is not written. It waits for (1) a review of the
test list below by a chemistry teacher (or someone who took Class 11–12 Chemistry) and (2) the PDF-path decision
(`docs/audit/u5-pdf-parity-spike.md`).

What exists today:
- **Flag only** — `app/services/ncert_chem_formula.py` `needs_formula_render(text, subject, chapter)`:
  Chemistry, or Science chemistry chapters; a word counts only if all of it parses as element symbols /
  `(…)` groups / counts and it has a count ≥ 2 or a charge. Tests: `tests/unit/test_ncert_chem_formula.py`.
  Not wired into any quality pass yet (Class 9/11/12 passes are out of scope).
- **Server PDF already subscripts** — `export_service._fix_chemical_formulas` applies `([A-Z][a-z]?)(\d+)` →
  `<sub>` to *every* subject: "A4 size paper" → A₄, "Q2" → Q₂ (seen in the U5 spike). Slice B should replace
  it with the same parser as the flag, not add a second rule.

Live data (28 Oct): formulas are stored as plain ASCII ("H2O", "CO2(g)", "[Ca(OH)2]"); no Class 10 Science row
contains Unicode subscripts (₂₃₄).

## Cases — expected output (teacher to confirm each line)

`<sub>`/`<sup>` below means "rendered lowered/raised", whatever the output path uses.

| # | Input | Expected | Rule |
|---|---|---|---|
| 1 | `H2SO4` | H<sub>2</sub>SO<sub>4</sub> | counts after an element or `)` are subscripts |
| 2 | `2H2O` | 2H<sub>2</sub>O | leading number = coefficient, stays full size |
| 3 | `NH4+` | NH<sub>4</sub><sup>+</sup> | trailing sign = charge |
| 4 | `Fe3+` | Fe<sup>3+</sup> | **ambiguous** — digit before a sign is the charge magnitude, not a count. Needs a teacher rule: "digit immediately followed by +/- at word end is charge" |
| 5 | `SO42-` | SO<sub>4</sub><sup>2−</sup> | **ambiguous** with #4: is it 4 + charge 2−, or 42? Proposal: last digit before the sign = charge, rest = count |
| 6 | `(SO4)2-` | (SO<sub>4</sub>)<sub>2</sub><sup>2−</sup>? or (SO<sub>4</sub>)<sup>2−</sup> | group charge — **teacher to decide**; NCERT writes SO<sub>4</sub><sup>2−</sup> without brackets |
| 7 | `2H+ + 2e- → H2` | 2H<sup>+</sup> + 2e<sup>−</sup> → H<sub>2</sub> | ionic equation. `e-` is an electron: lower-case `e` is not an element, so the parser alone will NOT raise it. Needs an explicit `e-` / `e−` rule |
| 8 | `C-14`, `14C` | leave as-is | isotopes — out of scope for now (<sup>14</sup>C is correct but rare in NCERT text) |
| 9 | `H2O(l)`, `NaCl(aq)`, `(s)`, `(g)` | H<sub>2</sub>O(l) — state at 0.85em | separate pass after formulas; never subscript inside the state bracket |
| 10 | `CuSO4·5H2O` | CuSO<sub>4</sub>·5H<sub>2</sub>O | `5` after `·` is a coefficient, not a subscript |
| 11 | `→`, `⇌`, `⟶`, `->`, `<=>` | one arrow style each (→ and ⇌) | normalise ASCII arrows; keep ⇌ distinct |
| 12 | `[H+] = 10^-3 M` | [H<sup>+</sup>] = 10<sup>−3</sup> M | concentration: the `10^-3` is math, not a formula — do not touch the number |
| 13 | `$...$`, `\(...\)`, `\[...\]` | untouched | KaTeX segments are rendered by KaTeX |
| 14 | `CH3COOH`, `C2H5OH` | CH<sub>3</sub>COOH, C<sub>2</sub>H<sub>5</sub>OH | organic condensed formulas |
| 15 | `2-methylpropane`, `CH3-CH2-CH3` | 2-methylpropane; CH<sub>3</sub>–CH<sub>2</sub>–CH<sub>3</sub> | IUPAC name untouched; hyphen-bonded chain: split on `-` only between two formula parts — **teacher to confirm**. The flag does NOT detect this form today |
| 16 | `A4 size`, `Q2.`, `Fig 2.3`, `Class XII`, `B5`, `C10` | untouched | not formulas (parser + denylist + "word before" rule) |
| 17 | `HCl`, `NaCl` | unchanged | nothing to render |
| 18 | `Mo`, `Sn`, `U` etc. | currently untouched | element list is hardcoded (40 symbols); extend if Class 11–12 needs more (Sn, Mo, Cs, Rb, Li…) |
| 19 | `x2`, `n2` in Maths | untouched by chemistry rules | Maths is gated out; its powers are a separate (KaTeX) problem |

## Open questions for the reviewer
1. Rows 4–6: charge vs count when a digit sits before `+`/`-`.
2. Row 7: is `e-` the only lower-case case, or also `e+` (positron, Class 12 nuclear — not in our chapters)?
3. Row 15: hyphen-bonded chains — render as en dash or leave the hyphen?
4. Do Class 11–12 chapters need more element symbols than the 40 hardcoded ones?
