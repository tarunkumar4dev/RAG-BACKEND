"""G0 font spike: margin strip rendered with PyMuPDF insert_htmlbox (no Gemini, synthetic page)."""
import io
import math
import sys
import time
from pathlib import Path

import pymupdf

OUT = Path(sys.argv[1])
FONTS = Path(sys.argv[2])          # NotoSans-*.ttf (downloaded)
DEJAVU = Path(sys.argv[3])         # backend app/fonts
CAVEAT = Path(sys.argv[4])         # frontend public/fonts/Caveat-Bold.ttf
OUT.mkdir(parents=True, exist_ok=True)

RED, AMBER = (0.827, 0.184, 0.184), (0.698, 0.416, 0.0)
TEST = "θ √ ² ³ ° ₹ × − ✓ ✗ ½"

# 1. Glyph coverage per font
cov = {}
for name, path in [("NotoSans", FONTS / "NotoSans-Regular.ttf"), ("DejaVuSans", DEJAVU / "DejaVuSans.ttf"),
                   ("Caveat", CAVEAT)]:
    f = pymupdf.Font(fontfile=str(path))
    cov[name] = {ch: bool(f.has_glyph(ord(ch))) for ch in TEST.split()}
for name, c in cov.items():
    print(f"{name:11s}", " ".join(f"{ch}{'+' if ok else '-'}" for ch, ok in c.items()))

# 2. Synthetic "scanned" student page 1400 x 1980 px (A4 ratio), rendered to JPEG like Stage A would
Wp, Hp = 1400, 1980
src = pymupdf.open()
pg = src.new_page(width=Wp, height=Hp)
pg.draw_rect(pg.rect, color=None, fill=(0.97, 0.96, 0.93))
for y in range(160, Hp - 60, 56):
    pg.draw_line((60, y), (Wp - 40, y), color=(0.75, 0.82, 0.92), width=1.2)
pg.draw_line((120, 0), (120, Hp), color=(0.9, 0.6, 0.6), width=1.5)
student = [
    (210, "Q11) P = V²/R = (12)² / 20"), (266, "   = 144/20 = 7.2"),
    (378, "Q12 Aerobic                 Anaerobic"), (434, "O2 ki presense me hota h    O2 ke bina"),
    (490, "mitocondria me              cytoplasm me"),
    (602, "Q13) 1/v − 1/u = 1/f ;  u = −20 cm, f = 15 cm"), (658, "   v = 60 cm, m = −v/u ... real & inverted"),
    (770, "Q3- (b)"),
]
arch = pymupdf.Archive(str(CAVEAT.parent))
for y, line in student:
    pg.insert_htmlbox(pymupdf.Rect(140, y - 44, Wp - 60, y + 14),
                      f"<p style='font-family:hand;font-size:40px;color:#1d2a6b;margin:0'>{line}</p>",
                      css=f"@font-face {{font-family: hand; src: url({CAVEAT.name});}}", archive=arch)
jpeg = src[0].get_pixmap(dpi=72).tobytes("jpeg", jpg_quality=82)
src.close()

# 3. Marked page: page image + margin strip M = 0.30 * Wp, all overlays vector
M = round(0.30 * Wp)
doc = pymupdf.open()
page = doc.new_page(width=Wp + M, height=Hp)
page.insert_image(pymupdf.Rect(0, 0, Wp, Hp), stream=jpeg)       # no re-encode
page.draw_rect(pymupdf.Rect(Wp, 0, Wp + M, Hp), color=None, fill=(1, 1, 1))
page.draw_line((Wp, 0), (Wp, Hp), color=(0.99, 0.65, 0.65), width=2)

css = (f"@font-face {{font-family: noto; src: url(NotoSans-Regular.ttf);}}"
       f"@font-face {{font-family: noto; font-weight: bold; src: url(NotoSans-Bold.ttf);}}"
       f"@font-face {{font-family: dejavu; src: url(DejaVuSans.ttf);}}"
       f"@font-face {{font-family: hand; src: url(Caveat-Bold.ttf);}}"
       "p {margin:0; padding:0}"
       ".m {font-family: hand; font-size: 54px; line-height: 1; color: #D32F2F}"
       ".r {font-family: noto, dejavu; font-size: 25px; line-height: 1.3; color: #D32F2F}"
       ".amber {color: #B26A00}")
fa = pymupdf.Archive()
fa.add(pymupdf.Archive(str(FONTS)))
fa.add(pymupdf.Archive(str(DEJAVU)))
fa.add(pymupdf.Archive(str(CAVEAT.parent)))

PAD = 22
BOX_W = M - 2 * PAD


def measure(html: str) -> float:
    """Dry run on a scratch page: height the html needs at BOX_W (never scaled down)."""
    tmp = pymupdf.open()
    p = tmp.new_page(width=BOX_W + 10, height=4000)
    spare, scale = p.insert_htmlbox(pymupdf.Rect(0, 0, BOX_W, 4000), html, css=css, archive=fa, scale_low=1)
    tmp.close()
    assert scale == 1
    return 4000 - spare


def tick(p, x, y, s=1.0, half=False):
    pts = [(x, y + 18 * s), (x + 14 * s, y + 34 * s), (x + 42 * s, y - 4 * s)]
    p.draw_polyline(pts, color=RED, width=5 * s, closePath=False, lineCap=1)
    if half:
        p.draw_line((x + 20 * s, y + 4 * s), (x + 34 * s, y + 26 * s), color=RED, width=5 * s, lineCap=1)


def cross(p, x, y, s=1.0):
    p.draw_line((x, y), (x + 34 * s, y + 34 * s), color=RED, width=5 * s, lineCap=1)
    p.draw_line((x + 34 * s, y), (x, y + 34 * s), color=RED, width=5 * s, lineCap=1)


def wavy(p, x0, x1, y, amp=5, step=14):
    pts, x = [], x0
    while x <= x1:
        pts.append((x, y + amp * math.sin((x - x0) / step * math.pi)))
        x += 3
    p.draw_polyline(pts, color=RED, width=2.4, closePath=False)


def circled(n: int) -> str:
    return chr(0x2460 + n - 1)


items = [   # desired_y = anchor.y_top * Hp
    {"y": 160, "sym": "half", "marks": "2.5/3",
     "remark": f"{circled(1)} Unit missing: P = 7.2 W (watt). Formula P = V²/R ✓ and substitution √ correct."},
    {"y": 330, "sym": "half", "marks": "2/3",
     "remark": f"{circled(1)} Only 2 differences (3 asked). Hinglish theek hai — 'O₂ ki presence me' is fine; "
               "spelling not penalised (mitocondria). Third: energy released — 38 ATP vs 2 ATP."},
    {"y": 560, "sym": "half", "marks": "2.5/3",
     "remark": f"{circled(1)} Nature incomplete: real, inverted and SAME SIZE (m = −1). Written under label 'Q12' "
               "— mapped to Q13 by content. θ = 30°, cost ₹50 (glyph test)."},
    {"y": 735, "sym": "cross", "marks": "0/1", "remark": "Ans: (c)"},
    {"y": 760, "sym": "review", "marks": "?", "remark": "Needs teacher review — low confidence reading."},
]

t0 = time.time()
y_cursor = PAD
for it in items:
    marks_html = f"<p class='m{' amber' if it['sym'] == 'review' else ''}'>{it['marks']}</p>"
    remark_html = f"<p class='r{' amber' if it['sym'] == 'review' else ''}'>{it['remark']}</p>"
    h_m = measure(marks_html)
    h_r = measure(remark_html)
    h_m = max(h_m, 44)
    h = h_m + 4 + h_r
    y = max(it["y"], y_cursor)
    x = Wp + PAD
    if it["sym"] == "half":
        tick(page, x, y + 2, half=True)
    elif it["sym"] == "cross":
        cross(page, x + 4, y + 6)
    page.insert_htmlbox(pymupdf.Rect(x + 60, y, Wp + M - PAD, y + h_m), marks_html, css=css, archive=fa, scale_low=1)
    page.insert_htmlbox(pymupdf.Rect(x, y + h_m + 4, Wp + M - PAD, y + h + 2), remark_html, css=css, archive=fa,
                        scale_low=1)
    if abs(y - it["y"]) > 12:     # pushed down → marker at the answer's height + dashed leader inside the strip
        page.draw_polyline([(Wp + 2, it["y"] + 8), (Wp + 12, it["y"] + 14), (Wp + 2, it["y"] + 20)], color=RED, fill=RED, width=1)
        page.draw_polyline([(Wp + 12, it["y"] + 14), (Wp + 12, y + 18), (x - 2, y + 18)], color=RED, width=1.2, dashes="[5 4] 0", closePath=False)
    it["drawn_y"], it["h"] = y, h
    y_cursor = y + h + 8

# wavy underline under the error line (Q11 "= 7.2", line bottom ≈ 276 px) + circled number
wavy(page, 200, 560, 280)
page.insert_htmlbox(pymupdf.Rect(568, 250, 620, 300), f"<p class='r'>{circled(1)}</p>", css=css, archive=fa)

# total top-right of page 1
page.insert_htmlbox(pymupdf.Rect(Wp - 420, 30, Wp - 30, 120),
                    "<p class='m' style='font-size:76px;text-align:right'>28.5/40</p>", css=css, archive=fa)
page.draw_line((Wp - 300, 118), (Wp - 34, 118), color=RED, width=3)
page.insert_htmlbox(pymupdf.Rect(Wp - 520, 122, Wp - 30, 160),
                    "<p class='r' style='text-align:right'>(provisional) · Not attempted: Q6</p>", css=css, archive=fa)
# footer
page.insert_htmlbox(pymupdf.Rect(Wp, Hp - 46, Wp + M, Hp - 10),
                    "<p class='r' style='color:#6B7280;font-size:20px;text-align:center'>Student 1 · 1/7</p>",
                    css=css, archive=fa)
ms = (time.time() - t0) * 1000

pdf = OUT / "spike_marked.pdf"
doc.subset_fonts()
doc.save(pdf, garbage=4, deflate=True)
print("jpeg_bytes", len(jpeg))
fonts_used = sorted({f[3] for f in doc[0].get_fonts()})
png = OUT / "spike_marked.png"
doc[0].get_pixmap(dpi=54).save(png)       # ~1300 px wide preview
doc.close()

overlap = any(a["drawn_y"] + a["h"] > b["drawn_y"] for a, b in zip(items, items[1:]))
print(f"render_ms={ms:.0f} pdf_bytes={pdf.stat().st_size} fonts={fonts_used} overlap={overlap}")
for it in items:
    print(f"  desired_y={it['y']:4d} drawn_y={it['drawn_y']:6.0f} h={it['h']:5.0f}")
