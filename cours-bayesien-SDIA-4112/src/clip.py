import pymupdf, re, os, json
from collections import Counter

PDF = r"C:\Users\HP\Downloads\EC _ EPD-SDIA-4112 – Statistique Bayésienne__Parco.pdf"
OUT = r"C:\Users\HP\Documents\cours-bayesien-SDIA-4112"
doc = pymupdf.open(PDF)

CODE_PAGES = set(range(13, 69)) | set(range(77, 120)) | set(range(149, 171))
rec = []
rights = []
for i, page in enumerate(doc):
    pn = i + 1
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            txt = "".join(s["text"] for s in l["spans"])
            if not txt.strip():
                continue
            x1 = l["bbox"][2]
            rights.append(x1)
            rec.append({"page": pn, "x1": round(x1, 1), "text": txt})

mx = max(r["x1"] for r in rec)
import statistics
# container right edge = the mode of the largest x1 values (clipped lines all end at the same x)
top = sorted(r["x1"] for r in rec)[-400:]
print("max x1:", round(mx, 1))
print("top-20 x1 values:", [round(v, 1) for v in sorted(set(top))[-20:]])

# container edge: the most frequent x1 among the top 300
edge = Counter(round(r["x1"]) for r in rec).most_common(8)
print("most common right edges:", edge)

EDGE = 612 - 2 * 71.0  # a4-ish? page is 612x792 => letter. margins from render:
# determine precisely: histogram of x1 in the top decile
c = Counter(round(r["x1"]) for r in rec if r["x1"] > 480)
print("histogram of x1>480:", sorted(c.items())[-25:])

clipped = [r for r in rec if r["x1"] > 500]
print(f"\nlines with x1>500: {len(clipped)} / {len(rec)}")
cp = Counter(r["page"] for r in clipped)
print("per page:", sorted(cp.items())[:40])
for r in clipped[:15]:
    print(f"  p{r['page']:>3} x1={r['x1']:>6} | {r['text'][-70:]!r}")

json.dump(rec, open(os.path.join(OUT, "src", "lines_geom.json"), "w", encoding="utf-8"))
