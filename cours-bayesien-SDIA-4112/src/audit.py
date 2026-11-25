import re, os, json
from collections import Counter

OUT = r"C:\Users\HP\Documents\cours-bayesien-SDIA-4112"
t = open(os.path.join(OUT, "src", "text_all.txt"), encoding="utf-8").read()
parts = re.split(r"\n===== PAGE (\d+) =====\n", t)[1:]
pg = {int(parts[i]): parts[i + 1] for i in range(0, len(parts), 2)}

# 1) longest code-ish lines : estimate clipping width
lens = Counter()
alllines = []
for p in sorted(pg):
    for ln in pg[p].split("\n"):
        alllines.append((p, ln))
        lens[len(ln.rstrip())] += 1
print("=== distribution of the longest line lengths (top 25) ===")
for L, c in sorted(lens.items(), reverse=True)[:25]:
    print(f"  len={L:>4}  count={c}")

# 2) heuristic clipped lines: length >= 95 and does not end in a 'closing' char
CLOSE = set('})\\].,;:"\'%#')
clipped = [(p, ln) for p, ln in alllines if len(ln.rstrip()) >= 95 and ln.rstrip()[-1] not in CLOSE]
print(f"\n=== suspected clipped lines: {len(clipped)} (out of {len(alllines)} lines) ===")
perpage = Counter(p for p, _ in clipped)
print("pages most affected:", perpage.most_common(12))
for p, ln in clipped[:12]:
    print(f"  p{p:>3} |{len(ln):>4}| ...{ln.strip()[-60:]!r}")

# 3) chapter coverage: find end of each chapter block by \end{document}
pages_list = sorted(pg)
ends = [p for p in pages_list if r"\end{document}" in pg[p]]
print("\n=== pages containing \\end{document} (end of a chapter file) ===", ends)

# 4) count frame titles
titles = []
for p in pages_list:
    for m in re.finditer(r"\\begin\{frame\}(?:\{[^\n]*\}|\[[^\]]*\])?", pg[p]):
        titles.append((p, m.group(0)))
print("\ntotal \\begin{frame} occurrences:", len(titles))
print("frames with explicit title:", sum(1 for _, s in titles if "{" in s))

# 5) includegraphics referenced files (existing? commented?)
files = Counter()
for p in pages_list:
    for m in re.finditer(r"(%?)\\includegraphics\[[^\]]*\]\{([^}]*)\}", pg[p]):
        files[(m.group(2), bool(m.group(1)))] += 1
print("\n=== \\includegraphics targets (commented?) ===")
for (f, c), n in sorted(files.items()):
    print(f"  {'COMMENTED' if c else 'ACTIVE   '} {f}  x{n}")
