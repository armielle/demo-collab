import re, json, os

OUT = r"C:\Users\HP\Documents\cours-bayesien-SDIA-4112"
txt = open(os.path.join(OUT, "src", "text_all.txt"), encoding="utf-8").read()
pages = re.split(r"\n===== PAGE (\d+) =====\n", txt)[1:]
pg = {}
for i in range(0, len(pages), 2):
    pg[int(pages[i])] = pages[i + 1]

# locate slides
slides = []
for p, body in pg.items():
    for m in re.finditer(r"Slide\s+(\d+)\s*[—\-–]\s*([^\n]{0,120})", body):
        slides.append((int(m.group(1)), m.group(2).strip(), p))

# chapter / section headings
chaps, secs = [], []
for p, body in pg.items():
    for m in re.finditer(r"(Chapitre\s+\d+\s*:[^\n]{0,90})", body):
        chaps.append((m.group(1).strip(), p))
    for m in re.finditer(r"(Section\s+\d+\.\d+\s*:[^\n]{0,90})", body):
        secs.append((m.group(1).strip(), p))

print("=== CHAPTERS ===")
seen = set()
for c, p in chaps:
    if (c, p) not in seen:
        seen.add((c, p)); print(f" p{p:>3} | {c}")
print("\n=== SECTIONS (unique, in order) ===")
seen = set()
for s, p in secs:
    if s not in seen:
        seen.add(s); print(f" p{p:>3} | {s}")
print("\n=== SLIDES:", len(slides), "===")
prev = 0
for n, t, p in slides:
    print(f" p{p:>3} | s{n:>3} | {t}")
