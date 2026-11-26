"""Repare les largeurs d'images cassees par re.sub (la sequence \\t devient une TABULATION).

Contexte : dans `re.sub(motif, remplacement, texte)`, les sequences d'echappement du
remplacement sont interpretees. Un remplacement construit comme
`f"width={x}\\textwidth"` devient donc `width=.86<TAB>extwidth`, ce qui fait echouer
XeTeX avec « Package calc Error: `t' invalid at this point ».

Repare : `width=NN<TAB>extwidth` -> `width=NN\\textwidth`, ainsi que la variante a
double antislash. Idempotent.
"""
import os, re, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")

TAB = re.compile(r"width=(\d*\.?\d+)\textwidth")          # \t = vraie tabulation
DBL = re.compile(r"width=(\d*\.?\d+)\\\\\\\\textwidth")   # double antislash

total = 0
for f in sorted(glob.glob(os.path.join(PARTS, "*.tex"))):
    src = open(f, encoding="utf-8", newline="").read()
    n1 = len(TAB.findall(src))
    n2 = len(DBL.findall(src))
    if not (n1 or n2):
        continue
    src = TAB.sub(lambda m: "width=" + m.group(1) + "\\textwidth", src)
    src = DBL.sub(lambda m: "width=" + m.group(1) + "\\textwidth", src)
    with open(f, "w", encoding="utf-8", newline="") as fh:
        fh.write(src)
    print(f"  {os.path.basename(f):<14} {n1} tabulation(s), {n2} double(s) antislash  -> repare")
    total += n1 + n2

print(f"\n{total} largeur(s) d'image reparee(s)")

# controle : plus aucune tabulation dans un width=
restes = 0
for f in sorted(glob.glob(os.path.join(PARTS, "*.tex"))):
    src = open(f, encoding="utf-8", newline="").read()
    restes += len(TAB.findall(src)) + len(DBL.findall(src))
    for i, l in enumerate(src.split("\n")):
        if "\t" in l:
            print(f"  ATTENTION tabulation residuelle : {os.path.basename(f)}:{i+1}")
print("verification :", restes, "cas restants")
