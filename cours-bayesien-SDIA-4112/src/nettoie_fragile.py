"""Retire l'option [fragile] des frames qui ne contiennent aucun environnement verbatim."""
import glob, os, re

PARTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "slides", "parts")
total = 0
for f in sorted(glob.glob(os.path.join(PARTS, "ch*.tex"))):
    src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    out, i = [], 0
    while True:
        a = src.find("\\begin{frame}", i)
        if a < 0:
            out.append(src[i:]); break
        b = src.find("\\end{frame}", a) + len("\\end{frame}")
        out.append(src[i:a]); out.append(src[a:b]); i = b
    modifie = False
    for k, bloc in enumerate(out):
        if not bloc.startswith("\\begin{frame}"):
            continue
        entete = bloc.split("\n")[0]
        if "fragile" not in entete:
            continue
        corps = bloc[:bloc.rfind("\\end{frame}")]
        if any(x in corps for x in ("lstlisting", "verbatim", "minted")):
            continue
        neuf = entete.replace("[fragile]", "").replace("[,fragile]", "").replace("[fragile,", "[")
        out[k] = neuf + bloc[len(entete):]
        modifie = True
        total += 1
        print(f"  {os.path.basename(f)} : [fragile] retire  {entete[:70]}")
    if modifie:
        open(f, "w", encoding="utf-8", newline="").write("".join(out).replace("\n", "\r\n"))
print(f"\n{total} frame(s) nettoyee(s)")
