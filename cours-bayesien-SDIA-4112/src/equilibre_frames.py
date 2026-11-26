"""Equilibre des environnements DANS chaque frame (et non par fichier)."""
import glob, os, re, sys

PARTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "slides", "parts")
JEUX = ["columns", "column", "itemize", "enumerate", "tabular", "kbox", "ebox", "wbox", "abox", "qbox",
        "solbox", "codebox", "figbox", "tcolorbox", "center", "minipage", "tikzpicture", "description"]
motif = sys.argv[1] if len(sys.argv) > 1 else "ch*.tex"
anomalies = 0
for f in sorted(glob.glob(os.path.join(PARTS, motif))):
    src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    i = 0
    while True:
        a = src.find("\\begin{frame}", i)
        if a < 0:
            break
        b = src.find("\\end{frame}", a)
        if b < 0:
            break
        corps = src[a:b]
        ligne = src[:a].count("\n") + 1
        titre = corps.split("}")[0][12:60] if "{" in corps else ""
        pb = []
        for j in JEUX:
            nb = len(re.findall(r"\\begin\{" + j + r"\}", corps))
            ne = len(re.findall(r"\\end\{" + j + r"\}", corps))
            if nb != ne:
                pb.append(f"{j}:{nb}/{ne}")
        if pb:
            anomalies += 1
            print(f"  {os.path.basename(f)}:{ligne:<5} {' '.join(pb):<28} {titre}")
        i = b + 1
print(f"\n{anomalies} frame(s) desequilibree(s)")
