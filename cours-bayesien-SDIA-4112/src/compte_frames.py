"""Compte les frames et verifie l'equilibre des environnements, fichier par fichier."""
import glob, os, re, sys

PARTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "slides", "parts")
JEUX = ["columns", "column", "itemize", "enumerate", "tabular", "kbox", "ebox", "wbox", "abox", "qbox",
        "solbox", "codebox", "figbox", "tcolorbox", "center", "minipage", "tikzpicture", "description"]

motif = sys.argv[1] if len(sys.argv) > 1 else "ch*.tex"
for f in sorted(glob.glob(os.path.join(PARTS, motif))):
    s = open(f, encoding="utf-8", newline="").read()
    b = len(re.findall(r"\\begin\{frame\}", s))
    e = len(re.findall(r"\\end\{frame\}", s))
    pb = []
    for j in JEUX:
        nb = len(re.findall(r"\\begin\{" + j + r"\}", s))
        ne = len(re.findall(r"\\end\{" + j + r"\}", s))
        if nb != ne:
            pb.append(f"{j} {nb}/{ne}")
    marque = " <<<" if (b != e or pb) else ""
    print(f"{os.path.basename(f):<14} frames({b}/{e})  " + ("; ".join(pb) if pb else "equilibre") + marque)
