"""Verifie l'ordre d'ouverture/fermeture des frames : une frame ne doit pas en contenir une autre."""
import glob, os, re, sys

PARTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "slides", "parts")
motif = sys.argv[1] if len(sys.argv) > 1 else "ch*.tex"
anomalies = 0
for f in sorted(glob.glob(os.path.join(PARTS, motif))):
    L = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n").split("\n")
    etat = None
    for n, l in enumerate(L, start=1):
        if "\\begin{frame}" in l:
            if etat is not None:
                print(f"  {os.path.basename(f)}:{n}  frame NESTED dans une frame ouverte en {etat}")
                anomalies += 1
            etat = n
        if "\\end{frame}" in l:
            if etat is None:
                print(f"  {os.path.basename(f)}:{n}  fermeture sans ouverture")
                anomalies += 1
            etat = None
    if etat is not None:
        print(f"  {os.path.basename(f)}  frame ouverte en {etat} jamais fermee")
        anomalies += 1
print(f"\n{anomalies} anomalie(s) de sequence")
