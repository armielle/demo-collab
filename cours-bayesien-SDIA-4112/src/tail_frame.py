"""Affiche la fin d'une frame (la ou se produit le débordement).

Usage : ./.venv/Scripts/python.exe src/tail_frame.py ch1_e.tex:2 [--n=20]
"""
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")
N = 20
for a in sys.argv[1:]:
    if a.startswith("--n="):
        N = int(a.split("=")[1])

for arg in [a for a in sys.argv[1:] if not a.startswith("--")]:
    fichier, ligne = arg.split(":")
    L = open(os.path.join(PARTS, fichier), encoding="utf-8", newline="").read().replace("\r\n", "\n").split("\n")
    i = int(ligne) - 1
    j = i
    while j < len(L) and r"\end{frame}" not in L[j]:
        j += 1
    utiles = [(k + 1, l) for k, l in enumerate(L[i:j + 1], start=i) if l.strip()]
    print(f"\n########## {arg} — {len(utiles)} lignes utiles ##########")
    print(f"  {utiles[0][0]:4} {utiles[0][1][:100]}")
    if len(utiles) > N + 1:
        print(f"  ...  ({len(utiles) - 1 - N} lignes passées)  ...")
    for k, l in utiles[len(utiles) - N:]:
        print(f"  {k:4} {l[:100]}")
