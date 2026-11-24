# -*- coding: utf-8 -*-
"""Trouve les variables de build (pré-calcul) utilisées dans les cellules
code des notebooks sans y être définies."""
import re

s = open("src/mk_tp.py", encoding="utf-8").read()
build_vars = ["cpu_parc", "p99_parc", "p99_hors_inc", "dispo", "indispo_h",
              "conso_tot", "top_conso", "corr_lc", "creneaux", "cpu_incident"]
marker = '("code", \'\'\''
cells = []
pos = 0
while True:
    i = s.find(marker, pos)
    if i < 0:
        break
    j = s.find("''')", i)
    cells.append(s[i + len(marker):j])
    pos = j + 1
print("nb cellules code :", len(cells))
probs = {}
for n, c in enumerate(cells, 1):
    for v in build_vars:
        if re.search(r"\b" + v + r"\b", c) and not re.search(r"\b" + v + r"\s*=", c):
            probs.setdefault(v, []).append(n)
for v, nums in probs.items():
    print(v, "-> utilisée sans définition dans cellules code", nums)
