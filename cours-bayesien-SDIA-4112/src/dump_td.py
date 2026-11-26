# -*- coding: utf-8 -*-
import pymupdf, os

for f in [r"C:\Users\HP\Downloads\TD_Estimateur_Bayes_Niveau1.pdf",
          r"C:\Users\HP\Downloads\TD_StatBayes_Chap1-2.pdf"]:
    d = pymupdf.open(f)
    name = os.path.basename(f)
    print("=" * 30, name, d.page_count, "pages")
    for i in range(d.page_count):
        print("--- page", i + 1)
        print(d[i].get_text())
