"""Produit une version allegee du cours complet, pour l'envoi par courriel.

Recompresse les images (JPEG, 120 ppp cible), sous-ensemble les polices et
re-enregistre le PDF avec ramasse-miettes. La version allegee est ecrite a cote
de l'originale, avec le suffixe `-leger.pdf`.

Usage :  ./.venv/Scripts/python.exe src/make_pdf_leger.py [chemin.pdf]
"""
import os, sys, time
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "out", "Cours-complet-Statistique-Bayesienne.pdf")
DST = SRC[:-4] + "-leger.pdf"

t0 = time.time()
doc = pymupdf.open(SRC)
avant = os.path.getsize(SRC)

doc.rewrite_images(dpi_threshold=150, dpi_target=120, quality=68, set_to_gray=False)
try:
    doc.subset_fonts()
    polices = "oui"
except Exception as e:                                     # pragma: no cover
    polices = f"non ({e.__class__.__name__})"

doc.save(DST, garbage=4, deflate=True, clean=True)
doc.close()
apres = os.path.getsize(DST)
d = pymupdf.open(DST)

print(f"source  : {os.path.basename(SRC)}  {avant/1024/1024:.2f} Mo")
print(f"allege  : {os.path.basename(DST)}  {apres/1024/1024:.2f} Mo  "
      f"({100*apres/avant:.0f} % de l'original)")
print(f"pages   : {d.page_count}   polices sous-ensemblessees : {polices}")
print(f"duree   : {time.time()-t0:.1f} s")
