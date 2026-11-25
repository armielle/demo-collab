"""Version allegee du document maitre CNN : recompresse les images, sous-ensemble les polices."""
import os
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "out", "Cours-CNN-complet.pdf")
DST = os.path.join(ROOT, "out", "Cours-CNN-complet-leger.pdf")

doc = pymupdf.open(SRC)
print("source :", os.path.getsize(SRC) / 1e6, "Mo,", doc.page_count, "pages")
doc.rewrite_images(dpi_threshold=150, dpi_target=120, quality=68, set_to_gray=False)
doc.subset_fonts(verbose=False)
doc.save(DST, garbage=4, deflate=True)
doc.close()
doc2 = pymupdf.open(DST)
print("leger  :", os.path.getsize(DST) / 1e6, "Mo,", doc2.page_count, "pages")
print("gain   :", round(100 * os.path.getsize(DST) / os.path.getsize(SRC)), "% de l'original")
