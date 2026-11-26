import pymupdf, os
PDF = r"C:\Users\HP\Downloads\EC _ EPD-SDIA-4112 – Statistique Bayésienne__Parco.pdf"
PAGES = r"C:\Users\HP\Documents\cours-bayesien-SDIA-4112\pages"
doc = pymupdf.open(PDF)
for pn in [1, 3, 14, 20, 43, 68, 77, 96, 119, 123, 148, 176]:
    p = doc[pn - 1]
    pix = p.get_pixmap(dpi=110)
    f = os.path.join(PAGES, f"p{pn:03d}.png")
    pix.save(f)
    print(f, pix.width, pix.height)
