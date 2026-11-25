"""Detecte les IMAGES rognees dans les PDF compiles.

Une figure trop haute pour la zone de texte n'est pas redimensionnee par
LaTeX : elle deborde sous la barre de pied de page (masquee) ou sous le bord
de la page (coupee net). Ce script liste, pour chaque page, les images dont le
bas depasse la limite utile.

Usage :  ./.venv/Scripts/python.exe src/check_images.py [--seuil=1]
"""
import os, sys, glob
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")


def bande_pied(page, defaut):
    """y0 de la ligne « n / N » du pied de page."""
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            t = "".join(s["text"] for s in l["spans"]).strip()
            if "/" in t and l["bbox"][1] > page.rect.height * 0.9:
                return l["bbox"][1]
    return defaut


def analyse(pdf, seuil):
    d = pymupdf.open(pdf)
    H = d[0].rect.height
    pied = bande_pied(d[0], H - 7.4)
    W = d[0].rect.width
    res = []
    for i, p in enumerate(d):
        for im in p.get_image_info():
            x0, y0, x1, y1 = im["bbox"]
            h, w = y1 - y0, x1 - x0
            if h < 12 or w < 12:             # icones, puces : ignorer
                continue
            bas = y1 - pied
            droite = x1 - (W - 4)            # bord de page (marge 4 pt)
            if bas > seuil or droite > seuil:
                res.append((i + 1, round(bas, 1), round(droite, 1),
                            round(w, 1), round(h, 1), round(y0, 1)))
    return dict(page=d.page_count, pied=round(pied, 1), H=round(H, 1), W=round(W, 1), images=res)


def main():
    seuil = 1.0
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            seuil = float(a.split("=")[1])
    total = 0
    for pdf in sorted(glob.glob(os.path.join(OUT, "Chapitre-*.pdf"))):
        r = analyse(pdf, seuil)
        print(f"\n=== {os.path.basename(pdf)} — pied a {r['pied']} pt, page {r['H']} pt ===")
        if not r["images"]:
            print("  aucune image rognee")
        for pn, bas, droite, w, h, y0 in sorted(r["images"], key=lambda x: -max(x[1], x[2])):
            quoi = []
            if bas > seuil: quoi.append(f"bas +{bas:.1f} pt")
            if droite > seuil: quoi.append(f"droite +{droite:.1f} pt")
            print(f"  p{pn:<3} {' / '.join(quoi):<28} | image {w:.0f}x{h:.0f} pt, haut a {y0:.0f}")
        total += len(r["images"])
    print(f"\nTOTAL images rognees : {total}")


if __name__ == "__main__":
    main()
