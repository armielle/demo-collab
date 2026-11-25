"""Controle qualite du deck CNN : debordements et contenu coupe, mesures dans le PDF.

Se base sur les lignes de texte et les rectangles (les encadres tcolorbox sont des
rectangles dessines). Les traces decoratifs (cercles des pages de garde) sont exclus.
"""
import os, sys, glob
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")


def mesure(pdf, seuil=6.0):
    d = pymupdf.open(pdf)
    W, H = d[0].rect.width, d[0].rect.height
    droite, bas, hors, coup = [], [], 0, []
    for pno, page in enumerate(d):
        # ---- lignes de texte
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                x0, y0, x1, y1 = ln["bbox"]
                if x1 > W - seuil:
                    droite.append((pno + 1, round(x1 - (W - seuil), 1), "DROITE"))
                if y1 > H - seuil:
                    bas.append((pno + 1, round(y1 - (H - seuil), 1), "BAS"))
                if y0 < -2 or x0 < -2:
                    hors += 1
        # ---- rectangles dessines (encadres tcolorbox)
        for dr in page.get_drawings():
            r = dr["rect"]
            if r.height > 200 or r.width > W * .95:      # decor / fonds de page
                continue
            if r.y1 > H + 1 and r.y0 < H:                # encadre qui sort par le bas
                coup.append((pno + 1, round(r.y1 - H, 1), "ENCADRE-BAS"))
            if r.x1 > W + 1:
                coup.append((pno + 1, round(r.x1 - W, 1), "ENCADRE-DROITE"))
    return {"pages": d.page_count, "droite": droite, "bas": bas,
            "hors": hors, "coup": coup}


def main():
    seuil = 6.0
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--seuil=" in " ".join(sys.argv):
        seuil = float(" ".join(sys.argv).split("--seuil=")[1].split()[0])
    fichiers = args or sorted(glob.glob(os.path.join(OUT, "Chapitre-*.pdf"))) + \
        [os.path.join(OUT, "Cours-CNN-complet.pdf")]
    total = 0
    for f in fichiers:
        r = mesure(f, seuil)
        n = len(r["droite"]) + len(r["bas"]) + len(r["coup"])
        total += n
        print(f"=== {os.path.basename(f)} — {r['pages']} pages ===")
        if n == 0:
            print("   aucun defaut au seuil de %.1f pt" % seuil)
        else:
            for p, d_, s_ in (r["droite"] + r["bas"] + r["coup"])[:12]:
                print(f"   p{p:<3} [{s_:>14}] +{d_:.1f} pt")
            if n > 12:
                print(f"   ... +{n - 12} autres")
    print(f"\nRESULTAT : {'PROPRE' if total == 0 else str(total) + ' DEFAUT(S)'}")


if __name__ == "__main__":
    main()
