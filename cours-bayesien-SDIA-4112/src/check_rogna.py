"""Detecte les elements ROGNES dans les PDF compiles.

Trois familles de rognage, mesurees sur le PDF (pas sur la source) :

1. hors page      : un texte ou un trace depasse le bord de la page -> coupe nette ;
2. sous le pied   : une ligne de corps descend dans la bande du pied de page,
                    qui est dessinee par-dessus -> ligne masquee ;
3. hors zone      : un texte depasse le bord droit de la zone de texte
                    (defaut esthetique, sans perte de contenu).

Usage :  ./.venv/Scripts/python.exe src/check_rogna.py [--seuil=2]
"""
import os, sys, glob
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")


def zone_droite(doc):
    """Bord droit de la zone de texte = mode des x1 des lignes de corps."""
    from collections import Counter
    c = Counter()
    for p in doc:
        w = p.rect.width
        for b in p.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t and l["bbox"][2] > 0.72 * w:
                    c[round(l["bbox"][2])] += 1
    return c.most_common(1)[0][0] if c else 0


def bande_pied(doc):
    """Haut de la bande de pied de page : y0 de la ligne « n / N »."""
    for p in doc:
        for b in p.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                t = "".join(s["text"] for s in l["spans"]).strip()
                if "/" in t and l["bbox"][1] > p.rect.height * 0.9:
                    return l["bbox"][1]
    return doc[0].rect.height


def analyse(pdf, seuil):
    d = pymupdf.open(pdf)
    W, H = d[0].rect.width, d[0].rect.height
    droite = zone_droite(d)
    pied = bande_pied(d)
    hors, sous, zone = [], [], []
    for i, p in enumerate(d):
        for b in p.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                t = "".join(s["text"] for s in l["spans"]).strip()
                if not t:
                    continue
                x0, y0, x1, y1 = l["bbox"]
                if x1 > W + 0.5 or y1 > H + 0.5 or x0 < -0.5:
                    hors.append((i + 1, round(max(x1 - W, y1 - H), 1), t[:48]))
                if 26 < y0 < pied and y1 > pied + seuil:
                    sous.append((i + 1, round(y1 - pied, 1), t[:48]))
                if x1 > droite + seuil:
                    zone.append((i + 1, round(x1 - droite, 1), t[:48]))
        # traces (rectangles) hors page
        for dr in p.get_drawings():
            r = dr["rect"]
            if r.x1 > W + 0.5 or r.y1 > H + 0.5 or r.x0 < -0.5:
                hors.append((i + 1, round(max(r.x1 - W, r.y1 - H), 1), "[trace]"))
    return dict(page=d.page_count, droite=droite, pied=round(pied, 1),
                W=round(W, 1), H=round(H, 1), hors=hors, sous=sous, zone=zone)


def main():
    seuil = 2.0
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            seuil = float(a.split("=")[1])
    total = [0, 0, 0]
    for pdf in sorted(glob.glob(os.path.join(OUT, "Chapitre-*.pdf"))):
        r = analyse(pdf, seuil)
        nom = os.path.basename(pdf)
        print(f"\n=== {nom} — {r['page']} pages | zone droite {r['droite']} pt | "
              f"pied a {r['pied']} pt | page {r['W']}x{r['H']} ===")
        print(f"  hors page (coupe nette)        : {len(r['hors'])}")
        for pn, dep, t in sorted(r["hors"], key=lambda x: -x[1])[:6]:
            print(f"      p{pn:<3} +{dep:5.1f} pt  {t}")
        print(f"  sous la bande du pied (masque) : {len(r['sous'])}")
        for pn, dep, t in sorted(r["sous"], key=lambda x: -x[1])[:6]:
            print(f"      p{pn:<3} +{dep:5.1f} pt  {t}")
        print(f"  hors zone droite (esthetique)  : {len(r['zone'])}")
        for pn, dep, t in sorted(r["zone"], key=lambda x: -x[1])[:6]:
            print(f"      p{pn:<3} +{dep:5.1f} pt  {t}")
        total[0] += len(r["hors"]); total[1] += len(r["sous"]); total[2] += len(r["zone"])
    print(f"\nTOTAL — hors page : {total[0]} | sous le pied : {total[1]} | hors zone droite : {total[2]}")


if __name__ == "__main__":
    main()
