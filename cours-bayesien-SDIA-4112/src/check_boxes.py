"""Detecte le contenu ROGNE dans les PDF compiles — par les rectangles, pas par le texte.

Pourquoi ce script existe : PyMuPDF ne renvoie que le texte qui intersecte la page.
Un encadre dont le contenu tombe sous le bord bas de la page disparait donc de
`get_text()` et echappe a un controle fonde sur les lignes de texte. Les rectangles
remplis (fonds d'encadres, images), eux, sont toujours listes.

Trois familles de rognage :
  BAS   : un encadre descend sous le haut du pied de page -> contenu masque ou hors page ;
  HAUT  : un texte remonte sous la barre de titre -> premiere ligne masquee ;
  DROITE: un rectangle depasse le bord droit de la zone de texte.

Usage :  ./.venv/Scripts/python.exe src/check_boxes.py [--seuil=1]
"""
import os, sys, glob
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")


def reperes(d):
    """(haut de la barre de titre, haut du pied de page, largeur page, hauteur page)"""
    titre, pied = 24.2, d[0].rect.height - 7.4
    for b in d[0].get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            t = "".join(s["text"] for s in l["spans"]).strip()
            y0, y1 = l["bbox"][1], l["bbox"][3]
            if t.startswith("Chapitre") and y0 < 25:
                titre = y1
            if "/" in t and y0 > d[0].rect.height * 0.9:
                pied = y0
    return titre, pied


def analyse(pdf, seuil):
    d = pymupdf.open(pdf)
    titre, pied = reperes(d)
    H, W = d[0].rect.height, d[0].rect.width
    bas, haut, droite, graves = [], [], [], []
    for i, p in enumerate(d):
        for dr in p.get_drawings():
            r = dr["rect"]
            if r.width < 60 or r.height < 6 or not dr.get("fill"):
                continue                      # traits, petites puces
            if r.x0 < 2 and r.x1 > W - 2:     # fond de page / barre de titre
                continue
            if r.height > 200:                # cercles decoratifs des pages de garde
                continue
            if r.y1 > pied + seuil:
                bas.append((i + 1, round(r.y1 - pied, 1), round(r.height, 1)))
            if r.y1 > pied + 8:
                graves.append((i + 1, round(r.y1 - pied, 1)))
        for b in p.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                t = "".join(s["text"] for s in l["spans"]).strip()
                if not t or "Chapitre" in t or "/" in t:
                    continue
                blanc = all(sp["color"] > 16000000 for sp in l["spans"] if sp["text"].strip())
                if blanc:
                    continue                  # titre de frame, ecrit en blanc
                if l["bbox"][1] < titre - 1:
                    haut.append((i + 1, round(titre - l["bbox"][1], 1), t[:44]))
                if l["bbox"][2] > W - 4:
                    droite.append((i + 1, round(l["bbox"][2] - (W - 4), 1), t[:44]))
    return dict(pages=d.page_count, titre=round(titre, 1), pied=round(pied, 1),
                H=round(H, 1), W=round(W, 1), bas=bas, haut=haut, droite=droite, graves=graves)


def main():
    seuil = 1.0
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            seuil = float(a.split("=")[1])
    tot = [0, 0, 0]
    for pdf in sorted(glob.glob(os.path.join(OUT, "Chapitre-*.pdf"))):
        r = analyse(pdf, seuil)
        print(f"\n=== {os.path.basename(pdf)} — {r['pages']} pages | titre finit {r['titre']} pt"
              f" | pied a {r['pied']} pt | page {r['W']}x{r['H']} ===")
        pages_bas = sorted({p for p, _, _ in r["bas"]})
        pages_graves = sorted({p for p, _ in r["graves"]})
        print(f"  encadrés touchant le pied : {len(r['bas'])} sur {len(pages_bas)} page(s)")
        print(f"  dont perte visible (>8 pt) : {len(r['graves'])} sur {len(pages_graves)} page(s)"
              f"  {pages_graves}")
        for pn, dep, h in sorted(r["bas"], key=lambda x: -x[1])[:8]:
            print(f"      p{pn:<3} +{dep:6.1f} pt  (encadré de {h:.0f} pt)")
        print(f"  textes sous la barre de titre : {len(r['haut'])}  {sorted({p for p,_,_ in r['haut']})}")
        for pn, dep, t in sorted(r["haut"], key=lambda x: -x[1])[:6]:
            print(f"      p{pn:<3} +{dep:6.1f} pt  {t}")
        print(f"  textes hors zone droite : {len(r['droite'])}")
        tot[0] += len(r["bas"]); tot[1] += len(r["haut"]); tot[2] += len(r["droite"])
    print(f"\nTOTAL — encadres coupés : {tot[0]} | textes sous le titre : {tot[1]} | hors zone droite : {tot[2]}")


if __name__ == "__main__":
    main()
