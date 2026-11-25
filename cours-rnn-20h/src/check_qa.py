"""Controle qualite du deck CNN : debordements et contenu coupe, mesures dans le PDF.

Le pied de page (barre avec le numero de slide) est exclu explicitement : c'est lui
qui produisait un defaut uniforme sur toutes les pages.
"""
import os, sys, re, glob
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
NUMERO = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")


def mesure(pdf, seuil=6.0):
    d = pymupdf.open(pdf)
    W, H = d[0].rect.width, d[0].rect.height
    droite, bas, coup = [], [], []
    pages_def = set()
    for pno, page in enumerate(d):
        lignes = []
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                lignes.append((ln["bbox"], " ".join(s["text"] for s in ln.get("spans", []))))
        pied = None
        for (x0, y0, x1, y1), t in lignes:
            if NUMERO.match(t):
                pied = y0
                break
        if pied is None:
            pied = H - 10
        for (x0, y0, x1, y1), t in lignes:
            if y0 >= pied - 1:
                continue
            if x1 > W - seuil:
                droite.append((pno + 1, round(x1 - (W - seuil), 1), "DROITE", t[:38]))
                pages_def.add(pno)
            if y1 > pied + 1:
                bas.append((pno + 1, round(y1 - (pied + 1), 1), "SOUS-PIED", t[:38]))
                pages_def.add(pno)
        for dr in page.get_drawings():
            r = dr["rect"]
            if r.height > 200 or r.width > W * .95:
                continue
            if r.y0 >= pied:
                continue
            if r.y1 > pied + 1:
                coup.append((pno + 1, round(r.y1 - (pied + 1), 1), "ENCADRE-BAS", ""))
                pages_def.add(pno)
            if r.x1 > W + 1:
                coup.append((pno + 1, round(r.x1 - W, 1), "ENCADRE-DROITE", ""))
                pages_def.add(pno)
    return {"pages": d.page_count, "droite": droite, "bas": bas, "coup": coup,
            "pages_def": sorted(pages_def)}


def main():
    seuil = 6.0
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--seuil=" in " ".join(sys.argv):
        seuil = float(" ".join(sys.argv).split("--seuil=")[1].split()[0])
    fichiers = args or sorted(glob.glob(os.path.join(OUT, "Chapitre-*.pdf")))
    total = 0
    rapport = {}
    for f in fichiers:
        r = mesure(f, seuil)
        n = len(r["droite"]) + len(r["bas"]) + len(r["coup"])
        total += n
        rapport[f] = r["pages_def"]
        print(f"=== {os.path.basename(f)} — {r['pages']} pages ===")
        if n == 0:
            print("   aucun defaut au seuil de %.1f pt" % seuil)
        else:
            for p, d_, s_, t in (r["droite"] + r["bas"] + r["coup"])[:12]:
                print(f"   p{p:<3} [{s_:>14}] +{d_:5.1f} pt  {t}")
            if n > 12:
                print(f"   ... +{n - 12} autres")
    print(f"\nRESULTAT : {'PROPRE' if total == 0 else str(total) + ' DEFAUT(S)'}")
    import json
    j = os.path.join(ROOT, "out", "journal", "qa_defects.json")
    os.makedirs(os.path.dirname(j), exist_ok=True)
    with open(j, "w", encoding="utf-8") as fh:
        json.dump(rapport, fh)


if __name__ == "__main__":
    main()
