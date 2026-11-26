"""Detecte les images plus larges que leur colonne (source LaTeX).

Une `\\includegraphics[width=.80\\textwidth]` placee dans une
`\\begin{column}{0.55\\textwidth}` est plus large que sa colonne : elle deborde
sur la colonne voisine, qui est dessinee par-dessus. Resultat a l'ecran : une
image visuellement ROGNEE, alors que son cadre reste dans la page (donc
invisible pour un controle qui ne regarde que les bords de page).

Usage :  ./.venv/Scripts/python.exe src/check_images_src.py
"""
import os, re, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")

COL = re.compile(r"\\begin\{column\}(?:\[[^\]]*\])?\{([\d.]+)\\textwidth\}")
IMG = re.compile(r"\\includegraphics\[[^\]]*width=(\d*\.?\d+)\\textwidth[^\]]*\]\{([^}]+)\}")
FRAME = re.compile(r"\\begin\{frame\}(\[[^\]]*\])?\{([^\n]*)\}")
FIN = r"\end{frame}"


def analyse(chemin):
    src = open(chemin, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    lignes = src.split("\n")
    res = []
    for m in FRAME.finditer(src):
        fin = src.find(FIN, m.end())
        bloc = src[m.start():fin]
        ligne = src[:m.start()].count("\n") + 1
        titre = m.group(2)[:46]
        # parcours sequentiel : on retient la colonne courante
        col = 1.0
        for k, l in enumerate(bloc.split("\n")):
            mc = COL.search(l)
            if mc:
                col = float(mc.group(1))
            for mi in IMG.finditer(l):
                w, nom = float(mi.group(1)), mi.group(2)
                if w > col + 0.02:
                    res.append((os.path.basename(chemin), ligne, titre, nom,
                                w, col, round(w - col, 2)))
    return res


def main():
    tout = []
    for f in sorted(glob.glob(os.path.join(PARTS, "ch*.tex"))):
        tout += analyse(f)
    print(f"{len(tout)} image(s) plus large(s) que leur colonne\n")
    for f, ligne, titre, nom, w, col, dep in sorted(tout, key=lambda x: -x[6]):
        print(f"  {f}:{ligne:<5} image {w:.2f} dans colonne {col:.2f} (+{dep:.2f})  {nom[:34]:<36} « {titre} »")


if __name__ == "__main__":
    main()
