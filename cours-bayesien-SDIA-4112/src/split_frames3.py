"""Scinde une frame dont l'unique bloc est un `columns` : une colonne par slide.

Dernier recours automatique, apres split_frames.py (figure) et split_frames2.py (dernier
bloc). Pour les frames restantes, le bloc trop haut est un `\\begin{columns}` : on repartit
ses colonnes sur deux slides.

Usage : ./.venv/Scripts/python.exe src/split_frames3.py
"""
import os, re, sys, glob
import check_boxes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PARTS = os.path.join(ROOT, "slides", "parts")
CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}
MARK = "% [cols]"
SEUIL = 8.0


def frames_of(src):
    out, pos = [], 0
    while True:
        i = src.find(r"\begin{frame}", pos)
        if i < 0:
            return out
        j = src.find(r"\end{frame}", i)
        if j < 0:
            return out
        out.append((i, j + len(r"\end{frame}")))
        pos = j


def decoupe_colonnes(corps):
    """Retourne la liste des colonnes (texte complet, \begin{column}...\end{column})."""
    cols, i = [], 0
    while True:
        a = corps.find(r"\begin{column}", i)
        if a < 0:
            return cols
        prof, k = 1, a + len(r"\begin{column}")
        while prof:
            o = corps.find(r"\begin{column}", k)
            e = corps.find(r"\end{column}", k)
            if e < 0:
                break
            if 0 <= o < e:
                prof += 1
                k = o + len(r"\begin{column}")
            else:
                prof -= 1
                k = e + len(r"\end{column}")
        cols.append(corps[a:k])
        i = k


def main():
    total = 0
    for ch in CH:
        pdf = os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")
        if not os.path.exists(pdf):
            continue
        r = check_boxes.analyse(pdf, 1.0)
        pages = sorted({p for p, dep in r["graves"] if dep > SEUIL}, reverse=True)
        if not pages:
            continue
        index = [(None, None, None)]
        for f in sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex"))):
            src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            index += [(f, a, b) for a, b in frames_of(src)]
        par_fichier = {}
        print(f"\n=== chapitre {ch} : {len(pages)} page(s) ===")
        for pn in pages:
            if pn - 1 >= len(index):
                continue
            f, a, b = index[pn - 1]
            if f is None:
                continue
            src = par_fichier.get(f)
            if src is None:
                src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            bloc = src[a:b]
            if MARK in bloc or "% [suite]" in bloc:
                continue
            m = re.match(r"(\\begin\{frame\}(?:\[[^\]]*\])?\{([^\n]*)\})\n", bloc)
            if not m:
                continue
            entete, titre = m.group(1), m.group(2)
            fin = bloc.rfind(r"\end{frame}")
            corps = bloc[m.end():fin]
            if r"\begin{columns}" not in corps:
                print(f"  p{pn:<3} pas de colonnes : a traiter a la main")
                continue
            cols = decoupe_colonnes(corps)
            if len(cols) < 2:
                print(f"  p{pn:<3} une seule colonne : a traiter a la main")
                continue
            moitie = (len(cols) + 1) // 2
            g1, g2 = cols[:moitie], cols[moitie:]
            avant = corps[:corps.index(r"\begin{columns}")].strip()
            apres = corps[corps.rindex(r"\end{columns}") + len(r"\end{columns}"):].strip()

            def groupe(cols_):
                return ("\\begin{columns}[t]\n" + "\n".join(cols_) + "\n\\end{columns}")

            a1 = "\n".join(x for x in [avant, groupe(g1)] if x)
            a2 = "\n".join(x for x in [groupe(g2), apres] if x)
            nouvelle = (f"{MARK}\n{entete}\n{a1}\n\\end{{frame}}\n\n"
                        f"{MARK}\n\\begin{{frame}}{{{titre} (suite)}}\n{a2}\n\\end{{frame}}")
            par_fichier[f] = src[:a] + nouvelle + src[b:]
            total += 1
            print(f"  p{pn:<3} colonnes reparties sur 2 slides ({len(cols)} colonnes)")
        for f, contenu in par_fichier.items():
            open(f, "w", encoding="utf-8", newline="").write(contenu.replace("\n", "\r\n"))
    print(f"\n{total} frame(s) scindee(s)")


if __name__ == "__main__":
    main()
