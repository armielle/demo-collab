"""Scinde une frame trop haute MEME quand son unique bloc est un `columns`.

Suite de split_frames.py (figure) et split_frames2.py (dernier bloc de premier niveau).
Ici on descend dans la colonne la plus longue et l'on deplace son dernier encadre sur une
slide « (suite) », la colonne etant refermee proprement dans la frame d'origine.

    python src/split_frames4.py [--dry-run] [--seuil=8]
"""
import os, re, sys, glob
import check_boxes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PARTS = os.path.join(ROOT, "slides", "parts")
CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}
MARK = "% [suite]"
BLOCS = ("columns", "tcolorbox", "kbox", "abox", "ebox", "wbox", "qbox", "solbox", "codebox",
         "itemize", "enumerate", "tabular", "center", "figbox", "minipage", "tikzpicture")


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


def spans(texte, nom, debut=0):
    """[(debut, fin, debut_contenu, fin_contenu)] des environnements `nom` de premier niveau."""
    res, i = [], debut
    while True:
        m = re.compile(r"\\begin\{" + nom + r"\}(\[[^\]]*\])?").search(texte, i)
        if not m:
            return res
        prof, k = 1, m.end()
        while prof:
            o = texte.find(r"\begin{" + nom + "}", k)
            c = texte.find(r"\end{" + nom + "}", k)
            if c < 0:
                return res
            if o >= 0 and o < c:
                prof += 1
                k = o + len(nom) + 8
            else:
                prof -= 1
                k = c + len(nom) + 6
        res.append((m.start(), k, m.end(), k - len(nom) - 6))
        i = k


def derniers_blocs(corps):
    """Blocs de premier niveau parmi BLOCS, avec les blocs internes (indices plus profonds)."""
    out = []
    for nom in BLOCS:
        for d, f, dc, fc in spans(corps, nom):
            out.append((d, f, nom))
    # on ne garde que les blocs racines : aucun autre bloc ne les contient
    racines = [(d, f, n) for d, f, n in out if not any(dd < d and f <= ff for dd, ff, _ in out)]
    return sorted(racines)


def coupe(corps):
    """(debut, fin, nom, fermetures) du bloc a deplacer, ou None."""
    racines = derniers_blocs(corps)
    if len(racines) >= 2:
        d, f, n = racines[-1]
        return d, f, n, ""
    if not racines or racines[0][2] != "columns":
        return None
    d_col, f_col, _ = racines[0]
    cols = spans(corps, "column", d_col)
    if len(cols) < 2:
        return None
    i = max(range(len(cols)), key=lambda k: cols[k][3] - cols[k][2])   # colonne la plus remplie
    cd, cf, cdc, cfc = cols[i]
    blocs = sorted((b for b in derniers_blocs(corps[cdc:cfc])))
    if len(blocs) < 2:
        return None
    d2, f2, n2 = blocs[-1]
    debut = cdc + d2
    return debut, debut, n2 + " (dans la colonne)", "\n\\end{column}\n\\end{columns}"


def main():
    dry = "--dry-run" in sys.argv
    seuil = 8.0
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            seuil = float(a.split("=")[1])
    total = 0
    for ch in CH:
        pdf = os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")
        if not os.path.exists(pdf):
            continue
        r = check_boxes.analyse(pdf, 1.0)
        pages = sorted({p for p, dep in r["graves"] if dep > seuil}, reverse=True)
        if not pages:
            print(f"=== chapitre {ch} : rien a scinder ===")
            continue
        index = [(None, None, None)]
        for f in sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex"))):
            src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            index += [(f, a, b) for a, b in frames_of(src)]
        par_fichier = {}
        print(f"\n=== chapitre {ch} : {len(pages)} page(s) a scinder ===")
        for pn in pages:
            if pn - 1 >= len(index):
                continue
            f, a, b = index[pn - 1]
            if f is None:
                continue
            src = par_fichier.get(f) or open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            bloc = src[a:b]
            if MARK in bloc:
                print(f"  p{pn:<3} deja scindee")
                continue
            entete = re.match(r"(\\begin\{frame\}(?:\[[^\]]*\])?\{[^\n]*\})\n", bloc)
            fin = bloc.rfind(r"\end{frame}")
            corps = bloc[entete.end():fin] if entete else None
            if not corps or not entete:
                continue
            c = coupe(corps)
            if not c:
                print(f"  p{pn:<3} impossible : bloc unique non decoupable")
                continue
            d, f_, nom, fermetures = c
            reste = corps[:d].rstrip()
            deplace = corps[d:].strip()
            titre = entete.group(1)
            t = re.match(r"\\begin\{frame\}(\[[^\]]*\])?\{([^\n]*)\}", titre)
            opts, nom_titre = t.group(1) or "", t.group(2)
            corps_a = reste + fermetures
            nouvelle = (f"{MARK}\n{titre}\n{corps_a}\n\\end{{frame}}\n\n{MARK}\n"
                        f"\\begin{{frame}}{opts}{{{nom_titre} (suite)}}\n{deplace}\n\\end{{frame}}")
            par_fichier[f] = src[:a] + nouvelle + src[b:]
            total += 1
            print(f"  p{pn:<3} scindee ({nom})")
        if not dry:
            for f, contenu in par_fichier.items():
                open(f, "w", encoding="utf-8", newline="").write(contenu.replace("\n", "\r\n"))
    print(f"\n{total} frame(s) scindee(s)" + (" (simulation)" if dry else ""))


if __name__ == "__main__":
    main()
