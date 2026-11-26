"""Separe le CONTENU d'une frame trop haute : deplace son dernier bloc sur une slide suivante.

Complement de split_frames.py (qui sort la figure). Ici, pour les frames dont le texte
seul depasse la page, on coupe au dernier bloc de premier niveau et l'on cree une frame
« (suite) ». Utilise en boucle avec la mesure :

    python src/split_frames2.py && python src/build.py && python src/check_boxes.py
"""
import os, re, sys, glob
import check_boxes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PARTS = os.path.join(ROOT, "slides", "parts")
CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}
MARK = "% [suite]"
SEUIL = 8.0
# environnements que l'on accepte de deplacer seuls
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


def top_level_blocks(corps):
    """[(debut, fin, nom)] des environnements de premier niveau dans le corps d'une frame."""
    res, i = [], 0
    while i < len(corps):
        m = re.compile(r"\\begin\{(\w+)\}").search(corps, i)
        if not m:
            break
        nom = m.group(1)
        if nom not in BLOCS:
            i = m.end()
            continue
        prof, k = 1, m.end()
        while prof and k < len(corps):
            o = corps.find(r"\begin{" + nom + "}", k)
            c = corps.find(r"\end{" + nom + "}", k)
            if c < 0:
                break
            if o >= 0 and o < c:
                prof += 1
                k = o + len(nom) + 8
            else:
                prof -= 1
                k = c + len(nom) + 6
        res.append((m.start(), k, nom))
        i = k
    return res


def main():
    dry = "--dry-run" in sys.argv
    total = 0
    for ch in CH:
        pdf = os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")
        if not os.path.exists(pdf):
            continue
        r = check_boxes.analyse(pdf, 1.0)
        pages = sorted({p for p, dep in r["graves"] if dep > SEUIL}, reverse=True)
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
            src = par_fichier.get(f)
            if src is None:
                src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            bloc = src[a:b]
            if MARK in bloc:
                continue
            entete = re.match(r"(\\begin\{frame\}(?:\[[^\]]*\])?\{[^\n]*\})\n", bloc)
            fin = bloc.rfind(r"\end{frame}")
            corps = bloc[entete.end():fin] if entete else None
            if not corps:
                continue
            bl = top_level_blocks(corps)
            if len(bl) < 2:
                print(f"  p{pn:<3} un seul bloc : a traiter a la main")
                continue
            d, fin_dernier, nom = bl[-1]
            reste = corps[:d].rstrip()
            deplace = corps[d:].strip()
            titre = entete.group(1)
            t = re.match(r"\\begin\{frame\}(\[[^\]]*\])?\{([^\n]*)\}", titre)
            opts, nom_titre = t.group(1) or "", t.group(2)
            nouvelle = f"{MARK}\n{titre}\n{reste}\n\\end{{frame}}\n\n{MARK}\n\\begin{{frame}}{opts}{{{nom_titre} (suite)}}\n{deplace}\n\\end{{frame}}"
            par_fichier[f] = src[:a] + nouvelle + src[b:]
            total += 1
            print(f"  p{pn:<3} scindee (dernier bloc : {nom})")
        if not dry:
            for f, contenu in par_fichier.items():
                open(f, "w", encoding="utf-8", newline="").write(contenu.replace("\n", "\r\n"))
    print(f"\n{total} frame(s) scindee(s)" + (" (simulation)" if dry else ""))


if __name__ == "__main__":
    main()
