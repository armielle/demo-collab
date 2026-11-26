"""Separe figure et contenu pour les frames dont le contenu ne tient pas dans la page.

Symptome : le corps de la frame est centre verticalement par Beamer ; quand le contenu
depasse la zone de texte, il deborde EN BAS (encadres dont le contenu part sous le bord
de la page : « texte rogne ») et parfois EN HAUT (sous la barre de titre).

Remede : sortir la figure de la frame et en faire une slide a part, en pleine largeur.
La frame d'origine, debarrassee de la figure, tient alors dans la page.

Ne traite QUE les pages signalees par check_boxes.py, en parcourant les pages par ordre
DECROISSANT pour que l'insertion d'une page ne decale pas les index suivants.
Idempotent : les frames traitees portent le marqueur `% [split]`.

Usage :
    ./.venv/Scripts/python.exe src/split_frames.py --dry-run
    ./.venv/Scripts/python.exe src/split_frames.py
"""
import os, re, sys, glob
import check_boxes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PARTS = os.path.join(ROOT, "slides", "parts")
CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}
MARK = "% [split]"
LARGEUR = ".86"
SEUIL_GRAVE = 8.0

FIGBOX = re.compile(
    r"\\begin\{center\}\s*\n(?:\\begin\{figbox\}\s*\n)?"
    r"(\\includegraphics\[[^\]]*\]\{[^}]+\})\s*\n"
    r"(?:\\end\{figbox\}\s*\n)?\\end\{center\}", re.M)
FIGSEUL = re.compile(r"^\s*\\includegraphics\[[^\]]*\]\{[^}]+\}\s*$", re.M)


def blocks(src):
    out, pos = [], 0
    while True:
        i = src.find(r"\begin{frame}", pos)
        if i < 0:
            return out
        j = src.find(r"\end{frame}", i)
        if j < 0:
            return out
        j += len(r"\end{frame}")
        out.append((i, j, src[i:j]))
        pos = j


def index_ch(ch):
    """Page PDF -> (fichier, debut, fin) de la frame (page 1 = page de garde)."""
    res = [(None, None, None)]
    for f in sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex"))):
        src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        res += [(f, a, b) for a, b, _ in blocks(src)]
    return res


def separer(bloc):
    m = FIGBOX.search(bloc)
    if m:
        image, retrait = m.group(1), bloc[m.start():m.end()]
    else:
        ms = list(FIGSEUL.finditer(bloc))
        if not ms:
            return None
        image, retrait = ms[0].group(0).strip(), bloc[ms[0].start():ms[0].end()]
    t = re.match(r"\\begin\{frame\}(\[[^\]]*\])?\{([^\n]*)\}", bloc)
    titre, opts = (t.group(2), t.group(1) or "") if t else ("?", "")
    image = re.sub(r"width=[\d.]+\\textwidth", f"width={LARGEUR}\\textwidth", image)
    tigre = titre.replace("--- la figure", "").strip()
    nouvelle = (f"{MARK}\n\\begin{{frame}}{opts}{{{tigre} --- la figure}}\n"
                f"\\begin{{center}}\n{image}\n\\end{{center}}\n\\end{{frame}}")
    reste = MARK + "\n" + re.sub(r"\n{3,}", "\n\n", bloc.replace(retrait, ""))
    return nouvelle, reste


def main():
    dry = "--dry-run" in sys.argv
    faits, manuels = 0, []
    for ch in CH:
        pdf = os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")
        if not os.path.exists(pdf):
            continue
        r = check_boxes.analyse(pdf, 1.0)
        pages = sorted({p for p, dep in r["graves"] if dep > SEUIL_GRAVE}, reverse=True)
        if not pages:
            print(f"=== chapitre {ch} : rien a separer ===")
            continue
        idx = index_ch(ch)
        par_fichier = {}
        print(f"\n=== chapitre {ch} : {len(pages)} page(s) a separer ===")
        for pn in pages:
            if pn - 1 >= len(idx):
                continue
            f, a, b = idx[pn - 1]
            if f is None:
                continue
            src = par_fichier.get(f)
            if src is None:
                src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            bloc = src[a:b]
            if MARK in bloc:
                continue
            res = separer(bloc)
            if res is None:
                manuels.append((os.path.basename(f), ch, pn, bloc))
                continue
            fig, reste = res
            par_fichier[f] = src[:a] + fig + "\n\n" + reste + src[b:]
            faits += 1
            titre = re.search(r"\{([^\n]*)\}", bloc).group(1)[:44]
            print(f"  p{pn:<3} separee  « {titre} »")
        if not dry:
            for f, contenu in par_fichier.items():
                open(f, "w", encoding="utf-8", newline="").write(contenu.replace("\n", "\r\n"))
    print(f"\n{faits} frame(s) separee(s)" + (" (simulation, rien ecrit)" if dry else ""))
    if manuels:
        print(f"\n{len(manuels)} frame(s) SANS figure en tete (autre remede requis) :")
        for f, ch, pn, _ in manuels:
            print(f"   ch{ch} p{pn:<3} {f}")


if __name__ == "__main__":
    main()
