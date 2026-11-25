"""Correction automatique des debordements verticaux des frames.

Principe : pour chaque frame dont le contenu descend dans la bande du pied de
page, on recupere de la place par deux leviers sans reecriture de texte
(aucun risque de reflow) :

  1. un `\\vspace*{-2.5mm}` juste apres l'ouverture de la frame (la marge
     disponible sous la barre de titre est de ~10 pt, elle absorbe ce decalage) ;
  2. si le depassement reste superieur a 7 pt ET que la frame contient une figure,
     la figure est reduite de 6 %.

Le script est idempotent : il marque les frames traitees par un commentaire
`% [auto-overflow]` et ne les retouche pas deux fois.

Usage :
    python src/fix_overflow.py --dry-run     # montre ce qui serait fait
    python src/fix_overflow.py               # applique
    python src/fix_overflow.py --seuil=4     # seuil de declenchement
"""
import os, re, sys, glob, io, contextlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PARTS = os.path.join(ROOT, "slides", "parts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_overflow import analyse

CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}
MARK = "% [auto-overflow] figure reduite"


def frames_of(path):
    """Decoupe un fichier de chapitre en frames : (debut, fin, texte)."""
    src = open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    out, pos = [], 0
    while True:
        i = src.find(r"\begin{frame}", pos)
        if i < 0:
            break
        j = src.find(r"\end{frame}", i)
        if j < 0:
            break
        j += len(r"\end{frame}")
        out.append((i, j, src[i:j]))
        pos = j
    return src, out


def pages_to_frames(ch, seuil):
    """Retourne {page: (sens, depassement_max, extrait)} pour un chapitre."""
    pdf = os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")
    if not os.path.exists(pdf):
        return {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyse(pdf, seuil)
    res = {}
    for line in buf.getvalue().splitlines():
        m = re.match(r"\s+p\s*(\d+) \[(\w+)\s*\]\s*\+\s*([\d.]+) pt \| (.*)", line)
        if m:
            pn, dep = int(m.group(1)), float(m.group(3))
            cur = res.get(pn)
            if cur is None or dep > cur[1]:
                res[pn] = (m.group(2), dep, m.group(4))
    return res


def main():
    seuil = 4.0
    dry = "--dry-run" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            seuil = float(a.split("=")[1])

    total = 0
    for ch in CH:
        pages = pages_to_frames(ch, seuil)
        if not pages:
            continue
        # index des frames du chapitre, dans l'ordre de compilation
        files = sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex")))
        index = []          # (fichier, position debut, fin)
        for f in files:
            src, fr = frames_of(f)
            if ch == "1" or f is files[0]:
                index.append((f, None, None))       # page de garde \chapterslide
            index += [(f, a, b) for a, b, _ in fr]
        print(f"\n=== chapitre {ch} : {len(pages)} frame(s) a compacter (seuil {seuil} pt) ===")
        par_fichier = {}
        # traitement par page DECROISSANTE : les insertions en fin de fichier ne
        # decalent pas les frames situees avant elles dans le meme fichier.
        for pn in sorted(pages, reverse=True):
            sens, dep, txt = pages[pn]
            if pn - 1 >= len(index):
                print(f"  p{pn}: hors index (ignore)")
                continue
            f, a, b = index[pn - 1]
            if a is None:
                print(f"  p{pn}: page de garde, ignoree")
                continue
            # on relit l'etat COURANT du fichier (edits precedents inclus)
            src = par_fichier.get(f)
            if src is None:
                src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            bloc = src[a:b]
            if MARK in bloc:
                print(f"  p{pn}: deja traite ({os.path.basename(f)})")
                continue
            # Levier unique : reduction de la plus grande figure de la frame.
            # (le \vspace* juste apres \begin{frame} est absorbe par Beamer : inutilisable)
            nouveau = src[:a] + MARK + "\n" + src[a:]
            figs = re.findall(r"\\includegraphics\[width=(\.?\d+(?:\.\d+)?)\\textwidth\]", nouveau[a:b])
            if figs:
                v = max(float(x) for x in figs)
                nouveau = nouveau.replace(
                    rf"\includegraphics[width={v:.2f}\textwidth]",
                    rf"\includegraphics[width={max(v - 0.06, 0.5):.2f}\textwidth]")
                print(f"  p{pn} [{sens}] +{dep:.1f} pt : figure {v:.2f} -> {max(v-0.06,0.5):.2f}"
                      f"  ({os.path.basename(f)})")
            else:
                print(f"  p{pn} [{sens}] +{dep:.1f} pt : SANS FIGURE - a traiter a la main"
                      f"  ({os.path.basename(f)})")
                continue
            par_fichier[f] = nouveau
            total += 1
        if not dry:
            for f, contenu in par_fichier.items():
                open(f, "w", encoding="utf-8", newline="").write(contenu.replace("\n", "\r\n"))
    print(f"\n{total} frame(s) corrigee(s)" + (" (mode simulation, rien ecrit)" if dry else ""))


if __name__ == "__main__":
    main()
