"""Table complete page PDF -> frame LaTeX (fichier, ligne, titre).

Complement de locate_page.py : celui-ci ne liste que les pages en defaut, alors
qu'ici on veut l'index entier du chapitre, pour retrouver n'importe quelle page
cit dans un rapport de mesure.

Usage :  ./.venv/Scripts/python.exe src/page_index.py 4 10 24 37
         ./.venv/Scripts/python.exe src/page_index.py 5 15 16 20 23 42 50
"""
import os, re, sys, glob, pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")
OUT = os.path.join(ROOT, "out")
CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}


def index(ch):
    """[(fichier, ligne 1-based du \\begin{frame}, titre), ...] dans l'ordre de compilation."""
    out = []
    for f in sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex"))):
        src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        for m in re.finditer(r"\\begin\{frame\}(\[[^\]]*\])?\{([^\n]*)\}", src):
            out.append((os.path.basename(f),
                        src[:m.start()].count("\n") + 1,
                        m.group(2)[:58]))
    return out


def main():
    ch = sys.argv[1]
    pages = [int(x) for x in sys.argv[2:]]
    idx = index(ch)
    pdf_pages = pymupdf.open(os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")).page_count
    print(f"chapitre {ch} : {len(idx)} frames, {pdf_pages} pages PDF "
          f"(decalage = {pdf_pages - len(idx)})")
    for pn in pages:
        # page 1 du PDF = page de garde (macro \chapterslide, pas un \begin{frame})
        k = pn - 2
        if k >= len(idx):
            print(f"  page {pn} : hors index")
            continue
        f, ligne, titre = idx[k]
        src = open(os.path.join(PARTS, f), encoding="utf-8", newline="").read().replace("\r\n", "\n")
        marque = " [rogna]" if "% [rogna]" in src.split("\n")[max(ligne - 2, 0):ligne + 3].__str__() else ""
        marque = " [rogna]" if "[rogna]" in "\n".join(src.split("\n")[max(ligne - 3, 0):ligne + 2]) else ""
        print(f"  page {pn:<3} -> {f}:{ligne:<5}{marque} « {titre} »")


if __name__ == "__main__":
    main()
