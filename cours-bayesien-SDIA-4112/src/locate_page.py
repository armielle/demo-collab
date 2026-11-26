"""Localise les debordements : page PDF -> fichier source + frame + titre.

Usage :
    python src/locate_page.py                 # tous les chapitres, seuil 10 pt
    python src/locate_page.py --seuil=6
    python src/locate_page.py 2               # seulement le chapitre 2
"""
import os, re, sys, glob, subprocess
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PARTS = os.path.join(ROOT, "slides", "parts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_overflow import analyse      # reutilise la meme mesure

CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}


def frame_index(ch):
    """Liste ordonnee des (fichier, numero de ligne) de chaque \\begin{frame}."""
    ix = []
    for f in sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex"))):
        src = open(f, encoding="utf-8").read()
        for m in re.finditer(r"\\begin\{frame\}", src):
            ix.append((os.path.basename(f), src[:m.start()].count("\n") + 1))
        # la page de garde \chapterslide est dans le premier fichier
        if "chapterslide" in src and ix and ix[0][0] == os.path.basename(f):
            ix.insert(0, (os.path.basename(f), src[:src.find("\\chapterslide")
                                                ].count("\n") + 1))
    return ix


def titre_de_page(doc, pn):
    page = doc[pn - 1]
    H = page.rect.height
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            y = l["bbox"][1]
            if y < H * 0.16:
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t:
                    return t
    return "(sans titre)"


def main():
    seuil = 10.0
    chs = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            seuil = float(a.split("=")[1])
    chs = chs or list(CH)
    total = 0
    for ch in chs:
        pdf = os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")
        if not os.path.exists(pdf):
            print(f"[ignore] {pdf} absent")
            continue
        ok, _ = analyse(pdf, seuil)
        if ok:
            print(f"chapitre {ch} : propre au seuil {seuil} pt")
            continue
        doc = pymupdf.open(pdf)
        # on refait le detail pour recuperer page + sens + mesure
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            analyse(pdf, seuil)
        detail = buf.getvalue()
        ix = frame_index(ch)
        pages = {}
        for line in detail.splitlines():
            m = re.match(r"\s+p\s*(\d+) \[(\w+)\s*\]\s*\+\s*([\d.]+) pt \| (.*)", line)
            if m:
                pn, sens, pt, txt = int(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
                pages.setdefault(pn, []).append((sens, pt, txt))
        print(f"\n=== CHAPITRE {ch} — {len(pages)} page(s) a corriger (seuil {seuil} pt) ===")
        for pn in sorted(pages):
            f, ln = ix[pn - 1] if pn - 1 < len(ix) else ("?", 0)
            print(f"  p{pn:>3}  {f}:{ln:<5} « {titre_de_page(doc, pn)[:62]} »")
            for sens, pt, txt in sorted(pages[pn], key=lambda t: -t[1])[:4]:
                print(f"        [{sens}] +{pt:5.1f} pt : {txt[:66]}")
            total += len(pages[pn])
        doc.close()
    print(f"\nTotal de lignes fautives : {total}")


if __name__ == "__main__":
    main()
