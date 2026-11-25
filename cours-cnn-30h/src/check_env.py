"""Controle de coherence des sources LaTeX du cours CNN.

Verifie : environnements equilibres, frames contenant lstlisting declarees
[fragile], figures referencees presentes sur le disque, ordre d'imbrication
(controle de pile).
"""
import os, re, sys, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")
FIGS = os.path.join(ROOT, "figures")
ASSETS = os.path.join(ROOT, "assets")

PUITS = ["frame", "columns", "column", "block", "center", "itemize", "enumerate",
         "table", "tabular", "figure", "tikzpicture", "lstlisting", "qbox", "abox",
         "ebox", "kbox", "wbox", "solbox", "codebox", "figbox", "tcolorbox",
         "scope", "axis", "minipage", "description"]


def analyse(txt, nom):
    pb = []
    # --- equilibre + pile
    pile = []
    for m in re.finditer(r"\\(begin|end)\{([A-Za-z*]+)\}", txt):
        sens, env = m.group(1), m.group(2)
        ligne = txt[:m.start()].count("\n") + 1
        if env not in PUITS and env not in ("document",):
            continue
        if sens == "begin":
            pile.append((env, ligne))
        else:
            if not pile:
                pb.append(f"  {nom} : \\end{{{env}}} orphelin (l.{ligne})")
            elif pile[-1][0] != env:
                pb.append(f"  {nom} : \\end{{{env}}} (l.{ligne}) ne correspond pas a "
                          f"\\begin{{{pile[-1][0]}}} (l.{pile[-1][1]})")
                while pile and pile[-1][0] != env:
                    pile.pop()
                if pile:
                    pile.pop()
            else:
                pile.pop()
    for env, ligne in pile:
        pb.append(f"  {nom} : \\begin{{{env}}} (l.{ligne}) jamais ferme")
    # --- frames avec lstlisting sans [fragile]
    for fm in re.finditer(r"\\begin\{frame\}(\[[^\]]*\])?\{", txt):
        debut = fm.start()
        ff = txt.find(r"\end{frame}", debut)
        corps = txt[debut:ff if ff > 0 else len(txt)]
        ligne = txt[:debut].count("\n") + 1
        opts = fm.group(1) or ""
        if "lstlisting" in corps and "fragile" not in opts:
            pb.append(f"  {nom} : frame l.{ligne} contient lstlisting sans [fragile]")
    # --- figures manquantes
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", txt):
        f = m.group(1).strip()
        if not (os.path.exists(os.path.join(FIGS, f)) or os.path.exists(os.path.join(ASSETS, f))
                or os.path.exists(os.path.join(ROOT, f))):
            pb.append(f"  {nom} : figure manquante {f}")
    # --- tabulations parasites (piege du re.sub)
    for i, ligne_txt in enumerate(txt.splitlines(), 1):
        if "\t" in ligne_txt:
            pb.append(f"  {nom} : tabulation parasite l.{i}")
    return pb


def main():
    fichiers = sorted(glob.glob(os.path.join(PARTS, "*.tex")))
    if not fichiers:
        print("aucun fichier parts/*.tex")
        return
    tous = []
    nframes = {}
    for f in fichiers:
        txt = open(f, encoding="utf-8").read()
        tous += analyse(txt, os.path.basename(f))
        n = len(re.findall(r"\\begin\{frame\}", txt))
        ch = os.path.basename(f).split("_")[0]
        nframes[ch] = nframes.get(ch, 0) + n
    print("Frames par chapitre :", nframes)
    if tous:
        print(f"PROBLEMES A CORRIGER ({len(tous)}) :")
        for p in tous:
            print(p)
    else:
        print("RESULTAT : TOUT EST COHERENT")


if __name__ == "__main__":
    main()
