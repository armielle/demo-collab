"""Corrige les lignes masquees par la bande du pied de page.

Pour chaque frame dont une ligne descend sous la bande du pied, on reduit la
hauteur du contenu sans jamais toucher au texte :

  - si la frame contient une figure : la plus grande figure est reduite ;
  - sinon : la frame passe en `\\small`, puis `\\footnotesize` si cela ne suffit pas.

Trois modes :
    --dry-run    montre ce qui serait fait, sans ecrire ;
    (defaut)     traite les frames pas encore marquees ;
    --escalade   reprend les frames deja marquees et durcit le traitement.

Boucle : appliquer -> recompiler -> remesurer (`check_rogna.py`) -> escalader
jusqu'a ce que le compteur « sous la bande du pied » tombe a zero.

    ./.venv/Scripts/python.exe src/fix_rogna.py
    ./.venv/Scripts/python.exe src/build.py
    ./.venv/Scripts/python.exe src/check_rogna.py
"""
import os, re, sys, glob
import check_rogna

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PARTS = os.path.join(ROOT, "slides", "parts")
CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}
MARK = "% [rogna]"
FIG = r"\\includegraphics\[width=(\.?\d+(?:\.\d+)?)\\textwidth\]"


def frames_of(path):
    src = open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    out, pos = [], 0
    while True:
        i = src.find(r"\begin{frame}", pos)
        if i < 0:
            break
        j = src.find(r"\end{frame}", i)
        if j < 0:
            break
        out.append((i, j + len(r"\end{frame}")))
        pos = j
    return src, out


def pages_rognees(ch, seuil):
    pdf = os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")
    if not os.path.exists(pdf):
        return {}
    r = check_rogna.analyse(pdf, seuil)
    res = {}
    for pn, dep, txt in r["sous"]:
        if pn not in res or dep > res[pn][0]:
            res[pn] = (dep, txt)
    return res


def traitements(ch, seuil, escalade):
    """Retourne {fichier: nouveau_contenu} apres application des corrections."""
    pages = pages_rognees(ch, seuil)
    if not pages:
        return {}, 0, []
    fichiers = sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex")))
    index = []
    for f in fichiers:
        src, fr = frames_of(f)
        if f is fichiers[0]:
            index.append((f, None, None))
        index += [(f, a, b) for a, b in fr]

    par_fichier, faits, lignes = {}, 0, []
    for pn in sorted(pages, reverse=True):
        dep, txt = pages[pn]
        if pn - 1 >= len(index):
            continue
        f, a, b = index[pn - 1]
        if a is None:
            continue
        src = par_fichier.get(f)
        if src is None:
            src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        bloc = src[a:b]
        marque = MARK in bloc
        if marque and not escalade:
            lignes.append(f"  p{pn:<3} +{dep:4.1f} pt : deja traite (utiliser --escalade) | {txt[:30]}")
            continue

        figs = re.findall(FIG, bloc)
        if r"\small" in bloc and marque:
            # passe precedente : on durcit d'un cran
            nouveau = src[:a] + bloc.replace(r"\small", r"\footnotesize", 1) + src[b:]
            action = "\\small -> \\footnotesize"
        elif figs:
            m = max(float(x) for x in figs)
            pas = 0.06 if marque else 0.08
            cible = max(m - pas, 0.5)
            nouveau = src[:a] + bloc.replace(
                rf"\includegraphics[width={m:.2f}\textwidth]",
                rf"\includegraphics[width={cible:.2f}\textwidth]", 1) + src[b:]
            if not marque:                      # marqueur pose AVANT la frame
                nouveau = src[:a] + MARK + "\n" + nouveau[a:]
            action = f"figure {m:.2f} -> {cible:.2f}"
        elif marque and r"\footnotesize" in bloc:
            # deja au maximum de l'escalade typographique : plus de levier automatique
            lignes.append(f"  p{pn:<3} +{dep:4.1f} pt : A RETOUCHER A LA MAIN       | {txt[:30]}")
            continue
        else:
            fin_ouv = src.find("\n", a)
            nouveau = src[:fin_ouv] + "\n" + MARK + "\n" + r"\small" + src[fin_ouv:]
            action = "passage en \\small"
        par_fichier[f] = nouveau
        faits += 1
        lignes.append(f"  p{pn:<3} +{dep:4.1f} pt : {action:<26} | {txt[:30]}")
    return par_fichier, faits, lignes


def main():
    dry = "--dry-run" in sys.argv
    escalade = "--escalade" in sys.argv
    seuil = 2.0
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            seuil = float(a.split("=")[1])

    total = 0
    for ch in CH:
        par_fichier, faits, lignes = traitements(ch, seuil, escalade)
        print(f"\n=== chapitre {ch} : {faits} frame(s) traitee(s)"
              + (" [escalade]" if escalade else "") + f" (seuil {seuil} pt) ===")
        for l in sorted(lignes, key=lambda s: -float(s.split("+")[1].split(" pt")[0])):
            print(l)
        total += faits
        if not dry:
            for f, contenu in par_fichier.items():
                open(f, "w", encoding="utf-8", newline="").write(contenu.replace("\n", "\r\n"))
    print(f"\n{total} frame(s) corrigee(s)" + (" (simulation, rien ecrit)" if dry else ""))


if __name__ == "__main__":
    main()
