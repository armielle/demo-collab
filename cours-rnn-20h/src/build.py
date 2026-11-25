"""Compilation des chapitres du cours CNN (Tectonic / XeLaTeX).

Usage : python src/build.py [numero_chapitre|all]
"""
import os, subprocess, sys, time, shutil, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SLIDES = os.path.join(ROOT, "slides")
PARTS = os.path.join(SLIDES, "parts")
OUT = os.path.join(ROOT, "out")
JOUR = os.path.join(OUT, "journal")
TECTONIC = os.path.expanduser("~/texbin/tectonic.exe")

CHAPTERS = {
    "1": ("ch1", "Chapitre-1-donnees-sequencielles"),
    "2": ("ch2", "Chapitre-2-architecture-RNN"),
    "3": ("ch3", "Chapitre-3-LSTM-GRU"),
    "4": ("ch4", "Chapitre-4-entrainement-et-applications"),
}
TITLES = {
    "1": "Donnees sequentielles",
    "2": "Architecture RNN",
    "3": "LSTM et GRU",
    "4": "Entrainement et applications",
}


def parts_du_chapitre(code):
    """Fichiers parts/chN_*.tex, dans l'ordre alphabetique."""
    return sorted(glob.glob(os.path.join(PARTS, f"{code}_*.tex")))


def ecrit_wrapper(nom, code, parts, maitre=False):
    lignes = ["% Fichier genere par build.py -- ne pas editer a la main",
              r"\input{preamble}"]
    if maitre:
        lignes += [r"\title{Reseaux de neurones convolutifs (CNN)}",
                   r"\subtitle{Cours 30 h --- Master 1 SDIA}",
                   r"\author{Dr. Noulapeu Ngaffo Armielle}",
                   r"\institute{Ecole Nationale Superieure Polytechnique de Douala\\"
                   r"Universite de Douala --- Departement Genie Informatique et Telecommunications}",
                   r"\date{Annee universitaire en cours}",
                   r"\setbeamertemplate{footline}{%",
                   r"  \vskip 10mm\leavevmode\hbox{%",
                   r"    \begin{beamercolorbox}[wd=.42\paperwidth,ht=1.9ex,dp=0.8ex,leftskip=.3cm]{author in head/foot}%",
                   r"      \color{white}\scriptsize Par Dr. Noulapeu Ngaffo Armielle",
                   r"    \end{beamercolorbox}%",
                   r"    \begin{beamercolorbox}[wd=.42\paperwidth,ht=1.9ex,dp=0.8ex,center]{title in head/foot}%",
                   r"      \color{white}\scriptsize\insertsectionhead",
                   r"    \end{beamercolorbox}%",
                   r"    \begin{beamercolorbox}[wd=.16\paperwidth,ht=1.9ex,dp=0.8ex,rightskip=.3cm]{date in head/foot}%",
                   r"      \color{white}\scriptsize\insertframenumber\,/\,\inserttotalframenumber",
                   r"    \end{beamercolorbox}}\vskip0pt}",
                   r"\begin{document}", r"\titlepage"]
    else:
        lignes += [r"\begin{document}"]
    for p in parts:
        lignes.append(r"\input{parts/" + os.path.basename(p)[:-4] + "}")
    lignes.append(r"\end{document}")
    chemin = os.path.join(SLIDES, nom)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    return chemin


def compile(tex, cible, journal):
    if os.path.exists(cible):
        os.remove(cible)
    # Tectonic nomme la sortie d'apres le fichier source : on nettoie puis on renomme.
    brut = os.path.join(OUT, os.path.basename(tex)[:-4] + ".pdf")
    if os.path.exists(brut):
        os.remove(brut)
    t0 = time.time()
    r = subprocess.run([TECTONIC, "-X", "compile", os.path.basename(tex),
                        "--outdir", OUT, "--keep-logs"],
                       cwd=SLIDES, capture_output=True, text=True, errors="replace")
    log = r.stdout + r.stderr
    with open(journal, "w", encoding="utf-8") as f:
        f.write(log)
    erreurs = [l.strip() for l in log.splitlines()
               if l.strip().startswith("error:") or l.strip().startswith("!")]
    duree = time.time() - t0
    if os.path.exists(brut):
        shutil.move(brut, cible)
        ko = os.path.getsize(cible) / 1024
        return True, duree, ko, erreurs, log
    return False, duree, 0.0, erreurs, log


def pages(pdf):
    try:
        import pymupdf
        return pymupdf.open(pdf).page_count
    except Exception:
        return -1


def main():
    os.makedirs(JOUR, exist_ok=True)
    cible = sys.argv[1] if len(sys.argv) > 1 else "all"
    codes = list(CHAPTERS) if cible == "all" else [cible]
    total = 0
    for c in codes:
        code, nom_pdf = CHAPTERS[c]
        parts = parts_du_chapitre(code)
        if not parts:
            print(f"[----] chapitre{c} : aucun fichier parts/{code}_*.tex")
            continue
        tex = ecrit_wrapper(f"_build_ch{c}.tex", code, parts)
        cible_pdf = os.path.join(OUT, nom_pdf + ".pdf")
        ok, duree, ko, err, log = compile(tex, cible_pdf,
                                          os.path.join(JOUR, f"compile_ch{c}.log"))
        np_pages = pages(cible_pdf) if ok else 0
        total += np_pages
        etat = "[OK ]" if ok else "[ECHEC]"
        print(f"{etat} chapitre{c:<2} {duree:6.1f}s {ko:9.1f} Ko {len(err):2d} erreur(s)"
              f" {np_pages:3d} p  ->  {nom_pdf}.pdf")
        for e in err[:6]:
            print("       " + e[:150])
    # document maitre
    if cible == "all":
        tous = []
        for c in CHAPTERS:
            tous += parts_du_chapitre(CHAPTERS[c][0])
        tex = ecrit_wrapper("_build_master.tex", "master", tous, maitre=True)
        cible_pdf = os.path.join(OUT, "Cours-RNN-complet.pdf")
        ok, duree, ko, err, log = compile(tex, cible_pdf,
                                          os.path.join(JOUR, "compile_master.log"))
        np_pages = pages(cible_pdf) if ok else 0
        etat = "[OK ]" if ok else "[ECHEC]"
        print(f"{etat} maitre     {duree:6.1f}s {ko:9.1f} Ko {len(err):2d} erreur(s)"
              f" {np_pages:3d} p  ->  Cours-RNN-complet.pdf")
        for e in err[:6]:
            print("       " + e[:150])
    print(f"\nTotal chapitres : {total} pages")


if __name__ == "__main__":
    main()
