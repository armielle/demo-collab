"""Compilation des supports de cours (Tectonic / XeLaTeX).

Usage :
    python src/build.py            # tous les chapitres presents + le document maitre
    python src/build.py 1 3        # seulement les chapitres 1 et 3
    python src/build.py --master   # seulement le document maitre
"""
import os, re, subprocess, sys, glob, time, json, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SLIDES = os.path.join(ROOT, "slides")
PARTS = os.path.join(SLIDES, "parts")
OUT = os.path.join(ROOT, "out")
TECTONIC = os.path.expanduser("~/texbin/tectonic.exe")
os.makedirs(OUT, exist_ok=True)

COURSE = {"title": "Statistique Bayésienne",
          "subtitle": "EPD-SDIA-4112 --- Master 1 Science des Données et Intelligence Artificielle",
          "author": "Enseignante : Dr. Noulapeu Ngaffo Armielle",
          "institute": "Département Génie Informatique et Télécommunications",
          "date": "Année universitaire 2026--2027"}

CHAPTERS = {
    "1": "Introduction et fondements bayésiens",
    "2": "Distributions a priori et a posteriori",
    "3": "Estimation et inférence bayésiennes",
    "4": "Comparaison de modèles et décision",
    "5": "Méthodes numériques (MCMC) et mini-projet",
}

SLUG = {
    "1": "fondements-bayesiens",
    "2": "a-priori-a-posteriori",
    "3": "estimation-inference",
    "4": "comparaison-decision",
    "5": "mcmc-mini-projet",
}

HEADER = r"""%% Fichier genere par src/build.py -- ne pas modifier a la main
\input{preamble.tex}
\title[%(titre_court)s]{%(titre)s}
\subtitle{%(subtitle)s}
\author{%(author)s}
\institute{%(institute)s}
\date{%(date)s}
\begin{document}
"""


def parts_of(ch):
    return sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex")))


def wrapper(name, chapters, titre, titre_court):
    body = HEADER % dict(titre_court=titre_court, titre=titre, **COURSE)
    for ch in chapters:
        for p in parts_of(ch):
            body += "\\input{parts/%s}\n" % os.path.basename(p)[:-4]
    body += "\\end{document}\n"
    path = os.path.join(SLIDES, f"_build_{name}.tex")
    open(path, "w", encoding="utf-8").write(body)
    return path


def compile_tex(texfile, tag):
    journal = os.path.join(OUT, "journal")
    os.makedirs(journal, exist_ok=True)
    log = os.path.join(journal, f"compile_{tag}.log")
    t0 = time.time()
    r = subprocess.run([TECTONIC, "-X", "compile", os.path.basename(texfile),
                        "--outdir", "../out", "--keep-logs"],
                       cwd=SLIDES, capture_output=True, text=True, timeout=3600)
    out = (r.stdout or "") + (r.stderr or "")
    open(log, "w", encoding="utf-8").write(out)
    # le journal TeX produit par --keep-logs : on le range aussi dans journal/
    tlog = os.path.join(OUT, os.path.basename(texfile)[:-4] + ".log")
    if os.path.exists(tlog):
        shutil.move(tlog, os.path.join(journal, os.path.basename(tlog)))
    pdf = os.path.join(OUT, os.path.basename(texfile)[:-4] + ".pdf")
    dt = time.time() - t0
    ok = os.path.exists(pdf)
    errs = [l for l in out.splitlines()
            if l.startswith("!") or "error" in l.lower() and "fontconfig" not in l.lower()]
    # nom final propre
    final = pdf
    if ok:
        m = re.match(r"_build_ch(\d)$", os.path.basename(texfile)[:-4])
        if m:
            final = os.path.join(OUT, f"Chapitre-{m.group(1)}-{SLUG[m.group(1)]}.pdf")
        elif os.path.basename(texfile) == "_build_all.tex":
            final = os.path.join(OUT, "Cours-complet-Statistique-Bayesienne.pdf")
        if final != pdf:
            # deplacement (et non copie) : evite de laisser un _build_chN.pdf doublon
            shutil.move(pdf, final)
    nfr = sum(len(re.findall(r"\\begin\{frame\}", open(p, encoding="utf-8").read()))
              for p in glob.glob(os.path.join(PARTS, "*ch*.tex")))
    print(f"[{'OK ' if ok else 'ECHEC'}] {tag:<16} {dt:6.1f}s  "
          f"{(os.path.getsize(final)/1024 if ok else 0):8.1f} Ko  {len(errs)} erreur(s)"
          f"  ->  {os.path.basename(final) if ok else 'aucun PDF'}")
    for e in errs[:12]:
        print("      ", e[:150])
    return ok, out


def frames_report():
    rep = {}
    for ch in CHAPTERS:
        n = 0
        for p in parts_of(ch):
            n += len(re.findall(r"\\begin\{frame\}", open(p, encoding="utf-8").read()))
        if n:
            rep[ch] = n
    return rep


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_master = "--master" in sys.argv or not args
    todo = args if args else [c for c in CHAPTERS if parts_of(c)]
    for ch in todo:
        if not parts_of(ch):
            print(f"[skip ] chapitre {ch} : aucun fichier parts/ch{ch}_*.tex")
            continue
        w = wrapper(f"ch{ch}", [ch], CHAPTERS[ch],
                    f"Chapitre {ch} --- {CHAPTERS[ch]}")
        compile_tex(w, f"chapitre{ch}")
    if want_master and len([c for c in CHAPTERS if parts_of(c)]) > 1:
        chs = [c for c in CHAPTERS if parts_of(c)]
        w = wrapper("all", chs, "Statistique Bayésienne --- support complet",
                    "Cours complet")
        compile_tex(w, "cours_complet")
    print("\nFrames par chapitre :", json.dumps(frames_report(), ensure_ascii=False))
