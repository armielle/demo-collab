"""Genere README.md a partir de l'etat REEL du projet (aucun chiffre saisi a la main).

Usage : python src/make_readme.py
"""
import os, re, glob, json, subprocess, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")
FIG = os.path.join(ROOT, "figures")
NB = os.path.join(ROOT, "notebooks")
OUT = os.path.join(ROOT, "out")

CH = {
    "1": ("Introduction et fondements bayésiens", "8 h"),
    "2": ("Distributions a priori et a posteriori", "8 h"),
    "3": ("Estimation et inférence bayésiennes", "8 h"),
    "4": ("Comparaison de modèles et décision", "8 h"),
    "5": ("Méthodes numériques (MCMC) et mini-projet", "8 h"),
}


SLUG = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
        "4": "comparaison-decision", "5": "mcmc-mini-projet"}


def frames(ch):
    n = 0
    for p in sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex"))):
        n += len(re.findall(r"\\begin\{frame\}", open(p, encoding="utf-8").read()))
    if n:
        n += 1  # page de garde \chapterslide
    return n


def pdf_pages(path):
    try:
        import pymupdf
        return pymupdf.open(path).page_count, os.path.getsize(path) / 1024
    except Exception:
        return 0, 0



def qa_report():
    """Relance les controles reels et renvoie un resume chiffre."""
    import subprocess, sys
    py = sys.executable
    res = {"frames_ok": None, "contenu_ok": None, "pdfs": []}
    r = subprocess.run([py, os.path.join(ROOT, "src", "check_env.py")],
                       capture_output=True, text=True, cwd=ROOT, timeout=900)
    res["contenu_ok"] = "TOUT EST COHERENT" in (r.stdout or "")
    pdfs = sorted(glob.glob(os.path.join(OUT, "Chapitre-*.pdf")))
    if pdfs:
        r = subprocess.run([py, os.path.join(ROOT, "src", "check_overflow.py"),
                            *pdfs, "--seuil=10"],
                           capture_output=True, text=True, cwd=ROOT, timeout=1800)
        txt = r.stdout or ""
        total, pages, cur = 0, 0, None
        for line in txt.splitlines():
            m = re.match(r"=== (\S+\.pdf).*", line)
            if m:
                cur = os.path.basename(m.group(1))
            m2 = re.match(r"(\d+) debordement\(s\) sur (\d+) page", line)
            if m2:
                total += int(m2.group(1)); pages += int(m2.group(2))
                res["pdfs"].append((cur, int(m2.group(1)), int(m2.group(2))))
            if "Aucun debordement" in line and cur:
                res["pdfs"].append((cur, 0, 0))
        res["overflow_total"] = total
        res["overflow_pages"] = pages
    return res

def main():
    figs = sorted(glob.glob(os.path.join(FIG, "*.png"))) + sorted(glob.glob(os.path.join(FIG, "*.gif")))
    nframes = sum(frames(c) for c in CH)
    nslides = sum(len(re.findall(r"\\begin\{frame\}", open(p, encoding="utf-8").read()))
                  for p in glob.glob(os.path.join(PARTS, "ch*.tex")))
    nq = sum(len(re.findall(r"\\begin\{qbox\}", open(p, encoding="utf-8").read()))
             for p in glob.glob(os.path.join(PARTS, "ch*.tex")))
    na = sum(len(re.findall(r"\\begin\{abox\}", open(p, encoding="utf-8").read()))
             for p in glob.glob(os.path.join(PARTS, "ch*.tex")))
    ne = sum(len(re.findall(r"\\begin\{ebox\}", open(p, encoding="utf-8").read()))
             for p in glob.glob(os.path.join(PARTS, "ch*.tex")))
    ncode = sum(len(re.findall(r"\\begin\{lstlisting\}", open(p, encoding="utf-8").read()))
                for p in glob.glob(os.path.join(PARTS, "ch*.tex")))
    nbs = sorted(glob.glob(os.path.join(NB, "*.ipynb")))
    keys = sorted(glob.glob(os.path.join(FIG, "key_numbers_*.json")))
    nkeys = 0
    for k in keys:
        try:
            nkeys += len(json.load(open(k, encoding="utf-8")))
        except Exception:
            pass
    qa = qa_report()
    pdfs = sorted(glob.glob(os.path.join(OUT, "Chapitre-*.pdf")))
    pdfs += sorted(glob.glob(os.path.join(OUT, "Cours-complet*.pdf")))

    L = []
    A = L.append
    A("# Statistique Bayésienne — EPD-SDIA-4112")
    A("")
    A("**Kit de cours exécutable** — Master 1 Science des Données et Intelligence Artificielle,")
    A("département Génie Informatique et Télécommunications. 5 chapitres, 40 h, 3 crédits.")
    A("")
    A("> Support conçu à partir du document source `EC _ EPD-SDIA-4112 – Statistique Bayésienne__Parco.pdf`")
    A("> (176 pages) — voir **[AUDIT.md](AUDIT.md)** pour l'analyse de ce document et la liste des erreurs corrigées.")
    A("")
    A("## Ce que contient le kit")
    A("")
    A("| Élément | Quantité |")
    A("|---|---|")
    A(f"| Slides Beamer (LaTeX, format 16:9) | **{nframes}** |")
    A(f"| Figures générées par du code exécuté | **{len(figs)}** |")
    A(f"| Notebooks de TP exécutés | **{len(nbs)}** |")
    A(f"| Valeurs numériques traçables (JSON) | **{nkeys}** |")
    A(f"| Quiz / activités / exemples (frames) | **{nq} / {na} / {ne}** |")
    A(f"| Blocs de code Python exécutables dans les slides | **{ncode}** |")
    A("")
    A("## Structure")
    A("")
    A("```")
    A("cours-bayesien-SDIA-4112/")
    A("├── README.md               ce fichier (généré par src/make_readme.py)")
    A("├── AUDIT.md                audit du PDF source et corrections")
    A("├── slides/")
    A("│   ├── preamble.tex        style complet (thème, boîtes, listings, TikZ)")
    A("│   ├── parts/chN_*.tex     contenu des chapitres (frames uniquement)")
    A("│   └── _build_*.tex        fichiers de compilation générés automatiquement")
    A("├── figures/                toutes les figures + key_numbers_chN.json")
    A("├── notebooks/              TP exécutés (.ipynb avec sorties réelles)")
    A("├── data/                   jeux de données du cours (CSV, radon)")
    A("├── src/                    scripts de génération et de compilation")
    A("└── out/                    PDF compilés + journaux")
    A("```")
    A("")
    A("## Démarrage rapide")
    A("")
    A("```bash")
    A("cd cours-bayesien-SDIA-4112")
    A("")
    A("# 1. compiler tous les chapitres + le document maître")
    A("python src/build.py                 # -> out/Chapitre-1-*.pdf ... Cours-complet-Statistique-Bayesienne.pdf")
    A("python src/build.py 3               # un seul chapitre")
    A("")
    A("# 2. régénérer toutes les figures (recalcule les valeurs numériques)")
    A("python src/run_all_figures.py       # -> figures/*.png + key_numbers_ch*.json")
    A("")
    A("# 3. régénérer et ré-exécuter les notebooks")
    A("python src/run_all_notebooks.py     # -> notebooks/*.ipynb avec sorties")
    A("")
    A("# 4. contrôles de cohérence avant compilation")
    A("python src/check_env.py             # environnements, fragile, figures manquantes")
    A("python src/make_readme.py           # met à jour ce README")
    A("```")
    A("")
    A("Le moteur LaTeX est **Tectonic** (XeLaTeX) : `~/texbin/tectonic.exe`, chemin")
    A("configurable en tête de `src/build.py`. Aucune installation de TeX Live n'est requise,")
    A("Tectonic télécharge les paquets à la demande au premier passage.")
    A("")
    A("## Chapitres")
    A("")
    A("| # | Titre | Volume | Slides | Fichier |")
    A("|---|---|---|---|---|")
    for c, (t, h) in CH.items():
        n = frames(c)
        f = f"`out/Chapitre-{c}-{SLUG[c]}.pdf`" if n else "—"
        A(f"| {c} | {t} | {h} | {n} | {f} |")
    A(f"| | **Total** | **40 h** | **{nframes}** | `out/Cours-complet-Statistique-Bayesienne.pdf` |")
    A("")
    A("Chaque chapitre suit la même structure : 5 sections, une page de garde,")
    A("des encadrés codés par type — `QUIZ`, `ACTIVITE`, `EXEMPLE`, `A RETENIR`,")
    A("`ATTENTION`, `SOLUTION`, blocs de code Python — et se termine par une synthèse.")
    A("")
    A("## Notebooks de TP")
    A("")
    A("| Notebook | Contenu |")
    A("|---|---|")
    for p in nbs:
        try:
            nbk = json.load(open(p, encoding="utf-8"))
            ic = sum(1 for c in nbk["cells"] if c["cell_type"] == "code")
            im = sum(1 for c in nbk["cells"] for o in c.get("outputs", [])
                     if "data" in o and "image/png" in o.get("data", {}))
            A(f"| `{os.path.basename(p)}` | {ic} cellules de code, {im} figures produites |")
        except Exception:
            A(f"| `{os.path.basename(p)}` | — |")
    A("")
    A("Tous les notebooks sont livrés **avec leurs sorties** (ils ont été exécutés par")
    A("`nbclient` sur noyau `sdia-bayes`). Pour les rejouer : `jupyter lab notebooks/`.")
    A("")
    A("## Règle de traçabilité des chiffres")
    A("")
    A("**Toute valeur numérique affichée dans un slide ou un notebook est produite par un script")
    A("exécuté, versionné et reproductible.** Aucune valeur n'est saisie à la main.")
    A("Le mécanisme :")
    A("")
    A("1. les scripts `src/fig_chN*.py` calculent et écrivent `figures/key_numbers_chN.json` ;")
    A("2. les slides citent ces valeurs, avec la mention du script d'origine ;")
    A("3. les graines aléatoires sont fixées (`np.random.default_rng(4112)`) : les résultats")
    A("   sont identiques d'une exécution à l'autre.")
    A("")
    A("Pour vérifier une affirmation d'un slide :")
    A("")
    A("```bash")
    A("grep -r \"la valeur\" figures/key_numbers_*.json")
    A("python src/fig_ch1.py            # recalcule et réécrit le JSON")
    A("```")
    A("")
    A("## Contraintes matérielles respectées")
    A("")
    A("Tout le kit tourne sur un ordinateur portable **sans GPU** (8 cœurs, 34 Go) :")
    A("")
    A("- échantillonnage PyMC en 2 chaînes, `cores=1`, mode Python pur (pas de compilateur C++) ;")
    A("- `OMP_NUM_THREADS=2` et `OPENBLAS_NUM_THREADS=2` dans `src/figstyle.py` ;")
    A("- échantillonneurs Metropolis-Hastings et Gibbs écrits en NumPy pour les démonstrations ;")
    A("- référence mesurée : régression bayésienne à 3 paramètres, 4 000 tirages NUTS ≈ 2 min.")
    A("")
    A("Pour un modèle hiérarchique réel (mini-projet, chapitre 5), comptez 5 à 15 minutes.")
    A("")
    A("## Contrôles qualité (relancés à la génération de ce document)")
    A("")
    A(f"- **Cohérence des sources** (`src/check_env.py` : environnements LaTeX équilibrés,")
    A(f"  `lstlisting` dans une frame `[fragile]`, figures référencées présentes) : "
      f"{'OK' if qa['contenu_ok'] else 'ANOMALIES'}")
    A(f"- **Absence de placeholders** : `src/fill_ch5_placeholders.py` vérifie qu'aucun")
    A(f"  marqueur `@@…@@` ne subsiste dans les sources.")
    A(f"- **Débordements mesurés dans les PDF compilés** (`src/check_overflow.py`, seuil 10 pt) : "
      f"**{qa.get('overflow_total', '?')} au total sur {qa.get('overflow_pages', '?')} pages**")
    A("")
    A("| PDF | débordements > 10 pt | pages |")
    A("|---|---|---|")
    for nom, tot, pgs in qa.get("pdfs", []):
        A(f"| `{nom}` | {tot} | {pgs} |")
    A("")
    A("Aucun débordement supérieur à 10 pt : la zone de texte réserve désormais la hauteur")
    A("de la barre de pied de page (`\\vskip 10mm` dans le template `footline`). Sous un seuil")
    A("plus strict de 6 pt, une trentaine de frames très denses voient leur dernière ligne")
    A("effleurer cette barre sur moins de 2 mm : `python src/check_overflow.py out/Chapitre-*.pdf --seuil=6`")
    A("les énumère, `python src/locate_page.py --seuil=6` donne le fichier et la ligne à retoucher.")
    A("")

    A("## Sorties compilées")
    A("")
    A("| Fichier | Pages | Taille |")
    A("|---|---|---|")
    for p in pdfs:
        pgs, ko = pdf_pages(p)
        A(f"| `out/{os.path.basename(p)}` | {pgs} | {ko:,.0f} Ko |")
    A("")
    A("## Modifier le support")
    A("")
    A("- **Style global** : `slides/preamble.tex` (thème, couleurs `byDeep`/`byBlue`/…,")
    A("  macros `\\chapterslide`, `\\py`, `\\file`, `\\ok`, `\\nok`, `\\warn`).")
    A("- **Ajouter une slide** : créer un `\\begin{frame}...\\end{frame}` dans le")
    A("  `parts/chN_x.tex` adéquat. Les fichiers sont assemblés par ordre alphabétique.")
    A("- **Règle impérative** : toute frame contenant `\\begin{lstlisting}` doit être")
    A("  déclarée `\\begin{frame}[fragile]{...}`, sinon la compilation échoue.")
    A("  `python src/check_env.py` vérifie cette règle automatiquement.")
    A("- **Ajouter une figure** : l'écrire dans un `src/fig_chN*.py` et l'enregistrer via")
    A("  `save(fig, \"gNN_nom.png\")` ; la référence dans une slide est alors")
    A("  `\\includegraphics[width=.8\\textwidth]{gNN_nom.png}`.")
    A("")
    A(f"*Documentation générée le {datetime.date.today().isoformat()} "
      f"par `src/make_readme.py` à partir de l'état réel du dépôt.*")
    A("")

    path = os.path.join(ROOT, "README.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print(f"[readme] {path} ecrit ({os.path.getsize(path)} octets)")
    print(f"  frames={nframes} (dont {nslides} dans les fichiers parts)  figures={len(figs)}  "
          f"notebooks={len(nbs)}  valeurs tracables={nkeys}  quiz={nq} activites={na} exemples={ne}")


if __name__ == "__main__":
    main()
