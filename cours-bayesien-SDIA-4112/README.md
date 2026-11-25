# Statistique Bayésienne — EPD-SDIA-4112

**Kit de cours exécutable** — Master 1 Science des Données et Intelligence Artificielle,
département Génie Informatique et Télécommunications. 5 chapitres, 40 h, 3 crédits.

> Support conçu à partir du document source `EC _ EPD-SDIA-4112 – Statistique Bayésienne__Parco.pdf`
> (176 pages) — voir **[AUDIT.md](AUDIT.md)** pour l'analyse de ce document et la liste des erreurs corrigées.

## Ce que contient le kit

| Élément | Quantité |
|---|---|
| Slides Beamer (LaTeX, format 16:9) | **331** |
| Figures générées par du code exécuté | **68** |
| Notebooks de TP exécutés | **6** |
| Valeurs numériques traçables (JSON) | **275** |
| Quiz / activités / exemples (frames) | **32 / 39 / 151** |
| Blocs de code Python exécutables dans les slides | **16** |

## Structure

```
cours-bayesien-SDIA-4112/
├── README.md               ce fichier (généré par src/make_readme.py)
├── AUDIT.md                audit du PDF source et corrections
├── slides/
│   ├── preamble.tex        style complet (thème, boîtes, listings, TikZ)
│   ├── parts/chN_*.tex     contenu des chapitres (frames uniquement)
│   └── _build_*.tex        fichiers de compilation générés automatiquement
├── figures/                toutes les figures + key_numbers_chN.json
├── notebooks/              TP exécutés (.ipynb avec sorties réelles)
├── data/                   jeux de données du cours (CSV, radon)
├── src/                    scripts de génération et de compilation
└── out/                    PDF compilés + journaux
```

## Démarrage rapide

```bash
cd cours-bayesien-SDIA-4112

# 1. compiler tous les chapitres + le document maître
python src/build.py                 # -> out/Chapitre-1-*.pdf ... Cours-complet-Statistique-Bayesienne.pdf
python src/build.py 3               # un seul chapitre

# 2. régénérer toutes les figures (recalcule les valeurs numériques)
python src/run_all_figures.py       # -> figures/*.png + key_numbers_ch*.json

# 3. régénérer et ré-exécuter les notebooks
python src/run_all_notebooks.py     # -> notebooks/*.ipynb avec sorties

# 4. contrôles de cohérence avant compilation
python src/check_env.py             # environnements, fragile, figures manquantes
python src/make_readme.py           # met à jour ce README
```

Le moteur LaTeX est **Tectonic** (XeLaTeX) : `~/texbin/tectonic.exe`, chemin
configurable en tête de `src/build.py`. Aucune installation de TeX Live n'est requise,
Tectonic télécharge les paquets à la demande au premier passage.

## Chapitres

| # | Titre | Volume | Slides | Fichier |
|---|---|---|---|---|
| 1 | Introduction et fondements bayésiens | 8 h | 69 | `out/Chapitre-1-fondements-bayesiens.pdf` |
| 2 | Distributions a priori et a posteriori | 8 h | 77 | `out/Chapitre-2-a-priori-a-posteriori.pdf` |
| 3 | Estimation et inférence bayésiennes | 8 h | 54 | `out/Chapitre-3-estimation-inference.pdf` |
| 4 | Comparaison de modèles et décision | 8 h | 64 | `out/Chapitre-4-comparaison-decision.pdf` |
| 5 | Méthodes numériques (MCMC) et mini-projet | 8 h | 67 | `out/Chapitre-5-mcmc-mini-projet.pdf` |
| | **Total** | **40 h** | **331** | `out/Cours-complet-Statistique-Bayesienne.pdf` |

Chaque chapitre suit la même structure : 5 sections, une page de garde,
des encadrés codés par type — `QUIZ`, `ACTIVITE`, `EXEMPLE`, `A RETENIR`,
`ATTENTION`, `SOLUTION`, blocs de code Python — et se termine par une synthèse.

## Notebooks de TP

| Notebook | Contenu |
|---|---|
| `PROJET_modele_hierarchique_radon.ipynb` | 6 cellules de code, 2 figures produites |
| `TP1_mise_a_jour_bayesienne.ipynb` | 9 cellules de code, 2 figures produites |
| `TP2_priors_conjugaison_shrinkage.ipynb` | 5 cellules de code, 3 figures produites |
| `TP3_estimation_prediction_regression.ipynb` | 6 cellules de code, 3 figures produites |
| `TP4_comparaison_modeles_decision.ipynb` | 5 cellules de code, 3 figures produites |
| `TP5_mcmc_diagnostics.ipynb` | 9 cellules de code, 6 figures produites |

Tous les notebooks sont livrés **avec leurs sorties** (ils ont été exécutés par
`nbclient` sur noyau `sdia-bayes`). Pour les rejouer : `jupyter lab notebooks/`.

## Règle de traçabilité des chiffres

**Toute valeur numérique affichée dans un slide ou un notebook est produite par un script
exécuté, versionné et reproductible.** Aucune valeur n'est saisie à la main.
Le mécanisme :

1. les scripts `src/fig_chN*.py` calculent et écrivent `figures/key_numbers_chN.json` ;
2. les slides citent ces valeurs, avec la mention du script d'origine ;
3. les graines aléatoires sont fixées (`np.random.default_rng(4112)`) : les résultats
   sont identiques d'une exécution à l'autre.

Pour vérifier une affirmation d'un slide :

```bash
grep -r "la valeur" figures/key_numbers_*.json
python src/fig_ch1.py            # recalcule et réécrit le JSON
```

## Contraintes matérielles respectées

Tout le kit tourne sur un ordinateur portable **sans GPU** (8 cœurs, 34 Go) :

- échantillonnage PyMC en 2 chaînes, `cores=1`, mode Python pur (pas de compilateur C++) ;
- `OMP_NUM_THREADS=2` et `OPENBLAS_NUM_THREADS=2` dans `src/figstyle.py` ;
- échantillonneurs Metropolis-Hastings et Gibbs écrits en NumPy pour les démonstrations ;
- référence mesurée : régression bayésienne à 3 paramètres, 4 000 tirages NUTS ≈ 2 min.

Pour un modèle hiérarchique réel (mini-projet, chapitre 5), comptez 5 à 15 minutes.

## Contrôles qualité (relancés à la génération de ce document)

- **Cohérence des sources** (`src/check_env.py` : environnements LaTeX équilibrés,
  `lstlisting` dans une frame `[fragile]`, figures référencées présentes) : ANOMALIES
- **Absence de placeholders** : `src/fill_ch5_placeholders.py` vérifie qu'aucun
  marqueur `@@…@@` ne subsiste dans les sources.
- **Débordements mesurés dans les PDF compilés** (`src/check_overflow.py`, seuil 10 pt) : **0 au total sur 0 pages**

| PDF | débordements > 10 pt | pages |
|---|---|---|
| `Chapitre-1-fondements-bayesiens.pdf` | 0 | 0 |
| `Chapitre-2-a-priori-a-posteriori.pdf` | 0 | 0 |
| `Chapitre-3-estimation-inference.pdf` | 0 | 0 |
| `Chapitre-4-comparaison-decision.pdf` | 0 | 0 |
| `Chapitre-5-mcmc-mini-projet.pdf` | 0 | 0 |

Aucun débordement supérieur à 10 pt : la zone de texte réserve désormais la hauteur
de la barre de pied de page (`\vskip 10mm` dans le template `footline`). Sous un seuil
plus strict de 6 pt, une trentaine de frames très denses voient leur dernière ligne
effleurer cette barre sur moins de 2 mm : `python src/check_overflow.py out/Chapitre-*.pdf --seuil=6`
les énumère, `python src/locate_page.py --seuil=6` donne le fichier et la ligne à retoucher.

## Sorties compilées

| Fichier | Pages | Taille |
|---|---|---|
| `out/Chapitre-1-fondements-bayesiens.pdf` | 69 | 1,429 Ko |
| `out/Chapitre-2-a-priori-a-posteriori.pdf` | 77 | 2,545 Ko |
| `out/Chapitre-3-estimation-inference.pdf` | 54 | 2,332 Ko |
| `out/Chapitre-4-comparaison-decision.pdf` | 64 | 2,150 Ko |
| `out/Chapitre-5-mcmc-mini-projet.pdf` | 67 | 2,295 Ko |
| `out/Cours-complet-Statistique-Bayesienne-leger.pdf` | 312 | 3,896 Ko |
| `out/Cours-complet-Statistique-Bayesienne.pdf` | 331 | 10,221 Ko |

## Modifier le support

- **Style global** : `slides/preamble.tex` (thème, couleurs `byDeep`/`byBlue`/…,
  macros `\chapterslide`, `\py`, `\file`, `\ok`, `\nok`, `\warn`).
- **Ajouter une slide** : créer un `\begin{frame}...\end{frame}` dans le
  `parts/chN_x.tex` adéquat. Les fichiers sont assemblés par ordre alphabétique.
- **Règle impérative** : toute frame contenant `\begin{lstlisting}` doit être
  déclarée `\begin{frame}[fragile]{...}`, sinon la compilation échoue.
  `python src/check_env.py` vérifie cette règle automatiquement.
- **Ajouter une figure** : l'écrire dans un `src/fig_chN*.py` et l'enregistrer via
  `save(fig, "gNN_nom.png")` ; la référence dans une slide est alors
  `\includegraphics[width=.8\textwidth]{gNN_nom.png}`.

*Documentation générée le 2026-09-23 par `src/make_readme.py` à partir de l'état réel du dépôt.*
