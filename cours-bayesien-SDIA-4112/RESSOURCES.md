# Ressources du cours — Statistique Bayésienne (EPD-SDIA-4112)

Index complet de ce qui est livré dans `cours-bayesien-SDIA-4112/`, avec la nature de
chaque ressource et la façon de la rejouer.

## 1. En un coup d'œil

| Ressource | Quantité | Où |
|---|---|---|
| Slides (PDF compilés) | 6 fichiers, **253 slides** | `out/` |
| Notebooks de TP exécutés | **6** (sorties incluses) | `notebooks/` |
| Jeux de données | 2 | `data/` |
| Figures générées par code | **68** (67 PNG + 1 GIF) | `figures/` |
| Valeurs numériques traçables | **275** clés réparties en 8 JSON | `figures/key_numbers_ch*.json` |
| Scripts Python | 38 utiles (+3 de travail) | `src/` |
| Logo institutionnel | 1 | `assets/logo_enspd.png` |
| Tableau de bord HTML autonome | 1 | `out/tableau-de-bord.html` |

Toute valeur numérique affichée dans les slides ou les notebooks sort d'un script exécuté
(graines fixées) et est enregistrée dans un JSON par chapitre : **aucun chiffre saisi à la main**.

## 2. Les slides

| Fichier | Chapitre | Volume | Slides |
|---|---|---|---|
| `out/Chapitre-1-fondements-bayesiens.pdf` | Principes, théorème de Bayes | 8 h | 50 |
| `out/Chapitre-2-a-priori-a-posteriori.pdf` | A priori, conjugaison, shrinkage | 8 h | 51 |
| `out/Chapitre-3-estimation-inference.pdf` | Estimation, HDI, prédiction, PPC | 8 h | 50 |
| `out/Chapitre-4-comparaison-decision.pdf` | Facteurs de Bayes, WAIC, décision | 8 h | 51 |
| `out/Chapitre-5-mcmc-mini-projet.pdf` | MCMC, diagnostics, mini-projet | 8 h | 51 |
| `out/Cours-complet-Statistique-Bayesienne.pdf` | Document maître | 40 h | **253** |

Sources LaTeX : `slides/preamble.tex` + `slides/parts/ch{1..5}_{a..f}.tex`.
Chaque chapitre se termine par une page de synthèse qui renvoie désormais explicitement
au TP correspondant.

## 3. Les travaux pratiques — `notebooks/`

Six notebooks, **livrés avec leurs sorties** (exécutés par `nbclient`, noyau `sdia-bayes`).
Ils se lisent tels quels ; pour les rejouer : `jupyter lab notebooks/`.

| Notebook | Chapitre | Contenu | Cellules de code |
|---|---|---|---|
| `TP1_mise_a_jour_bayesienne.ipynb` | 1 | Le problème fondateur ; trois façons d'obtenir le même posterior ; mise à jour séquentielle ; exercice de synthèse (taux de conversion d'un site) — **4 zones « À vous »** | 9 |
| `TP2_priors_conjugaison_shrinkage.ipynb` | 2 | Les quatre conjugaisons vérifiées par le calcul ; information apportée par un prior ; contrôle prédictif **a priori** ; shrinkage hiérarchique | 5 |
| `TP3_estimation_prediction_regression.ipynb` | 3 | Trois estimateurs / trois pertes ; HDI contre queues égales et couverture réelle ; prédiction a posteriori ; régression bayésienne contre moindres carrés | 6 |
| `TP4_comparaison_modeles_decision.ipynb` | 4 | Facteur de Bayes par échantillonnage ; évolution du BF avec $n$ ; décision à coûts asymétriques ; A/B test avec arrêt anticipé | 5 |
| `TP5_mcmc_diagnostics.ipynb` | 5 | Monte-Carlo et l'erreur en $1/\sqrt{N}$ ; Metropolis-Hastings écrit à la main ; effet de la largeur de proposition ; Gibbs sur une normale bivariée ; le même problème avec NUTS ; diagnostics — **2 zones « À vous »** | 9 |
| `PROJET_modele_hierarchique_radon.ipynb` | 5 (évaluation) | Mini-projet : les six étapes du pipeline bayésien sur les données radon, avec la grille d'évaluation (20 points) | 6 |

### Versions PDF pour impression — `out/`

Les six notebooks sont aussi exportés en PDF (nbconvert → Edge sans interface), consignes
et barème en première page, formules composées et figures incluses :

| PDF | Pages |
|---|---|
| `out/TP1-mise-a-jour-bayesienne.pdf` | 8 |
| `out/TP2-priors-conjugaison-shrinkage.pdf` | 8 |
| `out/TP3-estimation-prediction-regression.pdf` | 7 |
| `out/TP4-comparaison-modeles-decision.pdf` | 7 |
| `out/TP5-mcmc-diagnostics.pdf` | 10 |
| `out/Mini-projet-modele-hierarchique-radon.pdf` | 9 |

Régénération : `python src/export_tp_pdf.py`. Le texte des consignes vit dans
`src/tp_consignes.py` (un seul endroit à modifier pour les six TP) ; le script
`python src/tp_consignes.py` le propage dans les notebooks (insertion idempotente).

## 4. Les données — `data/`

| Fichier | Contenu | Usage |
|---|---|---|
| `radon_minnesota_srrs2.dat` | Relevés de radon du Minnesota (12 777 lignes, 25 colonnes) | Mini-projet : modèle hiérarchique par comté |
| `rendement_12points.csv` | 12 observations simulées (`x`, `y`) | Chapitre 1 et TP3 : régression bayésienne |

## 5. Figures et valeurs traçables

- `figures/*.png` + `figures/*.gif` — **68 figures**, toutes produites par un script de `src/`.
- `figures/key_numbers_ch*.json` — 8 fichiers, **275 clés** : chaque nombre cité dans un
  slide est retrouvable dans le JSON de son chapitre (`key_numbers_ch1.json`,
  `key_numbers_ch1b.json`, `key_numbers_ch1c.json`, `key_numbers_ch1d.json`,
  `key_numbers_ch2.json`, `key_numbers_ch3.json`, `key_numbers_ch4.json`,
  `key_numbers_ch5.json`).

## 6. Les scripts — `src/`

| Rôle | Scripts |
|---|---|
| Génération des figures par chapitre | `figstyle.py`, `fig_ch1.py`, `fig_ch1b_pymc.py`, `fig_ch1d_pymc.py`, `fig_ch2.py`, `fig_ch2b_pymc.py`, `fig_ch2c_poids.py`, `fig_ch3.py`, `fig_ch3b_pymc.py`, `fig_ch4.py`, `fig_ch4b.py`, `fig_ch5.py`, `fig_ch5b_pymc.py`, `fig_ch5c_nb.py`, `tbl_ch1.py` |
| Création / exécution des notebooks | `mknb.py`, `nb_tp1.py`, `nb_tp2.py`, `nb_tp34.py`, `nb_tp5.py`, `nb_projet.py` |
| Compilation et orchestration | `build.py`, `run_all_figures.py`, `run_all_notebooks.py` |
| Contrôle qualité | `check_env.py` (cohérence des sources), `check_overflow.py` (débordements mesurés dans les PDF), `locate_page.py` (page PDF → fichier + ligne), `fix_overflow.py` (compactage automatique idempotent) |
| Documentation | `make_readme.py`, `make_dashboard.py` |
| Audit du PDF d'origine | `extract.py`, `outline.py`, `render.py`, `audit.py`, `clip.py`, `bisect.py`, `tex_probe.py`, `fill_ch5_placeholders.py` |

## 7. Rejouer le kit

```bash
cd ~/Documents/cours-bayesien-SDIA-4112

# 1. tout compiler (5 chapitres + document maître)
./.venv/Scripts/python.exe src/build.py

# 2. recalculer toutes les figures et les valeurs numériques
./.venv/Scripts/python.exe src/run_all_figures.py

# 3. ré-exécuter les six notebooks et régénérer leurs sorties
./.venv/Scripts/python.exe src/run_all_notebooks.py

# 4. contrôles avant diffusion
./.venv/Scripts/python.exe src/check_env.py
./.venv/Scripts/python.exe src/check_overflow.py out/Chapitre-*.pdf --seuil=10

# 5. documentation
./.venv/Scripts/python.exe src/make_readme.py
./.venv/Scripts/python.exe src/make_dashboard.py
```

## 8. Environnement

- Python 3.11 dans `.venv/` (numpy, scipy, matplotlib, pandas, PyMC, ArviZ, nbclient, PyMuPDF).
- Noyau Jupyter `sdia-bayes` (« SDIA Bayes ») pour exécuter les notebooks.
- Compilation LaTeX : **Tectonic** (`~/texbin/tectonic.exe`), XeLaTeX.
- Machine sans GPU : 2 chaînes MCMC, threads limités à 2, priorité basse. Le mini-projet
  radon est calibré pour rester sous ~15 min sur ce matériel (8 comtés, 400 tirages par chaîne).

## 9. Ce que les étudiants doivent rendre

| Chapitre | Rendu |
|---|---|
| 1 | TP1 complété (zones « À vous ») + prior/vraisemblance d'un problème de son domaine |
| 2 | TP2 : changer le prior d'élicitation et recalculer le tableau de sensibilité |
| 3 | TP3 : PPC de la régression (points hors bande), figure de couverture à $n=5$ et $n=100$ |
| 4 | TP4 : tableau « trois critères + décision de coût » sur ses propres données |
| 5 | **Mini-projet** (20 points) : notebook ou script reproduisant les six étapes sur les données radon |
