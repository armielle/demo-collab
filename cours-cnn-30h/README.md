# Réseaux de neurones convolutifs (CNN) — Cours 30 h

**Master 1 Sciences des données et intelligence artificielle — ENSPD, Université de Douala**
**Enseignante : Dr. Noulapeu Ngaffo Armielle**

Kit de cours exécutable : slides Beamer compilées, visuels produits par du code exécuté,
notebooks de TP exécutés avec leurs sorties, tout en français.

## Livrables (dossier `out/`)

| Fichier | Contenu | Pages |
|---|---|:---:|
| `Cours-CNN-complet.pdf` | les 6 chapitres à la suite + page de titre | 197 |
| `Chapitre-1-du-MLP-aux-CNN.pdf` | du MLP aux CNN : pourquoi la convolution | 25 |
| `Chapitre-2-operation-de-convolution.pdf` | l'opération de convolution | 41 |
| `Chapitre-3-architecture-CNN.pdf` | architecture d'un réseau convolutif | 47 |
| `Chapitre-4-entrainement-regularisation.pdf` | entraînement, régularisation, transfert | 30 |
| `Chapitre-5-au-dela-de-la-classification.pdf` | CNN au-delà de la classification | 26 |
| `Chapitre-6-evaluation-et-projet.pdf` | évaluation, interprétabilité, projet | 27 |

## Ressources

| Élément | Quantité | Où |
|---|:---:|---|
| Figures produites par code | 41 | `figures/` |
| Valeurs numériques traçables | ~140 clés | `figures/key_numbers_ch*.json` |
| Notebooks de TP exécutés | 6 | `notebooks/` |
| Scripts Python | ~15 | `src/` |
| Sources LaTeX | 18 fichiers | `slides/parts/` |
| Plan pédagogique | — | `PLAN.md` |

## Structure des 30 h

| Ch. | Titre | Volume |
|:---:|---|:---:|
| 1 | Du MLP aux CNN : pourquoi la convolution | 3 h |
| 2 | L'opération de convolution | 6 h |
| 3 | Architecture d'un réseau convolutif | 6 h |
| 4 | Entraînement, régularisation et transfert | 5 h |
| 5 | CNN au-delà de la classification | 5 h |
| 6 | Évaluation, interprétabilité et projet | 5 h |

## Travaux pratiques

| Notebook | Objet | Machine |
|---|---|---|
| `TP1_noyaux_convolution.ipynb` | noyaux classiques, convolution à la main | CPU, rapide |
| `TP2_padding_stride_pooling.ipynb` | formule de dimension, pooling | CPU, rapide |
| `TP3_premier_CNN_MNIST.ipynb` | LeNet-5 entraîné sur MNIST (sous-ensemble) | CPU, ~3 min |
| `TP4_regularisation_augmentation.ipynb` | trois configurations comparées | CPU, ~5 min |
| `TP5_CNN_1D_series_temporelles.ipynb` | CNN 1D sur un ECG | CPU, rapide |
| `PROJET_classification_images.ipynb` | projet complet (CIFAR-10) + Grad-CAM | CPU, ~15 min |

**Tous les notebooks sont livrés avec leurs sorties exécutées.** Pour les rejouer :
installer le noyau (`python -m ipykernel install --user --name cnn-30h`) puis ouvrir
dans Jupyter (`jupyter lab notebooks/`).

### Note sur le projet final

Le projet du cours porte à l'origine sur **GTSRB** (panneaux routiers). Son serveur de
téléchargement étant injoignable depuis cette machine, le notebook livré travaille sur
**CIFAR-10** (60 000 images réelles, 10 classes) — **le code est identique** : la fin du
notebook montre la ligne à changer pour brancher GTSRB dès que le jeu est disponible localement.

## Reproduire

```bash
# figures (numpy/scipy/matplotlib uniquement)
python src/fig_ch1.py        # ... jusqu'a fig_ch6.py

# slides
python src/build.py all      # 6 chapitres + document maitre

# controle qualite
python src/check_env.py      # environnements, [fragile], figures manquantes
python src/check_qa.py --seuil=6   # debordements mesures dans les PDF

# notebooks
python src/nb_fast.py        # TP1, TP2, TP5
python src/nb_train.py       # TP3, TP4, PROJET (telechargements + entrainement)
```

## Contraintes matérielles

Tout est conçu pour une machine **sans GPU** : sous-ensembles de données, peu d'époques,
`torch.set_num_threads(2)`, priorité basse. Les figures n'utilisent que
numpy/scipy/matplotlib.

## Sources

CNRS — Formation FIDLE, séquences *Principes et concepts des réseaux de neurones
convolutifs* (2024 et 2025), licence CC BY-NC-ND 4.0. Le présent kit est une
restructuration pédagogique originale de ces contenus.
