# Index des ressources — Cours CNN 30 h

## 1. Supports de cours (`out/`)

| Fichier | Pages | Contenu |
|---|:---:|---|
| `Cours-CNN-complet.pdf` | 249 | les 6 chapitres + page de titre |
| `Chapitre-1-du-MLP-aux-CNN.pdf` | 30 | du MLP aux CNN : pourquoi la convolution |
| `Chapitre-2-operation-de-convolution.pdf` | 56 | l'opération de convolution (pas à pas, noyaux, padding, stride, dimensions) |
| `Chapitre-3-architecture-CNN.pdf` | 48 | couche convolutive, pooling, pipeline, LeNet, VGG, ResNet |
| `Chapitre-4-entrainement-regularisation.pdf` | 32 | entraînement, courbes, dropout, BN, augmentation, transfert |
| `Chapitre-5-au-dela-de-la-classification.pdf` | 44 | détection, segmentation, U-Net, CNN 1D, autres modalités |
| `Chapitre-6-evaluation-et-projet.pdf` | 30 | jeux de données, métriques, Grad-CAM, robustesse, projet |

## 2. Notebooks de TP (`notebooks/`)

| Notebook | Chapitre | Objet | État |
|---|:---:|---|---|
| `TP1_noyaux_convolution.ipynb` | 2 | noyaux classiques + convolution à la main | exécuté (2 figures) |
| `TP2_padding_stride_pooling.ipynb` | 2 | formule de dimension, padding, stride, pooling | exécuté |
| `TP3_premier_CNN_MNIST.ipynb` | 3 | LeNet-5 entraîné sur MNIST | exécuté |
| `TP4_regularisation_augmentation.ipynb` | 4 | trois configurations comparées | exécuté |
| `TP5_CNN_1D_series_temporelles.ipynb` | 5 | CNN 1D sur ECG | exécuté (3 figures) |
| `PROJET_classification_images.ipynb` | 6 | projet complet CIFAR-10 + Grad-CAM | exécuté |

Noyau Jupyter : `cnn-30h` (Python 3.11, PyTorch CPU).

## 3. Figures (`figures/`)

41 figures produites par `src/fig_ch1.py` … `src/fig_ch6.py`, toutes calculées
(numpy / scipy / matplotlib, graine 4112). Les valeurs citées dans les slides
proviennent des JSON `key_numbers_ch*.json`.

| Préfixe | Chapitre | Exemples |
|:---:|:---:|---|
| `f0x` | 1 | mur du MLP, matrice/tenseur, localité, invariance |
| `gx` | 2 | convolution pas à pas, noyaux classiques, effets, padding, stride, carte des dimensions, convolution 1D, champ récepteur |
| `hx` | 3 | paramètres, activations, pooling, pipeline, LeNet, architectures, hiérarchie des features |
| `ix` | 4 | courbes, dropout, batch norm, augmentation, transfert, learning rate |
| `jx` | 5 | détection/segmentation, U-Net, ECG, trafic, modalités |
| `kx` | 6 | GTSRB, confusion/métriques, top-k/ROC, filtres, Grad-CAM, robustesse |

## 4. Scripts (`src/`)

| Script | Rôle |
|---|---|
| `figstyle.py` | style commun aux figures |
| `fig_ch1.py` … `fig_ch6.py` | calcul + figures + JSON de traçabilité |
| `build.py` | compilation Tectonic (6 chapitres + maître) |
| `check_env.py` | environnements équilibrés, `[fragile]`, figures manquantes |
| `check_qa.py` | débordements mesurés dans les PDF compilés |
| `fix_debordement.py` | correcteur à leviers (image, scission, déplacement de bloc) |
| `nb_fast.py` | fabrique et exécute TP1, TP2, TP5 |
| `nb_train.py` | fabrique et exécute TP3, TP4, PROJET |

## 5. Données (`data/`)

| Fichier | Contenu |
|---|---|
| `face_gray.npy`, `face_rgb.npy` | image réelle de travail (768×1024) |
| `MNIST/`, `CIFAR10/` | jeux téléchargés par les TP |

## 6. Documentation

| Fichier | Contenu |
|---|---|
| `PLAN.md` | plan pédagogique des 30 h, détail des 6 chapitres |
| `README.md` | mode d'emploi, commandes de rejeu |
| `BRIEF.md` | règles de rédaction LaTeX (pour retoucher les slides) |
