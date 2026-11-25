# Réseaux de neurones convolutifs (CNN) — Cours 30 h — Master 1 SDIA

**Enseignante :** Dr. Noulapeu Ngaffo Armielle — ENSPD, Université de Douala
**Public :** Master 1 Sciences des données et intelligence artificielle
**Volume :** 30 h (18 h CM/TD projetés + 12 h TP machine)
**Sources de référence :** CNRS — Formation FIDLE, séquences *Principes et concepts des réseaux de neurones convolutifs* (2024 et 2025), licence CC BY-NC-ND 4.0.

---

## 1. Architecture générale du module

| Ch. | Titre | Volume | Slides | TP associé |
|:---:|---|:---:|:---:|:---:|
| 1 | Du MLP aux CNN : pourquoi la convolution | 3 h | ~28 | — |
| 2 | L'opération de convolution | 6 h | ~50 | TP1, TP2 |
| 3 | Architecture d'un réseau convolutif | 6 h | ~52 | TP3 |
| 4 | Entraînement, régularisation, transfert | 5 h | ~40 | TP4 |
| 5 | CNN au-delà de la classification | 5 h | ~38 | TP5 |
| 6 | Évaluation, interprétabilité, projet | 5 h | ~38 | TP6, projet |
| | **Total** | **30 h** | **~246** | 6 TP + 1 projet |

---

## 2. Détail par chapitre

### Chapitre 1 — Du MLP aux CNN : pourquoi la convolution (3 h)
- Rappels : perceptron, MLP, rétropropagation, descente de gradient.
- Une image est une **matrice** (niveaux de gris) ou un **tenseur** (RVB : 3 matrices empilées).
- Le mur du MLP : MNIST 28×28 = 784 entrées → ~0,8 M de paramètres ; une photo 24 Mpx RVB → **72 milliards** de paramètres.
- Trois défauts structurels : explosion du nombre de paramètres, connexions spatialement absurdes, dépendance à la position du motif.
- La réponse : **localité** (chaque neurone ne voit qu'un voisinage) + **partage de poids** (le même filtre balaye toute l'image) ⇒ invariance par translation et réduction drastique des paramètres.
- **QCM 1–5.**

### Chapitre 2 — L'opération de convolution (6 h)
- Définition du produit de convolution 2D discret : $(I \ast K)[i,j] = \sum_{m}\sum_{n} I[i+m, j+n]\,K[m,n]$.
- Mise en œuvre pas à pas à la main sur une matrice 5×5 avec un noyau 3×3.
- **Padding** (same/valid), **stride**, **dilation** ; formule de dimension de sortie.
- Champ récepteur et croissance avec la profondeur.
- Filtres classiques : flou (moyenneur, gaussien), netteté, contours (Sobel, Prewitt), Laplacien, emboss, « intensité ».
- Convolution **séparable**, coût de calcul, convolution 1D / 2D / 3D.
- **QCM 6–15.** **TP1** : appliquer des noyaux avec `scipy.signal.convolve2d`. **TP2** : padding/stride/pooling sur tenseurs.

### Chapitre 3 — Architecture d'un réseau convolutif (6 h)
- Couche convolutive : canaux d'entrée/sortie, tenseurs de poids, biais ; nombre de paramètres.
- Activation **ReLU** (et variantes : LeakyReLU, GELU, sigmoïde, tanh).
- **Pooling** : max, moyenne, global ; rôle de réduction et d'invariance.
- **Flatten** puis couches denses ; **softmax** et entropie croisée.
- Le pipeline complet : conv → ReLU → pool → … → flatten → dense → softmax.
- Architectures historiques : **LeNet-5** (1998), **AlexNet** (2012), **VGG**, blocs résiduels (**ResNet**), connexions sautées, 1×1, facteur d'échelle.
- CNN vs MLP : comparaison chiffrée des paramètres.
- **QCM 16–27.** **TP3** : premier CNN sur MNIST (PyTorch, CPU).

### Chapitre 4 — Entraînement, régularisation et transfert (5 h)
- Initialisation, lots (batchs), époques, learning rate, optimiseurs (SGD, momentum, Adam).
- Surapprentissage / sous-apprentissage : lecture des courbes.
- **Régularisation** : weight decay, dropout, batch normalization, early stopping, augmentation de données (rotations, translations, flips, bruit).
- **Transfert learning** : extraction de features vs fine-tuning ; jeux pré-entraînés (ImageNet).
- Erreurs classiques : normalisation oubliée, fuite de données, déséquilibre de classes.
- **QCM 28–36.** **TP4** : régularisation et augmentation.

### Chapitre 5 — CNN au-delà de la classification (5 h)
- **Détection d'objets** : en deux étapes (R-CNN, Faster R-CNN) vs en une étape (YOLO, SSD) ; boîtes englobantes, IoU, NMS.
- **Segmentation** : sémantique, par instance, panoptique ; FCN, **U-Net** (encodeur-décodeur, skip connections).
- **CNN 1D** : séries temporelles — ECG (tachycardie), trafic routier, météo (mistral), signaux industriels ; formes 1D des noyaux.
- Autres modalités : spectrogrammes audio, texte 1D, données 3D (imagerie médicale).
- **Landmarking** (morphométrie), comptage dense (crowd counting), régression par CNN.
- **QCM 37–45.** **TP5** : CNN 1D sur séries temporelles.

### Chapitre 6 — Évaluation, interprétabilité et projet (5 h)
- Jeux de données : MNIST, CIFAR-10, **GTSRB** (panneaux routiers, 43 classes, données réelles).
- Métriques : exactitude, précision/rappel, F1, matrice de confusion, top-k, courbe ROC.
- Validation : découpage train/val/test, validation croisée, fuite de données.
- **Interprétabilité** : visualisation des filtres (poids), des cartes d'activation (*feature maps*), hiérarchie bordes → parties → objets, **Grad-CAM**.
- Limites : robustesse, exemples adversariaux, biais de jeu de données.
- **QCM 46–54.** **TP6 + projet** : classifier les panneaux GTSRB.

---

## 3. Travaux pratiques (PyTorch, CPU)

| TP | Objet | Chapitre | Données |
|:---:|---|:---:|---|
| TP1 | Noyaux de convolution avec `scipy` | 2 | image synthétique + photo |
| TP2 | Padding, stride, pooling sur tenseurs | 2 | tenseurs jouets |
| TP3 | Premier CNN (LeNet) sur MNIST | 3 | MNIST (sous-ensemble) |
| TP4 | Régularisation et augmentation | 4 | MNIST / CIFAR-10 partiel |
| TP5 | CNN 1D sur séries temporelles | 5 | ECG synthétique + trafic |
| TP6 | Projet GTSRB (classification de panneaux) | 6 | GTSRB (sous-ensemble) |

**Contrainte matérielle :** tout tourne sur **CPU** (8 cœurs, pas de GPU). Les TP sont calibrés pour rester sous ~10 min chacun : sous-ensembles de données, peu d'époques, petits modèles.

---

## 4. Traçabilité

Aucun chiffre n'est saisi à la main : chaque valeur affichée provient d'un script exécuté (`src/fig_chN*.py`) et est consignée dans `figures/key_numbers_chN.json`. Graine fixée (4112) → résultats reproductibles.

## 5. Livrables

- 6 chapitres Beamer compilés + un document maître (PDF).
- ~6 TP exécutés (notebooks avec sorties) + export PDF imprimable.
- ~40 figures produites par code exécuté.
- Ce plan, un README et un index des ressources.
