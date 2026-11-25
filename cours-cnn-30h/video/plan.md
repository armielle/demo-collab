# Plan — Vidéo « La propagation dans un CNN, de l'image à la sortie dense »

**Public :** Master 1 SDIA (cours CNN 30 h). **Durée visée :** ~3 min 30. **Style :** 3B1B,
palette classique, police monospace (Consolas). **Sans LaTeX** (Text uniquement).

## Arc narratif (l'« aha »)

> Le même noyau de 9 poids balaye toute l'image ; à chaque étage il perd en résolution et
> gagne en abstraction ; à la fin, 16 nombres suffisent à décider.
> **Idée à faire sentir : la propagation avant n'est qu'une suite d'opérations locales.**

## Scènes

| # | Nom | Durée | Contenu |
|---|---|---|---|
| 1 | `S1_Ouverture` | 18 s | l'image réelle apparaît, titre, accroche « 9 nombres suffisent-ils ? » |
| 2 | `S2_ImageTenseur` | 25 s | grille 12×12 avec valeurs, un pixel = un nombre |
| 3 | `S3_Convolution` | 45 s | noyau 3×3 qui glisse, calcul détaillé d'un produit de convolution, puis carte d'activation complète (réelle) |
| 4 | `S4_ReLU` | 20 s | valeurs négatives (rouges) mises à zéro |
| 5 | `S5_Pooling` | 25 s | max pooling 2×2, fenêtre qui glisse, 10×10 → 5×5 |
| 6 | `S6_Hierarchie` | 25 s | contours → textures → parties, tailles décroissantes |
| 7 | `S7_FlattenDense` | 35 s | aplatissement 4×4 → 16, puis dense 16→8→3 avec impulsions |
| 8 | `S8_Softmax` | 30 s | barres de probabilités, argmax, récapitulatif du pipeline |

## Palette

`BG #1C1C1C` · primaire `#58C4DD` · secondaire `#83C167` · accent `#FFFF00` ·
rouge `#FC6255` · gris `#888888` · texte `#EAEAEA`

## Règles appliquées

- Opacité en couches : éléments actifs 1.0, contexte 0.4, structure 0.15.
- Pause après chaque révélation clé.
- Toutes les cartes affichées sont **calculées par du code exécuté** (`prep_donnees.py`,
  graine 4112) : aucune valeur inventée.
- La sortie dense est un réseau de démonstration : c'est dit à l'écran.

## Fichiers

- `prep_donnees.py` → `assets/*.png`, `assets/*.npy`, `assets/donnees.json`
- `script_cnn.py` → 8 scènes indépendantes
- Rendu : `-ql` pour itérer, `-qm` pour la version finale, puis concat ffmpeg.
