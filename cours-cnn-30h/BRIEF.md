# BRIEF DE RÉDACTION — Cours CNN 30 h (Master 1 SDIA)

**Tout en français. Beamer / XeLaTeX (Tectonic).** Les figures sont **déjà générées** :
ne jamais régénérer, ne jamais modifier `preamble.tex`, `build.py`, `check_env.py`, `figstyle.py`.

## 1. Règles absolues

1. **Aucun `\usepackage`** dans un `slides/parts/*.tex` — tout est dans le préambule.
2. Toute frame contenant `\begin{lstlisting}` **doit** être `\begin{frame}[fragile]{…}`.
3. Un espace typographique avant les unités et les `\%` : `95 \%`, `3 mm`.
4. Pas d'emoji. Utiliser `\ok`, `\nok`, `\warn` pour la validation.
5. Un fichier `parts/chN_x.tex` = **frames seules**, sans `\documentclass`, sans `\begin{document}`.
6. Densité : viser **~12–17 frames par fichier**, chacune lisible (ne pas surcharger).
7. Écrire les fichiers **un par un** et compiler après chaque (`build.py`) — un agent tué
   en cours de route ne laisse survivre que ce qui est écrit.
8. Ne pas inventer de chiffres : utiliser ceux de `figures/key_numbers_chN.json`.

## 2. Environnements d'encadré disponibles

`\begin{kbox}[Titre]` (à retenir, violet) · `\begin{wbox}[Titre]` (attention, rouge) ·
`\begin{ebox}[Titre]` (exemple, vert) · `\begin{abox}[Titre]` (activité, bleu) ·
`\begin{qbox}[Titre]` (quiz, orange) · `\begin{solbox}[Titre]` (solution, cyan) ·
`\begin{codebox}[Titre]` (code Python, gris) · `\begin{figbox}` (image encadrée).

Le titre entre crochets est **obligatoire** dans ce cours (`[Les trois idées clés]`).

Macros : `\py{code}` (code en ligne), `\file{nom.py}`, `\R`, `\N`, `\E`, `\given`.
Colonnes : `\begin{columns}[t] \begin{column}{0.48\textwidth} … \end{column} … \end{columns}`.

## 3. Figures disponibles (noms exacts, sans chemin)

| Ch. | Fichiers |
|:---:|---|
| 1 | `f01_mur_du_mlp.png` `f02_matrice_tenseur.png` `f03_localite_partage.png` `f04_invariance_translation.png` |
| 2 | `g01_convolution_pas_a_pas.png` `g02_glissement_noyau.png` `g03_noyaux_classiques.png` `g04_effets_noyaux.png` `g05_padding.png` `g06_stride.png` `g07_carte_dimensions.png` `g08_convolution_1d.png` `g09_champ_recepteur.png` `g10_multicanaux.png` `g11_conv_vs_corr.png` |
| 3 | `h01_parametres_couches.png` `h02_activations.png` `h03_pooling.png` `h04_pipeline.png` `h05_lenet.png` `h06_architectures.png` `h07_hierarchie_features.png` |
| 4 | `i01_courbes_apprentissage.png` `i02_dropout.png` `i03_batchnorm.png` `i04_augmentation.png` `i05_transfert.png` `i06_learning_rate.png` |
| 5 | `j01_detection_segmentation.png` `j02_trois_segmentations.png` `j03_une_deux_etapes.png` `j04_unet.png` `j05_cnn1d_ecg.png` `j06_cnn1d_trafic.png` `j07_autres_modalites.png` |
| 6 | `k01_gtsrb.png` `k02_confusion_metriques.png` `k03_topk_roc.png` `k04_filtres_activations.png` `k05_gradcam.png` `k06_robustesse.png` |

Usage : `\includegraphics[width=0.92\textwidth]{g01_convolution_pas_a_pas.png}`
**PIÈGE : une image plus large que sa colonne n'est pas redimensionnée par LaTeX.**
Dans une colonne `{0.48\textwidth}`, mettre au maximum `width=0.46\textwidth`.

## 4. Modèle de la page de garde (déjà écrit dans `ch1_a.tex`, ne pas le refaire)

Chaque chapitre commence par :
```
\section{Chapitre N --- Titre}
\chapterslide{Chapitre N}{Titre}{volume --- sous-titre}{\textbf{Objectifs.} …}
```

## 5. Répartition des chapitres et des fichiers

Le détail pédagogique complet est dans `PLAN.md` (lire les sections du chapitre concerné).

| Ch. | Fichiers à écrire | Contenu |
|:---:|---|---|
| 1 | `ch1_b` `ch1_c` `ch1_d` | mur du MLP (calculs), trois défauts, localité/partage de poids, invariance, quiz, TP1 |
| 2 | `ch2_a` `ch2_b` | convolution, calcul pas à pas, noyaux classiques, padding, stride, dimensions |
| 2 | `ch2_c` | filtres classiques, 1D, champ récepteur, multi-canaux, conv vs corrélation, quiz, TP1/TP2 |
| 3 | `ch3_a` `ch3_b` | couche conv, canaux, ReLU, pooling, flatten, dense, softmax, pipeline |
| 3 | `ch3_c` | LeNet, AlexNet, VGG, ResNet, nombre de paramètres, quiz, TP3 |
| 4 | `ch4_a` `ch4_b` | entraînement, learning rate, courbes, régularisation, augmentation, transfert, quiz, TP4 |
| 5 | `ch5_a` `ch5_b` | détection, segmentation, U-Net, CNN 1D, autres modalités, quiz, TP5 |
| 6 | `ch6_a` `ch6_b` | jeux de données, métriques, validation, interprétabilité, Grad-CAM, robustesse, quiz, projet |

**Chaque chapitre se termine par** : un encadré `\begin{qbox}` de 3 à 5 questions avec les
réponses dans un `\begin{solbox}` juste après, puis une frame « À retenir » et une frame
renvoyant au TP (`notebooks/TP…ipynb`).

## 6. Commandes

```bash
cd C:/Users/HP/Documents/cours-cnn-30h
./.venv/Scripts/python.exe src/check_env.py      # cohérence des sources
./.venv/Scripts/python.exe src/build.py 2        # compile le chapitre 2
```

Compiler **depuis la racine du projet**, jamais depuis `slides/`.
