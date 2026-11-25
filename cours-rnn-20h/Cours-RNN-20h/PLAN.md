# Réseaux de neurones récurrents (RNN) — Cours 20 h — Master 1 SDIA

**Enseignante :** Dr. Noulapeu Ngaffo Armielle — ENSPD, Université de Douala
**Public :** Master 1 Sciences des données et intelligence artificielle
**Volume :** 20 h (12 h CM/TD + 8 h TP machine) — *partie « données séquentielles »*
**Position dans le parcours :** fait suite au module « Réseaux de neurones convolutifs ».

## Architecture du module

| Ch. | Titre | Volume | TP associé |
|:---:|---|:---:|:---:|
| 1 | Données séquentielles : le problème | 4 h | TP1 |
| 2 | Le RNN : architecture et propagation | 6 h | TP2 |
| 3 | LSTM et GRU : dompter la mémoire | 5 h | TP3 |
| 4 | Entraînement, applications, projet | 5 h | TP4, projet |
| | **Total** | **20 h** | 4 TP + 1 projet |

## Détail par chapitre

### Chapitre 1 — Données séquentielles : le problème (4 h)
- Ce qu'est une séquence : texte, série temporelle, signal, trajectoire (cocinelle de FIDLE).
- Embeddings : mots → vecteurs ; plongement dense, sac de mots vs ordre.
- Pourquoi un réseau dense ou convolutif ne suffit plus : longueur variable, dépendances à distance, ordre.
- La cible : prédire la trajectoire d'une coccinelle, classer une critique de cinéma.
- QCM 1–6.

### Chapitre 2 — Le RNN : architecture et propagation (6 h)
- Le déroulement temporel (unrolling) : la même cellule réutilisée à chaque pas.
- Équations de la cellule : h_t = tanh(W_x x_t + W_h h_{t-1} + b).
- La boucle de rétropropagation à travers le temps (BPTT).
- Gradients qui s'évanouissent / explosent : la preuve par la dérivée itérée.
- Variantes d'entrée/sortie : many-to-one, many-to-many, seq2seq.
- QCM 7–18.

### Chapitre 3 — LSTM et GRU : dompter la mémoire (5 h)
- Les portes : oubli, entrée, sortie ; l'état de cellule comme convoyeur.
- Équations LSTM commentées ; pourquoi le gradient passe.
- GRU : deux portes, moins de paramètres.
- Bidirectionnel, multi-couches, seq2seq et attention naissante.
- QCM 19–28.

### Chapitre 4 — Entraînement, applications, projet (5 h)
- Clipping de gradient, troncature de BPTT, teacher forcing.
- Applications : classification de textes, prédiction de séries, génération.
- Métriques de séquence, échantillons de validation temporels (fuite).
- Projet : classifier des critiques de films / prédire une trajectoire.
- QCM 29–36.

## Travaux pratiques (PyTorch, CPU)

| TP | Objet | Chapitre |
|:---:|---|:---:|
| TP1 | Embeddings et séquences (mots → vecteurs) | 1 |
| TP2 | RNN simple : prédire une série temporelle | 2 |
| TP3 | LSTM/GRU : classifier des critiques | 3 |
| TP4 | Projet : prédiction de trajectoire | 4 |

**Contrainte matérielle :** CPU seul (pas de GPU), sous-ensembles de données, threads ≤ 2.

## Traçabilité

Aucun chiffre saisi à la main : chaque valeur sort d'un script exécuté
(`src/fig_chN*.py`) et est consignée dans `figures/key_numbers_chN.json`.
Graine fixée (4112).
