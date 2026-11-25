# Réseaux de neurones récurrents (RNN) — Cours 20 h

**Master 1 Sciences des données et intelligence artificielle — ENSPD, Université de Douala**
**Enseignante : Dr. Noulapeu Ngaffo Armielle**

Kit de cours exécutable : 4 chapitres Beamer compilés, 34 figures produites par du code
exécuté, 4 notebooks de TP exécutés avec leurs sorties. Suite du module CNN.

## Livrables (`out/`)

| Fichier | Contenu | Pages |
|---|---|:---:|
| `Cours-RNN-complet.pdf` | les 4 chapitres + page de titre | **72** |
| `Cours-RNN-complet-leger.pdf` | version allégée (1,08 Mo) | 72 |
| `Chapitre-1-donnees-sequencielles.pdf` | données séquentielles : le problème | 16 |
| `Chapitre-2-architecture-RNN.pdf` | architecture, propagation, gradients | 18 |
| `Chapitre-3-LSTM-GRU.pdf` | portes, convoyeur, variantes | 20 |
| `Chapitre-4-entrainement-et-applications.pdf` | entraînement, évaluation, projet | 17 |

## Structure des 20 h

| Ch. | Titre | Volume |
|:---:|---|:---:|
| 1 | Données séquentielles : le problème | 4 h |
| 2 | Le RNN : architecture et propagation | 6 h |
| 3 | LSTM et GRU : dompter la mémoire | 5 h |
| 4 | Entraînement, applications, projet | 5 h |

## Travaux pratiques

| Notebook | Objet | Durée |
|---|---|---|
| `TP1_embeddings_sequences.ipynb` | vocabulaire, indexation, padding, masque, embeddings | 1 h 30 |
| `TP2_RNN_serie_temporelle.ipynb` | cellule à la main, évanouissement mesuré, RNN sur série | 2 h |
| `TP3_LSTM_GRU_sentiment.ipynb` | RNN vs GRU vs LSTM vs bidirectionnel | 2 h |
| `PROJET_prediction_trajectoire.ipynb` | projet : trajectoire de coccinelle + référence naïve | 3 h |

Tous exécutés (sorties et figures incluses). Noyau Jupyter : `rnn-20h`.

## Figures et traçabilité

34 figures calculées par `src/fig_rnn_ch1.py` … `fig_rnn_ch4.py` (numpy/scipy/matplotlib,
graine 4112), et **~34 clés numériques** dans `figures/key_numbers_ch1..4.json`.
Aucune valeur saisie à la main : les gradients évanescents, les nombres de paramètres
et le RMSE de prévision sont calculés.

## Reproduire

```bash
python src/fig_rnn_ch1.py        # ... jusqu'a fig_rnn_ch4.py
python src/build.py all          # 4 chapitres + document maitre
python src/check_env.py          # coherence des sources
python src/check_qa.py --seuil=6 # debordements mesures dans les PDF
python src/nb_rnn_12.py          # TP1, TP2
python src/nb_rnn_34.py          # TP3, projet
python src/make_pdf_leger.py     # version allegee
```

## Contraintes

Machine **sans GPU** : modèles réduits, `torch.set_num_threads(2)`, corpus de démonstration.
Les TP restent sous quelques minutes chacun.

## Position dans le parcours

Ce module suppose acquis le module « Réseaux convolutifs » (convolution, pooling, entraînement,
régularisation). Il prépare les **Transformers** (attention, chapitre 3, section finale).
