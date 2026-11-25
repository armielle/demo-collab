# Index des ressources — Cours RNN 20 h

## 1. Supports (`out/`)

| Fichier | Pages |
|---|:---:|
| `Cours-RNN-complet.pdf` | 72 |
| `Cours-RNN-complet-leger.pdf` | 72 (1,08 Mo) |
| `Chapitre-1-donnees-sequencielles.pdf` | 16 |
| `Chapitre-2-architecture-RNN.pdf` | 18 |
| `Chapitre-3-LSTM-GRU.pdf` | 20 |
| `Chapitre-4-entrainement-et-applications.pdf` | 17 |

## 2. Notebooks (`notebooks/`)

| Notebook | Chapitre | Contenu |
|---|:---:|---|
| `TP1_embeddings_sequences.ipynb` | 1 | vocabulaire, indexation, padding/masque, embedding, moyenne masquée |
| `TP2_RNN_serie_temporelle.ipynb` | 2 | cellule écrite à la main, gradient mesuré, RNN sur sinusoïde |
| `TP3_LSTM_GRU_sentiment.ipynb` | 3 | comparaison RNN / GRU / LSTM / bidirectionnel |
| `PROJET_prediction_trajectoire.ipynb` | 4 | projet complet + référence naïve + RMSE |

## 3. Figures (`figures/`)

| Préfixe | Chapitre | Contenu |
|:---:|:---:|---|
| `r0x` | 1 | types de séquences, embeddings, ordre, padding, dépendance, fenêtre CNN, pipeline, coccinelle |
| `s0x` | 2 | déroulement, cellule, état caché, BPTT, gradients, clipping, formes E/S, seq2seq, paramètres |
| `t0x` | 3 | cellule LSTM, portes, convoyeur, survie du gradient, GRU, paramètres, bidirectionnel, empilement, attention |
| `u0x` | 4 | teacher forcing, BPTT tronquée, courbes, comparaison, prévision, température, fuite, projet |

## 4. Scripts (`src/`)

| Script | Rôle |
|---|---|
| `fig_rnn_ch1.py` … `fig_rnn_ch4.py` | calculs + figures + JSON de traçabilité |
| `build.py` | compilation Tectonic (4 chapitres + maître) |
| `check_env.py` | environnements, `[fragile]`, figures manquantes |
| `check_qa.py` | débordements mesurés dans les PDF |
| `fix_debordement.py` | correcteur à leviers |
| `nb_rnn_12.py`, `nb_rnn_34.py` | fabrique et exécute les notebooks |
| `make_pdf_leger.py` | version allégée |
| `livre.py` | dossier de livraison + archive |

## 5. Documentation

| Fichier | Contenu |
|---|---|
| `PLAN.md` | plan pédagogique des 20 h |
| `README.md` | mode d'emploi, commandes |
| `RESSOURCES.md` | ce fichier |
