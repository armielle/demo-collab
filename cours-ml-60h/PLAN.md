# Introduction au Machine Learning — 60 h

**Licence 3 Sciences des données et intelligence artificielle — ENSPD, Université de Douala**
**Enseignante : Dr. Noulapeu Ngaffo Armielle**

## Cadrage

| Élément | Valeur |
|---|---|
| Volume | **60 h** = 35 h CM/TD + 25 h TP |
| Public | Licence 3 SDIA (après Python, probabilités, algèbre linéaire) |
| Outils | Python, NumPy, pandas, scikit-learn, matplotlib (CPU uniquement) |
| Évaluation | TP notés 20 % · Examen final 40 % · Projet 40 % |
| Fil rouge | un même jeu de données tabulaires traverse les 8 chapitres, jusqu'au projet |

## Les 8 chapitres + projet

| Ch. | Titre | CM/TD | TP | Total |
|:--:|---|:--:|:--:|:--:|
| 1 | Qu'est-ce qu'apprendre ? Démarche et généralisation | 4 | 2 | **6 h** |
| 2 | Préparer les données sans se tromper | 2 | 2 | **4 h** |
| 3 | Régression et régularisation | 5 | 3 | **8 h** |
| 4 | Classification linéaire et évaluation | 5 | 3 | **8 h** |
| 5 | Arbres, ensembles et interprétabilité | 5 | 3 | **8 h** |
| 6 | SVM et modèles à noyaux | 3 | 2 | **5 h** |
| 7 | Apprentissage non supervisé | 5 | 3 | **8 h** |
| 8 | Fiabilité, éthique et mise en production | 4 | 1 | **5 h** |
| — | Projet intégrateur | 2 | 6 | **8 h** |
| | **Total** | **35** | **25** | **60 h** |

## Détail des chapitres

### Ch.1 — Qu'est-ce qu'apprendre ? (6 h : 4 + 2)
- Apprentissage supervisé, non supervisé, par renforcement : définitions et exemples réels.
- Le pipeline : données → variables → modèle → évaluation → décision.
- Risque empirique, généralisation, compromis biais-variance.
- Découpage train / validation / test, validation croisée k-fold.
- **Fuite de données** : le concept qui revient dans les 8 chapitres.
- **TP1** : premier pipeline scikit-learn, comparaison baseline / modèle.

### Ch.2 — Préparer les données (4 h : 2 + 2)
- Valeurs manquantes : suppression, imputation moyenne/médiane/KNN.
- Encodage : one-hot, ordinal, cible ; variables cycliques.
- Standardisation, normalisation, discrétisation, gestion des outliers.
- **Le prétraitement doit être dans le pipeline**, jamais avant le split.
- **TP2** : `ColumnTransformer` + `Pipeline`, mesure de l'effet d'une fuite.

### Ch.3 — Régression et régularisation (8 h : 5 + 3)
- Moindres carrés ordinaires : forme fermée, hypothèses, interprétation des coefficients.
- Descente de gradient : pas d'apprentissage, convergence, normalisation préalable.
- Ridge, Lasso, ElasticNet : biais introduit, sélection de variables, chemin de régularisation.
- Diagnostic : résidus, R², RMSE/MAE, multicolinéarité.
- **TP3** : prédiction sur données réelles, OLS vs Ridge vs Lasso, courbes de validation.

### Ch.4 — Classification linéaire et évaluation (8 h : 5 + 3)
- Régression logistique : sigmoïde, entropie croisée, frontière linéaire.
- Seuil de décision : ce n'est pas 0,5 par défaut, c'est un choix métier.
- Matrice de confusion, précision, rappel, F1, ROC/AUC, courbe précision-rappel.
- Classes déséquilibrées : pondération, rééchantillonnage, coût des erreurs.
- **TP4** : classification binaire, réglage du seuil, arbitrage précision/rappel chiffré.

### Ch.5 — Arbres, ensembles et interprétabilité (8 h : 5 + 3)
- CART : impureté de Gini, entropie, profondeur, élagage.
- Bagging, forêt aléatoire, importance des variables.
- Gradient boosting (histogram-based) : pourquoi il domine sur le tabulaire.
- Interprétabilité : importance globale, dépendance partielle, SHAP (introduction).
- **TP5** : arbre vs forêt vs boosting, accord entre importance et SHAP.

### Ch.6 — SVM et noyaux (5 h : 3 + 2)
- Marge maximale, vecteurs de support, marge souple (C).
- Astuce du noyau, noyau RBF, effet de γ.
- Quand le SVM bat les arbres : petit n, grande dimension, frontière nette.
- **TP6** : grille C/γ sur données non linéaires, comparaison à la forêt aléatoire.

### Ch.7 — Apprentissage non supervisé (8 h : 5 + 3)
- k-means : initialisation, choix de k (coude, silhouette), sensibilité à l'échelle.
- CAH (linkages), DBSCAN : densité, bruit, formes arbitraires.
- ACP : variance expliquée, projection, reconstruction ; t-SNE et UMAP pour visualiser.
- Détection d'anomalies : Isolation Forest.
- **TP7** : segmentation de profils + ACP + détection d'anomalies.

### Ch.8 — Fiabilité, éthique et mise en production (5 h : 4 + 1)
- Sélection de modèle et surapprentissage par recherche d'hyperparamètres.
- Biais, équité, variables sensibles, consentement, protection des données.
- Dérive des données (drift), surveillance, versionnage du modèle.
- Modèle interprétable vs modèle performant : l'arbitrage à assumer.
- **TP8** : audit d'un modèle — trouver la fuite, mesurer le biais, simuler une dérive.

### Projet intégrateur (8 h : 2 + 6)
- Groupes de 3 à 4, sujet réel (scolarité, département, données publiques).
- Exigences : baseline obligatoire, split honnête, métrique justifiée, application utilisable.
- 2 h de cadrage, 4 h d'accompagnement, 2 h de soutenances.

## Progression pédagogique

```
Ch.1 démarche ─► Ch.2 données ─► Ch.3 régression ─► Ch.4 classification
                                                          │
                        Ch.8 fiabilité ◄─ Ch.7 non supervisé ◄─ Ch.5 arbres ─► Ch.6 SVM
                                                          │
                                                    Projet intégrateur
```

## TP (8 notebooks exécutés, noyau `ml-60h`)

| # | Notebook | Chapitre |
|:--:|---|:--:|
| 1 | `TP1_premier_pipeline.ipynb` | 1 |
| 2 | `TP2_preparation_donnees.ipynb` | 2 |
| 3 | `TP3_regression_regularisation.ipynb` | 3 |
| 4 | `TP4_classification_seuil.ipynb` | 4 |
| 5 | `TP5_arbres_ensembles_shap.ipynb` | 5 |
| 6 | `TP6_svm_noyaux.ipynb` | 6 |
| 7 | `TP7_non_supervise.ipynb` | 7 |
| 8 | `TP8_audit_modele.ipynb` | 8 |

## Contraintes de production

- Machine **sans GPU** : tout tient sur CPU, threads limités à 2.
- Chaque chiffre du cours sort d'un script exécuté (graine fixée), stocké dans
  `figures/key_numbers_chN.json`.
- Chaque chapitre : slides Beamer compilées (Tectonic), figures calculées, TP exécuté.
- Livraison : dossier + archive ZIP + version allégée, vérifiés par SHA-256.
