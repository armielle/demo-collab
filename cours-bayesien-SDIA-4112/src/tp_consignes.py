"""Insere une page « Consignes et bareme » en tete de chaque notebook de TP.

Le texte vit ici (et non dans les notebooks) pour rester modifiable en un seul
endroit : relancer ce script propage la version courante dans les six notebooks.
L'insertion est idempotente : la cellule est reperee par son marqueur
`<!-- CONSIGNES -->` et remplacee si elle existe deja.

Usage :  ./.venv/Scripts/python.exe src/tp_consignes.py
"""
import os, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = os.path.join(ROOT, "notebooks")
MARQUE = "<!-- CONSIGNES -->"

ENTETE = """\
{marque}
# {titre}

**{chapitre}** — EPD-SDIA-4112 « Statistique Bayésienne », Master 1 SDIA
**Enseignante : Dr. Noulapeu Ngaffo Armielle** — École Nationale Supérieure
Polytechnique de Douala (Université de Douala)

---

## Objectifs

{objectifs}

## Ce qui est demandé

{demande}

## Barème — {points}

{bareme}

## Mode d'emploi

- **Noyau Jupyter** : `sdia-bayes` (Python 3.11, PyMC, ArviZ, NumPy, SciPy, Matplotlib).
- Ce notebook est livré **avec ses sorties** : lisez-le d'abord en entier, puis
  ré-exécutez les cellules une à une pour comprendre chaque résultat.
- Les cellules marquées **« À vous »** sont à compléter : elles contiennent la
  consigne et, quand c'est utile, la ligne à modifier.
- Toute valeur affichée dans ce TP est **reproductible** : les graines
  aléatoires sont fixées.
- Rendu : le notebook complété (`.ipynb`), avec vos sorties exécutées.
"""


def cellule(markdown):
    return {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(keepends=True)}


TEXTES = {
"TP1_mise_a_jour_bayesienne.ipynb": dict(
    titre="TP1 — Mise à jour bayésienne et conjugaison Beta-Binomial",
    chapitre="Chapitre 1 — Introduction et fondements bayésiens",
    objectifs="""\
- écrire une vraisemblance, un a priori et en déduire un posterior ;
- vérifier par le calcul que trois voies différentes (conjugaison, grille,
  échantillonnage) donnent la **même** loi a posteriori ;
- comprendre que le posterior d'aujourd'hui est le prior de demain.""",
    demande="""\
1. Parcourir les sections 1 à 3 et vérifier les sorties affichées.
2. Compléter les **quatre zones « À vous »** (elles se suivent dans le notebook).
3. Section 4 : refaire l'exercice de synthèse sur le taux de conversion d'un site
   en choisissant vous-même un a priori et en justifiant ce choix en une phrase.""",
    points="10 points",
    bareme="""\
| Critère | Points |
|---|---|
| Zone « À vous » 1 — posterior analytique | 2 |
| Zone « À vous » 2 — vérification par grille | 2 |
| Zone « À vous » 3 — mise à jour séquentielle | 2 |
| Zone « À vous » 4 — exercice de synthèse | 3 |
| Justification du choix d'a priori (une phrase, argumentée) | 1 |""",
),
"TP2_priors_conjugaison_shrinkage.ipynb": dict(
    titre="TP2 — Choix des a priori, conjugaison et shrinkage",
    chapitre="Chapitre 2 — Distributions a priori et a posteriori",
    objectifs="""\
- vérifier les quatre couples conjugués par le calcul direct ;
- quantifier l'information qu'apporte un a priori (taille d'échantillon virtuelle) ;
- faire un **contrôle prédictif a priori** — simuler avant d'avoir les données ;
- observer le **shrinkage** : comment un modèle hiérarchique emprunte de
  l'information entre groupes.""",
    demande="""\
1. Parcourir les sections 1 à 4 et commenter les figures obtenues.
2. **Exercice d'élicitation** : reprendre le tableau de sensibilité de la
   section 2 en changeant l'a priori, puis recalculer et comparer.
3. Section 4 : expliquer en trois lignes pourquoi l'intervalle d'un comté
   se rétrécit quand le modèle devient hiérarchique.""",
    points="10 points",
    bareme="""\
| Critère | Points |
|---|---|
| Vérification des quatre conjugaisons | 2 |
| Tableau de sensibilité recalculé | 3 |
| Contrôle prédictif a priori interprété | 2 |
| Explication du shrinkage (3 lignes) | 3 |""",
),
"TP3_estimation_prediction_regression.ipynb": dict(
    titre="TP3 — Estimation, intervalles, prédiction et régression bayésienne",
    chapitre="Chapitre 3 — Estimation et inférence bayésiennes",
    objectifs="""\
- relier chaque **perte** à son estimateur (quadratique → moyenne, absolue →
  médiane, 0–1 → mode) ;
- comparer le **HDI** à l'intervalle à queues égales, et mesurer la couverture
  réelle d'un intervalle fréquentiste ;
- distinguer la moyenne prédictive d'une observation future ;
- comparer la régression bayésienne aux moindres carrés.""",
    demande="""\
1. Parcourir les sections 1 à 4 et vérifier les valeurs numériques affichées.
2. Refaire la figure de **couverture** avec $n = 5$ puis $n = 100$ : la couverture
   de l'intervalle de Wald s'écarte-t-elle encore de 95 % ? Commenter.
3. Écrire un **contrôle prédictif a posteriori (PPC)** pour la régression :
   la statistique de test sera le **nombre de points hors de la bande de
   crédibilité à 90 %**. Comparer la valeur observée à la distribution simulée.""",
    points="10 points",
    bareme="""\
| Critère | Points |
|---|---|
| Relier pertes et estimateurs | 2 |
| Couverture à n = 5 et n = 100, commentée | 3 |
| PPC de la régression (statistique + interprétation) | 4 |
| Conclusion sur l'adéquation du modèle | 1 |""",
),
"TP4_comparaison_modeles_decision.ipynb": dict(
    titre="TP4 — Facteur de Bayes, critères prédictifs et décision bayésienne",
    chapitre="Chapitre 4 — Comparaison de modèles et décision",
    objectifs="""\
- calculer un **facteur de Bayes** par échantillonnage et l'interpréter sur
  l'échelle de Jeffreys ;
- voir comment le facteur de Bayes **s'accumule** avec $n$ ;
- décider avec des **coûts asymétriques** (et non avec un seuil de p-value) ;
- conduire un A/B test bayésien avec arrêt anticipé.""",
    demande="""\
1. Parcourir les sections 1 à 4 et vérifier les facteurs de Bayes affichés.
2. Reprendre l'analyse avec **vos propres données** (ou celles de votre domaine) :
   produire un tableau à cinq lignes « trois critères + décision de coût ».""",
    points="10 points",
    bareme="""\
| Critère | Points |
|---|---|
| Facteur de Bayes recalculé et interprété | 3 |
| Évolution avec n correctement lue | 2 |
| Décision à coûts asymétriques justifiée | 3 |
| Tableau « trois critères + décision » sur vos données | 2 |""",
),
"TP5_mcmc_diagnostics.ipynb": dict(
    titre="TP5 — Monte-Carlo, Metropolis-Hastings et diagnostics",
    chapitre="Chapitre 5 — Méthodes numériques (MCMC)",
    objectifs="""\
- mesurer l'erreur Monte-Carlo en $1/\\sqrt{N}$ ;
- écrire **Metropolis-Hastings à la main** et comprendre le rôle de la largeur
  de proposition ;
- implémenter un échantillonneur de **Gibbs** sur une normale bivariée ;
- retrouver le même résultat avec **NUTS** (PyMC) ;
- lire les diagnostics avant de croire un résultat : $\\hat{R}$, ESS, traces,
  autocorrélation.""",
    demande="""\
1. Parcourir les sections 1 à 7 et vérifier les diagnostics affichés.
2. Compléter les **zones « À vous »** (sections 3 et 6).
3. Section 6 : pour un échantillonnage qui échoue volontairement, expliquer
   **quel diagnostic** l'aurait détecté et pourquoi.""",
    points="10 points",
    bareme="""\
| Critère | Points |
|---|---|
| Erreur Monte-Carlo vérifiée en 1/√N | 2 |
| Metropolis-Hastings écrit à la main | 3 |
| Effet de la largeur de proposition | 2 |
| Diagnostics interprétés (R̂, ESS, traces) | 3 |""",
),
"PROJET_modele_hierarchique_radon.ipynb": dict(
    titre="Mini-projet — modèle hiérarchique bayésien sur les données radon",
    chapitre="Chapitre 5 — Méthodes numériques (MCMC) et mini-projet",
    objectifs="""\
Conduire **seul·e** une modélisation bayésienne complète, de la donnée brute à la
décision chiffrée, en justifiant chaque choix. Ce travail remplace l'examen final.""",
    demande="""\
Reproduire le pipeline en **six étapes** sur les données radon du Minnesota
(`data/radon_minnesota_srrs2.dat`), dans votre propre notebook ou script :

| Étape | Attendu |
|---|---|
| 0 | Lecture et exploration des données (effectifs par comté, distribution) |
| 1 | Écriture du modèle (vraisemblance + a priori, justifiés en une phrase chacun) |
| 2 | Contrôle **prédictif a priori** (simuler avant de regarder les données) |
| 3 | Échantillonnage : graine, version, nombre de chaînes et de tirages notés |
| 4 | Diagnostics **interprétés** : $\\hat{R}$, ESS, traces, autocorrélation |
| 5 | Contrôle prédictif a posteriori (PPC) |
| 6 | Visualisation du résultat et **phrase de décision chiffrée** |

Le notebook fourni sert de squelette et de grille de lecture : il montre les six
étapes, mais le rendu doit être le vôtre (données, choix, commentaires).""",
    points="20 points",
    bareme="""\
| Critère | Points |
|---|---|
| Étape 0 — lecture et exploration honnêtes | 2 |
| Étape 1 — modèle justifié (vraisemblance + a priori) | 4 |
| Étape 2 — contrôle prédictif a priori | 2 |
| Étape 3 — échantillonnage **reproductible** (graine, version) | 2 |
| Étape 4 — diagnostics interprétés, pas seulement affichés | 4 |
| Étape 5 — PPC | 2 |
| Étape 6 — décision chiffrée | 2 |
| Qualité de la rédaction et honnêteté sur les limites | 2 |""",
),
}


def main():
    for nom, t in TEXTES.items():
        chemin = os.path.join(NB, nom)
        nb = json.load(open(chemin, encoding="utf-8"))
        texte = ENTETE.format(marque=MARQUE, **t)
        nouvelle = cellule(texte)
        if nb["cells"] and MARQUE in "".join(nb["cells"][0].get("source", [])):
            nb["cells"][0] = nouvelle
            etat = "remplacee"
        else:
            nb["cells"].insert(0, nouvelle)
            etat = "inseree"
        json.dump(nb, open(chemin, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  {nom:<46} {etat}")
    print(f"\n{len(TEXTES)} notebooks : page de consignes en tete.")


if __name__ == "__main__":
    main()
