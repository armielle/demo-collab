"""Construit et exécute réellement le TP3 (corrigé officiel) avec nbclient."""
import os
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNEL = "sdia-bayes"

PREAMBLE = """%matplotlib inline
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
SEED = 5534                      # graine unique du projet (reproductibilité)
np.random.seed(SEED)
print("OK — graine", SEED)
"""

HEADER = """## EPD-GIT-5534 — TP 3 (corrigé officiel)

**Séance 3 — Détecter les incidents avant la panne**
**Enseignante : Dr. Noulapeu Ngaffo Armielle**

**Objectifs** : sur `nosys_features.csv` (34 560 fenêtres, 120 positives —
0,35 %), détecter **sans étiquette** (IsolationForest), confronter au réel,
entraîner des modèles supervisés (forêt *balanced*, gradient boosting,
régression logistique) avec un **split temporel** imposé
(train < 22/09/2026 = 24 192 lignes, test ≥ 22/09 = 10 368 lignes, 9 jours),
tracer les courbes précision-rappel, choisir un seuil sous **budget NOC**
(≤ 5 alertes/jour/serveur) et rédiger le tableau comparatif final.

**Barème (20 pts)** : B1 IsolationForest sans fuite — 3 pts · B2 recouvrement
+ commentaire honnête — 3 pts · B3 forêt balanced + matrice juste — 5 pts ·
B4 courbe PR + seuil sous budget — 5 pts · B5 comparatif + argumentation
métier — 3 pts · reproductibilité (graine 5534, split cité) — 1 pt.

*Corrigé officiel exécuté de bout en bout (noyau `sdia-bayes`, graine 5534).
Aucune statistique (standardisation, seuil, échantillonnage) n'est calculée
sur train + test mélangés : le futur ne sert qu'à juger.*
"""

CELLS = [
    ("md", HEADER),
    ("code", PREAMBLE),

    # ---------------- B1 ----------------
    ("md", "## Étape B1 — IsolationForest, sans jamais regarder la cible (25 min)\n\n"
           "Détection non supervisée : on entraîne sur le **passé uniquement** "
           "(fenêtres < 22/09) et on note tout le mois. `incident_30min` n'entre "
           "pas dans le calcul — on ne le regarde qu'à la fin, pour le contrôle."),
    ("code", """from sklearn.ensemble import IsolationForest

f = pd.read_csv("data/nosys_features.csv", parse_dates=["fenetre"])
X = f.drop(columns=["serveur", "fenetre", "incident_30min"])
X = X.fillna(0.0)                       # cpu_delta de la 1re fenêtre de chaque serveur

train = f["fenetre"] < "2026-09-22"     # split TEMPOREL imposé
print("train :", train.sum(), "lignes | test :", (~train).sum(), "lignes")

iso = IsolationForest(n_estimators=300, contamination=0.01,
                      random_state=SEED, n_jobs=2).fit(X[train])
f["score_anomalie"] = -iso.score_samples(X)

top20 = f.nlargest(20, "score_anomalie")
print(top20[["serveur", "fenetre", "cpu_moy", "lat_p99", "incident_30min"]].to_string(index=False))
print("top20 pré-incidents :", int(top20["incident_30min"].sum()), "/ 20")"""),
    ("code", """fig, ax = plt.subplots(figsize=(9, 3.2))
ax.hist(f.loc[train, "score_anomalie"], bins=60, alpha=.6, label="train (<22/09)")
ax.hist(f.loc[~train, "score_anomalie"], bins=60, alpha=.6, label="test (≥22/09)")
ax.axvline(f.loc[train, "score_anomalie"].quantile(0.99), color="crimson",
           ls="--", label="seuil q99 train")
ax.set_xlabel("score d'anomalie (-score_samples)")
ax.set_title("IsolationForest : distribution des scores")
ax.legend()
plt.tight_layout(); plt.show()"""),
    ("md", "**Livrable B1** — les **20 fenêtres les plus anormales** sont toutes "
           "**20/20 des pré-incidents** (`incident_30min = 1`), alors qu'aucune "
           "étiquette n'est entrée dans le calcul : le non-supervisé repère "
           "naturellement les configurations rarement vues (CPU en montée rapide, "
           "latence p99 hors norme). C'est un signe que les signaux du TP 2 "
           "portent l'information — pas une preuve de performance."),

    # ---------------- B2 ----------------
    ("md", "## Étape B2 — Recouvrement détections / alarmes réelles (20 min)\n\n"
           "Seuil = quantile 0,99 des scores **de train uniquement** (pas de fuite), "
           "puis confrontation au réel sur le test."),
    ("code", """seuil = f.loc[train, "score_anomalie"].quantile(0.99)
f["anomalie"] = f["score_anomalie"] > seuil

test = f[~train]
n_anom = int(test["anomalie"].sum())
n_vrais = int(test.loc[test["anomalie"], "incident_30min"].sum())
print("anomalies test :", n_anom, "| dont vrais pré-incidents :", n_vrais)
print(pd.crosstab(test["anomalie"], test["incident_30min"]))

# recouvrement : par fenêtre, combien de serveurs détectés à la fois ?
cov = (test.groupby("fenetre")["anomalie"].sum()
           .value_counts().sort_index())
print(cov.to_string())"""),
    ("code", """fig, ax = plt.subplots(figsize=(6.4, 3.2))
cov.plot.bar(ax=ax, color="C0")
ax.set_xlabel("serveurs détectés simultanément (même fenêtre)")
ax.set_ylabel("nb de fenêtres")
ax.set_title("Recouvrement des détections non supervisées (test)")
plt.tight_layout(); plt.show()"""),
    ("md", "**Livrable B2** — le seuil q99 déclenche **174 anomalies sur le test**, "
           "dont seulement **40 correspondent à un incident à venir** : les **134** "
           "autres sont des anomalies bénignes (batch, backup, pics). Chaque fausse "
           "alerte coûte une nuit d'ingénieur : **le non-supervisé seul ne suffit pas** "
           "— 23 % des anomalies valent le coup d'être envoyées au NOC. Le recouvrement "
           "montre que les détections se concentrent sur les mêmes fenêtres pour "
           "plusieurs serveurs : les incidents sont synchrones (cause partagée, "
           "constat du TP 1)."),

    # ---------------- B3 ----------------
    ("md", "## Étape B3 — Forêt balanced, gradient boosting, matrice de confusion (30 min)\n\n"
           "Split temporel strict, `class_weight='balanced'`, graine 5534. "
           "Piège attendu : avec `balanced`, le seuil 0,50 n'est plus calibré — "
           "la forêt prédit 7 positifs… tous faux (précision 0,00). "
           "**Ce n'est pas un bug**, c'est le déplacement du seuil effectif."),
    ("code", """from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

Xtr, ytr = X[train], f.loc[train, "incident_30min"]
Xte, yte = X[~train], f.loc[~train, "incident_30min"]

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=SEED, n_jobs=2).fit(Xtr, ytr)
pred_rf = rf.predict(Xte)
print("Forêt balanced — matrice de confusion (test) :")
print(confusion_matrix(yte, pred_rf))
print("précision %.2f | rappel %.2f | F1 %.2f"
      % (precision_score(yte, pred_rf, zero_division=0),
         recall_score(yte, pred_rf, zero_division=0),
         f1_score(yte, pred_rf, zero_division=0)))"""),
    ("code", """gb = GradientBoostingClassifier(n_estimators=300, random_state=SEED).fit(Xtr, ytr)
proba_gb = gb.predict_proba(Xte)[:, 1]
pred_gb = (proba_gb >= 0.5).astype(int)
print("Gradient boosting — matrice de confusion (test) :")
print(confusion_matrix(yte, pred_gb))
print("précision %.2f | rappel %.2f | F1 %.3f"
      % (precision_score(yte, pred_gb, zero_division=0),
         recall_score(yte, pred_gb, zero_division=0),
         f1_score(yte, pred_gb, zero_division=0)))

lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
proba_lr = lr.predict_proba(Xte)[:, 1]
pred_lr = (proba_lr >= 0.5).astype(int)
print("Régression logistique — précision %.3f | rappel %.3f | F1 %.3f"
      % (precision_score(yte, pred_lr, zero_division=0),
         recall_score(yte, pred_lr, zero_division=0),
         f1_score(yte, pred_lr, zero_division=0)))"""),
    ("md", "**Livrable B3** — les 4 cadres, commentés en coût métier :\n\n"
           "- **Forêt balanced** : 7 alertes, **toutes fausses** (FP = 7, TP = 0) — "
           "7 nuits d'ingénieur gaspillées et **64 incidents manqués** (FN) : le "
           "seuil 0,50 décalé par `balanced` rend le modèle inutilisable tel quel.\n"
           "- **Gradient boosting** : TP = 40, FP = 0, FN = 24 — **précision 1,00, "
           "rappel 0,625, F1 0,769** : chaque alerte est vraie, mais 24 incidents "
           "sur 64 passent encore.\n"
           "- Coût métier : un FN = un SLA qui tombe sans prévenir ; un FP = une "
           "nuit d'astreinte perdue. Avec 0,35 % de positifs, la précision vaut "
           "l'or — d'où le choix du boosting, à calibrer au B4."),

    # ---------------- B4 ----------------
    ("md", "## Étape B4 — Courbe PR et seuil métier sous budget NOC (30 min)\n\n"
           "Budget : ≤ **5 alertes/jour/serveur** — test = 9 jours × 8 serveurs, "
           "donc au plus 5 × 8 × 9 = **360 alertes** sur tout le test."),
    ("code", """from sklearn.metrics import precision_recall_curve, average_precision_score

fig, ax = plt.subplots(figsize=(6.8, 4))
for nom, proba in [("Gradient boosting", proba_gb),
                   ("Régression logistique", proba_lr),
                   ("Forêt balanced", rf.predict_proba(Xte)[:, 1])]:
    prec, rec, thr = precision_recall_curve(yte, proba)
    ap = average_precision_score(yte, proba)
    ax.plot(rec, prec, label=f"{nom} (AP = {ap:.2f})")
ax.axvline(0.625, color="k", ls=":", lw=1, label="rappel du corrigé (0,625)")
ax.set_xlabel("rappel"); ax.set_ylabel("précision")
ax.set_title("Courbes précision–rappel (test temporel)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

print("AP gradient boosting : %.3f" % average_precision_score(yte, proba_gb))"""),
    ("code", """JOURS, SERVEURS = 9, 8
def alertes_jour_serveur(s, proba):
    return (proba >= s).sum() / JOURS / SERVEURS

rows = []
for s in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    p = (proba_gb >= s).astype(int)
    rows.append((s, alertes_jour_serveur(s, proba_gb),
                 precision_score(yte, p, zero_division=0),
                 recall_score(yte, p, zero_division=0)))
tab = pd.DataFrame(rows, columns=["seuil", "alertes/jour/serveur", "précision", "rappel"])
display(tab)

# seuil recommandé : le plus élevé qui garde le meilleur rappel sous budget
ok = tab[tab["alertes/jour/serveur"] <= 5]
seuil_choisi = ok.loc[ok["rappel"].idxmax(), "seuil"]
n_alertes = int((proba_gb >= seuil_choisi).sum())
print("seuil choisi : %.2f -> %d alertes = %.2f/jour/serveur"
      % (seuil_choisi, n_alertes, n_alertes / JOURS / SERVEURS))"""),
    ("md", "**Livrable B4** — la courbe PR du gradient boosting domine nettement "
           "(AP ≈ 0,63). Le tableau seuil → (alertes/jour/serveur, précision, rappel) "
           "montre que le budget NOC (≤ 5/jour/serveur) est respecté avec une énorme "
           "marge : **40 alertes sur 9 jours = 0,56/jour/serveur**, précision "
           "**1,00**, rappel **0,625**. Le seuil recommandé est le plus strict qui "
           "conserve ce rappel : chaque alerte est vraie, le NOC n'est jamais "
           "dérangée pour rien, et 40 des 64 incidents sont anticipés de 10 à 30 min."),

    # ---------------- B5 ----------------
    ("md", "## Étape B5 — Tableau comparatif et décision de déploiement\n\n"
           "Comparaison finale (test temporel) + validation par **CV temporelle** "
           "(TimeSeriesSplit sur le train — jamais de KFold mélangé)."),
    ("code", """from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score as ap_score

def evaluer(nom, pred, proba):
    return {"modèle": nom,
            "précision": round(precision_score(yte, pred, zero_division=0), 3),
            "rappel": round(recall_score(yte, pred, zero_division=0), 3),
            "F1": round(f1_score(yte, pred, zero_division=0), 3),
            "AP": round(ap_score(yte, proba), 3),
            "alertes/j/srv": round((pred == 1).sum() / JOURS / SERVEURS, 2)}

tscv = TimeSeriesSplit(n_splits=3)
def cv_f1(modele):
    scores = []
    for tr_i, va_i in tscv.split(Xtr):
        m = modele.__class__(**{**modele.get_params()})
        if "n_jobs" in m.get_params() and m.get_params()["n_jobs"] is not None \
                and m.__class__.__name__ != "LogisticRegression":
            m.set_params(n_jobs=2)
        m.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        pv = (m.predict_proba(Xtr.iloc[va_i])[:, 1] >= 0.5).astype(int)
        scores.append(f1_score(ytr.iloc[va_i], pv, zero_division=0))
    return np.mean(scores), np.std(scores)

tab_cmp = pd.DataFrame([
    evaluer("IsolationForest (repère, non supervisé)",
            f.loc[~train, "anomalie"].astype(int).values, f.loc[~train, "score_anomalie"]),
    evaluer("Forêt balanced (seuil 0,5)", pred_rf, rf.predict_proba(Xte)[:, 1]),
    evaluer("Régression logistique (seuil 0,5)", pred_lr, proba_lr),
    evaluer("Gradient boosting (seuil 0,5)", pred_gb, proba_gb),
])
display(tab_cmp)

for nom, mod in [("Forêt balanced", rf), ("Gradient boosting", gb),
                 ("Régression logistique", lr)]:
    mu, sd = cv_f1(mod)
    print("CV temporelle F1 — %s : %.3f ± %.3f" % (nom, mu, sd))"""),
    ("md", "**Livrable B5 — décision de déploiement.**\n\n"
           "On déploie le **gradient boosting au seuil choisi au B4** : meilleure "
           "AP du panel (≈ 0,63), **précision 1,00** au seuil retenu (zéro fausse "
           "alerte — decisive pour l'adhésion du NOC), rappel **0,625**, budget "
           "largement respecté (0,56 alerte/jour/serveur contre 5 autorisées). "
           "La forêt *balanced* au seuil 0,5 ne détecte **rien** (précision 0) : "
           "elle resterait exploitable uniquement avec un seuil recalibré. "
           "Sur ce jeu, la régression logistique au seuil 0,5 atteint elle aussi "
           "précision 1,00 / rappel 0,625 (le corrigé officiel, avec d'autres choix "
           "de régularisation, obtenait 328 FP) ; la CV temporelle la confirme ici, "
           "mais on retient le **gradient boosting**, plus robuste aux interactions "
           "non linéaires entre signaux et confirmé par la CV (F1 ≈ 0,69).\n\n"
           "**Ce que l'on n'a PAS détecté** : 24 incidents sur 64 (rappel 0,625), "
           "dont vraisemblablement ceux qui montent trop vite ou trop discrètement "
           "pour laisser une signature dans les signaux de 10 min. Trois parades, "
           "par ordre de coût : (1) réduire la fenêtre de prédiction (prédire à "
           "t+10 au lieu de t+30) pour rattraper les montées rapides ; (2) enrichir "
           "les signaux (variabilité intra-fenêtre, latence par service) ; "
           "(3) garder l'IsolationForest en filet non supervisé, accepté à "
           "faible précision, pour les phénomènes hors distribution. "
           "Reproductibilité : graine **5534** partout, split temporel "
           "**train < 22/09/2026 (24 192) / test ≥ 22/09 (10 368)** cité et appliqué."),
    ("code", """# reproductibilité : chiffres clés vs corrigé officiel (figures/key_numbers_git3.json)
attendu = {"GB précision": 1.0, "GB rappel": 0.625, "GB F1": 0.769,
           "AP GB": 0.629, "anomalies test (iso)": 174,
           "vrais pré-incidents parmi anomalies": 40,
           "alertes au seuil": 40, "alertes/jour/serveur": 0.56}
obtenu = {
    "GB précision": round(precision_score(yte, pred_gb, zero_division=0), 3),
    "GB rappel": round(recall_score(yte, pred_gb, zero_division=0), 3),
    "GB F1": round(f1_score(yte, pred_gb, zero_division=0), 3),
    "AP GB": round(average_precision_score(yte, proba_gb), 3),
    "anomalies test (iso)": n_anom,
    "vrais pré-incidents parmi anomalies": n_vrais,
    "alertes au seuil": n_alertes,
    "alertes/jour/serveur": round(n_alertes / JOURS / SERVEURS, 2),
}
print(pd.DataFrame({"attendu": attendu, "obtenu": obtenu}).to_string())"""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for kind, src in CELLS:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                        else nbf.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": "SDIA Bayes",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    client = NotebookClient(nb, timeout=600, kernel_name=KERNEL,
                            resources={"metadata": {"path": ROOT}},
                            allow_errors=False)
    client.execute()
    name = "TP3_anomalies_classification.ipynb"
    path = os.path.join(ROOT, "notebooks", name)
    nbf.write(nb, path)
    imgs = sum(1 for c in nb.cells for o in c.get("outputs", [])
               if "data" in o and "image/png" in o.get("data", {}))
    ncode = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"[nb] {name} — {len(nb.cells)} cellules ({ncode} code, {imgs} figures) "
          f"{os.path.getsize(path)/1024:.1f} Ko")


if __name__ == "__main__":
    main()
