"""Figures du chapitre 1 -- Introduction au Machine Learning.

Toutes les valeurs sont calculees ici (graine 4112) et ecrites dans
figures/key_numbers_ch1.json. Aucun chiffre saisi a la main.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

from figstyle import C, save, FIGDIR

rng = np.random.default_rng(4112)
cles = {}

# ==================================================== 1. pipeline
fig, ax = plt.subplots(figsize=(11, 2.3))
ax.set_xlim(0, 11); ax.set_ylim(0, 2.3); ax.axis("off")
etapes = [("Données", "brutes"), ("Préparation", "nettoyage, features"),
          ("Modèle", "entraînement"), ("Évaluation", "test, métriques"),
          ("Décision", "seuil, action")]
for i, (t1, t2) in enumerate(etapes):
    x = .35 + i * 2.18
    ax.add_patch(FancyBboxPatch((x, .75), 1.75, .85, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=C["blue"] if i < 4 else C["green"], ec="none", alpha=.14))
    ax.add_patch(FancyBboxPatch((x, .75), 1.75, .85, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc="none", ec=C["blue"] if i < 4 else C["green"], lw=1.4))
    ax.text(x + .875, 1.32, t1, ha="center", va="center", fontsize=11, fontweight="bold",
            color=C["deep"])
    ax.text(x + .875, 1.0, t2, ha="center", va="center", fontsize=8, color=C["grey"])
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + 1.83, 1.17), (x + 2.12, 1.17),
                                     arrowstyle="-|>", mutation_scale=14, color=C["grey"], lw=1.4))
ax.add_patch(FancyArrowPatch((10.1, .70), (10.1, .30), arrowstyle="-", color=C["orange"], lw=1.2))
ax.add_patch(FancyArrowPatch((10.1, .30), (1.2, .30), arrowstyle="-", color=C["orange"], lw=1.2))
ax.add_patch(FancyArrowPatch((1.2, .30), (1.2, .70), arrowstyle="-|>", mutation_scale=14,
                             color=C["orange"], lw=1.2))
ax.text(5.6, .16, "réentraîner : nouvelles données, nouveau seuil, nouvelle métrique",
        ha="center", fontsize=8.5, color=C["orange"], style="italic")
save(fig, "g01_pipeline.png")
cles["pipeline_etapes"] = [e[0] for e in etapes]

# ==================================================== 2. trois apprentissages
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

# supervise
ax = axes[0]
X = np.r_[rng.normal((1.6, 1.6), .55, (40, 2)), rng.normal((3.4, 3.4), .55, (40, 2))]
y = np.r_[np.zeros(40), np.ones(40)]
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, color=C["blue"], alpha=.8, label="classe 0")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, color=C["red"], alpha=.8, label="classe 1")
xx = np.linspace(0, 5, 10)
ax.plot(xx, 5 - xx, color=C["deep"], lw=2, ls="--")
ax.set_title("Supervisé\non connaît les étiquettes")
ax.legend(fontsize=8, loc="upper right"); ax.set_xticks([]); ax.set_yticks([])

# non supervise
ax = axes[1]
centres = [(1.5, 3.4), (3.4, 3.6), (2.4, 1.4)]
for k, cen in enumerate(centres):
    P = rng.normal(cen, .42, (45, 2))
    ax.scatter(P[:, 0], P[:, 1], s=18, alpha=.75, color=[C["blue"], C["cyan"], C["orange"]][k])
ax.set_title("Non supervisé\naucune étiquette")
ax.set_xticks([]); ax.set_yticks([])

# renforcement
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.add_patch(FancyBboxPatch((1.2, 6.0), 3.0, 1.6, boxstyle="round,pad=.08,rounding_size=.15",
                            fc=C["purple"], ec="none", alpha=.15))
ax.text(2.7, 6.8, "Agent", ha="center", fontsize=11, fontweight="bold")
ax.add_patch(FancyBboxPatch((5.6, 6.0), 3.2, 1.6, boxstyle="round,pad=.08,rounding_size=.15",
                            fc=C["green"], ec="none", alpha=.15))
ax.text(7.2, 6.8, "Environnement", ha="center", fontsize=10, fontweight="bold")
ax.add_patch(FancyArrowPatch((4.25, 7.2), (5.55, 7.2), arrowstyle="-|>", mutation_scale=14,
                             color=C["blue"], lw=1.6, connectionstyle="arc3,rad=-.25"))
ax.text(4.9, 7.9, "action", ha="center", fontsize=9, color=C["blue"])
ax.add_patch(FancyArrowPatch((5.55, 6.4), (4.25, 6.4), arrowstyle="-|>", mutation_scale=14,
                             color=C["orange"], lw=1.6, connectionstyle="arc3,rad=-.25"))
ax.text(4.9, 5.6, "état, récompense", ha="center", fontsize=9, color=C["orange"])
ax.set_title("Renforcé\non apprend par essai-erreur")
save(fig, "g02_types.png")

# ==================================================== 3. surapprentissage
degres = np.arange(1, 15)
n_tr, n_te = 160, 200
x_tr = np.sort(rng.uniform(0, 1, n_tr))
y_tr = np.sin(2 * np.pi * x_tr) + rng.normal(0, .22, n_tr)
x_te = np.sort(rng.uniform(0, 1, n_te))
y_te = np.sin(2 * np.pi * x_te) + rng.normal(0, .22, n_te)
err_tr, err_te = [], []
for d in degres:
    coef = np.polyfit(x_tr, y_tr, d)
    err_tr.append(np.mean((np.polyval(coef, x_tr) - y_tr) ** 2))
    err_te.append(np.mean((np.polyval(coef, x_te) - y_te) ** 2))
err_tr, err_te = np.array(err_tr), np.array(err_te)
best = int(degres[np.argmin(err_te)])
cles["surapprentissage"] = {"degre_optimal_test": best,
                            "erreur_train_min": round(float(err_tr.min()), 4),
                            "erreur_test_min": round(float(err_te.min()), 4),
                            "erreur_test_deg14": round(float(err_te[-1]), 4),
                            "ratio_train_test_deg14": round(float(err_tr[-1] / err_te[-1]), 6)}

fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.semilogy(degres, err_tr, "o-", color=C["blue"], lw=1.8, ms=4, label="erreur d'entraînement")
ax.semilogy(degres, err_te, "s-", color=C["red"], lw=1.8, ms=4, label="erreur de test")
ax.axvline(best, color=C["green"], ls="--", lw=1.2)
ax.annotate(f"optimum : degré {best}\ntest = {err_te[best-1]:.3f}",
            (best, err_te[best - 1]), xytext=(best + 1.2, err_te[best - 1] * 3),
            fontsize=9, color=C["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["green"], lw=1))
ax.set_xlabel("complexité du modèle (degré du polynôme)")
ax.set_ylabel("erreur quadratique moyenne (échelle log)")
ax.set_title("Sous-apprentissage, bon compromis, surapprentissage")
ax.legend(loc="upper center")
save(fig, "g03_surapprentissage.png")

# ==================================================== 4. biais-variance
def vraie(x):
    return np.sin(2 * np.pi * x)


x_grid = np.linspace(.05, .95, 60)
R = 120
preds = {d: [] for d in (1, 3, 9, 15)}
for _ in range(R):
    xb = np.sort(rng.uniform(0, 1, 35))
    yb = vraie(xb) + rng.normal(0, .25, 35)
    for d in preds:
        preds[d].append(np.polyval(np.polyfit(xb, yb, d), x_grid))
bv = {}
for d, P in preds.items():
    P = np.array(P)
    moy = P.mean(0)
    biais2 = float(np.mean((moy - vraie(x_grid)) ** 2))
    var = float(np.mean(P.var(0)))
    bv[d] = {"biais2": round(biais2, 4), "variance": round(var, 4),
             "erreur_totale": round(biais2 + var, 4)}
cles["biais_variance"] = bv

fig, ax = plt.subplots(figsize=(8.6, 4.0))
ds = list(bv)
ax.set_yscale("log")
ax.bar(np.arange(len(ds)) - .22, [max(bv[d]["biais2"], 1e-4) for d in ds], width=.42,
       color=C["blue"], label="biais²")
ax.bar(np.arange(len(ds)) + .22, [max(bv[d]["variance"], 1e-4) for d in ds], width=.42,
       color=C["orange"], label="variance")
ax.plot(np.arange(len(ds)), [bv[d]["erreur_totale"] for d in ds], "o-", color=C["red"],
        lw=2, ms=6, label="erreur totale")
ax.set_xticks(np.arange(len(ds)))
ax.set_xticklabels([f"degré {d}" for d in ds])
ax.set_ylabel("erreur (échelle log, 60 points de test)")
ax.set_title("Décomposition biais-variance (120 rééchantillonnages)")
ax.legend()
save(fig, "g04_biais_variance.png")

# ==================================================== 5. decoupage
fig, ax = plt.subplots(figsize=(10, 1.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 3); ax.axis("off")
parts = [(0, 60, "Entraînement", "60 % — apprendre les paramètres", C["blue"]),
         (60, 20, "Validation", "20 % — choisir", C["orange"]),
         (80, 20, "Test", "20 % — mesurer une fois", C["green"])]
for x0, w, t1, t2, col in parts:
    ax.add_patch(Rectangle((x0, 1.1), w, .95, fc=col, ec="white", lw=2, alpha=.85))
    ax.text(x0 + w / 2, 1.72, t1, ha="center", color="white", fontsize=10.5, fontweight="bold")
    ax.text(x0 + w / 2, 1.34, t2, ha="center", color="white", fontsize=7.5)
ax.annotate("", (0, .8), (60, .8), arrowprops=dict(arrowstyle="<->", color=C["grey"]))
ax.text(30, .45, "on ajuste", ha="center", fontsize=8.5, color=C["grey"])
ax.annotate("", (60, .8), (80, .8), arrowprops=dict(arrowstyle="<->", color=C["grey"]))
ax.text(70, .45, "on choisit", ha="center", fontsize=8.5, color=C["grey"])
ax.annotate("", (80, .8), (100, .8), arrowprops=dict(arrowstyle="<->", color=C["grey"]))
ax.text(90, .45, "on mesure", ha="center", fontsize=8.5, color=C["grey"])
save(fig, "g05_split.png")
cles["split"] = {"train": 60, "validation": 20, "test": 20}

# ==================================================== 6. k-fold
fig, ax = plt.subplots(figsize=(9.2, 3.4))
ax.set_xlim(-1, 11.2); ax.set_ylim(-.3, 6.1); ax.axis("off")
K = 5
for i in range(K):
    y = 5 - i * 1.05
    for j in range(K):
        col = C["green"] if j == i else C["blue"]
        ax.add_patch(Rectangle((j * 2, y), 1.85, .72, fc=col, ec="white", lw=1.6,
                               alpha=.85 if j == i else .25))
    ax.text(-.25, y + .36, f"pli {i+1}", ha="right", va="center", fontsize=9, color=C["deep"])
    ax.text(10.15, y + .36, f"score {i+1}", ha="left", va="center", fontsize=8.5, color=C["grey"])
ax.text(4.6, -.15, "5 plis : chacun sert une fois de validation",
        ha="center", fontsize=9.5, color=C["grey"], style="italic")
save(fig, "g06_kfold.png")
cles["kfold"] = {"K": K}

# ==================================================== 7. fuite de donnees
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

n, p, k = 90, 2000, 12
Xf = rng.normal(size=(n, p))
yf = rng.integers(0, 2, n)          # etiquette PUREMENT aleatoire : aucune information
cv = StratifiedKFold(5, shuffle=True, random_state=4112)

# (a) fuite : selection des variables sur TOUTES les donnees, puis validation croisee
sel = SelectKBest(f_classif, k=k).fit(Xf, yf)
X_sel = sel.transform(Xf)
score_fuite = float(np.mean(cross_val_score(LogisticRegression(max_iter=1000), X_sel, yf, cv=cv)))

# (b) propre : la selection vit DANS le pipeline, donc dans chaque pli
from sklearn.pipeline import Pipeline
pipe = Pipeline([("sel", SelectKBest(f_classif, k=k)),
                 ("clf", LogisticRegression(max_iter=1000))])
score_propre = float(np.mean(cross_val_score(pipe, Xf, yf, cv=cv)))
cles["fuite_selection"] = {"n": n, "p": p, "k": k,
                           "score_avec_fuite": round(score_fuite, 3),
                           "score_propre": round(score_propre, 3),
                           "ecart_points": round(100 * (score_fuite - score_propre), 1)}

fig, ax = plt.subplots(figsize=(6.6, 3.6))
b = ax.bar(["sélection\navant le split", "sélection\ndans le pipeline"],
           [score_fuite, score_propre], color=[C["red"], C["green"]], alpha=.85, width=.55)
ax.axhline(.5, color=C["grey"], ls="--", lw=1.2)
ax.text(1.42, .515, "hasard = 0,5", fontsize=8.5, color=C["grey"], ha="right")
for bb, v in zip(b, [score_fuite, score_propre]):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .02, f"{v:.1%}", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.0); ax.set_ylabel("exactitude en validation croisée")
ax.set_title(f"Étiquettes aléatoires, {p} variables de bruit :\nla fuite fabrique un score")
save(fig, "g07_fuite.png")

# ==================================================== 8. baseline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

Xi, yi = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=.3, random_state=4112, stratify=yi)
base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
rf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
acc_base = float(accuracy_score(yte, base.predict(Xte)))
acc_rf = float(accuracy_score(yte, rf.predict(Xte)))
cles["baseline_cancer"] = {"baseline": round(acc_base, 3), "foret": round(acc_rf, 3),
                         "gain_points": round(100 * (acc_rf - acc_base), 1)}

fig, ax = plt.subplots(figsize=(6.4, 3.4))
b = ax.bar(["classe majoritaire\n(baseline)", "régression logistique"],
           [acc_base, acc_rf], color=[C["grey"], C["blue"]], alpha=.85, width=.55)
for bb, v in zip(b, [acc_base, acc_rf]):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .015, f"{v:.1%}", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05); ax.set_ylabel("exactitude sur le test")
ax.set_title(f"Sans baseline, {acc_rf:.1%} ne veut rien dire")
save(fig, "g08_baseline.png")

# ==================================================== JSON
with open(os.path.join(FIGDIR, "key_numbers_ch1.json"), "w", encoding="utf-8") as f:
    json.dump(cles, f, ensure_ascii=False, indent=2)

print("\n--- chiffres cles ---")
print("degre optimal :", best, "| test :", round(float(err_te[best-1]), 4),
      "| train :", round(float(err_tr[best-1]), 4))
print("biais-variance :", json.dumps(bv, ensure_ascii=False))
print(f"fuite : {score_fuite:.3f} vs propre {score_propre:.3f} "
      f"(+{100*(score_fuite-score_propre):.1f} points)")
print(f"cancer : baseline {acc_base:.3f} vs logistique {acc_rf:.3f}")
