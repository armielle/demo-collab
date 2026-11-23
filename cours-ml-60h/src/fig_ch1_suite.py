"""Figures complementaires du chapitre 1 (densification 4 h).

Ajoute g09..g12 et complete figures/key_numbers_ch1.json.
Toutes les valeurs sont calculees ici (graine 4112).
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

from figstyle import C, save, FIGDIR

rng = np.random.default_rng(4112)
JSON = os.path.join(FIGDIR, "key_numbers_ch1.json")
cles = json.load(open(JSON, encoding="utf-8")) if os.path.exists(JSON) else {}

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, LeaveOneOut)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])

# ============================================ g09 courbe d'apprentissage
tailles = [20, 40, 60, 90, 120, 180, 250, 350]
tr_moy, te_moy, te_ec = [], [], []
for n in tailles:
    tr, te = [], []
    for rep in range(8):
        rs = 4112 + rep
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n, test_size=150,
                                             random_state=rs, stratify=y)
        m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
        m.fit(Xtr, ytr)
        tr.append(accuracy_score(ytr, m.predict(Xtr)))
        te.append(accuracy_score(yte, m.predict(Xte)))
    tr_moy.append(np.mean(tr)); te_moy.append(np.mean(te)); te_ec.append(np.std(te))
tr_moy, te_moy, te_ec = map(np.array, (tr_moy, te_moy, te_ec))
cles["courbe_apprentissage"] = {
    "tailles": tailles,
    "train": [round(float(v), 3) for v in tr_moy],
    "test": [round(float(v), 3) for v in te_moy],
    "ecart_final": round(float(tr_moy[-1] - te_moy[-1]), 3)}

fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.plot(tailles, tr_moy, "o-", color=C["blue"], lw=1.8, ms=4, label="entraînement")
ax.plot(tailles, te_moy, "s-", color=C["red"], lw=1.8, ms=4, label="test")
ax.fill_between(tailles, te_moy - te_ec, te_moy + te_ec, color=C["red"], alpha=.15)
ax.annotate("l'écart se referme\nquand les données arrivent",
            (250, (tr_moy[6] + te_moy[6]) / 2), xytext=(150, .72), fontsize=9,
            color=C["grey"], arrowprops=dict(arrowstyle="->", color=C["grey"]))
ax.set_xlabel("taille du jeu d'entraînement (patients)")
ax.set_ylabel("exactitude")
ax.set_title("Courbe d'apprentissage : plus de données réduit le surapprentissage")
ax.set_ylim(.55, 1.02); ax.legend(loc="lower right")
save(fig, "g09_courbe_apprentissage.png")

# ============================================ g10 variabilite des estimations
def estimation(mode, graine, reps):
    rs = np.random.RandomState(graine)
    sc = []
    for _ in range(reps):
        if mode == "holdout":
            idx = rs.permutation(len(y))
            ntr = int(.7 * len(y))
            tr, te = idx[:ntr], idx[ntr:]
            m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
            m.fit(X[tr], y[tr]); sc.append(accuracy_score(y[te], m.predict(X[te])))
        else:
            cv = {"cv5": StratifiedKFold(5, shuffle=True, random_state=rs.randint(1e6)),
                  "cv10": StratifiedKFold(10, shuffle=True, random_state=rs.randint(1e6)),
                  "loo": LeaveOneOut()}[mode]
            sc.append(cross_val_score(Pipeline([("sc", StandardScaler()),
                                                ("clf", LogisticRegression(max_iter=1000))]),
                                      X, y, cv=cv).mean())
    return np.array(sc)

res = {}
REPS = {"holdout": 200, "cv5": 100, "cv10": 60, "loo": 8}
for mode, lbl in [("holdout", "hold-out 70/30"), ("cv5", "validation croisée 5"),
                  ("cv10", "validation croisée 10"), ("loo", "leave-one-out")]:
    res[mode] = estimation(mode, 4112, REPS[mode])
cles["variabilite_estimation"] = {
    lbl: {"moyenne": round(float(res[mode].mean()), 4),
          "ecart_type": round(float(res[mode].std()), 4),
          "min": round(float(res[mode].min()), 3), "max": round(float(res[mode].max()), 3)}
    for mode, lbl in [("holdout", "hold-out 70/30"), ("cv5", "validation croisée 5"),
                      ("cv10", "validation croisée 10"), ("loo", "leave-one-out")]}

fig, ax = plt.subplots(figsize=(8.6, 4.0))
labels = ["hold-out\n70/30", "CV 5 plis", "CV 10 plis", "leave-one-out"]
data = [res["holdout"], res["cv5"], res["cv10"], res["loo"]]
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=.55, showfliers=False)
for patch, col in zip(bp["boxes"], [C["red"], C["blue"], C["cyan"], C["green"]]):
    patch.set_facecolor(col); patch.set_alpha(.30)
for med in bp["medians"]:
    med.set_color(C["deep"]); med.set_linewidth(1.6)
ax.set_ylabel("exactitude estimée (répétitions)")
ax.set_title("La même donnée, quatre estimations : la variance change")
save(fig, "g10_variabilite.png")

# ============================================ g11 desequilibre
n = 2000
yimb = np.r_[np.zeros(int(.95 * n)), np.ones(int(.05 * n))]
pred_tout0 = np.zeros(n)
acc0 = accuracy_score(yimb, pred_tout0)
rec0 = recall_score(yimb, pred_tout0, zero_division=0)

# modele "equilibre" : 90 % d'exactitude, 80 % de rappel sur la classe rare
tp, fn = 80, 20
fp, tn = 180, 1720
ytrue = np.r_[np.ones(tp + fn), np.zeros(fp + tn)]
ypred = np.r_[np.ones(tp), np.zeros(fn), np.ones(fp), np.zeros(tn)]
acc1 = accuracy_score(ytrue, ypred); rec1 = recall_score(ytrue, ypred)
cles["desequilibre"] = {"prevalence": .05,
                        "tout_majoritaire": {"exactitude": round(float(acc0), 3),
                                             "rappel": round(float(rec0), 3)},
                        "modele_utile": {"exactitude": round(float(acc1), 3),
                                         "rappel": round(float(rec1), 3)}}

fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(2)
ax.bar(x - .19, [acc0, acc1], width=.36, color=C["grey"], alpha=.85, label="exactitude")
ax.bar(x + .19, [rec0, rec1], width=.36, color=C["red"], alpha=.85, label="rappel (classe rare)")
for i, (a, r) in enumerate([(acc0, rec0), (acc1, rec1)]):
    ax.text(i - .19, a + .02, f"{a:.0%}", ha="center", fontsize=10, fontweight="bold")
    ax.text(i + .19, r + .02, f"{r:.0%}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["modèle qui prédit\ntoujours « sain »", "modèle réellement\nutile"])
ax.set_ylim(0, 1.05); ax.set_ylabel("score")
ax.set_title("Prévalence 5 % : 95 % d'exactitude et zéro cas détecté")
ax.legend(loc="upper right")
save(fig, "g11_desequilibre.png")

# ============================================ g12 descente de gradient
def f(w):
    return (w - 3) ** 2 + 1

def grad(w):
    return 2 * (w - 3)

fig, ax = plt.subplots(figsize=(8.4, 3.9))
ws = np.linspace(-4, 10, 400)
ax.plot(ws, f(ws), color=C["deep"], lw=2)
traj = {}
for lr, col in [(0.02, C["blue"]), (0.15, C["green"]), (1.05, C["red"])]:
    w, chemin = -3.2, [-3.2]
    for _ in range(12 if lr < 1 else 6):
        w = w - lr * grad(w)
        chemin.append(w)
    traj[f"{lr}"] = [round(float(v), 3) for v in chemin]
    ax.plot(chemin, [f(v) for v in chemin], "o-", color=col, ms=3.5, lw=1.3,
            label=f"pas = {lr}")
cles["descente_gradient"] = {k: v[:5] for k, v in traj.items()}
ax.set_xlabel("paramètre w"); ax.set_ylabel("perte J(w)")
ax.set_title("Descente de gradient : trop petit, correct, divergent")
ax.set_ylim(0, 60); ax.legend()
save(fig, "g12_descente_gradient.png")

with open(JSON, "w", encoding="utf-8") as fjson:
    json.dump(cles, fjson, ensure_ascii=False, indent=2)

print("\n--- chiffres ajoutes ---")
print("apprentissage, ecart final :", cles["courbe_apprentissage"]["ecart_final"])
print("variabilite :", json.dumps(cles["variabilite_estimation"], ensure_ascii=False))
print("desequilibre :", json.dumps(cles["desequilibre"], ensure_ascii=False))
print("gradient (5 premiers points) :", json.dumps(cles["descente_gradient"], ensure_ascii=False))
