"""Figures du chapitre 4 -- Classification et evaluation (Licence 3 SDIA)."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

from figstyle import C, save, FIGDIR

rng = np.random.default_rng(4112)
cles = {}

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve

# ==================================================== l01 : sigmoide
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
z = np.linspace(-8, 8, 400)
axes[0].plot(z, 1 / (1 + np.exp(-z)), color=C["blue"], lw=2.2)
axes[0].axhline(.5, color=C["grey"], ls="--", lw=1)
axes[0].axvline(0, color=C["grey"], ls="--", lw=1)
axes[0].set_xlabel("z = $\\beta_0 + x^T\\beta$"); axes[0].set_ylabel("probabilité")
axes[0].set_title("Sigmoïde : de $\\mathbb{R}$ vers $[0,1]$")
p = np.linspace(.01, .99, 200)
axes[1].plot(p, np.log(p / (1 - p)), color=C["orange"], lw=2.2)
axes[1].set_xlabel("probabilité"); axes[1].set_ylabel("log-odds")
axes[1].set_title("Log-odds : la droite que le modèle ajuste")
save(fig, "l01_sigmoide.png")

# ==================================================== donnees
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=4112, stratify=y)

modele = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
proba = modele.predict_proba(Xte)[:, 1]
pred = (proba >= .5).astype(int)

acc = float(accuracy_score(yte, pred)); prec = float(precision_score(yte, pred))
rec = float(recall_score(yte, pred)); f1 = float(f1_score(yte, pred))
cm = confusion_matrix(yte, pred)
fpr, tpr, seuils = roc_curve(yte, proba)
auc_val = float(auc(fpr, tpr))
cl = {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
      "f1": round(f1, 4), "auc": round(auc_val, 4),
      "matrice": cm.tolist()}
cles["logistique"] = cl

# ==================================================== l02 : frontiere 2D
X2 = np.r_[rng.normal((1.6, 1.6), .75, (80, 2)), rng.normal((3.4, 3.4), .75, (80, 2))]
y2 = np.r_[np.zeros(80), np.ones(80)]
m2 = LogisticRegression().fit(X2, y2)
xx, yy = np.meshgrid(np.linspace(-1, 6, 300), np.linspace(-1, 6, 300))
grille = np.c_[xx.ravel(), yy.ravel()]
zz = m2.predict_proba(grille)[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
cf = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=.55)
ax.contour(xx, yy, zz, levels=[.5], colors="k", linewidths=1.6)
ax.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], s=16, color=C["blue"], edgecolor="white", lw=.4)
ax.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], s=16, color=C["red"], edgecolor="white", lw=.4)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("Frontière linéaire et probabilité estimée")
plt.colorbar(cf, ax=ax, fraction=.046, label="P(y = 1)")
save(fig, "l02_frontiere.png")

# ==================================================== l03 : matrice de confusion
fig, ax = plt.subplots(figsize=(5.2, 4.0))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else C["deep"], fontweight="bold")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["prédit 0", "prédit 1"]); ax.set_yticklabels(["réel 0", "réel 1"])
ax.set_title(f"Matrice de confusion (n = {len(yte)})")
ax.grid(False)
save(fig, "l03_matrice.png")

# ==================================================== l04 : effet du seuil
seuils_e = np.linspace(.05, .95, 60)
precs, recs, f1s = [], [], []
for s in seuils_e:
    p_ = (proba >= s).astype(int)
    precs.append(precision_score(yte, p_, zero_division=0))
    recs.append(recall_score(yte, p_))
    f1s.append(f1_score(yte, p_, zero_division=0))
best_f1 = int(np.argmax(f1s))
cles["seuil"] = {"seuil_optimal_f1": round(float(seuils_e[best_f1]), 3),
                 "f1_max": round(float(f1s[best_f1]), 3),
                 "precision_a_0_5": round(float(precs[np.argmin(np.abs(seuils_e - .5))]), 3),
                 "rappel_a_0_5": round(float(recs[np.argmin(np.abs(seuils_e - .5))]), 3)}

fig, ax = plt.subplots(figsize=(8.2, 3.8))
ax.plot(seuils_e, precs, color=C["blue"], lw=2, label="précision")
ax.plot(seuils_e, recs, color=C["orange"], lw=2, label="rappel")
ax.plot(seuils_e, f1s, color=C["green"], lw=2, ls="--", label="$F_1$")
ax.axvline(.5, color=C["grey"], ls=":", lw=1.2)
ax.axvline(seuils_e[best_f1], color=C["red"], ls="--", lw=1)
ax.set_xlabel("seuil de décision"); ax.set_ylabel("score")
ax.set_title("Le seuil déplace le compromis précision / rappel")
ax.legend(loc="lower right")
save(fig, "l04_seuil.png")

# ==================================================== l05 : ROC
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.plot(fpr, tpr, color=C["blue"], lw=2.2, label=f"AUC = {auc_val:.3f}")
ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1, label="hasard")
ax.set_xlabel("taux de faux positifs"); ax.set_ylabel("taux de vrais positifs")
ax.set_title("Courbe ROC")
ax.legend(loc="lower right")
save(fig, "l05_roc.png")

# ==================================================== l06 : precision-rappel (desequilibre)
n0, n1 = 1900, 100
Xd = rng.normal(size=(n0 + n1, 2))
Xd[n0:] += 1.1
yd = np.r_[np.zeros(n0), np.ones(n1)]
Xtr_d, Xte_d, ytr_d, yte_d = train_test_split(Xd, yd, test_size=.3, random_state=4112, stratify=yd)

m_d = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight=None))]).fit(Xtr_d, ytr_d)
p_d = m_d.predict_proba(Xte_d)[:, 1]
pred_d = (p_d >= .5).astype(int)
pr_d, rc_d, _ = precision_recall_curve(yte_d, p_d)
ap_d = float(average_precision_score(yte_d, p_d))

acc_d = float(accuracy_score(yte_d, pred_d)); f1_d = float(f1_score(yte_d, pred_d))
# baseline : tout negatif
acc_base = float((yte_d == 0).mean())
cles["desequilibre"] = {"prevalence": round(float(yd.mean()), 3),
                        "accuracy_modele": round(acc_d, 4),
                        "f1_modele": round(f1_d, 4),
                        "accuracy_tout_negatif": round(acc_base, 4),
                        "average_precision": round(ap_d, 4)}

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(rc_d, pr_d, color=C["red"], lw=2.2, label=f"AP = {ap_d:.3f}")
ax.axhline(yte_d.mean(), color=C["grey"], ls="--", lw=1, label=f"hasard = {yte_d.mean():.2f}")
ax.set_xlabel("rappel"); ax.set_ylabel("précision")
ax.set_title(f"Déséquilibre ({100*yd.mean():.0f} % de positifs) : la courbe PR est le bon outil")
ax.legend()
save(fig, "l06_pr.png")

# ==================================================== l07 : accuracy vs f1 avec ponderation
m_bal = Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(Xtr_d, ytr_d)
pred_bal = m_bal.predict(Xte_d)
acc_b = float(accuracy_score(yte_d, pred_bal)); f1_b = float(f1_score(yte_d, pred_bal))
rec_b = float(recall_score(yte_d, pred_bal))
cles["ponderation"] = {"sans_poids": {"accuracy": round(acc_d, 3), "f1": round(f1_d, 3),
                                      "rappel": round(float(recall_score(yte_d, pred_d)), 3)},
                       "avec_poids": {"accuracy": round(acc_b, 3), "f1": round(f1_b, 3),
                                      "rappel": round(rec_b, 3)}}

fig, ax = plt.subplots(figsize=(7.4, 3.5))
x = np.arange(3); w = .35
sans = [acc_d, f1_d, recall_score(yte_d, pred_d)]
avec = [acc_b, f1_b, rec_b]
ax.bar(x - w/2, sans, w, color=C["grey"], alpha=.85, label="sans pondération")
ax.bar(x + w/2, avec, w, color=C["green"], alpha=.85, label="class_weight='balanced'")
for i, (a, b) in enumerate(zip(sans, avec)):
    ax.text(i - w/2, a + .015, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, b + .015, f"{b:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["exactitude", "$F_1$", "rappel"])
ax.set_ylim(0, 1.08); ax.set_title("La pondération sacrifie l'exactitude pour le rappel")
ax.legend(fontsize=8)
save(fig, "l07_ponderation.png")

# ==================================================== l08 : calibration
frac_pos, moy_pred = calibration_curve(yte, proba, n_bins=6, strategy="quantile")
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1, label="calibration parfaite")
ax.plot(moy_pred, frac_pos, "o-", color=C["blue"], lw=2, label="modèle")
ax.set_xlabel("probabilité annoncée"); ax.set_ylabel("fréquence observée")
ax.set_title("Diagramme de fiabilité")
ax.legend(loc="upper left")
save(fig, "l08_calibration.png")
cles["calibration"] = {"ecart_max": round(float(np.max(np.abs(frac_pos - moy_pred))), 3)}

# ==================================================== l09 : multiclasse
from sklearn.datasets import load_digits
Xd_, yd_ = load_digits(return_X_y=True)
garder = np.isin(yd_, [3, 8, 9])
Xw, yw = Xd_[garder], yd_[garder]
yw = np.searchsorted([3, 8, 9], yw)      # 0, 1, 2
Xtr_w, Xte_w, ytr_w, yte_w = train_test_split(Xw, yw, test_size=.3, random_state=4112, stratify=yw)
mw = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=3000))]).fit(Xtr_w, ytr_w)
pred_w = mw.predict(Xte_w)
cm_w = confusion_matrix(yte_w, pred_w)
rappel_par_classe = [round(float(recall_score(yte_w, pred_w, labels=[k], average=None)[0]), 3)
                     for k in range(3)]
cles["multiclasse"] = {"accuracy": round(float(accuracy_score(yte_w, pred_w)), 3),
                       "rappel_par_classe": rappel_par_classe,
                       "matrice": cm_w.tolist()}

fig, ax = plt.subplots(figsize=(5.4, 4.2))
im = ax.imshow(cm_w, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm_w[i, j]}", ha="center", va="center", fontsize=15,
                color="white" if cm_w[i, j] > cm_w.max()/2 else C["deep"], fontweight="bold")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(["3", "8", "9"]); ax.set_yticklabels(["3", "8", "9"])
ax.set_xlabel("prédit"); ax.set_ylabel("réel")
ax.set_title(f"Chiffres 3 / 8 / 9 (n = {len(yte_w)}), exactitude {accuracy_score(yte_w, pred_w):.2f}")
ax.grid(False)
save(fig, "l09_multiclasse.png")

with open(os.path.join(FIGDIR, "key_numbers_ch4.json"), "w", encoding="utf-8") as f:
    json.dump(cles, f, ensure_ascii=False, indent=2)

print("\n--- chiffres cles ---")
for k, v in cles.items():
    print(k, ":", v)
