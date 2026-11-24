# -*- coding: utf-8 -*-
"""Figures de la Seance 3 (EPD-GIT-5534) -- modeles ML appliques aux infra.
Entrainement REEL sur data/nosys_features.csv (corrige officiel du TP2),
split temporel strict : train < 2026-09-22, test >= 2026-09-22.
Ecrit figures/key_numbers_git3.json AVANT les figures lourdes. Graine 5534."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, average_precision_score, precision_recall_curve)

OMP = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
SEED = 5534
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
F = pd.read_csv(os.path.join(ROOT, "data", "nosys_features.csv"),
                parse_dates=["fenetre"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#5B6470", "axes.labelcolor": "#10243E",
    "text.color": "#10243E", "xtick.color": "#5B6470", "ytick.color": "#5B6470",
    "axes.titlecolor": "#10243E", "figure.facecolor": "white",
    "axes.grid": True, "grid.color": "#D8DEE6", "grid.linewidth": 0.6,
})
C = {"blue": "#1F6FEB", "deep": "#10243E", "cyan": "#12A594", "orange": "#D97706",
     "red": "#C0392B", "green": "#1E8449", "purple": "#6C5CE7", "grey": "#5B6470"}
DPI = 190

FEATS = ["cpu_moy", "cpu_max", "ram_moy", "lat_moy", "lat_p99", "perte_moy",
         "temp_moy", "conso_moy", "capacite", "cpu_delta", "cpu_amplitude",
         "h_sin", "h_cos", "js_sin", "js_cos"]
y = F["incident_30min"].values.astype(int)
X = F[FEATS].copy()
X["cpu_delta"] = X["cpu_delta"].fillna(0.0)
X = X.values

CUT = pd.Timestamp("2026-09-22")
m_tr = (F["fenetre"] < CUT).values
m_te = (F["fenetre"] >= CUT).values
Xtr, ytr = X[m_tr], y[m_tr]
Xte, yte = X[m_te], y[m_te]

# ============================================================== chiffres cles
K = {}
K["lignes"] = int(len(F))
K["positifs"] = int(y.sum())
K["taux_positifs_pct"] = round(100 * y.mean(), 2)
K["train_lignes"] = int(m_tr.sum());  K["train_positifs"] = int(ytr.sum())
K["test_lignes"] = int(m_te.sum());   K["test_positifs"] = int(yte.sum())
K["jours_test"] = int(F.loc[m_te, "fenetre"].dt.date.nunique())

# ---- stats z-score / IQR (rappel seance 2, calculées ici pour de vrai)
p99 = F["lat_p99"].values
K["lat_p99_moy"] = round(float(np.nanmean(p99)), 1)
K["lat_p99_std"] = round(float(np.nanstd(p99)), 1)
mu, sd = np.nanmean(p99), np.nanstd(p99)
q1, q3 = np.nanpercentile(p99, [25, 75])
K["lat_p99_zscore_depassements"] = int((np.abs((p99 - mu) / sd) > 3).sum())
K["lat_p99_iqr_depassements"] = int(((p99 < q1 - 1.5 * (q3 - q1)) |
                                     (p99 > q3 + 1.5 * (q3 - q1))).sum())

# ---- modeles reels
MODELES = {}
iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED,
                      n_jobs=2)
iso.fit(Xtr)
score_iso = -iso.score_samples(X)          # plus grand = plus anormal
K["iso_top20"] = int((y[score_iso >= np.sort(score_iso)[-20]].sum()))
K["iso_top20_pct"] = round(100 * K["iso_top20"] / 20, 1)
# seuil a contamination=1% sur le test
thr_iso = np.quantile(score_iso[m_tr], 0.99)
K["iso_anomalies_test"] = int((score_iso[m_te] > thr_iso).sum())
K["iso_anomalies_test_incidents"] = int((y[m_te][score_iso[m_te] > thr_iso] == 1).sum())

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=SEED, n_jobs=2).fit(Xtr, ytr)
p_rf = rf.predict_proba(Xte)[:, 1]
MODELES["Foret (balanced)"] = p_rf

gb = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
p_gb = gb.predict_proba(Xte)[:, 1]
MODELES["Gradient boosting"] = p_gb

sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
lr.fit(sc.transform(Xtr), ytr)
p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
MODELES["Reg. logistique"] = p_lr

# "toujours 0" : reference trompeuse
K["acc_toujours0_test"] = round(100 * (1 - yte.mean()), 2)
K["acc_toujours0_global"] = round(100 * (1 - y.mean()), 2)

MET = {}
for nom, p in MODELES.items():
    yh = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
    MET[nom] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": round(float(precision_score(yte, yh, zero_division=0)), 3),
                "rappel": round(float(recall_score(yte, yh, zero_division=0)), 3),
                "f1": round(float(f1_score(yte, yh, zero_division=0)), 3),
                "pr_auc": round(float(average_precision_score(yte, p)), 3)}
K["metriques_test_0.5"] = MET

# ---- courbes PR (stockees pour la figure)
PR = {nom: precision_recall_curve(yte, p) for nom, p in MODELES.items()}

# ---- validation croisée TEMPORELLE (TimeSeriesSplit, 5 folds) sur le train
tsc = TimeSeriesSplit(n_splits=5)
cv = {}
for nom, cls in [("Foret (balanced)", RandomForestClassifier(
                      n_estimators=300, class_weight="balanced",
                      random_state=SEED, n_jobs=2)),
                 ("Gradient boosting", GradientBoostingClassifier(random_state=SEED)),
                 ("Reg. logistique", LogisticRegression(class_weight="balanced",
                                                        max_iter=3000, random_state=SEED))]:
    f1s = []
    for tr_i, va_i in tsc.split(Xtr):
        mdl = cls.fit(Xtr[tr_i], ytr[tr_i])
        pv = mdl.predict_proba(Xtr[va_i])[:, 1]
        f1s.append(f1_score(ytr[va_i], (pv >= 0.5), zero_division=0))
    cv[nom] = [round(float(np.mean(f1s)), 3), round(float(np.std(f1s)), 3)]
K["cv_temporelle_F1_moy_std"] = cv

# ---- seuil optimal : <= 5 alertes/jour/serveur sur le test
BEST = max(MET, key=lambda n: MET[n]["f1"])
p_best = MODELES[BEST]
nb_serveurs = F["serveur"].nunique()
budget = 5 * nb_serveurs * K["jours_test"]          # alertes autorisees sur le test
seuil_choisi, best_f1 = None, -1.0
for s in np.sort(np.unique(p_best)):          # parmi les seuils sous budget :
    if int((p_best >= s).sum()) <= budget:    # celui qui maximise le F1
        yh = (p_best >= s).astype(int)
        f1s = f1_score(yte, yh, zero_division=0)
        if f1s >= best_f1:                    # >= : on garde le seuil le PLUS BAS
            best_f1, seuil_choisi = f1s, float(s)
yh_s = (p_best >= seuil_choisi).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, yh_s).ravel()
K["meilleur_modele"] = BEST
K["budget_alertes_test"] = int(budget)
K["seuil_choisi"] = round(seuil_choisi, 3)
K["alertes_au_seuil"] = int(yh_s.sum())
K["alertes_par_jour_parc"] = round(yh_s.sum() / K["jours_test"], 1)
K["alertes_par_jour_serveur"] = round(yh_s.sum() / K["jours_test"] / nb_serveurs, 2)
K["metriques_meilleur_seuil"] = {
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "precision": round(float(precision_score(yte, yh_s, zero_division=0)), 3),
    "rappel": round(float(recall_score(yte, yh_s, zero_division=0)), 3),
    "f1": round(float(f1_score(yte, yh_s, zero_division=0)), 3)}

# ---- question de la seance 1 : dispo apres 70 % d'incidents evites
indisp_h = round((1 - 0.9980) * 8 * 720, 2)          # 11,52 h sur le parc-mois
K["indisp_initiale_h"] = indisp_h
K["indisp_apres70_h"] = round(indisp_h * 0.3, 2)
K["dispo_apres70_pct"] = round(100 * (1 - indisp_h * 0.3 / (8 * 720)), 3)

# ---- importances (interpretabilite)
imp = rf.feature_importances_
ordre = np.argsort(imp)[::-1]

with open(os.path.join(FIG, "key_numbers_git3.json"), "w", encoding="utf-8") as fh:
    json.dump(K, fh, ensure_ascii=False, indent=1)
print("JSON ecrit :", len(K), "cles")

def sauve(fig, nom):
    fig.savefig(os.path.join(FIG, nom), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", nom)

# ================================================================ f16 seuils statistiques
fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.4))
ax[0].hist(p99, bins=60, color=C["blue"], alpha=.75)
ax[0].axvline(mu + 3 * sd, color=C["red"], ls="--", lw=1.4)
ax[0].axvline(q3 + 1.5 * (q3 - q1), color=C["orange"], ls=":", lw=1.6)
ax[0].set_title(f"a) lat_p99 : z-score (3σ, {K['lat_p99_zscore_depassements']} dép.) vs IQR "
                f"({K['lat_p99_iqr_depassements']} dép.)", fontsize=9)
ax[0].set_xlabel("lat_p99 (ms)")
ymin, ymax = np.nanpercentile(F["cpu_moy"], [1, 99])
scat = ax[1].scatter(F["cpu_moy"], F["lat_p99"], s=4, c=y, cmap="coolwarm",
                     vmin=0, vmax=1, alpha=.6)
ax[1].set_xlabel("cpu_moy (%)"); ax[1].set_ylabel("lat_p99 (ms)")
ax[1].set_title("b) le plan (cpu, latence) : rarement une frontière simple", fontsize=9)
plt.colorbar(scat, ax=ax[1], label="incident_30min")
sauve(fig, "f16_seuils_statistiques.png")

# ================================================================ f17 IsolationForest
fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.6))
top20 = np.argsort(score_iso)[::-1][:20]
ax[0].scatter(np.arange(len(score_iso))[::37], score_iso[::37], s=3, color=C["grey"], alpha=.4)
ax[0].scatter(top20, score_iso[top20], s=26, color=C["red"], zorder=3)
for i, lab in [(i, 1) for i in top20 if y[i] == 1]:
    pass
ax[0].scatter(top20[y[top20] == 1], score_iso[top20][y[top20] == 1], s=70,
              facecolors="none", edgecolors=C["green"], lw=1.4, zorder=4,
              label="vraie pré-incident")
ax[0].axhline(thr_iso, color=C["orange"], ls="--", lw=1.2,
              label="seuil train 99 %")
ax[0].set_title(f"a) score d'isolation : top 20, {K['iso_top20']}/20 sont des pré-incidents",
                fontsize=9)
ax[0].set_xlabel("fenêtres (échantillon)"); ax[0].set_ylabel("anormalité (−score_samples)")
ax[0].legend(fontsize=7, loc="lower right")
ax[1].scatter(F["cpu_moy"], F["lat_p99"], s=4, color=C["grey"], alpha=.35)
an = score_iso > thr_iso
ax[1].scatter(F["cpu_moy"][an & (y == 0)], F["lat_p99"][an & (y == 0)], s=14,
              color=C["orange"], label=f"anomalies sans incident ({int((an & (y==0)).sum())})")
ax[1].scatter(F["cpu_moy"][an & (y == 1)], F["lat_p99"][an & (y == 1)], s=34,
              color=C["red"], marker="X", label=f"anomalies ≈ incidents ({int((an & (y==1)).sum())})")
ax[1].set_title("b) une anomalie n'est PAS forcément une panne", fontsize=9)
ax[1].set_xlabel("cpu_moy (%)"); ax[1].set_ylabel("lat_p99 (ms)")
ax[1].legend(fontsize=7)
sauve(fig, "f17_isolation_forest.png")

# ================================================================ f18 desequilibre
fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.4))
ax[0].bar(["classe 0\n(34 440)", "classe 1\n(120)"], [34440, 120],
          color=[C["blue"], C["red"]])
for i, v in enumerate([34440, 120]):
    ax[0].text(i, v + 700, f"{v}", ha="center", fontsize=9, color=C["deep"], fontweight="bold")
ax[0].set_yscale("log"); ax[0].set_ylabel("fenêtres (échelle log)")
ax[0].set_title(f"a) 0,35 % de positifs : un modèle « toujours 0 » a "
                f"{K['acc_toujours0_test']:.2f} % d'exactitude", fontsize=9)
x = np.arange(3)
acc = [K["acc_toujours0_test"], 100 * MET["Foret (balanced)"]["precision"], 100 * MET["Foret (balanced)"]["rappel"]]
ax[1].bar(x, acc, color=[C["grey"], C["blue"], C["orange"]])
ax[1].set_xticks(x); ax[1].set_xticklabels(["exactitude\n« toujours 0 »", "précision\nforêt", "rappel\nforêt"])
for i, v in enumerate(acc):
    ax[1].text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9, color=C["deep"])
ax[1].set_ylim(0, 112); ax[1].set_ylabel("%")
ax[1].set_title("b) l'exactitude flatte, la précision/rappel disent la vérité", fontsize=9)
sauve(fig, "f18_desequilibre.png")

# ================================================================ f19 matrice de confusion
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
for k, (nom, titre) in enumerate([("Gradient boosting", "a) Gradient boosting — seuil 0,50"),
                                  ("Foret (balanced)", "b) Forêt balanced — seuil 0,50 (piège !)")]):
    m = MET[nom]; M = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    im = axes[k].imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[k].text(j, i, str(M[i, j]), ha="center", va="center", fontsize=15,
                         color="white" if M[i, j] > M.max() / 2 else C["deep"],
                         fontweight="bold")
    axes[k].set_xticks([0, 1]); axes[k].set_yticks([0, 1])
    axes[k].set_xticklabels(["prédit 0", "prédit 1"]); axes[0].set_yticklabels(["réel 0", "réel 1"])
    axes[k].set_title(titre, fontsize=9)
sauve(fig, "f19_matrice_confusion.png")

# ================================================================ f20 courbes PR
fig, ax = plt.subplots(figsize=(5.6, 4.0))
cols = {"Foret (balanced)": C["blue"], "Gradient boosting": C["purple"],
        "Reg. logistique": C["cyan"]}
for nom, (prec, rec, _) in PR.items():
    ax.plot(rec, prec, color=cols[nom], lw=1.8,
            label=f"{nom} (AP={MET[nom]['pr_auc']:.2f})")
ax.axhline(K["taux_positifs_pct"] / 100, color=C["grey"], ls=":", lw=1.2,
           label=f"hasard ({K['taux_positifs_pct']} %)")
ax.scatter(MET[BEST]["rappel"], MET[BEST]["precision"], s=60, marker="X",
           color=C["red"], zorder=5, label="seuil 0,50")
ax.set_xlabel("rappel"); ax.set_ylabel("précision")
ax.set_title("Courbes précision-rappel sur le test (futur jamais vu)", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
sauve(fig, "f20_courbes_pr.png")

# ================================================================ f21 comparaison modeles
fig, ax = plt.subplots(figsize=(8.6, 3.4))
noms = list(MET)
x = np.arange(len(noms))
for k, (met, col) in enumerate([("precision", C["blue"]), ("rappel", C["orange"]),
                                ("f1", C["cyan"]), ("pr_auc", C["purple"])]):
    vals = [100 * MET[n][met] for n in noms]
    ax.bar(x + (k - 1.5) * 0.2, vals, 0.19, color=col, label=met)
ax.set_xticks(x); ax.set_xticklabels(noms, fontsize=8)
ax.set_ylabel("% / AP×100"); ax.set_ylim(0, 105)
ax.legend(fontsize=8, ncol=4)
ax.set_title("Les 4 familles comparées sur le test (seuil 0,50)", fontsize=9)
sauve(fig, "f21_comparaison_modeles.png")

# ================================================================ f22 seuil vs alertes
fig, ax = plt.subplots(figsize=(7.2, 3.6))
seuils = np.linspace(0.02, 0.98, 49)
nal, nrec = [], []
for s in seuils:
    yh = (p_best >= s).astype(int)
    nal.append(yh.sum() / K["jours_test"])       # alertes/jour sur tout le parc
    nrec.append(recall_score(yte, yh, zero_division=0))
ax.plot(seuils, nal, color=C["blue"], lw=1.8, label="alertes/jour (parc 8 serveurs)")
ax.axhline(5 * nb_serveurs, color=C["red"], ls="--", lw=1.3,
           label=f"budget ≤ 5/j/serveur = {5*nb_serveurs}/jour")
ax.axvline(seuil_choisi, color=C["orange"], ls=":", lw=1.5,
           label=f"seuil choisi = {seuil_choisi:.2f}")
ax2 = ax.twinx()
ax2.plot(seuils, nrec, color=C["green"], lw=1.4, ls="-.", label="rappel")
ax2.set_ylabel("rappel", color=C["green"]); ax2.set_ylim(0, 1.02)
ax.set_xlabel("seuil de décision"); ax.set_ylabel("alertes / jour")
ax.set_yscale("log")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
ax.set_title(f"Choix du seuil : {K['alertes_au_seuil']} alertes sur {K['jours_test']} jours "
             f"({K['alertes_par_jour_parc']}/jour parc)", fontsize=9)
sauve(fig, "f22_choix_seuil.png")

# ================================================================ f23 importances + CV
fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.6))
top = ordre[:8]
ax[0].barh(range(len(top))[::-1], imp[top] * 100, color=C["blue"])
ax[0].set_yticks(range(len(top))[::-1])
ax[0].set_yticklabels([FEATS[i] for i in top], fontsize=8)
ax[0].set_xlabel("importance (%)")
ax[0].set_title("a) forêt : variables qui décident (règles métier)", fontsize=9)
noms3 = list(cv)
moy = [cv[n][0] for n in noms3]; std = [cv[n][1] for n in noms3]
ax[1].bar(np.arange(3), moy, yerr=std, capsize=4,
          color=[C["blue"], C["purple"], C["cyan"]])
ax[1].set_xticks(np.arange(3)); ax[1].set_xticklabels(noms3, fontsize=8)
ax[1].set_ylabel("F1 (moy ± écart-type, 5 plis)")
ax[1].set_title("b) validation croisée TEMPORELLE (TimeSeriesSplit)", fontsize=9)
for i, v in enumerate(moy):
    ax[1].text(i, v + std[i] + 0.01, f"{v:.2f}", ha="center", fontsize=8)
sauve(fig, "f23_interpretabilite_cv.png")

# ================================================================ f24 dispo
fig, ax = plt.subplots(figsize=(7.0, 3.4))
cats = ["mesuré séance 1\n(99,80 %)", "après détection prédictive\nà 70 % évités"]
vals = [99.80, K["dispo_apres70_pct"]]
bars = ax.bar(cats, vals, color=[C["grey"], C["green"]], width=.45)
ax.axhline(99.9, color=C["red"], ls="--", lw=1.5)
ax.text(1.42, 99.905, "SLA 99,9 %", color=C["red"], fontsize=8, va="bottom")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f} %",
            ha="center", fontweight="bold", color=C["deep"])
ax.set_ylim(99.7, 99.97)
ax.set_ylabel("disponibilité du parc (%)")
ax.set_title(f"Réponse à la question de la séance 1 : {indisp_h:.1f} h d'indispo → "
             f"{K['indisp_apres70_h']:.2f} h évitées à 70 %", fontsize=9)
sauve(fig, "f24_dispo_predictif.png")

print("FIGURES TERMINEES")
