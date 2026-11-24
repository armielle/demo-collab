# -*- coding: utf-8 -*-
"""Figures de la Seance 4 (EPD-GIT-5534) -- supervision, optimisation, prediction.
Calcule depuis data/nosys_metriques.csv et data/nosys_features.csv (graine 5534).
SARIMA reellement ajuste (statsmodels SARIMAX) sur l'agregat horaire du parc,
harmoniques hebdomadaires en exogenes, MAE mesuree sur 6 jours de test.
Ecrit figures/key_numbers_git4.json AVANT les figures."""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, recall_score

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
np.random.seed(5534)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
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

def sauve(fig, nom):
    fig.savefig(os.path.join(FIG, nom), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", nom)

# ============================================================ 1. serie horaire
M = pd.read_csv(os.path.join(ROOT, "data", "nosys_metriques.csv"),
                parse_dates=["horodatage"])
# trafic agrege = charge CPU moyenne du parc, par heure (720 points)
tr = (M.set_index("horodatage")["cpu_pct"].resample("1h").mean())
assert len(tr) == 720
train, test = tr.iloc[:576], tr.iloc[576:]          # 24 jours / 6 jours

# stationnarite : test ADF (brut, diff d=1, diff saisonniere s=24)
adf_raw = adfuller(tr.values, autolag="AIC")
adf_d1 = adfuller(tr.diff().dropna().values, autolag="AIC")
adf_d24 = adfuller((tr - tr.shift(24)).dropna().values, autolag="AIC")

# decomposition additive, periode 24 h
dec = seasonal_decompose(tr, model="additive", period=24)

K = {}
K["serie"] = {
    "n_points": int(len(tr)), "granularite": "1 h", "periode_saison": 24,
    "n_train": int(len(train)), "n_test": int(len(test)),
    "cpu_moy_parc": round(float(tr.mean()), 2),
    "adf_stat_brut": round(float(adf_raw[0]), 3), "adf_p_brut": round(float(adf_raw[1]), 4),
    "adf_stat_d1": round(float(adf_d1[0]), 3), "adf_p_d1": round(float(adf_d1[1]), 8),
    "adf_stat_d24": round(float(adf_d24[0]), 3), "adf_p_d24": round(float(adf_d24[1]), 8),
}

# ============================================================ 2. SARIMA reel
# exogenes : harmoniques de Fourier de la SAISON HEBDOMADAIRE (periode 168 h),
# connues a l'avance comme l'heure elle-meme -- legere et decisives ici
t0 = np.arange(len(tr))
HARM = np.column_stack(
    [f(t0 * 2 * np.pi * j / 168) for j in range(1, 4) for f in (np.sin, np.cos)])
Xa, Xb = HARM[:576], HARM[576:]

cands = [(0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 1)]   # grille legere par AIC
aics = {}
for order in cands:
    m = SARIMAX(train.values, exog=Xa, order=order, seasonal_order=(1, 0, 1, 24),
                enforce_stationarity=False, enforce_invertibility=False)
    r = m.fit(disp=False, maxiter=300)
    aics[str(order)] = round(float(r.aic), 1)
best_order = min(aics, key=aics.get)
best_order = tuple(int(x) for x in best_order.strip("()").split(","))

mod = SARIMAX(train.values, exog=Xa, order=best_order, seasonal_order=(1, 0, 1, 24),
              enforce_stationarity=False, enforce_invertibility=False)
res = mod.fit(disp=False, maxiter=300)
fc = res.get_forecast(steps=len(test), exog=Xb)
pred = fc.predicted_mean
ci = fc.conf_int(alpha=0.05)
mae = float(np.mean(np.abs(pred - test.values)))
mae_persist = float(np.mean(np.abs(
    np.tile(train.iloc[-24:].values, int(np.ceil(len(test) / 24)))[:len(test)] - test.values)))
mae_semaine = float(np.mean(np.abs(
    np.tile(train.iloc[-len(test):].values, 1) - test.values)))
resid = res.resid
lb = acorr_ljungbox(resid, lags=[24], return_df=True)
K["sarima"] = {
    "ordre": list(best_order), "saisonnier": [1, 0, 1, 24],
    "aics": aics, "aic_meilleur": aics[str(best_order)],
    "mae_test": round(mae, 3), "mae_persist_jour": round(mae_persist, 3),
    "mae_semaine_precedente": round(mae_semaine, 3),
    "gain_vs_persist_pct": round(100 * (1 - mae / mae_persist), 1),
    "gain_vs_semaine_pct": round(100 * (1 - mae / mae_semaine), 1),
    "ljungbox_p_lag24": round(float(lb["lb_pvalue"].iloc[0]), 4),
    "cpu_moy_parc": round(float(tr.mean()), 2),
}

# ============================================================ 3. modele de panne
F = pd.read_csv(os.path.join(ROOT, "data", "nosys_features.csv"),
                parse_dates=["fenetre"])
feats = ["cpu_moy", "cpu_max", "ram_moy", "lat_moy", "lat_p99", "perte_moy",
         "temp_moy", "conso_moy", "cpu_delta", "cpu_amplitude", "h_sin", "h_cos"]
F2 = F.dropna(subset=feats + ["incident_30min"]).copy()
cut = pd.Timestamp("2026-09-22")
tr_m, te_m = F2[F2["fenetre"] < cut], F2[F2["fenetre"] >= cut]

def fit_gb(Xtr, ytr, Xte):
    gbc = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                     learning_rate=0.06, random_state=5534)
    gbc.fit(Xtr, ytr)
    return gbc.predict_proba(Xte)[:, 1]

y = te_m["incident_30min"].values
p = fit_gb(tr_m[feats].values, tr_m["incident_30min"].values, te_m[feats].values)
cm = confusion_matrix(y, (p >= 0.5).astype(int))
recall = recall_score(y, (p >= 0.5).astype(int))
prec = cm[1, 1] / max(cm[1, 1] + cm[0, 1], 1)

# LE PIEGE n.1 : meme modele + l'alarme COURANTE dans les features (fuite)
al = (M.assign(f10=M["horodatage"].dt.floor("10min"))
        .groupby(["serveur", "f10"])["alarme"].max().rename("alarme_10min").reset_index())
F3 = F2.merge(al, left_on=["serveur", "fenetre"], right_on=["serveur", "f10"])
tr_l, te_l = F3[F3["fenetre"] < cut], F3[F3["fenetre"] >= cut]
y_l = te_l["incident_30min"].values
p_l = fit_gb(tr_l[feats + ["alarme_10min"]].values, tr_l["incident_30min"].values,
             te_l[feats + ["alarme_10min"]].values)
cm_leak = confusion_matrix(y_l, (p_l >= 0.5).astype(int))
recall_leak = recall_score(y_l, (p_l >= 0.5).astype(int))
prec_leak = cm_leak[1, 1] / max(cm_leak[1, 1] + cm_leak[0, 1], 1)

# le piege en un chiffre : P(incident | alarme courante) vs P(incident | sans alarme)
n_al = int((te_l["alarme_10min"] == 1).sum())
pos_al = int(y_l[te_l["alarme_10min"].values == 1].sum())
n_sans = int((te_l["alarme_10min"] == 0).sum())
pos_sans = int(y_l[te_l["alarme_10min"].values == 0].sum())

# KFold aleatoire (structure fausse ici) vs decoupage temporel strict
from sklearn.model_selection import StratifiedKFold
X_all, y_all = F2[feats].values, F2["incident_30min"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5534)
kf_recs = []
for i_tr, i_te in skf.split(X_all, y_all):
    pk = fit_gb(X_all[i_tr], y_all[i_tr], X_all[i_te])
    kf_recs.append(round(float(recall_score(y_all[i_te], (pk >= 0.5).astype(int))), 3))
kfold_recall = float(np.mean(kf_recs))

# courbe rappel vs nb d'alertes par jour (test = 9,16 jours de fenetres 10 min)
n_jours = round(float((te_m["fenetre"].max() - te_m["fenetre"].min())
                      / pd.Timedelta(days=1) + 1.0 / 6), 2)
ordre = np.argsort(-p)
pos_total = int(y.sum())
ks, recs = [], []
for k in range(1, 49):                      # k alertes par jour
    n_sel = min(int(round(k * n_jours)), len(y))
    recs.append(round(float(y[ordre[:n_sel]].sum() / pos_total), 4))
    ks.append(k)
# point d'operation : plus petit k avec rappel >= 0.60
k_op = next((k for k, r in zip(ks, recs) if r >= 0.60), ks[-1])
K["panne"] = {
    "n_train": int(len(tr_m)), "n_test": int(len(te_m)),
    "pos_train": int(tr_m["incident_30min"].sum()), "pos_test": int(y.sum()),
    "jours_test": n_jours,
    "confusion_vn_fp_fn_vp": [int(x) for x in cm.ravel()],
    "recall_test": round(float(recall), 3), "precision_test": round(float(prec), 3),
    "cm_fuite_vn_fp_fn_vp": [int(x) for x in cm_leak.ravel()],
    "recall_fuite": round(float(recall_leak), 3), "precision_fuite": round(float(prec_leak), 3),
    "n_fenetres_alarme_test": n_al, "pos_si_alarme": pos_al,
    "p_incident_si_alarme": round(pos_al / max(n_al, 1), 3),
    "n_sans_alarme_test": n_sans, "pos_sans_alarme": pos_sans,
    "p_incident_sans_alarme_pct": round(100 * pos_sans / max(n_sans, 1), 3),
    "kfold_recall_moy": round(kfold_recall, 3), "kfold_recalls": kf_recs,
    "k_op_alertes_par_jour": k_op, "recall_op": recs[ks.index(k_op)],
    "rappel_k3": recs[ks.index(3)], "rappel_k6": recs[ks.index(6)],
    "rappel_k12": recs[ks.index(12)], "rappel_max_topk": max(recs),
}

# ============================================================ 4. saturation disque
# hypothese explicite : metriques + logs occupent 40 % du disque apres 30 jours
# de retention -> +1,333 %/jour ; seuil d'alerte operationnel 90 %
usage0, jours_ecoules, seuil = 40.0, 30.0, 90.0
taux = usage0 / jours_ecoules                 # % par jour
jours_restants = (seuil - usage0) / taux
date_sat = pd.Timestamp("2026-10-01") + pd.Timedelta(days=jours_restants)
K["saturation"] = {
    "usage_initial_pct": usage0, "taux_pct_par_jour": round(taux, 3),
    "seuil_pct": seuil, "jours_restants": round(float(jours_restants), 1),
    "date_saturation": date_sat.strftime("%d/%m/%Y"),
    "date_saturation_iso": str(date_sat.date()),
}
# disponibilite : resultat officiel de la question posee en seance 1
# 11,3 h d'indispo cumulee ; 70 % evitees a 30 min -> 3,39 h restantes ;
# fenetre de mesure du dossier : 8 serveurs x 720 h = 5 760 heures-serveur
K["disponibilite"] = {
    "indispo_h_mois": 11.3, "part_evitee": 0.7,
    "indispo_restante_h": round(11.3 * 0.3, 2),
    "heures_serveur_mois": 8 * 30 * 24,
    "dispo_pct": round(100 * (1 - 11.3 * 0.3 / (8 * 30 * 24)), 3),
}

with open(os.path.join(FIG, "key_numbers_git4.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, ensure_ascii=False, indent=1)
print("JSON ecrit")

# ================================================================ f21 decompo + ACF/PACF
fig, ax = plt.subplots(2, 2, figsize=(11.2, 5.8))
ax[0, 0].plot(tr.index, tr.values, color=C["blue"], lw=1.0)
ax[0, 0].set_title("a) Trafic agrégé du parc : CPU moyen par heure, 30 jours", fontsize=9)
ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax[0, 0].set_ylabel("CPU (%)")
ax[0, 1].plot(tr.index, dec.trend, color=C["purple"], lw=1.4, label="tendance")
ax[0, 1].plot(tr.index, dec.seasonal, color=C["cyan"], lw=.7, label="saison (24 h)")
ax[0, 1].axhline(0, color=C["grey"], lw=.6)
ax[0, 1].legend(fontsize=7.5)
ax[0, 1].set_title("b) Décomposition additive (période 24 h)", fontsize=9)
ax[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
lags = 48
ac = acf(tr.values, nlags=lags)
pa = pacf(tr.values, nlags=lags, method="ywm")
ax[1, 0].bar(range(lags + 1), ac, color=C["blue"], width=.7)
ax[1, 0].axhline(1.96 / np.sqrt(len(tr)), color=C["red"], ls="--", lw=.9)
ax[1, 0].axhline(-1.96 / np.sqrt(len(tr)), color=C["red"], ls="--", lw=.9)
for s in range(24, lags + 1, 24):
    ax[1, 0].axvline(s, color=C["orange"], lw=.8, alpha=.8)
ax[1, 0].set_title("c) ACF : pics répétés toutes les 24 h (orange) => saison", fontsize=9)
ax[1, 0].set_xlabel("décalage (h)")
ax[1, 1].bar(range(1, lags + 1), pa[1:], color=C["green"], width=.7)
ax[1, 1].axhline(1.96 / np.sqrt(len(tr)), color=C["red"], ls="--", lw=.9)
ax[1, 1].axhline(-1.96 / np.sqrt(len(tr)), color=C["red"], ls="--", lw=.9)
ax[1, 1].set_title("d) PACF : chute après le premier pic => AR(1)", fontsize=9)
ax[1, 1].set_xlabel("décalage (h)")
fig.tight_layout()
sauve(fig, "f21_decomposition_acf.png")

# ================================================================ f22 prevision SARIMA
fig, ax = plt.subplots(figsize=(11.2, 3.6))
w = 240                       # fenetre affichee : 10 derniers jours train + test
ax.plot(tr.index[-w:], tr.values[-w:], color=C["grey"], lw=.9, label="entraînement (réalisé)")
ax.plot(test.index, test.values, color=C["deep"], lw=1.6, label="réalisé (6 jours de test)")
ax.plot(test.index, pred, color=C["red"], lw=1.6, ls="--",
        label=f"prévision SARIMA {best_order}×(1,0,1)₂₄ — MAE {mae:.2f} pts CPU")
ax.fill_between(test.index, ci[:, 0], ci[:, 1], color=C["red"], alpha=.15,
                label="intervalle de confiance 95 %")
ax.axvline(test.index[0], color=C["orange"], lw=1.2, ls=":")
ax.annotate("frontière train/test\n(ici : JAMAIS de shuffle)", xy=(test.index[0], tr.max() * .96),
            fontsize=7.5, color=C["orange"], ha="left", xytext=(6, 0),
            textcoords="offset points")
ax.legend(fontsize=7.6, loc="upper left")
ax.set_ylabel("CPU (%)")
ax.set_title("Prévision SARIMA vs réalisé — agrégat horaire du parc, frontière du 22/09", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
sauve(fig, "f22_sarima_prevision.png")

# ================================================================ f23 rappel vs alertes/jour
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.6))
ax[0].plot(ks, [100 * r for r in recs], color=C["blue"], lw=1.8, marker="o", ms=2.5)
ax[0].axhline(60, color=C["green"], ls="--", lw=1, label="objectif : rappel 60 %")
ax[0].axvline(k_op, color=C["orange"], ls="--", lw=1,
              label=f"point d'opération : {k_op} alertes/jour")
ax[0].set_xlabel("alertes par jour (k)"); ax[0].set_ylabel("rappel (%)")
ax[0].set_title("a) Rappel vs budget d'alertes (jeu de test)", fontsize=9)
ax[0].legend(fontsize=7.6)
# b) la fuite en un chiffre : P(incident | alarme courante) vs P(incident | sans)
lab2 = ["alarme courante\nsonne (40 fen.)", "pas d'alarme\ncourante (10 304 fen.)"]
pv = [100 * pos_al / max(n_al, 1), 100 * pos_sans / max(n_sans, 1)]
bars = ax[1].bar(lab2, pv, color=[C["red"], C["grey"]], width=.5, log=True)
for b, v in zip(bars, pv):
    txt = "100 %" if v > 99.99 else f"{v:.2f}".replace(".", ",") + " %"
    ax[1].text(b.get_x() + b.get_width() / 2, v * 1.35, txt,
               ha="center", fontsize=9.5, fontweight="bold")
ax[1].set_ylim(0.05, 300); ax[1].set_ylabel("P(incident dans 30 min)  (éch. log)")
ax[1].set_title("b) Le piège n°1 : l'alarme courante EST l'étiquette", fontsize=9)
fig.tight_layout()
sauve(fig, "f23_rappel_alertes.png")

# ================================================================ f24 saturation
fig, ax = plt.subplots(figsize=(11.2, 3.4))
jours = np.arange(0, 75)
usage = usage0 + taux * jours
ax.plot(pd.Timestamp("2026-10-01") + pd.to_timedelta(jours, unit="D"), usage,
        color=C["blue"], lw=1.8, label=f"projection : +{taux:.2f} %/jour (hypothèse explicite)")
ax.axhline(seuil, color=C["red"], ls="--", lw=1.2, label="seuil d'alerte 90 %")
ax.axhline(100, color=C["deep"], ls=":", lw=1, label="disque plein 100 %")
ax.axvline(date_sat, color=C["orange"], lw=1.5)
ax.annotate(f"90 % atteint le\n{date_sat.strftime('%d/%m/%Y')}\n(+{jours_restants:.0f} jours)",
            xy=(date_sat, seuil), xytext=(30, -30), textcoords="offset points",
            fontsize=8.5, color=C["orange"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["orange"]))
ax.axvspan(pd.Timestamp("2026-10-01"), date_sat, color=C["green"], alpha=.06)
ax.set_ylim(0, 105); ax.set_ylabel("disque utilisé (%)")
ax.set_title("De la prévision à la décision : date de saturation du stockage (projection linéaire)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
sauve(fig, "f24_saturation.png")

print("TERMINE -- figures seance 4")
print(json.dumps({"ordre": best_order, "mae": round(mae, 3),
                  "mae_persist": round(mae_persist, 3), "mae_semaine": round(mae_semaine, 3),
                  "recall": round(float(recall), 3), "precision": round(float(prec), 3),
                  "recall_leak": round(float(recall_leak), 3), "precision_leak": round(float(prec_leak), 3),
                  "k_op": k_op, "date_sat": K["saturation"]["date_saturation"],
                  "dispo": K["disponibilite"]["dispo_pct"],
                  "adf": [K["serie"]["adf_p_brut"], K["serie"]["adf_p_d1"], K["serie"]["adf_p_d24"]],
                  "lb_p": K["sarima"]["ljungbox_p_lag24"]}, ensure_ascii=False))
