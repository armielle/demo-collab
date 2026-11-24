# -*- coding: utf-8 -*-
"""Figures de la Seance 2 (EPD-GIT-5534) -- preparation des donnees.
Calcule depuis data/nosys_metriques_brutes.csv (defauts semes, graine 5534).
Ecrit figures/key_numbers_git2.json avant les figures lourdes."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
B = pd.read_csv(os.path.join(ROOT, "data", "nosys_metriques_brutes.csv"),
                parse_dates=["horodatage"])
CLEAN = pd.read_csv(os.path.join(ROOT, "data", "nosys_metriques.csv"),
                    parse_dates=["horodatage"])
COLS = ["cpu_pct", "ram_pct", "latence_ms", "paquets_perdus_pct", "temperature_c"]

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

# ============================================================ chiffres cles
K = {}
K["lignes_brutes"] = int(len(B))
K["doublons_exactes"] = int(B.duplicated().sum())
K["nan_total"] = int(B[COLS].isna().sum().sum())
K["nan_pct"] = round(100 * K["nan_total"] / (len(B) * len(COLS)), 3)
K["nan_par_colonne"] = {c: int(B[c].isna().sum()) for c in COLS}
K["temp_max_brute"] = float(B["temperature_c"].max())
K["cpu_max_brute"] = float(B["cpu_pct"].max())
K["lat_max_brute"] = float(B["latence_ms"].max())
K["temp_max_normale"] = float(CLEAN["temperature_c"].max())
K["cpu_max_normale"] = float(CLEAN["cpu_pct"].max())
K["lat_max_normale"] = float(CLEAN["latence_ms"].max())
# granularite mixte : fraction de pas de 15 min par serveur
frac15 = B.groupby("serveur")["horodatage"].apply(
    lambda h: (h.sort_values().diff().dt.total_seconds() == 900).mean())
K["pas_5min"] = int((B.groupby("serveur")["horodatage"].apply(
    lambda h: (h.sort_values().diff().dt.total_seconds() == 300).mean()) > 0.9).sum())
K["serveurs_15min"] = sorted(frac15[frac15 > 0.9].index.tolist())
K["pas_15min"] = int((frac15 > 0.9).sum())
# derive srv-06 : pente de temperature sur le mois (regression lineaire simple)
s6 = B[(B["serveur"] == "srv-06") & B["temperature_c"].between(0, 80)].dropna(
    subset=["temperature_c"])
t = (s6["horodatage"] - s6["horodatage"].min()).dt.total_seconds() / 86400
pente = float(np.polyfit(t, s6["temperature_c"], 1)[0])
K["derive_srv06_C_par_jour"] = round(pente, 3)
s6c = CLEAN[CLEAN["serveur"] == "srv-06"]
tc = (s6c["horodatage"] - s6c["horodatage"].min()).dt.total_seconds() / 86400
K["pente_normale_srv06"] = float(round(np.polyfit(tc, s6c["temperature_c"], 1)[0], 3))
# imputation : erreur moyenne sur les NaN artificiels (ffill vs interpolation)
test = CLEAN[CLEAN["serveur"] == "srv-01"].copy().reset_index(drop=True)
mask = np.zeros(len(test), dtype=bool)
rng = np.random.default_rng(5534)
mask[rng.choice(len(test), 600, replace=False)] = True
cache = test.loc[mask, "cpu_pct"].copy()
test.loc[mask, "cpu_pct"] = np.nan
s = test.set_index("horodatage")["cpu_pct"]
hmm = test.loc[mask, "horodatage"]
ffill = s.ffill().reindex(hmm).values
interp = s.interpolate(limit_direction="both").reindex(hmm).values
K["err_ffill_cpu"] = round(float(np.nanmean(np.abs(ffill - cache.values))), 3)
K["err_interp_cpu"] = round(float(np.nanmean(np.abs(interp - cache.values))), 3)
K["ecart_type_cpu_normal"] = round(float(CLEAN["cpu_pct"].std()), 2)
# etiquette incident<=30min : par serveur, sur la grille 5 min, max des
# 6 fenetres [t ; t+25 min] puis agregation en fenetres de 10 min
lab = CLEAN[["serveur", "horodatage", "alarme"]].sort_values(["serveur", "horodatage"])
etiq = []
for srv, g in lab.groupby("serveur"):
    a = g["alarme"].values.astype(bool)
    fut = a.copy()
    for k in range(1, 6):                      # t+5 ... t+25 min
        fut[:-k] |= a[k:]
    etiq.append(pd.DataFrame({"horodatage": g["horodatage"].values,
                              "serveur": srv, "futur": fut.astype(int)}))
etiq = pd.concat(etiq)
e10 = (etiq.pivot(index="horodatage", columns="serveur", values="futur")
       .resample("10min").max())
K["nb_fenetres_10min"] = int(e10.size)
K["nb_fenetres_incident_futur"] = int((e10 == 1).values.sum())
K["taux_positifs_pct"] = round(100 * K["nb_fenetres_incident_futur"] / K["nb_fenetres_10min"], 2)

with open(os.path.join(FIG, "key_numbers_git2.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, ensure_ascii=False, indent=1)
print("JSON ecrit :", len(K), "cles")

def sauve(fig, nom):
    fig.savefig(os.path.join(FIG, nom), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", nom)

# ================================================================ f08 defauts
fig, ax = plt.subplots(2, 2, figsize=(11.2, 5.6))
# a. temperature brute : pics
t6 = B[(B["serveur"] == "srv-06") & B["temperature_c"].notna()].set_index("horodatage")
ax[0, 0].plot(t6.index, t6["temperature_c"], color=C["blue"], lw=.8)
ax[0, 0].set_yscale("symlog")
ax[0, 0].set_title("a) srv-06 température brute : capteur décalé (échelle log)", fontsize=9)
ax[0, 0].set_ylabel("°C")
# b. cpu brute : pics > 100
m1 = B[(B["serveur"] == "srv-01") & B["cpu_pct"].notna()].set_index("horodatage")
ax[0, 1].plot(m1.index, m1["cpu_pct"], color=C["cyan"], lw=.8)
ax[0, 1].axhline(100, color=C["red"], ls="--", lw=1)
ax[0, 1].set_ylim(-5, 480)
ax[0, 1].set_title("b) srv-01 CPU : valeurs > 100 % impossibles", fontsize=9)
ax[0, 1].set_ylabel("%")
# c. trous : horodatages manquants srv-01
h = B[B["serveur"] == "srv-01"]["horodatage"].sort_values()
gaps = h.diff().dt.total_seconds() / 60
idx_gaps = np.where(gaps > 5)[0]
ax[1, 0].plot(h, np.arange(len(h)), color=C["green"], lw=.8)
for i in idx_gaps:
    ax[1, 0].axvspan(h.iloc[i - 1], h.iloc[i], color=C["red"], alpha=.30)
ax[1, 0].set_title("c) srv-01 : trous d'échantillonnage (zones rouges)", fontsize=9)
ax[1, 0].set_ylabel("indice de la mesure")
ax[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
# d. derive srv-06
ax[1, 1].plot(t6.index, t6["temperature_c"], color=C["purple"], lw=.8)
ax[1, 1].set_ylim(25, 45)
ax[1, 1].axhline(K["temp_max_normale"], color=C["orange"], ls="--", lw=1)
ax[1, 1].set_title("d) srv-06 : dérive de capteur (+%.2f °C/jour)" % K["derive_srv06_C_par_jour"], fontsize=9)
ax[1, 1].set_ylabel("°C")
ax[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
sauve(fig, "f08_defauts.png")

# ================================================================ f09 reechantillonnage
jour = pd.Timestamp("2026-09-09")
s1b = B[(B["serveur"] == "srv-01") & (B["horodatage"].dt.date == jour.date())] \
    .set_index("horodatage")["cpu_pct"]
s1c = CLEAN[(CLEAN["serveur"] == "srv-01") & (CLEAN["horodatage"].dt.date == jour.date())] \
    .set_index("horodatage")["cpu_pct"]
fig, ax = plt.subplots(figsize=(11.2, 3.4))
ax.plot(s1c.index, s1c.values, color=C["green"], lw=1.4, label="référence propre (5 min)")
ax.plot(s1b.index, s1b.values, "o", ms=3, color=C["red"], alpha=.7,
        label="collecte brute : pas irréguliers, NaN, pics")
reg = s1b.resample("5min").mean().interpolate(limit_direction="both")
ax.plot(reg.index, reg.values, color=C["blue"], lw=1.6, ls="--", label="après rééchantillonnage + interpolation")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Du brut au signal exploitable — srv-01, jour de l'incident (09/09)", fontsize=10)
ax.set_ylabel("CPU (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
sauve(fig, "f09_reechantillonnage.png")

# ================================================================ f10 moyennes mobiles
fig, ax = plt.subplots(figsize=(11.2, 3.4))
ax.plot(s1c.index, s1c.values, color=C["grey"], lw=.7, alpha=.6, label="5 min (bruité)")
for n, col in [("15min", C["cyan"]), ("60min", C["purple"]), ("6h", C["orange"])]:
    mm = s1c.resample("5min").mean().rolling(n, min_periods=1).mean()
    ax.plot(mm.index, mm.values, lw=1.8, color=col, label=f"moyenne mobile {n}")
ax.legend(fontsize=8); ax.set_ylabel("CPU (%)")
ax.set_title("Moyennes mobiles : extraire la tendance sans perdre les incidents", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
sauve(fig, "f10_moyennes_mobiles.png")

# ================================================================ f11 encodage cyclique
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.4))
heure = np.arange(24)
ax[0].plot(heure, np.sin(2 * np.pi * heure / 24), color=C["blue"], label="sin(2πh/24)")
ax[0].plot(heure, np.cos(2 * np.pi * heure / 24), color=C["orange"], label="cos(2πh/24)")
ax[0].set_title("Encodage cyclique de l'heure", fontsize=9.5)
ax[0].set_xlabel("heure"); ax[0].legend(fontsize=8)
prof = CLEAN.groupby(CLEAN["horodatage"].dt.hour)["cpu_pct"].mean()
ax[1].plot(prof.index, prof.values, color=C["green"], lw=1.8, marker="o", ms=3)
ax[1].set_title("CPU moyen par heure (référence NOSYS)", fontsize=9.5)
ax[1].set_xlabel("heure"); ax[1].set_ylabel("CPU (%)")
sauve(fig, "f11_cyclique.png")

# ================================================================ f12 outliers : z vs IQR
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.4))
x = B["latence_ms"].dropna()
x_ok = x[(x > 0) & (x < 400)]                     # sans pics capteur
z = (x_ok - x_ok.mean()) / x_ok.std()
q1, q3 = x_ok.quantile([.25, .75]); iqr = q3 - q1
ax[0].hist(z, bins=60, color=C["blue"], alpha=.8)
ax[0].axvline(3, color=C["red"], ls="--", lw=1.2); ax[0].axvline(-3, color=C["red"], ls="--", lw=1.2)
ax[0].set_title("z-score : |z| > 3 => %.1f %% des points" % (100 * (z.abs() > 3).mean()), fontsize=9.5)
ax[0].set_xlabel("z-score")
ax[1].hist(x_ok, bins=60, color=C["cyan"], alpha=.8)
ax[1].axvline(q3 + 1.5 * iqr, color=C["red"], ls="--", lw=1.2)
ax[1].axvline(q1 - 1.5 * iqr, color=C["red"], ls="--", lw=1.2)
ax[1].set_title("IQR : bornes [%.1f ; %.1f] ms" % (q1 - 1.5 * iqr, q3 + 1.5 * iqr), fontsize=9.5)
ax[1].set_xlabel("latence (ms)")
sauve(fig, "f12_outliers.png")

# ================================================================ f13 imputation
fig, ax = plt.subplots(figsize=(11.2, 3.2))
w = s1c.iloc[60:180]
trou = w.copy(); trou.iloc[20:36] = np.nan
ax.plot(w.index, w.values, color=C["grey"], lw=.9, alpha=.8, label="réel (masqué)")
trou2 = trou.interpolate(limit_direction="both")
ax.plot(trou2.index, trou2.values, color=C["blue"], lw=2, label="interpolation linéaire")
ff = trou.ffill()
ax.plot(ff.index, ff.values, color=C["orange"], lw=1.4, ls="--", label="forward fill")
ax.axvspan(w.index[20], w.index[35], color=C["red"], alpha=.10)
ax.legend(fontsize=8); ax.set_ylabel("CPU (%)")
ax.set_title("Imputer un trou de 80 min : l'interpolation suit la tendance, le ffill produit un palier", fontsize=9.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
sauve(fig, "f13_imputation.png")

# ================================================================ f14 standardisation
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.2))
ok = CLEAN[["cpu_pct", "ram_pct", "latence_ms"]].dropna()
ax[0].hist(ok["cpu_pct"], bins=50, color=C["blue"], alpha=.75, label="cpu")
ax[0].hist(ok["ram_pct"], bins=50, color=C["orange"], alpha=.55, label="ram")
ax[0].hist(ok["latence_ms"] * 10, bins=50, color=C["cyan"], alpha=.55, label="latence x10")
ax[0].legend(fontsize=8); ax[0].set_title("Échelles incompatibles avant standardisation", fontsize=9.5)
zok = (ok - ok.mean()) / ok.std()
ax[1].hist(zok["cpu_pct"], bins=50, color=C["blue"], alpha=.75, label="cpu")
ax[1].hist(zok["ram_pct"], bins=50, color=C["orange"], alpha=.55, label="ram")
ax[1].hist(zok["latence_ms"], bins=50, color=C["cyan"], alpha=.55, label="latence")
ax[1].legend(fontsize=8); ax[1].set_title("Après z-score : moyenne 0, écart-type 1", fontsize=9.5)
sauve(fig, "f14_standardisation.png")

# ================================================================ f15 etiquette incident
fig, ax = plt.subplots(figsize=(11.2, 2.8))
s1c2 = CLEAN[CLEAN["serveur"] == "srv-01"].set_index("horodatage").sort_index()
w5 = s1c2["alarme"]
fut = w5.copy().astype(bool)
for k in range(1, 6):
    fut.iloc[:-k] |= w5.values[k:] == 1
w10 = w5.resample("10min").max()
f10 = fut.resample("10min").max()
ax.fill_between(w10.index, 0, 1, where=w10 == 1, color=C["red"], alpha=.5, step="post",
                label="alarme présente")
ax.fill_between(w10.index, 0, .6, where=f10 == 1, color=C["orange"], alpha=.35, step="post",
                label="étiquette « incident <= 30 min »")
ax.set_yticks([0, 1]); ax.set_yticklabels(["non", "oui"])
ax.legend(fontsize=8, loc="upper right"); ax.set_ylim(0, 1.2)
ax.set_title("Construire l'étiquette supervisée (srv-01, mois complet) : fenêtre 10 min, futur 30 min", fontsize=9.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
sauve(fig, "f15_etiquette.png")

print("TERMINE -- figures seance 2")
print(json.dumps({k: K[k] for k in ["lignes_brutes", "doublons_exactes", "nan_total",
      "nan_pct", "temp_max_brute", "derive_srv06_C_par_jour", "serveurs_15min",
      "err_ffill_cpu", "err_interp_cpu", "nb_fenetres_10min",
      "nb_fenetres_incident_futur", "taux_positifs_pct"]}, ensure_ascii=False, indent=1))
