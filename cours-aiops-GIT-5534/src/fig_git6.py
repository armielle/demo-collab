# -*- coding: utf-8 -*-
"""Pipeline REEL de la Seance 6 (EPD-GIT-5534) -- industrialisation.

Entraîne une forêt aléatoire sur le train (< 2026-09-22), la sauvegarde avec
joblib, la recharge, mesure la latence réelle de scoring (perf_counter sur
1 000 prédictions : moyenne + p99), l'AUC sur le test, puis l'AUC par
tranches de 2 jours du test (dérive mesurée).
Ecrit figures/key_numbers_git6.json AVANT les figures. Graine 5534, threads 2."""
import json, os, time
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "out")
SEED = 5534
SPLIT = pd.Timestamp("2026-09-22")

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

# ------------------------------------------------------------------ données
F = pd.read_csv(os.path.join(ROOT, "data", "nosys_features.csv"),
                parse_dates=["fenetre"])
FEN = ["cpu_moy", "cpu_max", "ram_moy", "lat_moy", "lat_p99", "perte_moy",
       "temp_moy", "conso_moy", "cpu_delta", "cpu_amplitude",
       "h_sin", "h_cos", "js_sin", "js_cos"]
F = F.dropna(subset=FEN + ["incident_30min"]).sort_values("fenetre")
y = F["incident_30min"].astype(int)
train = F["fenetre"] < SPLIT
test = F["fenetre"] >= SPLIT
Xtr, ytr = F.loc[train, FEN], y[train]
Xte, yte = F.loc[test, FEN], y[test]

K = {}
K["n_train"] = int(len(Xtr)); K["pos_train"] = int(ytr.sum())
K["n_test"] = int(len(Xte)); K["pos_test"] = int(yte.sum())
K["n_features"] = len(FEN)

# ------------------------------------------------- entraînement + joblib
clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                             class_weight="balanced", random_state=SEED,
                             n_jobs=2)
t0 = time.perf_counter()
clf.fit(Xtr.values, ytr.values)
K["temps_entrainement_s"] = round(time.perf_counter() - t0, 2)

# impuretés -> moyennes zéro pour un joblib propre
for att in ("tree_",):
    pass
mod_path = os.path.join(OUT, "nosys_forest.joblib")
joblib.dump({"modele": clf, "variables": FEN, "seance": 6, "graine": SEED},
            mod_path)
K["taille_joblib_ko"] = round(os.path.getsize(mod_path) / 1024, 1)

# rechargement
bundle = joblib.load(mod_path)
mod = bundle["modele"]
assert bundle["variables"] == FEN

# ------------------------------------------------- AUC test réel
proba = mod.predict_proba(Xte.values)[:, 1]
K["auc_test"] = round(float(roc_auc_score(yte, proba)), 4)
auc_tr = roc_auc_score(ytr, mod.predict_proba(Xtr.values)[:, 1])
K["auc_train"] = round(float(auc_tr), 4)

# ------------------------------------------------- latence réelle : 1 000 prédictions
Xl = Xte.values
for _ in range(20):                       # warm-up
    mod.predict_proba(Xl[:32])
lat = []
for i in range(1000):
    x = Xl[i * 7 % len(Xl): i * 7 % len(Xl) + 1]
    t0 = time.perf_counter()
    mod.predict_proba(x)
    lat.append(time.perf_counter() - t0)
lat = np.array(lat) * 1000.0              # ms
K["latence_moyenne_ms"] = round(float(lat.mean()), 3)
K["latence_p99_ms"] = round(float(np.percentile(lat, 99)), 3)
K["fenetres_par_minute_1coeur"] = int(round(60_000 / lat.mean()))
# scoring par lots (100 fenêtres d'un coup)
t0 = time.perf_counter()
for _ in range(10):
    mod.predict_proba(Xl[:100])
lot = (time.perf_counter() - t0) / 10 * 1000
K["latence_lot100_ms"] = round(float(lot), 3)
K["fenetres_par_minute_lot100"] = int(round(100 * 60_000 / lot))

# ------------------------------------------------- dérive : fenêtres de 2 jours du test
# Les positifs du test sont concentrés sur la journée du 22/09 (incident
# majeur) : sur les tranches sans incident, l'AUC n'est PAS calculable --
# c'est exactement le problème réel du monitoring. On mesure alors la
# dérive de la distribution des scores (proba moyenne, taux d'alertes).
Fte = F.loc[test].copy()
Fte["proba"] = proba
Fte["y"] = yte.values
tranches, aucs, alertes, probas, ns = [], [], [], [], []
d0 = Fte["fenetre"].min().normalize()
d1 = Fte["fenetre"].max().normalize()
jour = d0
while jour <= d1:
    m = (Fte["fenetre"] >= jour) & (Fte["fenetre"] < jour + pd.Timedelta(days=2))
    g = Fte[m]
    if len(g) > 50:
        tranches.append(jour.strftime("%d/%m"))
        if g["y"].nunique() == 2:
            aucs.append(round(float(roc_auc_score(g["y"], g["proba"])), 4))
        else:
            aucs.append(None)                    # pas d'incident -> AUC n/d
        alertes.append(round(float((g["proba"] >= 0.5).mean()), 4))
        probas.append(round(float(g["proba"].mean()), 4))
        ns.append(int(len(g)))
    jour += pd.Timedelta(days=2)
aucs_ok = [a for a in aucs if a is not None]
K["derive_tranches"] = tranches
K["derive_auc"] = aucs                # None = pas d'incident dans la tranche
K["derive_auc_mesurable"] = aucs_ok
K["taux_alertes_par_tranche"] = alertes
K["proba_moyenne_par_tranche"] = probas
K["tailles_tranches"] = ns
K["auc_min_tranche"] = min(aucs_ok); K["auc_max_tranche"] = max(aucs_ok)

with open(os.path.join(FIG, "key_numbers_git6.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, ensure_ascii=False, indent=1)
print("JSON ecrit :", len(K), "cles")

def sauve(fig, nom):
    fig.savefig(os.path.join(FIG, nom), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  ->", nom)

# ================================================= f29 : dérive mesurée par tranche
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.6))
# a) AUC par tranche (n/d quand la tranche ne contient aucun incident)
for i, a in enumerate(aucs):
    if a is None:
        ax[0].bar(i, 0.5, color=C["grey"], alpha=.25, width=.6, hatch="///",
                  edgecolor=C["grey"])
        ax[0].text(i, .52, "n/d", ha="center", fontsize=8, color=C["grey"])
    else:
        ax[0].bar(i, a, color=C["blue"], alpha=.85, width=.6)
        ax[0].text(i, a + .012, f"{a:.2f}", ha="center", fontsize=8, color=C["deep"])
ax[0].axhline(K["auc_test"], color=C["orange"], ls="--", lw=1.4,
              label=f"AUC test globale = {K['auc_test']:.3f}")
ax[0].axhline(0.5, color=C["grey"], lw=1)
ax[0].set_xticks(range(len(tranches))); ax[0].set_xticklabels(tranches, fontsize=8, rotation=45)
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("AUC")
ax[0].set_title("a) AUC par tranche de 2 jours (test)", fontsize=9.5)
ax[0].legend(fontsize=7.5, loc="upper right")
# b) dérive de la distribution des scores
ax[1].plot(range(len(tranches)), probas, "o-", color=C["purple"], lw=1.6,
           label="proba moyenne prédite")
ax[1].plot(range(len(tranches)), alertes, "s--", color=C["red"], lw=1.6,
           label="taux d'alertes (seuil 0,5)")
ax[1].set_xticks(range(len(tranches))); ax[1].set_xticklabels(tranches, fontsize=8, rotation=45)
ax[1].set_ylabel("proportion")
ax[1].set_title("b) Dérive des scores : que surveiller sans incident", fontsize=9.5)
ax[1].legend(fontsize=7.5)
sauve(fig, "f29_auc_tranches.png")

# ================================================= f30 : latence de scoring
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
b = ax[0].bar(["moyenne", "p99"], [lat.mean(), np.percentile(lat, 99)],
              color=[C["cyan"], C["orange"]], width=.5)
for r, v in zip(b, [lat.mean(), np.percentile(lat, 99)]):
    ax[0].text(r.get_x() + r.get_width() / 2, v * 1.06, f"{v:.2f} ms",
               ha="center", fontsize=9, color=C["deep"])
ax[0].set_ylim(0, np.percentile(lat, 99) * 1.3)
ax[0].set_ylabel("ms par fenêtre")
ax[0].set_title("Latence réelle : 1 000 prédictions individuelles", fontsize=9.5)
ax[1].bar(["1 par 1", "lot de 100"],
          [60_000 / lat.mean(), 100 * 60_000 / lot],
          color=[C["blue"], C["purple"]], width=.5)
ax[1].set_ylabel("fenêtres / minute")
ax[1].set_title("Capacité de scoring d'un seul cœur", fontsize=9.5)
sauve(fig, "f30_latence.png")

print("TERMINE -- figures seance 6")
print(json.dumps({k: K[k] for k in ["auc_test", "auc_train", "latence_moyenne_ms",
      "latence_p99_ms", "fenetres_par_minute_1coeur", "fenetres_par_minute_lot100",
      "taille_joblib_ko", "temps_entrainement_s", "auc_min_tranche",
      "auc_max_tranche"]}, ensure_ascii=False, indent=1))
