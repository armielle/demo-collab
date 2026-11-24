# -*- coding: utf-8 -*-
"""Seance 5 (EPD-GIT-5534) -- Securite et fiabilite : detection d'intrusion REELLE.
Pipeline complet depuis data/nosys_logs.csv (graine 5534, threads 2) :
  1. parse + agregation par (machine, fenetre 10 min)
  2. features d'attaque (echecs consecutifs, utilisateurs distincts,
     entropie de Shannon des comptes testes, part de comptes sensibles, nuit)
  3. IsolationForest non supervise + classement top-N + precision@N
  4. recouvrement avec les sessions d'attaque reelles
     (bdd-principal, 02:00-04:00 le 05/10/2026)
  5. classification supervisee (foret aleatoire sur l'etiquette session-attaque)
Ecrit figures/key_numbers_git5.json AVANT les figures."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import entropy as shannon
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
SEED = 5534
MSG_ECHEC = "Echec d'authentification (mot de passe invalide)"

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

# ============================================================ 1. parse
L = pd.read_csv(os.path.join(ROOT, "data", "nosys_logs.csv"), parse_dates=["horodatage"])
L["echec"] = (~L["auth_reussie"].astype(bool)).astype(int)
L["succes"] = 1 - L["echec"]
L["fenetre"] = L["horodatage"].dt.floor("10min")
L["heure"] = L["horodatage"].dt.hour
SENSIBLES = {"root", "admin", "inconnu-1", "inconnu-2"}

# ============================================================ 2. agregation par (machine, 10 min)
g = L.groupby(["machine", "fenetre"])
S = pd.DataFrame({
    "n_evenements": g.size(),
    "echecs": g["echec"].sum(),
    "succes": g["succes"].sum(),
    "utilisateurs": g["utilisateur"].nunique(),
    "comptes_sensibles": g.apply(
        lambda d: int(d.loc[d["echec"] == 1, "utilisateur"].isin(SENSIBLES).sum()),
        include_groups=False),
})
# entropie de Shannon (base 2) des comptes testes dans la fenetre
S["entropie"] = g["utilisateur"].apply(
    lambda u: float(shannon(list(pd.Series(list(u)).value_counts().values), base=2)))
# echecs consecutifs max dans la fenetre (ordre horodatage)
def max_run(d):
    e = d.sort_values("horodatage")["echec"].values
    best = cur = 0
    for v in e:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best
S["echecs_consecutifs"] = g.apply(max_run, include_groups=False)
S["taux_echec"] = S["echecs"] / S["n_evenements"]
S["heure"] = S.index.get_level_values("fenetre").hour
S["nuit"] = ((S["heure"] < 6) | (S["heure"] >= 22)).astype(int)
S = S.reset_index()

# ============================================================ 3. etiquette session-attaque
att_mask = ((S["machine"] == "bdd-principal")
            & (S["fenetre"] >= pd.Timestamp("2026-10-05 02:00"))
            & (S["fenetre"] < pd.Timestamp("2026-10-05 04:00")))
S["attaque"] = att_mask.astype(int)

# ============================================================ chiffres cles (JSON AVANT figures)
K = {}
K["lignes_logs"] = int(len(L))
K["machines"] = int(L["machine"].nunique())
K["echecs_total_7j"] = int(L["echec"].sum())
K["echecs_par_jour"] = round(K["echecs_total_7j"] / 7, 1)
K["echecs_attaque"] = int(L["echec"][
    L["machine"].eq("bdd-principal")
    & L["horodatage"].between(pd.Timestamp("2026-10-05 02:00"),
                              pd.Timestamp("2026-10-05 04:00"),
                              inclusive="left")].sum())
K["nb_sessions"] = int(len(S))
K["nb_sessions_attaque"] = int(S["attaque"].sum())
K["taux_sessions_attaque_pct"] = round(100 * K["nb_sessions_attaque"] / K["nb_sessions"], 3)
att = S[S["attaque"] == 1]
base = S[S["attaque"] == 0]
K["echecs_session_attaque_moy"] = round(float(att["echecs"].mean()), 1)
K["echecs_session_attaque_max"] = int(att["echecs"].max())
K["echecs_session_normale_max"] = int(base["echecs"].max())
K["entropie_moy_attaque"] = round(float(att["entropie"].mean()), 2)
K["entropie_moy_normale"] = round(float(base["entropie"].mean()), 2)
K["entropie_attaque_max"] = round(float(att["entropie"].max()), 2)
K["utilisateurs_attaque_moy"] = round(float(att["utilisateurs"].mean()), 1)
K["succes_session_attaque"] = int(att["succes"].sum())
K["part_comptes_sensibles_attaque_pct"] = round(
    100 * float(att["comptes_sensibles"].sum()) / max(int(att["echecs"].sum()), 1), 1)

# ============================================================ 4. IsolationForest non supervise
FEATS = ["echecs", "taux_echec", "utilisateurs", "entropie",
         "echecs_consecutifs", "comptes_sensibles", "nuit"]
X = S[FEATS].values
iso = IsolationForest(n_estimators=200, contamination=0.005, random_state=SEED, n_jobs=2)
iso.fit(X)
S["score_anomalie"] = -iso.decision_function(X)
S = S.sort_values("score_anomalie", ascending=False).reset_index(drop=True)
S["rang"] = np.arange(1, len(S) + 1)

def precision_at(n):
    return round(float(S.loc[:n - 1, "attaque"].mean()), 3)
K["precision_at_30"] = precision_at(30)
K["sessions_attaque_top30"] = int(S.loc[:29, "attaque"].sum())
K["rappel_top30"] = round(K["sessions_attaque_top30"] / K["nb_sessions_attaque"], 3)
K["rang_pire_session_attaque"] = int(S.loc[S["attaque"] == 1, "rang"].max())
K["rang_meilleure_session_normale"] = int(S.loc[S["attaque"] == 0, "rang"].min())
prec_n = [(n, precision_at(n)) for n in range(1, 61)]
K["precision_at_10"] = precision_at(10)
K["precision_at_50"] = precision_at(50)
K["baseline_precision_at_30"] = round(K["nb_sessions_attaque"] / K["nb_sessions"], 5)

# ============================================================ 5. classification supervisee
X = S[FEATS].values          # S est trie par rang : X et etiquettes alignes
y = S["attaque"].values
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=SEED)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=SEED, n_jobs=2)
rf.fit(Xtr, ytr)
pred = rf.predict(Xte)
from sklearn.metrics import precision_score, recall_score
K["rf_precision"] = round(float(precision_score(yte, pred, zero_division=0)), 3)
K["rf_recall"] = round(float(recall_score(yte, pred)), 3)
K["rf_positives_test"] = int(yte.sum())
K["rf_features_importance"] = {f: round(float(i), 3) for f, i in
                               sorted(zip(FEATS, rf.feature_importances_),
                                      key=lambda t: -t[1])}

with open(os.path.join(FIG, "key_numbers_git5.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, ensure_ascii=False, indent=1)
print("JSON ecrit :", len(K), "cles")
print(json.dumps(K, indent=1, ensure_ascii=False))

def sauve(fig, nom):
    fig.savefig(os.path.join(FIG, nom), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", nom)

# ============================================================ f25 : echecs par fenetre 10 min, 7 jours
fig, ax = plt.subplots(2, 1, figsize=(11.2, 5.4), gridspec_kw={"height_ratios": [1, 1]})
# a) parc entier : total des echecs par fenetre 10 min
tot = L.groupby("fenetre")["echec"].sum()
ax[0].fill_between(tot.index, tot.values, color=C["blue"], alpha=.35, lw=0)
ax[0].plot(tot.index, tot.values, color=C["blue"], lw=.7)
t0 = pd.Timestamp("2026-10-05 02:00"); t1 = pd.Timestamp("2026-10-05 04:00")
ax[0].axvspan(t0, t1, color=C["red"], alpha=.25)
ax[0].annotate("intrusion brute force :\n125 echecs en 2 h", xy=(t0 + pd.Timedelta("20min"), 16.5),
               xytext=(pd.Timestamp("2026-10-01 12:00"), 16.2), fontsize=9, color=C["red"],
               arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.2))
ax[0].set_ylabel("echecs / 10 min (parc)")
ax[0].set_title("a) Echecs d'authentification du parc, 7 jours (grille 10 min)", fontsize=9)
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
# b) zoom bdd-principal, 5/10 00 h - 06 h
z = S[(S["machine"] == "bdd-principal")
      & S["fenetre"].between(pd.Timestamp("2026-10-05 00:00"), pd.Timestamp("2026-10-05 06:00"))]
norm = z[z["attaque"] == 0]; attz = z[z["attaque"] == 1]
ax[1].bar(norm["fenetre"], norm["echecs"], width=0.006, color=C["blue"], label="sessions normales")
ax[1].bar(attz["fenetre"], attz["echecs"], width=0.006, color=C["red"], label="sessions d'attaque")
for _, r in attz.iterrows():
    ax[1].text(r["fenetre"], r["echecs"] + .3, str(int(r["echecs"])), ha="center",
               fontsize=7, color=C["red"])
ax[1].set_ylabel("echecs / 10 min (bdd-principal)")
ax[1].set_title("b) Zoom : bdd-principal, nuit du 05/10 (02 h - 04 h) --- le pic est evident",
                fontsize=9)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
sauve(fig, "f25_echecs_7j.png")

# ============================================================ f26 : entropie vs echecs
fig, ax = plt.subplots(figsize=(7.4, 4.6))
nb = base.sample(min(2500, len(base)), random_state=SEED)
ax.scatter(nb["echecs"], nb["entropie"], s=14, color=C["blue"], alpha=.35,
           label="sessions normales (%d, echantillon)" % len(nb))
ax.scatter(att["echecs"], att["entropie"], s=42, color=C["red"], marker="D",
           label="sessions d'attaque (%d)" % len(att))
ax.axhline(K["entropie_moy_normale"], color=C["grey"], ls="--", lw=.9)
ax.axvline(K["echecs_session_normale_max"] + .5, color=C["orange"], ls="--", lw=.9)
ax.text(K["echecs_session_normale_max"] + .8, 3.05, "aucune session normale\ndepasse 4 echecs",
        fontsize=8, color=C["orange"])
ax.set_xlabel("echecs d'authentification dans la fenetre")
ax.set_ylabel("entropie de Shannon des comptes testes (bits)")
ax.set_title("Deux dimensions separment l'attaque du bruit de fond", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
sauve(fig, "f26_entropie_echecs.png")

# ============================================================ f27 : precision@N de l'IF
Ns = np.array([n for n, _ in prec_n])
Ps = np.array([p for _, p in prec_n])
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(Ns, Ps, color=C["purple"], lw=1.8, marker="o", ms=3,
        label="IsolationForest (score d'anomalie)")
ax.axhline(K["baseline_precision_at_30"] * 30, color=C["grey"], ls=":", lw=1.2,
           label="hasard (0,12 %% des sessions sont attaque)".replace("%%", "%"))
ax.axvline(30, color=C["orange"], ls="--", lw=1.1)
ax.annotate("precision@30 = %.2f\n(%d/30 sessions d'attaque retrouvees)"
            % (K["precision_at_30"], K["sessions_attaque_top30"]),
            xy=(30, K["precision_at_30"]), xytext=(36, 0.16), fontsize=9, color=C["deep"],
            arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.2))
ax.set_xlabel("N : nombre de sessions les plus anormales examinees")
ax.set_ylabel("precision@N")
ax.set_ylim(-0.03, 1.03)
ax.set_title("L'analyste ne regarde que le haut du classement", fontsize=10)
ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
sauve(fig, "f27_precision_at_n.png")

# ============================================================ f28 : non-supervise vs supervise
fig, ax = plt.subplots(figsize=(7.4, 4.4))
labels = ["IF\nprecision@30", "IF\nrappel (top 30)",
          "Foret supervisee\nprecision (test)", "Foret supervisee\nrappel (test)"]
vals = [K["precision_at_30"], K["rappel_top30"], K["rf_precision"], K["rf_recall"]]
cols = [C["purple"], C["purple"], C["green"], C["green"]]
bars = ax.bar(labels, vals, color=cols, width=.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + .02, "%.2f" % v, ha="center",
            fontsize=10, fontweight="bold", color=C["deep"])
ax.axhline(K["baseline_precision_at_30"], color=C["red"], ls=":", lw=1.2,
           label="classe majoritaire (0,0012)")
ax.set_ylim(0, 1.12)
ax.set_ylabel("score")
ax.set_title("Non-supervise (sans etiquettes) vs supervise (avec etiquettes)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
sauve(fig, "f28_supervise_vs_non.png")

print("FIGURES OK")
