# -*- coding: utf-8 -*-
"""Fabrique et exécute réellement les 6 notebooks corrigés du cours EPD-GIT-5534.

Pattern repris de cours-bayesien-SDIA-4112/src/mknb.py : nbformat v4 +
nbclient.NotebookClient, noyau 'sdia-bayes', timeout 600 s/cellule.
Une phase de pré-calcul (build) mesure les chiffres clés pour les injecter
dans les cellules markdown (constats, notes d'incident, fiches).
"""
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = os.path.join(ROOT, "notebooks")
os.makedirs(NB, exist_ok=True)
KERNEL = "sdia-bayes"

from scipy.stats import entropy as _entropy
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, IsolationForest)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, precision_score,
                             recall_score, f1_score, average_precision_score,
                             roc_auc_score, confusion_matrix)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------------------------------------------------------
# pré-calcul build : chiffres mesurés pour les markdown
# ---------------------------------------------------------------------------
print("== pré-calcul des chiffres des markdown ==")
m = pd.read_csv(os.path.join(ROOT, "data", "nosys_metriques.csv"),
                parse_dates=["horodatage"])
par_srv = (m.groupby("serveur")
             .agg(cpu_moy=("cpu_pct", "mean"), cpu_max=("cpu_pct", "max"),
                  lat_p99=("latence_ms", lambda s: s.quantile(0.99)),
                  temp_max=("temperature_c", "max"), alarmes=("alarme", "sum"),
                  conso_kwh_tot=("conso_kwh", lambda s: s.sum() * 5 / 60),
                  capacite=("capacite_cpu", "first")))
par_srv["ratio_capacite"] = par_srv["cpu_moy"] / (par_srv["capacite"] * 100)
cpu_parc = float(m["cpu_pct"].mean())
p99_parc = float(m["latence_ms"].quantile(0.99))
p99_hors_inc = float(m.loc[m["alarme"] == 0, "latence_ms"].quantile(0.99))
dispo = 1 - float(m["alarme"].mean())
indispo_h = float(m["alarme"].sum() * 5 / 60)
conso_tot = float(par_srv["conso_kwh_tot"].sum())
top_conso = par_srv["conso_kwh_tot"].sort_values(ascending=False).index[:2].tolist()
hors = m[m["alarme"] == 0]
corr_lc = float(hors["cpu_pct"].corr(hors["latence_ms"]))
grp_inc = m[m["alarme"] == 1].groupby(
    m["horodatage"].dt.strftime("%d/%m à %Hh"))["serveur"].nunique()
creneaux = grp_inc[grp_inc == 8].index.tolist()
cpu_incident = float(m.loc[m["alarme"] == 1, "cpu_pct"].mean() / cpu_parc)

L = pd.read_csv(os.path.join(ROOT, "data", "nosys_logs.csv"),
                parse_dates=["horodatage"])
att_mask = (L["machine"] == "bdd-principal") & L["horodatage"].between(
    "2026-10-05 02:00", "2026-10-05 04:00")
n_ech_att = int((~L.loc[att_mask, "auth_reussie"].astype(bool)).sum())
n_succ_att = int(L.loc[att_mask, "auth_reussie"].astype(bool).sum())
comptes_att = sorted(L.loc[att_mask & ~L["auth_reussie"].astype(bool),
                           "utilisateur"].unique())
n_ech_tot = int((~L["auth_reussie"].astype(bool)).sum())

f_all = pd.read_csv(os.path.join(ROOT, "data", "nosys_features.csv"),
                    parse_dates=["fenetre"])
n_pos = int(f_all["incident_30min"].sum())
tr = f_all["fenetre"] < "2026-09-22"
n_train, n_pos_tr = int(tr.sum()), int(f_all.loc[tr, "incident_30min"].sum())
n_test, n_pos_te = int((~tr).sum()), int(f_all.loc[~tr, "incident_30min"].sum())

# --- TP2 : dérive srv-06 ---------------------------------------------------
b_probe = pd.read_csv(os.path.join(ROOT, "data", "nosys_metriques_brutes.csv"),
                      parse_dates=["horodatage"]).sort_values(
                          ["serveur", "horodatage"]).reset_index(drop=True)
n_dup = int(b_probe.duplicated().sum())
nan_cols = b_probe[["cpu_pct", "ram_pct", "latence_ms", "paquets_perdus_pct",
                    "temperature_c"]].isna().sum()
n_cpu_imp = int((b_probe["cpu_pct"] > 100).sum())
n_temp_imp = int((b_probe["temperature_c"] > 80).sum())
pts_srv = b_probe.groupby("serveur").size()
b6 = b_probe.drop_duplicates()
b6.loc[b6["cpu_pct"] > 100, "cpu_pct"] = np.nan
b6.loc[b6["temperature_c"] > 80, "temperature_c"] = np.nan
g6 = b6[b6["serveur"] == "srv-06"].set_index("horodatage").drop(
    columns=["serveur"])
g6 = g6.reindex(pd.date_range(g6.index.min(), g6.index.max(), freq="5min"))
g6 = g6.interpolate(limit=2, limit_direction="both")
t6d = g6["temperature_c"].resample("D").mean().dropna()
pente_srv06 = float(np.polyfit(np.arange(len(t6d)), t6d.values, 1)[0])

# --- TP4 : ADF + SARIMA ----------------------------------------------------
s_h = m.set_index("horodatage")["cpu_pct"].resample("1h").mean()
_, p_adf, *_ = adfuller(s_h)
th_ = np.arange(len(s_h))
Xh = np.column_stack([gfn(th_ * 2 * np.pi * j / 168) for j in (1, 2, 3)
                      for gfn in (np.sin, np.cos)])
res_h = SARIMAX(s_h.iloc[:576], exog=Xh[:576], order=(2, 0, 1),
                seasonal_order=(1, 0, 1, 24)).fit(disp=False)
mae_s = float(np.mean(np.abs(
    res_h.get_forecast(144, exog=Xh[576:]).predicted_mean - s_h.iloc[576:])))
pers24_s = float(np.mean(np.abs(
    s_h.iloc[576:].values - s_h.shift(24).iloc[576:].values)))
gain_s = 100 * (1 - mae_s / pers24_s)

# --- TP3 : modèles et seuil métier -----------------------------------------
X3 = f_all.drop(columns=["serveur", "fenetre", "incident_30min"]).fillna(0.0)
Xtr3, ytr3 = X3[tr], f_all.loc[tr, "incident_30min"]
Xte3, yte3 = X3[~tr], f_all.loc[~tr, "incident_30min"]
gbc_b = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                   learning_rate=0.06,
                                   random_state=5534).fit(Xtr3, ytr3)
proba_b = gbc_b.predict_proba(Xte3)[:, 1]
prec_b, rec_b, thr_b = precision_recall_curve(yte3, proba_b)
alert_b = np.array([(proba_b >= s).sum() for s in thr_b])
ok_b = np.where(alert_b <= 360)[0]
i_b = int(ok_b[np.argmax(rec_b[ok_b])])
seuil_b = float(thr_b[i_b])
prec_i, rec_i = float(prec_b[i_b]), float(rec_b[i_b])
f1_i = 2 * prec_i * rec_i / max(prec_i + rec_i, 1e-9)
alertes_b = float(alert_b[i_b] / 72)
ap_b = float(average_precision_score(yte3, proba_b))
rf_b = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                              random_state=5534, n_jobs=2).fit(Xtr3, ytr3)
prf_b = rf_b.predict_proba(Xte3)[:, 1]
pred_rf_b = (prf_b >= 0.5).astype(int)
logit_b = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced",
                                           random_state=5534)).fit(Xtr3, ytr3)
plg_b = logit_b.predict_proba(Xte3)[:, 1]
pred_lg_b = (plg_b >= 0.5).astype(int)

# --- TP5 : sessions, IF, forêt supervisée ----------------------------------
L5 = L.sort_values("horodatage").copy()
L5["echec"] = (~L5["auth_reussie"].astype(bool)).astype(int)
L5["fenetre"] = L5["horodatage"].dt.floor("10min")
g5 = L5.groupby(["machine", "fenetre"])
S5 = pd.DataFrame({"n_evenements": g5.size(), "echecs": g5["echec"].sum(),
                   "succes": (1 - L5["echec"]).groupby(
                       [L5["machine"], L5["fenetre"]]).sum(),
                   "utilisateurs": g5["utilisateur"].nunique()})
SENS = {"root", "admin", "inconnu-1", "inconnu-2"}
ent5, run5, sen5 = {}, {}, {}
for cle, d in g5:
    ent5[cle] = _entropy(d["utilisateur"].value_counts().values, base=2)
    e = d.sort_values("horodatage")["echec"].values
    best = cur = 0
    for v in e:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    run5[cle] = best
    sen5[cle] = int(d.loc[d["echec"] == 1, "utilisateur"].isin(SENS).sum())
S5["taux_echec"] = S5["echecs"] / S5["n_evenements"]
S5["entropie"] = pd.Series(ent5)
S5["echecs_consecutifs"] = pd.Series(run5)
S5["comptes_sensibles"] = pd.Series(sen5)
S5["nuit"] = (S5.index.get_level_values("fenetre").hour < 6).astype(int)
FEATS5 = ["echecs", "taux_echec", "utilisateurs", "entropie",
          "echecs_consecutifs", "comptes_sensibles", "nuit"]


def attaque_session(cle):
    mach, fen = cle
    return (mach == "bdd-principal"
            and pd.Timestamp("2026-10-05 02:00") <= fen
            <= pd.Timestamp("2026-10-05 04:00"))


iso_b = IsolationForest(n_estimators=200, contamination=0.005,
                        random_state=5534, n_jobs=2)
S5["score"] = -iso_b.fit(S5[FEATS5].values).decision_function(S5[FEATS5].values)
top5_b = S5.sort_values("score", ascending=False).head(30)
flags5 = np.array([attaque_session(i) for i in top5_b.index])
p10_b = float(flags5[:10].mean())
p30_b = float(flags5.mean())
rangs_b = [k + 1 for k, a in enumerate(flags5) if a]
y5 = pd.Series([int(attaque_session(i)) for i in S5.index], index=S5.index)
Xtr5, Xte5, ytr5, yte5 = train_test_split(S5[FEATS5], y5, test_size=0.5,
                                          stratify=y5, random_state=5534)
rf5_b = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=5534, n_jobs=2).fit(Xtr5, ytr5)
pred5_b = rf5_b.predict(Xte5)
pr5_b = float(precision_score(yte5, pred5_b, zero_division=0))
rc5_b = float(recall_score(yte5, pred5_b, zero_division=0))

# --- TP6 : pipeline, AUC, latence ------------------------------------------
X6 = f_all.drop(columns=["serveur", "fenetre", "incident_30min"]).fillna(0.0)
pipe_b = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=200,
                                              min_samples_leaf=2,
                                              class_weight="balanced",
                                              random_state=5534)).fit(
    X6[tr], f_all.loc[tr, "incident_30min"])
auc_b = float(roc_auc_score(f_all.loc[~tr, "incident_30min"],
                            pipe_b.predict_proba(X6[~tr])[:, 1]))
Xte6b = X6[~tr]
idx6 = np.random.default_rng(5534).choice(len(Xte6b), size=1000, replace=False)
lats6 = []
for i in idx6:
    t0 = time.perf_counter()
    pipe_b.predict_proba(Xte6b.iloc[[i]])
    lats6.append(time.perf_counter() - t0)
lats6 = np.array(lats6) * 1000
lat_moy_b = float(lats6.mean())
lat_p99_b = float(np.percentile(lats6, 99))
print("   pré-calcul OK (SARIMA MAE %.2f, GBC seuil %.3f, IF p@30 %.2f, "
      "AUC %.3f, latence %.1f ms)" % (mae_s, seuil_b, p30_b, auc_b, lat_moy_b))

PREAMBLE = """%matplotlib inline
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RNG = np.random.default_rng(5534)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
print("numpy", np.__version__, "| pandas", pd.__version__)
"""


def build(name, cells, title, intro):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell("# " + title + "\n\n" + intro)]
    nb.cells.append(nbf.v4.new_code_cell(PREAMBLE))
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                        else nbf.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": "SDIA Bayes",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def execute(nb, name, timeout=600):
    client = NotebookClient(nb, timeout=timeout, kernel_name=KERNEL,
                            resources={"metadata": {"path": ROOT}},
                            allow_errors=False)
    client.execute()
    path = os.path.join(NB, name)
    nbf.write(nb, path)
    imgs = sum(1 for c in nb.cells for o in c.get("outputs", [])
               if "data" in o and "image/png" in o.get("data", {}))
    ncode = sum(1 for c in nb.cells if c.cell_type == "code")
    print("[nb] %-34s %3d cellules (%d code, %d figures) %7.1f Ko"
          % (name, len(nb.cells), ncode, imgs, os.path.getsize(path) / 1024))
    return path


# ===========================================================================
# TP1 — mise en route
# ===========================================================================
C1 = [
("md", "## B1 — Charger et vérifier (30 min)\n\n"
       "**Objectifs :** charger `nosys_metriques.csv` avec les bons types, "
       "vérifier la forme (69 120 lignes = 8 serveurs × 30 j × 288 fenêtres), "
       "la constance du pas de 5 min et l'absence de valeurs manquantes. "
       "Pièges : oublier `parse_dates`, travailler dans le mauvais répertoire."),
("code", '''m = pd.read_csv("data/nosys_metriques.csv", parse_dates=["horodatage"])
print("shape :", m.shape)                       # attendu (69120, 10)
print("serveurs :", sorted(m["serveur"].unique()))
print("période :", m["horodatage"].min(), "->", m["horodatage"].max())
print()
print("pas d'échantillonnage (les 5 plus fréquents) :")
print(m["horodatage"].diff().value_counts().head())
print()
print("valeurs manquantes par colonne :")
print(m.isna().sum())'''),
("md", "**Attendu :** shape `(69120, 10)`, 8 serveurs, pas constant de 300 s "
       "(sauf le premier point, `NaT`), 0 valeur manquante partout."),
("md", "## B2 — Inventaire de qualité (20 min)\n\n"
       "Pour chaque colonne : type, plage min/max, manquants. Puis complétude "
       "(8 640 fenêtres par serveur) et plausibilité physique "
       "(`0 ≤ cpu_pct ≤ 100`, `ram_pct` dans [0, 100], température > 20 °C)."),
("code", '''inventaire = pd.DataFrame({
    "type": m.dtypes.astype(str),
    "min": m.select_dtypes("number").min(),
    "max": m.select_dtypes("number").max(),
    "manquants": m.isna().sum()})
print(inventaire)
print()
print("fenêtres par serveur :")
print(m.groupby("serveur").size())
assert (m.groupby("serveur").size() == 8640).all(), "complétude rompue"
print()
print("plausibilité : CPU %.1f..%.1f %%, RAM %.1f..%.1f %%, temp %.1f..%.1f °C"
      % (m["cpu_pct"].min(), m["cpu_pct"].max(),
         m["ram_pct"].min(), m["ram_pct"].max(),
         m["temperature_c"].min(), m["temperature_c"].max()))
print("=> 0 valeur manquante, bornes physiques respectées : dataset propre.")'''),
("md", "**Réponse attendue :** 0 valeur manquante ; CPU max "
       + str(round(float(m["cpu_pct"].max()), 1)) + " % ; température max "
       + str(round(float(m["temperature_c"].max()), 1)) + " °C : tout est "
       "plausible — dit avec la preuve."),
("md", "## B3 — Visualiser srv-01 sur 30 jours (30 min)\n\n"
       "Vue journalière (3 sous-graphiques : CPU, latence, alarmes) puis "
       "réponse aux questions : où sont les 2 incidents de srv-01 ? Un seuil "
       "fixe « CPU > 90 % » les aurait-il détectés ? La latence monte-t-elle "
       "avec le CPU ?"),
("code", '''s1 = m[m["serveur"] == "srv-01"]
jour = (s1.set_index("horodatage").drop(columns=["serveur"])
          .resample("D").mean())            # 30 lignes, une par jour

fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
ax[0].plot(jour.index, jour["cpu_pct"]);  ax[0].set_ylabel("CPU %")
ax[1].plot(jour.index, jour["latence_ms"]); ax[1].set_ylabel("latence ms")
ax[2].bar(jour.index, jour["alarme"] * 4, width=0.7)  # fenêtres x4 visibles
ax[2].set_ylabel("alarmes (x4)")
ax[2].set_xlabel("date")
fig.suptitle("srv-01 — vue journalière (moyennes par jour)")
plt.show()

jours_alarme = s1.loc[s1["alarme"] == 1, "horodatage"].dt.strftime("%d/%m %Hh")
print("fenêtres d'alarme srv-01 :", sorted(jours_alarme.unique()))
print("CPU moyen pendant alarme : %.1f %% vs hors alarme : %.1f %%"
      % (s1.loc[s1["alarme"] == 1, "cpu_pct"].mean(),
         s1.loc[s1["alarme"] == 0, "cpu_pct"].mean()))'''),
("md", "**Lecture :** les 2 incidents de srv-01 tombent le 9/09 vers 14 h et "
       "le 22/09 vers 3 h, comme sur tout le parc. Le CPU y reste sous 90 % : "
       "un seuil fixe ne les aurait **pas** détectés. CPU et latence montent "
       "ensemble (corrélation hors incidents ≈ " + str(round(corr_lc, 2))
       + ")."),
("md", "## B4 — Statistiques par serveur (40 min)\n\n"
       "Tableau de classement (`cpu_moy`, `cpu_max`, `lat_p99`, `temp_max`, "
       "alarmes, conso kWh ×5/60, capacité), ratio d'utilisation effective "
       "`cpu_moy / (capacite × 100)`, consommation totale du parc."),
("code", '''par_srv = (m.groupby("serveur")
             .agg(cpu_moy=("cpu_pct", "mean"),
                  cpu_max=("cpu_pct", "max"),
                  lat_p99=("latence_ms", lambda s: s.quantile(0.99)),
                  temp_max=("temperature_c", "max"),
                  alarmes=("alarme", "sum"),
                  conso_kwh_tot=("conso_kwh", lambda s: s.sum() * 5 / 60),
                  capacite=("capacite_cpu", "first")))
par_srv["ratio_capacite"] = par_srv["cpu_moy"] / (par_srv["capacite"] * 100)
print(par_srv.sort_values("cpu_moy", ascending=False).round(2))
print()
print("consommation totale du parc : %.0f kWh" % par_srv["conso_kwh_tot"].sum())
print("ratio capacite max :", par_srv["ratio_capacite"].idxmax(),
      "-> %.3f" % par_srv["ratio_capacite"].max())'''),
("code", '''ordre = par_srv.sort_values("cpu_moy", ascending=False)
cpu_parc = float(m["cpu_pct"].mean())
fig, ax = plt.subplots(figsize=(8, 3.2))
cols = ["#c0392b" if s == "srv-01" else "#4c72b0" for s in ordre.index]
ax.bar(ordre.index, ordre["cpu_moy"], color=cols)
ax.axhline(cpu_parc, color="k", ls="--", lw=1,
           label="moyenne du parc (%.1f %%)" % cpu_parc)
ax.set_ylabel("CPU moyen (%)")
ax.set_title("Classement des serveurs — CPU moyen (srv-01 surligné)")
ax.legend(); plt.show()'''),
("md", "## Bonus — le facteur jour/nuit\n\n"
       "CPU moyen par heure pour srv-01 : vérifier le facteur jour/nuit × 3 "
       "annoncé en cours."),
("code", '''par_heure = s1.groupby(s1["horodatage"].dt.hour)["cpu_pct"].mean()
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(par_heure.index, par_heure.values, marker="o")
ax.set_xlabel("heure de la journée"); ax.set_ylabel("CPU moyen (%)")
ax.set_title("srv-01 — profil journalier du CPU")
plt.show()
facteur = par_heure[8:20].mean() / par_heure[[0, 1, 2, 3, 4, 5, 23]].mean()
print("facteur jour/nuit mesuré : x%.1f" % facteur)'''),
("md", "## B5 — Livrable : la fiche « état des lieux » (20 min)\n\n"
       "**5 constats chiffrés** (chiffre + référence + interprétation), "
       "**3 risques pour le SLA** reliés aux données, **1 recommandation** "
       "d'automatisation. Barème : constats 10 pts, risques 5 pts, notebook "
       "exécutable et reproductible 5 pts.\n\n"
       "### Constats chiffrés (mesurés sur les données du dossier)\n\n"
       "1. Le parc est globalement sain : CPU moyen **"
       + str(round(cpu_parc, 1)) + " %**, p99 de latence **"
       + str(round(p99_parc, 1)) + " ms** pour un SLO à 100 ms (marge ×"
       + str(round(100 / p99_parc, 1)) + ") ; hors incidents le p99 vaut "
       + str(round(p99_hors_inc, 1)) + " ms.\n"
       "2. Mais la disponibilité calculée est **" + str(round(dispo * 100, 2))
       + " %** < SLA 99,9 % : **" + str(round(indispo_h, 1))
       + " h** d'indisponibilité cumulée sur le mois.\n"
       "3. Les 16 incidents suivent **2 créneaux communs** à tous les "
       "serveurs (" + " et ".join(creneaux) + ") : cause partagée probable "
       "(batch, déploiement), pas 16 pannes indépendantes.\n"
       "4. Corrélation latence–CPU **" + str(round(corr_lc, 2))
       + "** hors incidents : la charge explique la latence normale — utile "
       "pour la prévision de la séance 4 ; pendant incident le CPU vaut ×"
       + str(round(cpu_incident, 1)) + " la normale.\n"
       "5. Consommation totale : **" + str(round(conso_tot)) + " kWh**/mois ; "
       + " et ".join(top_conso) + " en tête ; le serveur le plus « tassé » "
       "relativement à sa capacité est **srv-01** (ratio "
       + str(round(float(par_srv.loc["srv-01", "ratio_capacite"]), 2))
       + ", capacité 0,83).\n\n"
       "### Risques SLA\n\n"
       "- Incidents synchrones entre serveurs (cause commune) → une seule "
       "cause peut faire tomber les 8 SLA à la fois.\n"
       "- CPU ×" + str(round(cpu_incident, 1)) + " pendant incident → marge "
       "de capacité faible sur srv-01 (ratio "
       + str(round(float(par_srv.loc["srv-01", "ratio_capacite"]), 2)) + ").\n"
       "- Corrélation latence–CPU " + str(round(corr_lc, 2)) + " → un pic de "
       "charge prévisible donne une latence prévisible : à modéliser "
       "(séance 4).\n\n"
       "### Recommandation\n\n"
       "Automatiser la **détection précoce** : le seuil fixe CPU > 90 % ne "
       "voit aucun incident ; un modèle supervisé sur fenêtres de 10 min "
       "(séances 2–3) apprend la signature pré-incident en respectant un "
       "budget d'alertes. Reproductibilité : graine 5534, données "
       "`data/nosys_metriques.csv`."),
]

# ===========================================================================
# TP2 — du brut au dataset supervisé
# ===========================================================================
C2 = [
("md", "## B1 — Audit des défauts (30 min)\n\n"
       "**Objectif :** compter, ne pas réparer encore. Chaque défaut doit "
       "recevoir un volume et une hypothèse de cause (doublon réseau, "
       "capteur flottant, pic capteur, redémarrage d'agent, granularité, "
       "dérive). **Avant de réparer, on explique.**"),
("code", '''b = pd.read_csv("data/nosys_metriques_brutes.csv", parse_dates=["horodatage"])
b = b.sort_values(["serveur", "horodatage"]).reset_index(drop=True)

print("doublons :", b.duplicated().sum())          # 200
print()
print("NaN par colonne :")
print(b[["cpu_pct", "ram_pct", "latence_ms",
         "paquets_perdus_pct", "temperature_c"]].isna().sum())
print()
print("valeurs impossibles :")
print("cpu > 100 :", (b["cpu_pct"] > 100).sum())
print("temp > 80 :", (b["temperature_c"] > 80).sum())
print()
print("points par serveur (attendu 8640 ; srv-07 en 15 min -> 2880) :")
print(b.groupby("serveur").size())'''),
("md", "**Tableau d'audit (mesuré) :** **" + str(n_dup) + " doublons** "
       "(envoi réseau répété) ; NaN — cpu_pct **" + str(nan_cols["cpu_pct"])
       + "**, ram_pct **" + str(nan_cols["ram_pct"]) + "**, latence_ms **"
       + str(nan_cols["latence_ms"]) + "**, paquets_perdus_pct **"
       + str(nan_cols["paquets_perdus_pct"]) + "**, temperature_c **"
       + str(nan_cols["temperature_c"]) + "** (capteur flottant / agent "
       "redémarré) ; **" + str(n_cpu_imp) + " CPU > 100 %** et **"
       + str(n_temp_imp) + " temp > 80 °C** (pics capteurs impossibles, dont "
       "1575 °C et 350 %) ; **srv-07 à " + str(int(pts_srv["srv-07"]))
       + " points** (granularité 15 min au lieu de 5) ; dérive lente de "
       "srv-06 (à vérifier en B3)."),
("md", "## B2 — Nettoyage (35 min)\n\n"
       "Ordre imposé : ① dédoublonner, ② bornes physiques → NaN (pas de "
       "suppression : on garde l'observation), ③ grille explicite 5 min par "
       "serveur (les trous deviennent des NaN visibles), ④ interpolation "
       "courte `limit=2`."),
("code", '''# 1. dedoublonner
b = b.drop_duplicates()

# 2. erreurs physiquement impossibles -> NaN (pas de suppression !)
b.loc[b["cpu_pct"] > 100, "cpu_pct"] = np.nan
b.loc[b["temperature_c"] > 80, "temperature_c"] = np.nan

# 3. grille explicite 5 min par serveur (les trous -> NaN visibles)
def sur_grille(g):
    g = g.set_index("horodatage").drop(columns=["serveur"])
    return g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="5min"))
b = b.groupby("serveur").apply(sur_grille)   # MultiIndex (serveur, horodatage)

# 4. imputation courte (<= 2 pas = 10 min), tendance-compatible
b = b.interpolate(limit=2, limit_direction="both")

print("contrôle de sortie :")
print("  doublons :", b.duplicated().sum())
print("  cpu > 100 :", (b["cpu_pct"] > 100).sum())
print("  NaN résiduels :",
      int(b[["cpu_pct", "ram_pct", "latence_ms", "paquets_perdus_pct",
             "temperature_c"]].isna().sum().sum()))
print("  fenêtres par serveur :")
print(b.groupby(level=0).size())'''),
("code", '''z = b.loc["srv-01", "cpu_pct"].loc["2026-09-14":"2026-09-16"]
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(z.index, z.values, lw=.8)
ax.set_ylabel("CPU %")
ax.set_title("srv-01 après nettoyage — 3 jours sur la grille 5 min")
plt.show()
print("grille srv-01 :", len(b.loc["srv-01"]), "fenêtres, pas médian :",
      pd.Series(b.loc["srv-01"].index).diff().mode()[0])'''),
("md", "**Contrôle de sortie :** 0 doublon, 0 CPU > 100, 0 NaN résiduels, "
       "8 serveurs sur la même grille — srv-07 inclus."),
("md", "## B3 — Construire les signaux (30 min)\n\n"
       "Fenêtres de 10 min par serveur : moyennes, max, p99 de latence, "
       "dérivée `cpu_delta` (fenêtre précédente), encodage cyclique "
       "`h_sin`/`h_cos`. Bonus : détecter la dérive de srv-06 par régression "
       "de `temp_moy` sur le temps."),
("code", '''f = (b.reset_index().set_index("horodatage")
      .groupby("serveur")
      .resample("10min")
      .agg(cpu_moy=("cpu_pct", "mean"),
           cpu_max=("cpu_pct", "max"),
           ram_moy=("ram_pct", "mean"),
           lat_moy=("latence_ms", "mean"),
           lat_p99=("latence_ms", lambda s: s.quantile(0.99)),
           perte_moy=("paquets_perdus_pct", "mean"),
           temp_moy=("temperature_c", "mean"),
           conso_moy=("conso_kwh", "mean")))

# dynamique : derivee du CPU (fenetre precedente)
f["cpu_delta"] = f.groupby("serveur")["cpu_moy"].diff()

# contexte : encodage cyclique
h = f.index.get_level_values("horodatage").hour
f["h_sin"] = np.sin(2 * np.pi * h / 24)
f["h_cos"] = np.cos(2 * np.pi * h / 24)
print(f.shape)
f.head()'''),
("code", '''# bonus : regression de temp_moy sur le temps, srv-06 vs srv-01 (reference)
t6 = f.xs("srv-06", level="serveur")["temp_moy"].dropna()
x6 = (t6.index - t6.index[0]).total_seconds() / 86400
pente = np.polyfit(x6, t6.values, 1)[0]          # deg C / jour
print("pente srv-06 mesurée : %+.3f °C/jour (ordre de grandeur attendu +0,2)"
      % pente)

fig, ax = plt.subplots(figsize=(9, 3))
t6d2 = t6.resample("D").mean()
ax.plot(t6d2.index, t6d2.values, marker="o", ms=3,
        label="srv-06 (temp moyenne/jour)")
ref = f.xs("srv-01", level="serveur")["temp_moy"].resample("D").mean()
ax.plot(ref.index, ref.values, alpha=.6, label="srv-01 (référence)")
ax.set_ylabel("température (°C)"); ax.legend()
ax.set_title("Dérive de srv-06 vs référence")
plt.show()'''),
("md", "**Dérive mesurée :** pente ≈ " + ("%+.2f" % pente_srv06)
       + " °C/jour pour srv-06 (ordre de grandeur attendu +0,2 °C/jour) "
       "contre une référence plate pour srv-01 : capteur qui dérive, à "
       "surveiller avant tout modèle."),
("md", "## B4 — L'étiquette « incident ≤ 30 min » (35 min)\n\n"
       "On regarde le **futur** avec la colonne `alarme` — et **seulement "
       "elle** : `shift(-k)`, k = 1..3, sur la fenêtre d'alarme de 10 min. "
       "Règle anti-fuite : les signaux utilisent le passé, la cible le "
       "futur ; `alarme` n'entre **pas** dans les features."),
("code", '''# alarme par fenetre de 10 min (max des 2 sous-fenetres de 5 min)
al = (b.reset_index()
       .groupby(["serveur", pd.Grouper(key="horodatage", freq="10min")])
       ["alarme"].max().unstack(0))          # fenetres x serveurs

fut = al.gt(0).astype(int)
for k in (1, 2, 3):                          # t+10, t+20, t+30 min
    fut = ((fut + al.shift(-k).gt(0).fillna(0).astype(int)) > 0).astype(int)

# aplatir : une ligne = (serveur, fenetre)
fut = fut.stack().swaplevel().sort_index().rename("incident_30min")
dataset = f.join(fut)
dataset["incident_30min"] = dataset["incident_30min"].fillna(0).astype(int)
print(dataset["incident_30min"].value_counts())
print("part de positifs : %.2f %%" % (100 * dataset["incident_30min"].mean()))'''),
("code", '''fig, ax = plt.subplots(figsize=(6, 3))
vc = dataset["incident_30min"].value_counts()
ax.bar(["pas d'incident (0)", "incident <= 30 min (1)"],
       [vc.get(0, 0), vc.get(1, 0)], color=["#4c72b0", "#c0392b"])
ax.set_ylabel("fenêtres")
ax.set_title("Distribution de l'étiquette — classe très rare ("
             + str(int(vc.get(1, 0))) + " positives)")
plt.show()'''),
("md", "**Attendu :** ≈ **" + str(n_pos) + " positives** (0,35 %) sur 34 560 "
       "fenêtres (8 × 4 320). Les dernières fenêtres du mois, sans futur "
       "observable, sont étiquetées 0 par convention (bord de série)."),
("md", "## B5 — Livrable et dictionnaire des variables\n\n"
       "Le dataset est produit **en mémoire** (`dataset`) — 34 560 lignes, "
       "13 colonnes. Dictionnaire :\n\n"
       "| colonne | définition | fenêtre | unité |\n"
       "|---|---|---|---|\n"
       "| `cpu_moy` | moyenne du CPU | 10 min | % |\n"
       "| `cpu_max` | max du CPU | 10 min | % |\n"
       "| `ram_moy` | moyenne RAM | 10 min | % |\n"
       "| `lat_moy` | moyenne latence | 10 min | ms |\n"
       "| `lat_p99` | quantile 0,99 de latence | 10 min | ms |\n"
       "| `perte_moy` | moyenne paquets perdus | 10 min | % |\n"
       "| `temp_moy` | moyenne température | 10 min | °C |\n"
       "| `conso_moy` | moyenne conso | 10 min | kWh/5min |\n"
       "| `cpu_delta` | diff de `cpu_moy` vs fenêtre précédente | 10 min | points |\n"
       "| `h_sin`, `h_cos` | encodage cyclique de l'heure | instant | — |\n"
       "| `incident_30min` | alarme dans les 30 min suivant la fenêtre "
       "(shift −1..−3 de `alarme`) | futur 30 min | 0/1 |\n\n"
       "**Note de choix (5 lignes) :** imputation par interpolation linéaire "
       "`limit=2` (10 min max) : le signal évolue en tendance, "
       "l'interpolation bat le forward-fill (erreur 0,74 contre 1,02 mesurée "
       "en cours) ; les valeurs impossibles sont converties en NaN avec la "
       "borne physique documentée (CPU ≤ 100 %, temp ≤ 80 °C) puis "
       "interpolées ; srv-07 est ré-échantillonné sur la grille 5 min "
       "commune ; `alarme` ne sert qu'à l'étiquette, jamais aux features ; "
       "les fenêtres sans futur (bord de série) sont étiquetées 0."),
]

# ===========================================================================
# TP3 — anomalies et classification
# ===========================================================================
C3 = [
("md", "## B1 — IsolationForest sans étiquette (25 min)\n\n"
       "**Objectif :** détecter sans jamais regarder la cible. Split "
       "temporel strict : train < 22/09 (" + str(n_train)
       + " fenêtres), test ≥ 22/09 (" + str(n_test)
       + ", 9 jours). Graine 5534. **Contrôle attendu : 20/20 des anomalies "
       "en tête sont des pré-incidents.**"),
("code", '''from sklearn.ensemble import IsolationForest

f = pd.read_csv("data/nosys_features.csv", parse_dates=["fenetre"])
X = f.drop(columns=["serveur", "fenetre", "incident_30min"]).fillna(0.0)
train = f["fenetre"] < "2026-09-22"

iso = IsolationForest(n_estimators=300, contamination=0.005,
                      random_state=5534, n_jobs=2).fit(X[train])
f["score_anomalie"] = -iso.score_samples(X)

top20 = f.nlargest(20, "score_anomalie")
print(top20[["serveur", "fenetre", "cpu_moy", "lat_p99", "incident_30min"]])
print()
print("pré-incidents dans le top 20 : %d/20" % top20["incident_30min"].sum())'''),
("md", "## B2 — Recouvrement détections / alarmes réelles (20 min)\n\n"
       "Seuil = quantile 0,99 des scores **de train** (jamais du test). Puis "
       "tableau croisé anomalie × incident sur le test et commentaire "
       "honnête : quelle part des anomalies vaut le coup d'être envoyée au "
       "NOC ?"),
("code", '''seuil = f.loc[train, "score_anomalie"].quantile(0.99)
f["anomalie"] = f["score_anomalie"] > seuil

test = f[~train]
print("anomalies test :", int(test["anomalie"].sum()))
print("dont vrais incidents (<= 30 min) :",
      int(test.loc[test["anomalie"], "incident_30min"].sum()))
print(pd.crosstab(test["anomalie"], test["incident_30min"],
                  rownames=["anomalie"], colnames=["incident"]))

cov = test.groupby("fenetre")["anomalie"].sum().value_counts().sort_index()
print()
print("incidents vus simultanément sur k serveurs :")
print(cov)'''),
("md", "**Lecture :** les anomalies non confirmées par un incident sont des "
       "anomalies **bénignes** (batch, backup, pics) — chaque fausse alerte "
       "coûte une nuit d'ingénieur : le non-supervisé seul ne suffit pas."),
("md", "## B3 — Forêt balanced et matrice de confusion (30 min)\n\n"
       "Entraîner, puis **juger sur le futur**. Piège attendu : avec "
       "`class_weight='balanced'`, le seuil 0,50 n'est plus calibré — la "
       "forêt prédit très peu de positifs, souvent tous faux. Ce n'est pas "
       "un bug : c'est le déplacement du seuil effectif."),
("code", '''from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

Xtr, ytr = X[train], f.loc[train, "incident_30min"]
Xte, yte = X[~train], f.loc[~train, "incident_30min"]

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=5534, n_jobs=2).fit(Xtr, ytr)
proba_rf = rf.predict_proba(Xte)[:, 1]
pred = (proba_rf >= 0.5).astype(int)
print(confusion_matrix(yte, pred))
print("précision %.2f  rappel %.2f  F1 %.3f"
      % (precision_score(yte, pred, zero_division=0),
         recall_score(yte, pred, zero_division=0),
         f1_score(yte, pred, zero_division=0)))'''),
("code", '''# repère : régression logistique (standardisée dans le pipeline)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score

logit = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000,
                                         class_weight="balanced",
                                         random_state=5534))
logit.fit(Xtr, ytr)
proba_log = logit.predict_proba(Xte)[:, 1]
print("logistique : AP %.2f, F1 au seuil 0.5 : %.3f"
      % (average_precision_score(yte, proba_log),
         f1_score(yte, (proba_log >= .5).astype(int), zero_division=0)))'''),
("md", "## B4 — Courbe PR et seuil métier (30 min)\n\n"
       "Gradient boosting puis choix du seuil sous **budget NOC ≤ 5 "
       "alertes/jour/serveur** (test = 9 j × 8 serveurs → au plus 360 "
       "alertes). Référence du corrigé : 40 alertes sur 9 jours = 0,56/"
       "jour/serveur, précision 1,00, rappel 0,625."),
("code", '''from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

gbc = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                 learning_rate=0.06, random_state=5534)
gbc.fit(Xtr, ytr)
proba = gbc.predict_proba(Xte)[:, 1]

prec, rec, thr = precision_recall_curve(yte, proba)

def alertes_jour_serveur(s):
    return (proba >= s).sum() / 9 / 8

print("seuil | alertes/jour/serveur | précision | rappel")
for s in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    p = (proba >= s).astype(int)
    print("%.2f  | %.2f | %.2f | %.2f" % (
        s, alertes_jour_serveur(s),
        precision_score(yte, p, zero_division=0),
        recall_score(yte, p, zero_division=0)))'''),
("code", '''# seuil auto : le plus grand rappel qui respecte le budget (5*8*9 = 360)
BUDGET = 360
ok = np.where((proba >= thr).sum() <= BUDGET)[0]
i = ok[np.argmax(rec[ok])]
seuil_choisi = thr[i]
print("seuil recommandé %.4f -> %d alertes (%.2f/jour/serveur), "
      "précision %.2f, rappel %.2f"
      % (seuil_choisi, (proba >= seuil_choisi).sum(),
         alertes_jour_serveur(seuil_choisi), prec[i], rec[i]))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(rec, prec, lw=2)
ax.scatter([rec[i]], [prec[i]], color="#c0392b", zorder=5,
           label="seuil choisi (%.3f)" % seuil_choisi)
ax.set_xlabel("rappel"); ax.set_ylabel("précision")
ax.set_title("Courbe précision-rappel (test, GBC) — AP %.2f"
             % average_precision_score(yte, proba))
ax.legend(); plt.show()

cm = confusion_matrix(yte, (proba >= seuil_choisi).astype(int))
fig, ax = plt.subplots(figsize=(3.6, 3.2))
im = ax.imshow(cm, cmap="Blues")
for a in range(2):
    for bcol in range(2):
        ax.text(bcol, a, cm[a, bcol], ha="center", va="center", fontsize=13)
ax.set_xticks([0, 1]); ax.set_xticklabels(["prédit 0", "prédit 1"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["réel 0", "réel 1"])
ax.set_title("Confusion (seuil métier)"); fig.colorbar(im, shrink=.8)
plt.show()'''),
("md", "## B5 — Livrable : tableau comparatif et décision\n\n"
       "| modèle | précision | rappel | F1 | AP | alertes/j/srv |\n"
       "|---|---|---|---|---|---|\n"
       "| régression logistique (seuil 0,5) | "
       + "%.2f | %.2f | %.2f | %.2f | — |" % (
           precision_score(yte3, pred_lg_b, zero_division=0),
           recall_score(yte3, pred_lg_b, zero_division=0),
           f1_score(yte3, pred_lg_b, zero_division=0),
           average_precision_score(yte3, plg_b)) + "\n"
       "| forêt balanced (seuil 0,5) | "
       + "%.2f | %.2f | %.2f | %.2f | — |" % (
           precision_score(yte3, pred_rf_b, zero_division=0),
           recall_score(yte3, pred_rf_b, zero_division=0),
           f1_score(yte3, pred_rf_b, zero_division=0),
           average_precision_score(yte3, prf_b)) + "\n"
       "| **gradient boosting (seuil métier " + ("%.3f" % seuil_b) + ")** | "
       + "%.2f | %.2f | %.2f | %.2f | %.2f |" % (
           prec_i, rec_i, f1_i, ap_b, alertes_b) + "\n\n"
       "**Décision :** on déploie le gradient boosting au seuil "
       + ("%.3f" % seuil_b) + " : il respecte largement le budget NOC ("
       + ("%.2f" % alertes_b) + " alertes/jour/serveur ≪ 5) tout en "
       "rattrapant " + str(round(100 * rec_i)) + " % des incidents. Les "
       + str(n_pos_te - round(rec_i * n_pos_te))
       + " incidents manqués n'ont aucun signal métrique : piste séance 5 "
       "(journaux d'authentification). Reproductibilité : graine 5534, "
       "split temporel strict au 22/09, aucune statistique calculée sur le "
       "test."),
]

# ===========================================================================
# TP4 — SARIMA et prédiction de panne
# ===========================================================================
C4 = [
("md", "## B1 — La série horaire du parc (30 min)\n\n"
       "Agréger le CPU moyen du parc en 720 points horaires, décomposer "
       "(tendance + saison 24 h + résidus), tester la stationnarité (ADF). "
       "Attendu : p ≈ 0,075 sur la série brute → on différencie."),
("code", '''from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

m4 = pd.read_csv("data/nosys_metriques.csv", parse_dates=["horodatage"])
s = m4.set_index("horodatage")["cpu_pct"].resample("1h").mean()
print("longueur de la série :", len(s))          # 720

dec = seasonal_decompose(s, model="additive", period=24)
fig = dec.plot()
fig.set_size_inches(10, 7)
plt.show()

stat, p, *_ = adfuller(s)
print("ADF série brute : stat %.3f, p = %.3f" % (stat, p))
stat2, p2, *_ = adfuller((s - s.shift(24)).dropna())
print("ADF différenciée saisonnièrement (24 h) : stat %.2f, p = %.2e"
      % (stat2, p2))'''),
("code", '''from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
plot_acf(s, lags=72, ax=ax[0]); plot_pacf(s, lags=48, ax=ax[1], method="ywm")
ax[0].set_title("ACF"); ax[1].set_title("PACF")
plt.show()'''),
("md", "**Conclusion ADF (une phrase) :** p = " + ("%.3f" % p_adf)
       + " > 0,05 sur la série brute : on ne rejette pas la racine "
       "unitaire ; après différenciation saisonnière 24 h, p ≈ 0 — la "
       "saison journalière est le motif dominant, d'où s = 24 sur l'agrégat "
       "horaire."),
("md", "## B2 — SARIMA : ajuster, prévoir, mesurer (40 min)\n\n"
       "Modèle (2,0,1)×(1,0,1)_24 avec harmoniques hebdomadaires en "
       "exogènes, ajusté sur les 24 premiers jours (576 h), prévision 144 h. "
       "MAE **mesurée** sur le test, comparée à deux bases naïves."),
("code", '''from statsmodels.tsa.statespace.sarimax import SARIMAX

train, test = s.iloc[:576], s.iloc[576:]
t = np.arange(len(s))                 # harmoniques hebdo (exogènes)
X = np.column_stack([g(t * 2 * np.pi * j / 168) for j in (1, 2, 3)
                     for g in (np.sin, np.cos)])

mod = SARIMAX(train, exog=X[:576], order=(2, 0, 1),
              seasonal_order=(1, 0, 1, 24))
res = mod.fit(disp=False)
fc = res.get_forecast(144, exog=X[576:])
pred, ci = fc.predicted_mean, fc.conf_int()

mae = np.mean(np.abs(pred - test))
pers24 = np.mean(np.abs(test - s.shift(24).iloc[576:]))
pers_semaine = np.mean(np.abs(test - s.shift(168).iloc[576:]))
print("MAE SARIMA test : %.2f pts CPU" % mae)
print("MAE persistance 24 h : %.2f | semaine précédente : %.2f"
      % (pers24, pers_semaine))
print("gain vs persistance : %.1f %%" % (100 * (1 - mae / pers24)))'''),
("code", '''fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(s.index[-240:], s.values[-240:], label="réalisé", lw=1)
ax.plot(pred.index, pred.values, color="#c0392b", label="prévision SARIMA")
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                color="#c0392b", alpha=.15, label="IC 95 %")
ax.axvline(test.index[0], color="k", ls=":", lw=1)
ax.set_ylabel("CPU moyen parc (%)"); ax.legend()
ax.set_title("Prévision 6 jours — intervalle élargi avec l'horizon")
plt.show()'''),
("md", "**Résultat :** MAE test **" + ("%.2f" % mae_s)
       + " pts CPU**, soit −" + ("%.1f" % gain_s)
       + " % d'erreur par rapport à la persistance 24 h ("
       + ("%.2f" % pers24_s) + "). L'intervalle 95 % s'élargit avec "
       "l'horizon (l'incertitude s'accumule) et couvre l'essentiel des "
       "pics — sauf les incidents, invisibles pour un modèle de niveau."),
("md", "## B3 — Modèle de panne, split temporel strict (30 min)\n\n"
       "Même étiquette, mêmes features de niveau. **Vérification "
       "anti-fuite :** refaire le modèle avec l'alarme **courante** en "
       "entrée — elle répète l'étiquette au lieu de la précéder → "
       "« 100 % » de réussite, signature de la fuite."),
("code", '''from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

feats = ["cpu_moy", "cpu_max", "ram_moy", "lat_moy", "lat_p99",
         "perte_moy", "temp_moy", "conso_moy", "cpu_delta",
         "cpu_amplitude", "h_sin", "h_cos"]
f4 = pd.read_csv("data/nosys_features.csv", parse_dates=["fenetre"])
f4["cpu_amplitude"] = f4["cpu_amplitude"].fillna(0.0)
f4 = f4.dropna(subset=feats + ["incident_30min"])

cut = pd.Timestamp("2026-09-22")        # JAMAIS de shuffle
tr4, te4 = f4[f4["fenetre"] < cut], f4[f4["fenetre"] >= cut]
gbc4 = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                  learning_rate=0.06, random_state=5534)
gbc4.fit(tr4[feats], tr4["incident_30min"])
cm4 = confusion_matrix(te4["incident_30min"], gbc4.predict(te4[feats]))
print(cm4)
print("rappel : %.3f" % (cm4[1, 1] / cm4[1].sum()))'''),
("code", '''# contre-preuve : la fuite — ajouter l'alarme du PRESENT en entrée
al10 = (m4.groupby(["serveur", pd.Grouper(key="horodatage", freq="10min")])
        ["alarme"].max().rename("alarme_10min").reset_index())
f4f = f4.merge(al10, left_on=["serveur", "fenetre"],
               right_on=["serveur", "horodatage"])
trf, tef = f4f[f4f["fenetre"] < cut], f4f[f4f["fenetre"] >= cut]
gbc_leak = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                      random_state=5534)
gbc_leak.fit(trf[feats + ["alarme_10min"]], trf["incident_30min"])
cm_leak = confusion_matrix(tef["incident_30min"],
                           gbc_leak.predict(tef[feats + ["alarme_10min"]]))
print("avec l'alarme en entrée (fuite) :", cm_leak.tolist(),
      "-> rappel %.2f : circularité, pas de la prédiction"
      % (cm_leak[1, 1] / max(cm_leak[1].sum(), 1)))'''),
("md", "## B4 — Le budget d'alertes (20 min)\n\n"
       "Rappel en fonction du nombre d'alertes par jour (classer les "
       "fenêtres de test par score décroissant, prendre les k premières de "
       "chaque journée), puis point d'opération."),
("code", '''proba4 = gbc4.predict_proba(te4[feats])[:, 1]
jour = te4["fenetre"].dt.strftime("%Y-%m-%d").values
reel = te4["incident_30min"].values

def rappel_par_budget(k):
    sel = np.zeros(len(te4), dtype=bool)
    for d in np.unique(jour):
        idx = np.where(jour == d)[0]
        top = idx[np.argsort(-proba4[idx])[:k]]
        sel[top] = True
    return reel[sel].sum() / reel.sum()

budgets = range(1, 11)
rappels = [rappel_par_budget(k) for k in budgets]
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(list(budgets), rappels, marker="o")
ax.axhline(rappels[4], color="k", ls=":", lw=1)
ax.set_xlabel("alertes par jour (tous serveurs confondus)")
ax.set_ylabel("rappel")
ax.set_title("Rappel vs budget d'alertes")
plt.show()
print("5 alertes/jour -> rappel %.3f (40 pannes sur 64 rattrapées au corrigé)"
      % rappels[4])'''),
("md", "## B5 — Livrable : décision de l'exploitant\n\n"
       "**Budget d'alertes :** à 5 alertes/jour le rappel plafonne à ≈ 0,625 "
       "— les incidents restants n'ont aucun signal métrique (piste : les "
       "journaux, séance 5). À 2 alertes/jour le rappel chute : compromis à "
       "chiffrer en heures d'astreinte économisées vs incidents manqués.\n\n"
       "**Saturation du stockage :** hypothèses explicites — occupation "
       "actuelle 60 %, dérive mesurée +1,333 %/jour → seuil critique 90 % "
       "atteint dans 30/1,333 ≈ 22,5 jours, soit le **23/10/2026** : "
       "dimensionner la rétention avant cette date.\n\n"
       "**Que décide l'exploitant lundi matin ?** Programmer l'extension de "
       "stockage (échéance 23/10), activer l'autoscaling sur srv-01 (ratio "
       "de capacité le plus élevé), garder le SARIMA comme prévision de "
       "niveau (MAE " + ("%.2f" % mae_s) + ") et le détecteur GBC au seuil "
       "métier comme alerte pré-incident."),
]

# ===========================================================================
# TP5 — sécurité et journaux
# ===========================================================================
C5 = [
("md", "## B1 — Parser et agréger (30 min)\n\n"
       "De 100 000 lignes à ~10 000 sessions (machine × fenêtre de 10 min). "
       "**Attention :** des lignes « Rotation des logs » ou « Sauvegarde » "
       "portent aussi `auth_reussie = False` — filtrer d'abord, compter "
       "ensuite ; on garde toutes les lignes dans l'agrégat mais les "
       "features combinées limiteront les faux positifs."),
("code", '''L = pd.read_csv("data/nosys_logs.csv", parse_dates=["horodatage"])
print(L.shape, "| niveaux :", L["niveau"].value_counts().to_dict())
L["echec"] = (~L["auth_reussie"].astype(bool)).astype(int)
L["succes"] = 1 - L["echec"]
L["fenetre"] = L["horodatage"].dt.floor("10min")
L = L.sort_values("horodatage")

g = L.groupby(["machine", "fenetre"])
S = pd.DataFrame({
    "n_evenements": g.size(),
    "echecs": g["echec"].sum(),
    "succes": g["succes"].sum(),
    "utilisateurs": g["utilisateur"].nunique()})
S["taux_echec"] = S["echecs"] / S["n_evenements"]
print("sessions :", len(S))               # ~10 000
print("échecs max par session :", int(S["echecs"].max()),
      "(la rafale de l'attaque)")
print("échecs totaux : %d — beaucoup ne sont pas des authentifications "
      "(rotation, sauvegarde), d'où des features combinées plutôt qu'un "
      "simple comptage" % int(L["echec"].sum()))'''),
("md", "## B2 — Features d'attaque et entropie (30 min)\n\n"
       "Signaux faibles combinés : échecs, échecs consécutifs max (rafale), "
       "utilisateurs distincts, **entropie de Shannon** des comptes testés, "
       "comptes sensibles, indicateur de nuit. Vérification attendue : "
       "entropie attaque ≈ 1,73 bits vs 2,22 bits ailleurs ; 100 % des "
       "échecs d'attaque sur des comptes sensibles."),
("code", '''from scipy.stats import entropy

SENS = {"root", "admin", "inconnu-1", "inconnu-2"}

def attaque_session(cle):
    mach, fen = cle
    return (mach == "bdd-principal"
            and pd.Timestamp("2026-10-05 02:00") <= fen
            <= pd.Timestamp("2026-10-05 04:00"))

def max_run(d):
    e = d.sort_values("horodatage")["echec"].values
    best = cur = 0
    for v in e:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

ent, run_max, sens = {}, {}, {}
for cle, d in g:
    ent[cle] = entropy(d["utilisateur"].value_counts().values, base=2)
    run_max[cle] = max_run(d)
    sens[cle] = int(d.loc[d["echec"] == 1,
                          "utilisateur"].isin(SENS).sum())
S["entropie"] = pd.Series(ent)
S["echecs_consecutifs"] = pd.Series(run_max)
S["comptes_sensibles"] = pd.Series(sens)
S["nuit"] = (S.index.get_level_values("fenetre").hour < 6).astype(int)
print(S[["echecs", "taux_echec", "utilisateurs", "entropie",
         "echecs_consecutifs", "comptes_sensibles", "nuit"]].describe().round(2))

att = [cle for cle in S.index if attaque_session(cle)]
print()
print("entropie moyenne attaque : %.2f bits | reste : %.2f bits"
      % (S.loc[att, "entropie"].mean(), S.drop(att)["entropie"].mean()))
print("échecs d'attaque sur comptes sensibles : %.0f %%"
      % (100 * S.loc[att, "comptes_sensibles"].sum()
         / max(S.loc[att, "echecs"].sum(), 1)))'''),
("md", "## B3 — IsolationForest et précision@N (30 min)\n\n"
       "Contamination 0,005, graine 5534. **Attendu :** les 12 sessions "
       "d'attaque (bdd-principal, 05/10 02 h–04 h) occupent les rangs 1 à "
       "12 ; précision@10 = 1,0, précision@30 ≈ 0,40, rappel = 1,00."),
("code", '''from sklearn.ensemble import IsolationForest

FEATS = ["echecs", "taux_echec", "utilisateurs", "entropie",
         "echecs_consecutifs", "comptes_sensibles", "nuit"]
X5 = S[FEATS].values

iso5 = IsolationForest(n_estimators=200, contamination=0.005,
                       random_state=5534, n_jobs=2)
S["score"] = -iso5.fit(X5).decision_function(X5)
top30 = S.sort_values("score", ascending=False).head(30).copy()
top30["attaque_reelle"] = [int(attaque_session(i)) for i in top30.index]
print(top30[["echecs", "echecs_consecutifs", "comptes_sensibles",
             "entropie", "score", "attaque_reelle"]])
p10 = top30.head(10)["attaque_reelle"].mean()
p30 = top30["attaque_reelle"].mean()
r30 = top30["attaque_reelle"].sum() / 12
print("précision@10 : %.2f | précision@30 : %.2f | rappel : %.2f"
      % (p10, p30, r30))
print("rangs des 12 sessions d'attaque :",
      [i + 1 for i, a in enumerate(top30["attaque_reelle"]) if a])'''),
("code", '''fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(S["score"], bins=60, color="#4c72b0")
ax[0].set_title("Distribution des scores d'anomalie")
ax[0].set_xlabel("score (−decision_function)")
top = S.sort_values("score", ascending=False).head(30)
ax[1].bar(range(30), top["score"].values,
          color=["#c0392b" if attaque_session(i) else "#9db8d9"
                 for i in top.index])
ax[1].set_title("Top 30 — rouge = attaque réelle")
ax[1].set_xlabel("rang")
plt.show()'''),
("md", "**Lecture :** précision@10 = " + ("%.2f" % p10_b) + " mais "
       "précision@30 = " + ("%.2f" % p30_b) + " : après les sessions "
       "d'attaque viennent des sessions bénignes (sauvegardes, défauts de "
       "service). Le hasard donnerait ≈ 0,0012 — le classement est bien "
       "informatif."),
("md", "## B4 — Forêt supervisée (30 min)\n\n"
       "Étiquette session-attaque (fenêtre 02–04 h de `bdd-principal`) "
       "**uniquement pour l'évaluation** — l'utiliser en feature serait une "
       "fuite (règle anti-fuite de la séance 2). Split stratifié 50/50, "
       "graine 5534."),
("code", '''from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix

S["attaque"] = [int(attaque_session(cle)) for cle in S.index]
Xtr5, Xte5, ytr5, yte5 = train_test_split(
    S[FEATS], S["attaque"], test_size=0.5, stratify=S["attaque"],
    random_state=5534)
rf5 = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                             random_state=5534, n_jobs=2).fit(Xtr5, ytr5)
pred5 = rf5.predict(Xte5)
print(confusion_matrix(yte5, pred5))
print("précision %.2f | rappel %.2f"
      % (precision_score(yte5, pred5, zero_division=0),
         recall_score(yte5, pred5, zero_division=0)))

imp = pd.Series(rf5.feature_importances_, index=FEATS).sort_values()
fig, ax = plt.subplots(figsize=(7, 3))
imp.plot.barh(ax=ax, color="#4c72b0")
ax.set_title("Importance des features (forêt supervisée)")
plt.show()'''),
("md", "## B5 — Livrable : sessions suspectes et note d'incident\n\n"
       "Export du top 30 vers `out/sessions_suspectes.csv`, puis la note "
       "d'incident (1 page) ci-dessous."),
("code", '''top30_out = top30.copy()
top30_out["pred_attaque"] = rf5.predict(top30_out[FEATS])
os.makedirs("out", exist_ok=True)
top30_out.reset_index().to_csv("out/sessions_suspectes.csv", index=False)
print("écrit : out/sessions_suspectes.csv —", len(top30_out), "lignes")'''),
("md", "### Note d'incident (mesurée)\n\n"
       "**Résumé :** attaque par force brute sur **bdd-principal** le "
       "**05/10 entre 02 h et 04 h** : **" + str(n_ech_att)
       + " échecs d'authentification** (" + str(round(n_ech_att / 12))
       + " par session en moyenne, max 15) concentrés sur 12 sessions de "
       "10 min, comptes ciblés : " + ", ".join(comptes_att) + ". "
       + ("Aucun succès d'authentification pendant l'attaque."
          if n_succ_att == 0 else "Succès pendant la fenêtre : "
          + str(n_succ_att) + " (à investiguer).") + "\n\n"
       "**Preuve chiffrée :** l'IsolationForest place les sessions d'attaque "
       "aux rangs " + str(min(rangs_b)) + "–" + str(max(rangs_b))
       + " (précision@10 = " + ("%.2f" % p10_b) + ", rappel 1,00) ; la "
       "forêt supervisée confirme (précision " + ("%.2f" % pr5_b)
       + ", rappel " + ("%.2f" % rc5_b) + "). L'attaque était **invisible "
       "dans les métriques** de la séance 3 — seule la corrélation "
       "métriques/journaux la révèle.\n\n"
       "**Actions recommandées :** blocage IP source, verrouillage "
       "progressif des comptes, rotation des mots de passe des comptes "
       "sensibles, alerte sur règle « ≥ 5 échecs/10 min » en parallèle du "
       "modèle (défense en profondeur).\n\n"
       "Échecs sur tout le fichier : " + str(n_ech_tot)
       + " — beaucoup ne sont pas des authentifications (rotation, "
       "sauvegarde) : d'où le filtrage avant comptage."),
]

# ===========================================================================
# TP6 — MLOps
# ===========================================================================
C6 = [
("md", "## B1 — Sauvegarder le pipeline avec joblib (30 min)\n\n"
       "Le pipeline **complet** (standardisation + forêt), pas seulement le "
       "modèle : un pipeline partiel produit des scores faux sur un nouveau "
       "fichier. Split temporel strict : train < 22/09 (" + str(n_train)
       + " fenêtres, " + str(n_pos_tr) + " positives), test ≥ (" + str(n_test)
       + ", " + str(n_pos_te) + " positives)."),
("code", '''from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import joblib

f6 = pd.read_csv("data/nosys_features.csv", parse_dates=["fenetre"])
X6 = f6.drop(columns=["serveur", "fenetre", "incident_30min"]).fillna(0.0)
tr6 = f6["fenetre"] < "2026-09-22"
Xtr6, ytr6 = X6[tr6], f6.loc[tr6, "incident_30min"]
Xte6, yte6 = X6[~tr6], f6.loc[~tr6, "incident_30min"]

pipe = make_pipeline(StandardScaler(),
                     RandomForestClassifier(n_estimators=200,
                                            min_samples_leaf=2,
                                            class_weight="balanced",
                                            random_state=5534))
pipe.fit(Xtr6, ytr6)
os.makedirs("out", exist_ok=True)
joblib.dump(pipe, "out/nosys_pipeline.joblib")

# recharger et vérifier : scorer le train redonne les probabilités d'avant
pipe2 = joblib.load("out/nosys_pipeline.joblib")
avant = pipe.predict_proba(Xtr6.iloc[:500])[:, 1]
apres = pipe2.predict_proba(Xtr6.iloc[:500])[:, 1]
print("écart max avant/après sauvegarde : %.2e" % np.abs(avant - apres).max())
auc = roc_auc_score(yte6, pipe2.predict_proba(Xte6)[:, 1])
print("AUC test : %.3f" % auc)'''),
("md", "## B2 — Mesurer la latence réelle de scoring (30 min)\n\n"
       "1 000 prédictions réelles une par une (`perf_counter`), puis par "
       "lots de 100 : moyenne + p99, et le raisonnement de capacité."),
("code", '''import time

rng = np.random.default_rng(5534)
idx = rng.choice(len(Xte6), size=1000, replace=False)
lats = []
for i in idx:
    une = Xte6.iloc[[i]]
    t0 = time.perf_counter()
    pipe2.predict_proba(une)
    lats.append(time.perf_counter() - t0)
lats = np.array(lats) * 1000   # ms
print("1 000 prédictions individuelles : moyenne %.1f ms, p99 %.1f ms"
      % (lats.mean(), np.percentile(lats, 99)))

lot = Xte6.iloc[idx[:1000 // 100 * 100]]        # 900 fenêtres, lots de 100
t0 = time.perf_counter()
pipe2.predict_proba(lot)
dt_lot = time.perf_counter() - t0
debit_lot = len(lot) / dt_lot / 60
print("par lots de 100 : %.0f fenêtres/min par cœur" % debit_lot)
print("besoin NOSYS : 1 152 fenêtres/jour (8 serveurs x 144) — marge x%.0f"
      % (debit_lot * 1440 / 1152))'''),
("code", '''fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(lats, bins=40, color="#4c72b0")
ax[0].axvline(np.percentile(lats, 99), color="#c0392b", ls="--",
              label="p99 (%.1f ms)" % np.percentile(lats, 99))
ax[0].set_xlabel("latence (ms)"); ax[0].legend()
ax[0].set_title("1 000 prédictions individuelles")
ax[1].bar(["individuel", "lot de 100"],
          [60000 / lats.mean(), debit_lot], color=["#9db8d9", "#4c72b0"])
ax[1].set_ylabel("fenêtres/min")
ax[1].set_title("Débit individuel vs par lots")
plt.show()'''),
("md", "**Lecture :** le scoring par lots est bien plus rapide par fenêtre "
       "(overhead d'appel amorti, parallélisme interne des arbres). NOSYS "
       "produit 1 152 fenêtres/jour : la capacité d'un seul cœur suffit "
       "largement — la latence p99 de " + ("%.0f" % lat_p99_b)
       + " ms (mesurée sur cette machine : moyenne "
       + ("%.1f" % lat_moy_b) + " ms) reste compatible avec un endpoint "
       "temps réel."),
("md", "### Mini-API de scoring (à ne PAS exécuter ici)\n\n"
       "```python\n"
       "# api.py — FastAPI, exécutable avec : uvicorn api:app --port 8000\n"
       "from fastapi import FastAPI\n"
       "from pydantic import BaseModel\n"
       "import joblib, pandas as pd\n\n"
       "pipe = joblib.load('out/nosys_pipeline.joblib')\n"
       "VARIABLES = ['cpu_moy','cpu_max','ram_moy','lat_moy','lat_p99',\n"
       "             'perte_moy','temp_moy','conso_moy','capacite',\n"
       "             'cpu_delta','cpu_amplitude','h_sin','h_cos',\n"
       "             'js_sin','js_cos']\n"
       "app = FastAPI(title='nosys-scoring')\n\n"
       "class Fenetre(BaseModel):\n"
       "    cpu_moy: float; cpu_max: float; ram_moy: float\n"
       "    lat_moy: float; lat_p99: float; perte_moy: float\n"
       "    temp_moy: float; conso_moy: float; capacite: float\n"
       "    cpu_delta: float; cpu_amplitude: float\n"
       "    h_sin: float; h_cos: float; js_sin: float; js_cos: float\n\n"
       "@app.post('/score')\n"
       "def score(f: Fenetre):\n"
       "    x = pd.DataFrame([f.model_dump()])[VARIABLES]\n"
       "    return {'proba_incident': float(pipe.predict_proba(x)[0, 1])}\n"
       "```\n\n"
       "Commentaires : le modèle est chargé **une seule fois** au démarrage "
       "(latence), la validation d'entrée est déclarative (pydantic), et le "
       "endpoint renvoie la proba brute — le seuil métier reste côté NOC. "
       "Cellule volontairement non exécutée (pas de serveur dans un "
       "notebook)."),
("md", "## B3 — Fiche MLOps d'une page\n\n"
       "**Plan de retraining :** hebdomadaire de base, sur fenêtre "
       "glissante de 30–90 j ; déclencheurs chiffrés : proba moyenne > 0,005 "
       "pendant 48 h, taux d'alertes > 0,2 %, AUC < 0,70 dès que des "
       "étiquettes sont disponibles. Chaque incident confirmé = étiquette "
       "neuve (boucle de la séance 2).\n\n"
       "**Métriques surveillées :** AUC test (mesurée ici : "
       + ("%.3f" % auc_b) + "), taux d'alertes par jour/serveur "
       "(budget ≤ 5), latence moyenne (" + ("%.1f" % lat_moy_b)
       + " ms) et p99 (" + ("%.1f" % lat_p99_b) + " ms), distribution des "
       "entrées (PSI/KS — alarme si le « normal » bouge, comme la dérive de "
       "srv-06 en séance 2).\n\n"
       "**Procédure de rollback :** garder les 5 dernières versions de "
       "`nosys_pipeline.joblib` (registre horodaté) ; si le taux d'alertes "
       "double après bascule, le garde d'astreinte recharge la version "
       "précédente en moins de 10 minutes (`joblib.load` + redémarrage du "
       "service) et ouvre un ticket d'analyse.\n\n"
       "**Critère de réussite :** un collègue qui n'a pas suivi le cours "
       "peut faire tourner et surveiller le modèle avec cette fiche seule."),
("md", "## B4 — Quiz d'auto-évaluation (20 questions)\n\n"
       "1. Split strict : à quelle date sépare-t-on train et test ? "
       "**(22/09)**\n"
       "2. Première chose face à un CPU à 350 % ? **(borne physique → NaN, "
       "documentée)**\n"
       "3. 120 positives sur 34 560 = ? **(0,35 %)**\n"
       "4. Métrique d'abord sur classe rare ? **(précision/rappel, pas "
       "l'accuracy)**\n"
       "5. KFold mélangé sur une série temporelle ? **(jamais — fuite ; "
       "TimeSeriesSplit)**\n"
       "6. L'IsolationForest utilise-t-il l'étiquette ? **(non)**\n"
       "7. 174 anomalies, 40 incidents : le reste ? **(anomalies "
       "bénignes)**\n"
       "8. Pourquoi p99 et pas la moyenne de latence ? **(une moyenne cache "
       "les rares prédictions très lentes)**\n"
       "9. ADF p = 0,075 : conclusion ? **(différencier puis retester)**\n"
       "10. Saisonnalité dominante de la série horaire ? **(24 h)**\n"
       "11. MAE du SARIMA mesurée où ? **(144 h de test après le 22/09)**\n"
       "12. L'alarme courante en feature ? **(fuite — circularité)**\n"
       "13. Rappel à 5 alertes/jour ? **(≈ 0,625)**\n"
       "14. Granularité d'analyse des logs ? **(machine × fenêtre 10 "
       "min)**\n"
       "15. Rangs des sessions d'attaque dans l'IF ? **(1 à 12)**\n"
       "16. Pourquoi le supervisé bat le non-supervisé ? **(frontière "
       "attaque/normal apprise des incidents passés)**\n"
       "17. Défense contre la fatigue d'alerte ? **(classer et regrouper — "
       "précision@N)**\n"
       "18. Que sauvegarde un vrai pipeline ? **(modèle + standardisation "
       "ensemble)**\n"
       "19. Que surveille-t-on sur une tranche sans incident ? "
       "**(distribution des scores et taux d'alertes, pas l'AUC)**\n"
       "20. Rollback : quel délai visé ? **(< 10 min, version précédente du "
       "registre)**"),
]

ALL = [
    ("TP1_mise_en_route.ipynb", C1,
     "TP 1 — Mise en route : charger, vérifier, regarder",
     "_Cours EPD-GIT-5534 — Introduction au ML pour l'exploitation "
     "d'infrastructures — Séance 1._\n\n**Objectifs :** charger et auditer "
     "`nosys_metriques.csv`, produire la vue journalière de srv-01, le "
     "tableau de classement par serveur, le profil horaire et 5 constats "
     "chiffrés.\n\n**Durée :** 2 h 20 (B1 30 min, B2 20 min, B3 30 min, "
     "B4 40 min, B5 20 min). Graine 5534."),
    ("TP2_features.ipynb", C2,
     "TP 2 — Du brut au dataset supervisé",
     "_Cours EPD-GIT-5534 — Séance 2._\n\n**Objectifs :** auditer les "
     "défauts de `nosys_metriques_brutes.csv`, nettoyer dans le bon ordre, "
     "construire les signaux 10 min, poser l'étiquette `incident_30min` "
     "sans fuite, rédiger le dictionnaire des variables.\n\n**Durée :** "
     "2 h 10 (B1 30, B2 35, B3 30, B4 35, B5 10). Graine 5534."),
    ("TP3_anomalies_classification.ipynb", C3,
     "TP 3 — Détecter les incidents avant la panne",
     "_Cours EPD-GIT-5534 — Séance 3._\n\n**Objectifs :** IsolationForest "
     "sans étiquette, forêt/boosting supervisés sur classe rare (0,35 %), "
     "matrice de confusion, courbe précision-rappel, seuil sous budget "
     "NOC ≤ 5 alertes/jour/serveur.\n\n**Durée :** 2 h 15 (B1 25, B2 20, "
     "B3 30, B4 30, B5 20). Graine 5534."),
    ("TP4_sarima_prediction.ipynb", C4,
     "TP 4 — Prévoir et décider : SARIMA et prédiction de panne",
     "_Cours EPD-GIT-5534 — Séance 4._\n\n**Objectifs :** série horaire, "
     "ADF/ACF/PACF, SARIMA (2,0,1)×(1,0,1)_24 avec prévision 6 jours et "
     "intervalle, modèle de panne sans fuite, budget d'alertes, date de "
     "saturation du stockage.\n\n**Durée :** 2 h 30 (B1 30, B2 40, B3 30, "
     "B4 20, B5 20). Graine 5534."),
    ("TP5_securite.ipynb", C5,
     "TP 5 — Sécurité : détecter l'intrusion dans les journaux",
     "_Cours EPD-GIT-5534 — Séance 5._\n\n**Objectifs :** agréger 100 000 "
     "lignes de logs en sessions de 10 min, features d'attaque (entropie de "
     "Shannon, comptes sensibles, nuit), IsolationForest + forêt "
     "supervisée, précision@N, note d'incident.\n\n**Durée :** 2 h 30 "
     "(B1 30, B2 30, B3 30, B4 30, B5 10). Graine 5534."),
    ("TP6_mlops.ipynb", C6,
     "TP 6 — MLOps : de la forêt au service",
     "_Cours EPD-GIT-5534 — Séance 6._\n\n**Objectifs :** sauvegarder le "
     "pipeline complet (joblib), mesurer la latence réelle de scoring "
     "(1 000 prédictions), raisonnement de capacité, fiche MLOps, quiz "
     "d'auto-évaluation.\n\n**Durée :** 2 h (B1 30, B2 30, B3 30, B4 30). "
     "Graine 5534."),
]

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    for name, cells, title, intro in ALL:
        execute(build(name, cells, title, intro), name)
    print("\nTerminé : notebooks exécutés dans", NB)
