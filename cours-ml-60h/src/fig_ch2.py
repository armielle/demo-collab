"""Figures du chapitre 2 -- Preparer les donnees (Licence 3 SDIA).

Toutes les valeurs sont calculees ici (graine 4112) et ecrites dans
figures/key_numbers_ch2.json.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from figstyle import C, save, FIGDIR

rng = np.random.default_rng(4112)
cles = {}

# ==================================================== h01 : valeurs manquantes
n = 400
revenu = rng.lognormal(mean=10.5, sigma=.8, size=n)      # revenu parental, asymetrique
moyenne = rng.normal(11.5, 3.2, n).clip(0, 20)
absences = rng.poisson(9, n)
age = rng.normal(21, 1.8, n).round()

masque = rng.random((n, 4)) < np.array([.08, .22, .05, .30])
colonnes = ["Moyenne", "Revenu parental", "Absences", "Filière"]
taux = (masque.mean(0) * 100).round(1)

fig, ax = plt.subplots(figsize=(8.0, 3.4))
b = ax.barh(colonnes[::-1], taux[::-1], color=[C["blue"], C["cyan"], C["orange"], C["red"]][::-1],
            alpha=.85, height=.55)
for bb, v in zip(b, taux[::-1]):
    ax.text(v + .6, bb.get_y() + bb.get_height() / 2, f"{v:.1f} %", va="center", fontsize=9)
ax.set_xlabel("part de valeurs manquantes (%)"); ax.set_xlim(0, 36)
ax.set_title("Un jeu de données réel n'est jamais complet")
save(fig, "h01_manquants.png")
cles["manquants"] = {c: float(v) for c, v in zip(colonnes, taux)}

# ==================================================== h02 : imputation
vrai_moy = float(revenu.mean())
vrai_med = float(np.median(revenu))
obs = revenu.copy()
idx = rng.random(n) < .25
obs[idx] = np.nan
moy_obs = float(np.nanmean(obs))
med_obs = float(np.nanmedian(obs))
imput_moy = np.where(np.isnan(obs), moy_obs, obs)
imput_med = np.where(np.isnan(obs), med_obs, obs)
# biais sur la moyenne reconstruite
biais_moy = float(imput_moy.mean() - vrai_moy)
biais_med = float(imput_med.mean() - vrai_moy)
cles["imputation"] = {"taux_manquant": .25, "moyenne_vraie": round(vrai_moy, 1),
                      "ecart_type_vrai": round(float(revenu.std()), 1),
                      "ecart_type_apres_moyenne": round(float(imput_moy.std()), 1),
                      "ecart_type_apres_mediane": round(float(imput_med.std()), 1),
                      "mediane_vraie": round(vrai_med, 1),
                      "imputation_moyenne": round(float(imput_moy.mean()), 1),
                      "imputation_mediane": round(float(imput_med.mean()), 1),
                      "biais_moyenne": round(biais_moy, 1),
                      "biais_mediane": round(biais_med, 1)}

fig, ax = plt.subplots(figsize=(8.4, 3.8))
bins = np.linspace(0, np.percentile(revenu, 99), 45)
ax.hist(revenu, bins=bins, color=C["grey"], alpha=.35, label="revenu réel")
ax.axvline(vrai_moy, color=C["deep"], lw=2, label=f"moyenne vraie = {vrai_moy:,.0f}")
ax.axvline(float(imput_moy.mean()), color=C["red"], lw=2, ls="--",
           label=f"après imputation par la moyenne = {imput_moy.mean():,.0f}")
ax.axvline(float(imput_med.mean()), color=C["green"], lw=2, ls=":",
           label=f"après imputation par la médiane = {imput_med.mean():,.0f}")
ax.set_xlabel("revenu parental (unités arbitraires)"); ax.set_ylabel("effectif")
ax.set_title("Toute imputation déforme la distribution")
ax.legend(fontsize=8, loc="upper right")
save(fig, "h02_imputation.png")

# ==================================================== h03 : encodage
fig, ax = plt.subplots(figsize=(9.2, 2.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
ax.text(.2, 3.75, "Filière", fontsize=10, fontweight="bold")
ax.text(3.4, 3.75, "One-hot (3 colonnes)", fontsize=10, fontweight="bold")
ax.text(7.6, 3.75, "Ordinal", fontsize=10, fontweight="bold")
lignes = [("Informatique", [1, 0, 0], "0"),
          ("Télécom", [0, 1, 0], "1"),
          ("Réseaux", [0, 0, 1], "2")]
for k, (nom, vec, ordi) in enumerate(lignes):
    y = 3.0 - k * .8
    ax.text(.2, y, nom, fontsize=9)
    for j, v in enumerate(vec):
        x = 3.5 + j * .75
        ax.add_patch(Rectangle((x, y - .22), .6, .5,
                               fc=C["blue"] if v else "white", ec=C["grey"], lw=.8,
                               alpha=.85 if v else 1))
        ax.text(x + .3, y + .02, str(v), ha="center", fontsize=9,
                color="white" if v else C["grey"])
    ax.text(7.8, y, ordi, fontsize=9, ha="center")
ax.text(.2, .35, "One-hot : aucune relation d'ordre imposée.", fontsize=8.5, color=C["grey"])
ax.text(.2, .05, "Ordinal : suppose 0 < 1 < 2, ce qui est faux ici.", fontsize=8.5, color=C["red"])
save(fig, "h03_encodage.png")
cles["encodage"] = {"one_hot": "3 colonnes binaires", "ordinal": "1 colonne 0/1/2"}

# ==================================================== h04 : mise a l'echelle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=4112, stratify=y)
knn_brut = KNeighborsClassifier(5).fit(Xtr, ytr)
knn_std = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(5))]).fit(Xtr, ytr)
acc_brut = float(accuracy_score(yte, knn_brut.predict(Xte)))
acc_std = float(accuracy_score(yte, knn_std.predict(Xte)))
# amplitude des colonnes
etendues = (X.max(0) - X.min(0))
cles["echelle"] = {"knn_sans_normalisation": round(acc_brut, 3),
                   "knn_avec_normalisation": round(acc_std, 3),
                   "gain_points": round(100 * (acc_std - acc_brut), 1),
                   "colonne_la_plus_large": f"{etendues.max():.0f}",
                   "colonne_la_plus_etroite": f"{etendues.min():.3f}",
                   "ratio_etendues": round(float(etendues.max() / etendues.min()), 0)}

fig, ax = plt.subplots(figsize=(6.8, 3.5))
b = ax.bar(["sans\nnormalisation", "avec\nnormalisation"], [acc_brut, acc_std],
           color=[C["red"], C["green"]], alpha=.85, width=.55)
for bb, v in zip(b, [acc_brut, acc_std]):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .012, f"{v:.1%}", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.02); ax.set_ylabel("exactitude (kNN, k = 5)")
ax.set_title("Sans normalisation, une seule colonne décide des distances")
save(fig, "h04_echelle.png")

# ==================================================== h05 : outliers
base = rng.normal(50, 8, 300)
avec_out = np.r_[base, [120, 135, 150, 15, 8]]
q1, q3 = np.percentile(avec_out, [25, 75])
iqr = q3 - q1
borne_bas, borne_haut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
nb_iqr = int(((avec_out < borne_bas) | (avec_out > borne_haut)).sum())
z = (avec_out - avec_out.mean()) / avec_out.std()
nb_z = int((np.abs(z) > 3).sum())
cles["outliers"] = {"n": int(len(avec_out)), "iqr_borne_haut": round(float(borne_haut), 1),
                    "detectes_iqr": nb_iqr, "detectes_zscore": nb_z,
                    "vraies_valeurs_ajoutees": 5}

fig, ax = plt.subplots(figsize=(8.2, 3.2))
ax.boxplot([avec_out], vert=False, widths=.4, patch_artist=True,
           boxprops=dict(facecolor=C["blue"], alpha=.25))
ax.scatter(avec_out, np.ones_like(avec_out) + rng.normal(0, .035, len(avec_out)),
           s=12, color=C["deep"], alpha=.5)
ax.axvline(borne_haut, color=C["red"], ls="--", lw=1.4, label=f"borne IQR = {borne_haut:.0f}")
ax.set_yticks([]); ax.set_xlabel("valeur observée")
ax.set_title(f"Détection par IQR : {nb_iqr} points signalés (5 vraies anomalies ajoutées)")
ax.legend()
save(fig, "h05_outliers.png")

# ==================================================== h06 : fuite par encodage cible
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

n2 = 200
cats = np.repeat(np.arange(100), 2)          # 100 modalites, 2 lignes chacune
y2 = rng.integers(0, 2, n2)

def score_avec_fuite():
    moyennes = np.array([y2[cats == c].mean() for c in cats])   # calcul global !
    return cross_val_score(LogisticRegression(max_iter=1000),
                           moyennes.reshape(-1, 1), y2,
                           cv=KFold(5, shuffle=True, random_state=4112)).mean()

def score_propre():
    kf = KFold(5, shuffle=True, random_state=4112)
    pred = np.zeros(n2)
    for tr, te in kf.split(cats):
        moy = {c: y2[tr][cats[tr] == c].mean() for c in np.unique(cats[tr])}
        global_moy = y2[tr].mean()
        pred[te] = [moy.get(c, global_moy) for c in cats[te]]
    return cross_val_score(LogisticRegression(max_iter=1000),
                           pred.reshape(-1, 1), y2, cv=kf).mean()

s_fuite = float(score_avec_fuite())
s_propre = float(score_propre())
cles["fuite_encodage"] = {"modalites": 100, "lignes_par_modalite": 2,
                          "score_fuite": round(s_fuite, 3),
                          "score_propre": round(s_propre, 3),
                          "ecart_points": round(100 * (s_fuite - s_propre), 1)}

fig, ax = plt.subplots(figsize=(7.0, 3.4))
b = ax.bar(["encodage cible\nsur tout le jeu", "encodage cible\ndans les plis"],
           [s_fuite, s_propre], color=[C["red"], C["green"]], alpha=.85, width=.55)
ax.axhline(.5, color=C["grey"], ls="--", lw=1)
ax.text(1.42, .515, "hasard", fontsize=8.5, color=C["grey"], ha="right")
for bb, v in zip(b, [s_fuite, s_propre]):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .02, f"{v:.1%}", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.0); ax.set_ylabel("exactitude en validation croisée")
ax.set_title("Encodage cible : 100 modalités, étiquettes aléatoires")
save(fig, "h06_fuite_encodage.png")

# ==================================================== h07 : pipeline
fig, ax = plt.subplots(figsize=(10.5, 2.8))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 2.8); ax.axis("off")
blocs = [("Colonnes\nnumériques", "imputation médiane\n+ standardisation", C["blue"]),
         ("Colonnes\ncatégorielles", "one-hot encoding", C["cyan"]),
         ("Colonnes\ncycliques", "sin / cos", C["orange"])]
for i, (t1, t2, col) in enumerate(blocs):
    y = 1.85 - i * .95
    ax.add_patch(FancyBboxPatch((.2, y), 2.1, .65, boxstyle="round,pad=.05,rounding_size=.1",
                                fc=col, ec="none", alpha=.15))
    ax.text(1.25, y + .32, t1, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.add_patch(FancyArrowPatch((2.45, y + .32), (3.55, y + .32), arrowstyle="-|>",
                                 mutation_scale=12, color=C["grey"], lw=1.2))
    ax.add_patch(FancyBboxPatch((3.7, y), 2.5, .65, boxstyle="round,pad=.05,rounding_size=.1",
                                fc=col, ec=col, lw=1.1, alpha=.08))
    ax.text(4.95, y + .32, t2, ha="center", va="center", fontsize=8)
ax.add_patch(FancyBboxPatch((6.6, .55), 1.6, 1.6, boxstyle="round,pad=.06,rounding_size=.12",
                            fc=C["purple"], ec="none", alpha=.15))
ax.text(7.4, 1.35, "Column\nTransformer", ha="center", va="center", fontsize=9, fontweight="bold")
ax.add_patch(FancyArrowPatch((8.3, 1.35), (9.2, 1.35), arrowstyle="-|>",
                             mutation_scale=14, color=C["grey"], lw=1.4))
ax.add_patch(FancyBboxPatch((9.3, .95), 1.05, .8, boxstyle="round,pad=.05,rounding_size=.1",
                            fc=C["green"], ec="none", alpha=.2))
ax.text(9.82, 1.35, "Modèle", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(5.2, .12, "Un seul objet : il s'ajuste dans chaque pli, jamais avant le split.",
        ha="center", fontsize=8.5, color=C["grey"], style="italic")
save(fig, "h07_pipeline.png")

with open(os.path.join(FIGDIR, "key_numbers_ch2.json"), "w", encoding="utf-8") as f:
    json.dump(cles, f, ensure_ascii=False, indent=2)

print("\n--- chiffres cles ---")
print("manquants :", json.dumps(cles["manquants"], ensure_ascii=False))
print("imputation :", json.dumps(cles["imputation"], ensure_ascii=False))
print("echelle :", json.dumps(cles["echelle"], ensure_ascii=False))
print("outliers :", json.dumps(cles["outliers"], ensure_ascii=False))
print("fuite encodage :", json.dumps(cles["fuite_encodage"], ensure_ascii=False))
