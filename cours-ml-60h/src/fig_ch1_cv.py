"""Figure complementaire : validation croisee groupee (g13).

Montre qu'un decoupage aleatoire sur des individus repetes gonfle le score.
Graine 4112, valeurs ajoutees a key_numbers_ch1.json.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

from figstyle import C, save, FIGDIR

rng = np.random.default_rng(4112)
JSON = os.path.join(FIGDIR, "key_numbers_ch1.json")
cles = json.load(open(JSON, encoding="utf-8")) if os.path.exists(JSON) else {}

from sklearn.model_selection import KFold, GroupKFold, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 200 patients, 2 visites chacun. L'etiquette depend du patient, pas des mesures :
# aucune information generale n'est apprenable. Seule l'identite du patient distingue.
n_pat, visites = 200, 2
signature = rng.normal(size=(n_pat, 10))          # signature propre a chaque patient
y_pat = rng.integers(0, 2, n_pat)                 # etiquette aleatoire par patient

Xrep = np.repeat(signature, visites, axis=0) + rng.normal(0, 0.02, (n_pat * visites, 10))
yrep = np.repeat(y_pat, visites)
groupes = np.repeat(np.arange(n_pat), visites)

modele = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(1))])

s_aleatoire = cross_val_score(modele, Xrep, yrep,
                              cv=StratifiedKFold(5, shuffle=True, random_state=4112)).mean()
s_groupee = cross_val_score(modele, Xrep, yrep, cv=GroupKFold(5), groups=groupes).mean()

cles["validation_groupee"] = {"patients": n_pat, "visites_par_patient": visites,
                              "lignes": int(Xrep.shape[0]),
                              "score_decoupage_aleatoire": round(float(s_aleatoire), 4),
                              "score_decoupage_groupe": round(float(s_groupee), 4),
                              "ecart_points": round(100 * (s_aleatoire - s_groupee), 1)}

fig, ax = plt.subplots(figsize=(7.2, 3.5))
b = ax.bar(["découpage aléatoire\n(les 2 visites séparées)", "découpage groupé\n(patient entier d'un côté)"],
           [s_aleatoire, s_groupee], color=[C["red"], C["green"]], alpha=.85, width=.55)
ax.axhline(.5, color=C["grey"], ls="--", lw=1)
ax.text(-.42, .515, "hasard", fontsize=8.5, color=C["grey"])
for bb, v in zip(b, [s_aleatoire, s_groupee]):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .012, f"{v:.1%}", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05); ax.set_ylabel("exactitude (validation croisée, k = 5)")
ax.set_title("Étiquettes aléatoires : seul le découpage fait la différence")
save(fig, "g13_validation_groupee.png")

with open(JSON, "w", encoding="utf-8") as f:
    json.dump(cles, f, ensure_ascii=False, indent=2)

print(json.dumps(cles["validation_groupee"], ensure_ascii=False))
