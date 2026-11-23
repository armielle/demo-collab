"""Figures complementaires du chapitre 2 (densification).

h08 analyse exploratoire, h09 jointure qui duplique, h10 comparaison
d'imputations. Graine 4112, valeurs ecrites dans key_numbers_ch2.json.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

from figstyle import C, save, FIGDIR

rng = np.random.default_rng(4112)
JSON = os.path.join(FIGDIR, "key_numbers_ch2.json")
cles = json.load(open(JSON, encoding="utf-8")) if os.path.exists(JSON) else {}

n = 400
# trois notes correlees (un etudiant bon reste bon) + absences
latent = rng.normal(0, 1, n)
notes = np.column_stack([
    11 + 3 * latent + rng.normal(0, 1.2, n),
    10.5 + 2.8 * latent + rng.normal(0, 1.3, n),
    11.5 + 2.5 * latent + rng.normal(0, 1.4, n),
]).clip(0, 20)
absences = np.maximum(0, 9 - 2.5 * latent + rng.normal(0, 3, n)).round()

# ============================================ h08 : analyse exploratoire
fig, axes = plt.subplots(1, 4, figsize=(12.2, 2.9))
for j, ax in enumerate(axes[:3]):
    ax.hist(notes[:, j], bins=18, color=[C["blue"], C["cyan"], C["orange"]][j], alpha=.8)
    ax.set_title(f"Note {j+1}", fontsize=9.5)
    ax.set_xlabel("valeur"); ax.set_yticks([])
corr = np.corrcoef(np.column_stack([notes, absences]).T)
im = axes[3].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
axes[3].set_xticks(range(4)); axes[3].set_yticks(range(4))
axes[3].set_xticklabels(["N1", "N2", "N3", "Abs"], fontsize=8)
axes[3].set_yticklabels(["N1", "N2", "N3", "Abs"], fontsize=8)
for i in range(4):
    for j in range(4):
        axes[3].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                     color="white" if abs(corr[i, j]) > .6 else C["deep"])
axes[3].set_title("Corrélations", fontsize=9.5)
axes[3].grid(False)
fig.suptitle("Trois notes corrélées entre elles, et corrélées négativement aux absences",
             fontsize=10, y=1.02)
save(fig, "h08_exploration.png")
cles["exploration"] = {"corr_note1_note2": round(float(corr[0, 1]), 2),
                       "corr_note1_absences": round(float(corr[0, 3]), 2),
                       "corr_note2_absences": round(float(corr[1, 3]), 2)}

# ============================================ h09 : jointure qui duplique
# table etudiants (1 ligne) et table notes (1 a 3 semestres par etudiant)
etudiants = np.arange(400)
nb_sem = rng.choice([1, 2, 3], 400, p=[.3, .45, .25])
lignes_notes = np.repeat(etudiants, nb_sem)
reussite = rng.integers(0, 2, 400)
# moyenne de reussite : par etudiant vs par ligne de la jointure
moy_etudiant = float(reussite.mean())
moy_join = float(reussite[lignes_notes].mean())
cles["jointure"] = {"etudiants": 400, "lignes_apres_jointure": int(len(lignes_notes)),
                    "facteur": round(len(lignes_notes) / 400, 2),
                    "moyenne_par_etudiant": round(moy_etudiant, 3),
                    "moyenne_apres_jointure": round(moy_join, 3),
                    "ecart_points": round(100 * (moy_join - moy_etudiant), 1)}

fig, ax = plt.subplots(figsize=(7.4, 3.4))
b = ax.bar(["1 ligne par étudiant", "après jointure naïve"],
           [moy_etudiant * 100, moy_join * 100], color=[C["blue"], C["red"]],
           alpha=.85, width=.55)
for bb, v in zip(b, [moy_etudiant, moy_join]):
    ax.text(bb.get_x() + bb.get_width() / 2, v * 100 + 1, f"{v:.1%}", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylabel("taux de réussite (%)")
ax.set_title(f"La jointure multiplie les lignes par {len(lignes_notes)/400:.2f} "
             "et déplace la moyenne")
ax.set_ylim(0, 100)
save(fig, "h09_jointure.png")

# ============================================ h10 : comparaison d'imputations
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

X = notes.copy()
masque = rng.random(X.shape) < .20
X_masque = X.copy()
X_masque[masque] = np.nan

resultats = {}
for nom, imp in [
    ("moyenne", SimpleImputer(strategy="mean")),
    ("médiane", SimpleImputer(strategy="median")),
    ("kNN (k=5)", KNNImputer(n_neighbors=5)),
    ("itératif", IterativeImputer(random_state=4112, max_iter=20)),
]:
    Ximp = imp.fit_transform(X_masque)
    rmse = float(np.sqrt(np.mean((Ximp[masque] - X[masque]) ** 2)))
    resultats[nom] = round(rmse, 3)
cles["imputations_comparees"] = resultats

fig, ax = plt.subplots(figsize=(7.2, 3.4))
noms = list(resultats); vals = [resultats[k] for k in noms]
b = ax.bar(noms, vals, color=[C["red"], C["orange"], C["blue"], C["green"]], alpha=.85, width=.6)
for bb, v in zip(b, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .02, f"{v:.2f}", ha="center",
            fontsize=10.5, fontweight="bold")
ax.set_ylabel("RMSE de reconstruction")
ax.set_title("Retrouver 20 % de valeurs masquées : le modèle bat la moyenne")
ax.set_ylim(0, max(vals) * 1.25)
save(fig, "h10_imputations.png")

with open(JSON, "w", encoding="utf-8") as f:
    json.dump(cles, f, ensure_ascii=False, indent=2)

print("\n--- chiffres ajoutes ---")
print("exploration :", json.dumps(cles["exploration"], ensure_ascii=False))
print("jointure :", json.dumps(cles["jointure"], ensure_ascii=False))
print("imputations :", json.dumps(cles["imputations_comparees"], ensure_ascii=False))
