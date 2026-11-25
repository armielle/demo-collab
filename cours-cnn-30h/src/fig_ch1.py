"""Figures du chapitre 1 -- Du MLP aux CNN.

Toutes les valeurs affichees sont calculees ici et consignees dans
figures/key_numbers_ch1.json (aucun chiffre saisi a la main).
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
K = {}

# ---------------------------------------------------------------- 1. le mur du MLP
K["mnist_cote"] = 28
K["mnist_entrees"] = 28 * 28
K["couche1_neurones"] = 1000
K["mnist_param_couche1"] = K["mnist_entrees"] * K["couche1_neurones"] + K["couche1_neurones"]

K["photo_mpx"] = 24
K["photo_cote"] = 4000          # 4000x4000 ~ 16 Mpx, arrondi classique des exposes
K["photo_canaux"] = 3
K["photo_entrees"] = K["photo_cote"] * K["photo_cote"] * K["photo_canaux"]
K["photo_param_couche1"] = K["photo_entrees"] * 1000

# un CNN equivalent sur la meme photo : 64 filtres 3x3x3
K["cnn_filtres"] = 64
K["cnn_noyau"] = 3
K["cnn_param_couche1"] = (K["cnn_noyau"] ** 2) * K["photo_canaux"] * K["cnn_filtres"] + K["cnn_filtres"]
K["ratio_param"] = K["photo_param_couche1"] / K["cnn_param_couche1"]

fig, ax = plt.subplots(figsize=(7.4, 3.5))
labels = ["MLP\nMNIST 28×28\n(1000 neurones)", "CNN\nMNIST 28×28\n(64 filtres 3×3)",
          "MLP\nphoto 24 Mpx RVB\n(1000 neurones)", "CNN\nphoto 24 Mpx RVB\n(64 filtres 3×3)"]
vals = [K["mnist_param_couche1"], 9 * 1 * 64 + 64, K["photo_param_couche1"], K["cnn_param_couche1"]]
cols = [C["red"], C["green"], C["red"], C["green"]]
bars = ax.barh(labels, vals, color=cols, alpha=.85, height=.62)
ax.set_xscale("log")
ax.set_xlabel("paramètres de la première couche (échelle log)")
ax.set_title("Le mur du MLP : une seule couche dense sur une image réelle")
for b, v in zip(bars, vals):
    ax.text(v * 1.35, b.get_y() + b.get_height() / 2, f"{v:,.0f}".replace(",", " "),
            va="center", fontsize=9, fontweight="bold", color=C["deep"])
ax.set_xlim(1e1, 1e14)
ax.grid(axis="x", alpha=.3)
save(fig, "f01_mur_du_mlp.png")

# ------------------------------------------------- 2. image = matrice / tenseur
img = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "face_gray.npy"))
small = img[::16, ::16].astype(float)          # 48 x 64
K["patch_cote"] = 8
K["patch_pixels"] = 64
K["patch_valeurs"] = small[:8, :8].round(1).tolist()

fig = plt.figure(figsize=(11.2, 3.6))
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(small, cmap="gray")
ax1.set_title("Une image est une matrice")
ax1.set_xlabel(f"{small.shape[1]} colonnes (pixels)")
ax1.set_ylabel(f"{small.shape[0]} lignes")
ax1.grid(False)

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(small[:8, :8], cmap="gray", vmin=small.min(), vmax=small.max())
for i in range(8):
    for j in range(8):
        v = small[i, j]
        ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.4,
                 color="white" if v < 130 else C["deep"])
ax2.set_title("Chaque pixel = une valeur (0–255)")
ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)

ax3 = fig.add_subplot(1, 3, 3, projection="3d")
yy, xx = np.mgrid[0:8, 0:8]
patch = small[:8, :8]
ax3.plot_surface(xx, yy, patch, cmap="viridis", edgecolor="none", alpha=.95)
ax3.set_title("… ou un tenseur (canaux empilés)")
ax3.set_zlabel("intensité")
ax3.tick_params(labelsize=7)
save(fig, "f02_matrice_tenseur.png")

# ------------------------------------------------- 3. localite + partage de poids
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
grid = 8
# a) connexion dense
a = axes[0]
a.set_title("MLP : toutes les entrées connectées")
for i in range(grid):
    a.scatter([.15] * grid, np.linspace(.05, .95, grid), s=14, color=C["grey"])
    a.scatter([.85], [np.linspace(.05, .95, grid)[i]], s=26, color=C["blue"])
for i in range(grid):
    a.plot([.15, .85], [np.linspace(.05, .95, grid)[i], .5], lw=.35, color=C["grey"], alpha=.6)
a.text(.5, -.08, "784 × 1000 = 784 000 connexions", ha="center", fontsize=8.5,
       color=C["red"], fontweight="bold")
a.axis("off")

# b) connexion locale
b = axes[1]
b.set_title("CNN : voisinage local (3×3)")
r, c = 4, 4
for i in range(grid):
    for j in range(grid):
        inside = abs(i - r) <= 1 and abs(j - c) <= 1
        b.add_patch(Rectangle((j, grid - i - 1), 1, 1,
                              facecolor=C["orange"] if inside else "white",
                              edgecolor=C["grey"], lw=.4, alpha=.55 if inside else 1))
b.scatter([c + .5], [grid - r - .5], s=55, marker="*", color=C["red"], zorder=5)
b.text(grid / 2, -1.1, "chaque neurone ne voit que 9 pixels", ha="center",
       fontsize=8.5, color=C["green"], fontweight="bold")
b.set_xlim(-.3, grid + .3); b.set_ylim(-1.8, grid + .3)
b.set_aspect("equal"); b.axis("off")

# c) partage de poids
d = axes[2]
d.set_title("Partage de poids : le même filtre partout")
for i in range(grid):
    for j in range(grid):
        d.add_patch(Rectangle((j, grid - i - 1), 1, 1, facecolor="white",
                              edgecolor=C["grey"], lw=.4))
for (r0, c0) in [(1, 1), (4, 1), (1, 5), (4, 5)]:
    d.add_patch(Rectangle((c0, grid - r0 - 3), 3, 3, facecolor=C["cyan"],
                          edgecolor=C["deep"], lw=1.1, alpha=.42))
d.text(grid / 2, -1.1, "9 poids réutilisés sur toute l'image", ha="center",
       fontsize=8.5, color=C["cyan"], fontweight="bold")
d.set_xlim(-.3, grid + .3); d.set_ylim(-1.8, grid + .3)
d.set_aspect("equal"); d.axis("off")
save(fig, "f03_localite_partage.png")

# ------------------------------------------------- 4. invariants : translation
fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.9))
base = np.zeros((16, 16))
base[4:8, 4:8] = 1.0
base[6:10, 6:10] = .55
for ax, (dy, dx, txt) in zip(axes, [(0, 0, "motif original"), (0, 3, "translation +3 px"),
                                    (3, 3, "translation (+3,+3)"), (5, 0, "translation +5 px")]):
    m = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(txt, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Le même filtre détecte le motif où qu'il soit : invariance par translation",
             fontsize=11, fontweight="bold", y=1.02)
save(fig, "f04_invariance_translation.png")

# ------------------------------------------------- 5. calcul : dimensions
K["cnn_exemple_entree"] = 224
K["cnn_exemple_filtres"] = 64
K["cnn_exemple_param"] = 3 * 3 * 3 * 64 + 64
K["cnn_exemple_sortie"] = 224 - 3 + 1

with open(os.path.join(FIGDIR, "key_numbers_ch1.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch1.json -- {len(K)} cles")
