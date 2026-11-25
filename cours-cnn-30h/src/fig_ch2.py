"""Figures du chapitre 2 -- L'operation de convolution.

Toutes les valeurs sont calculees ici (numpy / scipy) et consignees dans
figures/key_numbers_ch2.json.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy import signal
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
K = {}
HERE = os.path.dirname(__file__)

# ============================================================ 1. pas a pas
I = np.array([
    [ 1,  2,  0,  1,  3],
    [ 4,  5,  1,  0,  2],
    [ 2,  3,  6,  1,  1],
    [ 0,  1,  4,  8,  2],
    [ 1,  2,  1,  3,  5]], dtype=float)
Kmat = np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]], dtype=float)

# correlation (convention des reseaux de neurones)
def corr2d(img, kern):
    kh, kw = kern.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kern)
    return out

O = corr2d(I, Kmat)
K["ex_in"] = I.astype(int).tolist()
K["ex_kernel"] = Kmat.astype(int).tolist()
K["ex_out"] = O.astype(int).tolist()
K["ex_premier"] = float(np.sum(I[:3, :3] * Kmat))
K["ex_produit_detail"] = [float(a * b) for a, b in zip(I[:3, :3].ravel(), Kmat.ravel())]

fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.3),
                         gridspec_kw={"width_ratios": [1, 1, 1.25, 1]})

def show_mat(ax, M, title, cmap="Blues", fmt="{:.0f}", fs=9.5, vmin=None, vmax=None):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fs, color=C["deep"], fontweight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

show_mat(axes[0], I, "Entrée $I$ (5×5)")
show_mat(axes[1], Kmat, "Noyau $K$ (3×3)", cmap="Oranges")
axes[1].add_patch(Rectangle((-.5, -.5), 3, 3, fill=False, edgecolor=C["red"], lw=2))
axes[2].axis("off")
axes[2].set_title("Premier élément de la sortie", fontsize=10)
txt = ("$\\sum$ (zone $\\times$ noyau)\n\n"
       + " + ".join(f"{v:.0f}" for v in K["ex_produit_detail"])
       + f"\n= {K['ex_premier']:.0f}")
axes[2].text(.5, .58, txt, ha="center", va="center", fontsize=11,
             transform=axes[2].transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=C["light"], edgecolor=C["grey"]))
axes[2].text(.5, .12,
             "Le noyau glisse ensuite\nsur toute l'image", ha="center", fontsize=9,
             transform=axes[2].transAxes, color=C["grey"], style="italic")
show_mat(axes[3], O, "Sortie (3×3)", cmap="Greens", fs=10)
save(fig, "g01_convolution_pas_a_pas.png")

# ============================================================ 2. glissement
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
positions = [(0, 0, "1ᵉʳ élément"), (0, 1, "2ᵉ élément"), (1, 0, "4ᵉ élément")]
for ax, (i, j, t) in zip(axes, positions):
    show_mat(ax, I, t, fmt="{:.0f}")
    ax.add_patch(Rectangle((j - .5, i - .5), 3, 3, fill=False, edgecolor=C["red"], lw=2.2))
    ax.text(j + 1, i + 1, f"{O[i, j]:.0f}", ha="center", va="center", fontsize=13,
            color=C["red"], fontweight="bold")
fig.suptitle("Le noyau balaye l'image : chaque position produit un pixel de la sortie",
             fontsize=11, fontweight="bold", y=1.03)
save(fig, "g02_glissement_noyau.png")

# ============================================================ 3. noyaux classiques
noyaux = {
    "Flou (moyenneur)": np.ones((3, 3)) / 9,
    "Flou gaussien": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16,
    "Netteté": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "Contours (Sobel X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "Contours (Sobel Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "Laplacien": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "Emboss (relief)": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
    "Rehaussement": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float) * 1.0,
    "Détection d'intensité": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], float),
}
fig, axes = plt.subplots(2, 5, figsize=(12.6, 5.0))
for ax, (nom, kk) in zip(axes.ravel(), noyaux.items()):
    lim = max(abs(kk).max(), 1e-9)
    ax.imshow(kk, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{kk[i, j]:.2g}", ha="center", va="center", fontsize=7.5,
                    color=C["deep"], fontweight="bold")
    ax.set_title(nom, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for ax in axes.ravel()[len(noyaux):]:
    ax.axis("off")
fig.suptitle("Les noyaux classiques du traitement d'image", fontsize=11.5,
             fontweight="bold", y=1.0)
save(fig, "g03_noyaux_classiques.png")
K["noyaux"] = {k: v.tolist() for k, v in noyaux.items()}

# ============================================================ 4. effets sur image
img = np.load(os.path.join(HERE, "..", "data", "face_gray.npy")).astype(float)
img = img[::2, ::2]                       # 384 x 512
K["image_convolution"] = list(img.shape)

def applique(im, kk):
    return signal.convolve2d(im, kk, mode="same", boundary="symm")

effets = [
    ("Originale", img),
    ("Flou (moyenneur)", applique(img, noyaux["Flou (moyenneur)"])),
    ("Gaussien", applique(img, noyaux["Flou gaussien"])),
    ("Netteté", applique(img, noyaux["Netteté"])),
    ("Sobel X (contours verticaux)", np.abs(applique(img, noyaux["Contours (Sobel X)"]))),
    ("Laplacien (contours)", np.abs(applique(img, noyaux["Laplacien"]))),
]
fig, axes = plt.subplots(1, 6, figsize=(13.6, 2.9))
for ax, (t, im) in zip(axes, effets):
    ax.imshow(im, cmap="gray")
    ax.set_title(t, fontsize=8.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Le même noyau appliqué à toute l'image : flou, netteté, contours",
             fontsize=11.5, fontweight="bold", y=1.03)
save(fig, "g04_effets_noyaux.png")

# ============================================================ 5. padding
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
petit = I.copy()
ax = axes[0]
show_mat(ax, petit, "Sans padding (valid)\nsortie 3×3", fmt="{:.0f}")
ax.add_patch(Rectangle((-.5, -.5), 3, 3, fill=False, edgecolor=C["red"], lw=1.8))
ax.add_patch(Rectangle((1.5, 1.5), 3, 3, fill=False, edgecolor=C["green"], lw=1.8, ls="--"))

pad = np.pad(petit, 1, constant_values=0)
ax = axes[1]
show_mat(ax, pad, "Padding = 1 (same)\nsortie 5×5", fmt="{:.0f}")
ax.add_patch(Rectangle((-.5, -.5), 7, 7, fill=False, edgecolor=C["green"], lw=1.8))
ax.add_patch(Rectangle((-.5, -.5), 3, 3, fill=False, edgecolor=C["red"], lw=1.8))

ax = axes[2]
ax.axis("off")
formules = ("$N_{out} = \\left\\lfloor \\dfrac{N_{in} - K + 2P}{S} \\right\\rfloor + 1$\n\n"
            "valid  : $P=0$  →  $5 \\to 3$\n"
            "same   : $P=(K-1)/2$  →  $5 \\to 5$\n"
            "stride 2 : $5 \\to 2$\n\n"
            "$N_{out} \\approx N_{in}/S$")
ax.text(.5, .5, formules, ha="center", va="center", fontsize=10.5,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=C["light"], edgecolor=C["grey"]))
axes[2].set_title("Formule de dimension", fontsize=10)
save(fig, "g05_padding.png")

K["formule_p_valid_same"] = 1
K["ex_valid"] = "5 -> 3"
K["ex_same"] = "5 -> 5"
K["ex_stride2"] = "5 -> 2"

# ============================================================ 6. stride
def corr2d_stride(img, kern, s):
    kh, kw = kern.shape
    h = (img.shape[0] - kh) // s + 1
    w = (img.shape[1] - kw) // s + 1
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(img[i * s:i * s + kh, j * s:j * s + kw] * kern)
    return out

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
for ax, st in zip(axes, [1, 2]):
    o = corr2d_stride(I, Kmat, st)
    show_mat(ax, I, f"stride = {st}  →  sortie {o.shape[0]}×{o.shape[1]}", fmt="{:.0f}")
    for i, j in [(0, 0), (0, st), (0, 2 * st), (st, 0), (2 * st, 0)]:
        if i + 3 <= I.shape[0] and j + 3 <= I.shape[1]:
            ax.add_patch(Rectangle((j - .5, i - .5), 3, 3, fill=False,
                                   edgecolor=C["red"] if st == 1 else C["orange"], lw=2))
    K[f"stride{st}_sortie"] = [int(o.shape[0]), int(o.shape[1])]
save(fig, "g06_stride.png")

# ============================================================ 7. formule : carte des tailles
Ks = np.arange(1, 12)
Ns = np.arange(8, 130)
P = 0
S = 1
Z = np.floor((Ns[:, None] - Ks[None, :] + 2 * P) / S) + 1
Z[Z < 0] = np.nan
fig, ax = plt.subplots(figsize=(7.6, 3.8))
im = ax.imshow(Z.T, aspect="auto", origin="lower", cmap="viridis",
               extent=[Ns[0], Ns[-1], Ks[0], Ks[-1]])
ax.set_xlabel("$N_{in}$ (côté de l'entrée)")
ax.set_ylabel("$K$ (côté du noyau)")
ax.set_title("Taille de sortie $N_{out}$ (valid, $S=1$)")
cb = fig.colorbar(im, ax=ax); cb.set_label("$N_{out}$", fontsize=9)
ax.grid(False)
ax.plot([28], [3], "o", color=C["red"], ms=9)
ax.annotate("MNIST 28×28, noyau 3×3  →  26×26", (28, 3), xytext=(40, 6.5),
            fontsize=9, color=C["red"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.3))
K["mnist_conv3x3_sortie"] = 26
save(fig, "g07_carte_dimensions.png")

# ============================================================ 8. convolution 1D
t = np.linspace(0, 20, 600)
x1 = np.sin(t) + .25 * RNG.normal(size=t.size)
noyau1 = np.hanning(21); noyau1 /= noyau1.sum()
noyau2 = np.array([-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1], float)
lisse = np.convolve(x1, noyau1, mode="same")
derive = np.convolve(x1, noyau2, mode="same")
K["conv1d_noyau_lissage"] = 21
K["conv1d_noyau_derive"] = 11
K["conv1d_sortie_formule"] = "N - K + 1"
K["conv1d_sortie"] = int(600 - 21 + 1)

fig, axes = plt.subplots(3, 1, figsize=(10.4, 5.2), sharex=True)
axes[0].plot(t, x1, color=C["grey"], lw=.9, label="signal brut")
axes[0].set_title("Signal 1D bruité", fontsize=10)
axes[1].plot(t, lisse, color=C["blue"], lw=1.6, label="lissage (noyau de Hann, 21)")
axes[1].set_title("Après convolution par un noyau de lissage", fontsize=10)
axes[2].plot(t, derive, color=C["red"], lw=1.1, label="dérivée (noyau de différence)")
axes[2].set_title("Après convolution par un noyau de dérivation (détection de fronts)", fontsize=10)
axes[2].set_xlabel("temps")
for a in axes:
    a.legend(loc="upper right", fontsize=8.5)
save(fig, "g08_convolution_1d.png")

# ============================================================ 9. champ recepteur
cotes = np.arange(1, 12)
champ = [1 + 2 * n for n in cotes]      # noyau 3x3, stride 1 : +2 par couche
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.plot(cotes, champ, "o-", color=C["blue"], lw=2, ms=7)
ax.set_xlabel("nombre de couches convolutives (noyau 3×3, stride 1)")
ax.set_ylabel("champ récepteur (pixels)")
ax.set_title("Le champ récepteur grandit de 2 pixels par couche")
for n in [1, 3, 5, 10]:
    ax.annotate(f"{2 * n + 1}×{2 * n + 1}", (n, champ[n - 1]),
                textcoords="offset points", xytext=(6, 6), fontsize=9,
                color=C["deep"], fontweight="bold")
K["champ_recepteur_5_couches"] = int(champ[4])
K["champ_recepteur_10_couches"] = int(champ[9])
save(fig, "g09_champ_recepteur.png")

# ============================================================ 10. multi-canaux
fig = plt.figure(figsize=(11.0, 3.4))
canaux = [(np.load(os.path.join(HERE, "..", "data", "face_rgb.npy"))[:, :, k])[::6, ::6]
          for k in range(3)]
for k, (ax, nom, cm) in enumerate(zip([fig.add_subplot(1, 4, i + 1) for i in range(4)],
                                      ["canal R", "canal V", "canal B", "somme pondérée"],
                                      ["Reds", "Greens", "Blues", "gray"])):
    if k < 3:
        ax.imshow(canaux[k], cmap=cm)
        ax.set_title(f"{nom} ({canaux[k].shape[0]}×{canaux[k].shape[1]})", fontsize=9.5)
    else:
        s = .299 * canaux[0] + .587 * canaux[1] + .114 * canaux[2]
        ax.imshow(s, cmap=cm)
        ax.set_title("niveaux de gris", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Une image couleur = 3 matrices empilées (tenseur 3 canaux)",
             fontsize=11.5, fontweight="bold", y=1.03)
K["canaux_rgb"] = [c.shape for c in canaux]
save(fig, "g10_multicanaux.png")

# ============================================================ 11. convolution vs correlation
kern = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
K["conv_vs_corr_kernel"] = kern.astype(int).tolist()
K["conv_vs_corr_flip"] = kern[::-1, ::-1].astype(int).tolist()
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
show_mat(axes[0], kern, "Noyau $K$", cmap="Oranges")
show_mat(axes[1], kern[::-1, ::-1], "Noyau retourné $K'$", cmap="Oranges")
axes[2].axis("off")
axes[2].set_title("Deux conventions", fontsize=10)
axes[2].text(.5, .55,
             "corrélation croisée : $K$\n(convention des CNN — PyTorch)\n\n"
             "convolution : $K$ retourné\n(traitement du signal)",
             ha="center", va="center", fontsize=9.6, transform=axes[2].transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=C["light"], edgecolor=C["grey"]))
save(fig, "g11_conv_vs_corr.png")

with open(os.path.join(FIGDIR, "key_numbers_ch2.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch2.json -- {len(K)} cles")
