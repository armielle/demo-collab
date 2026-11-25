"""Figures du chapitre 3 -- Architecture d'un reseau convolutif."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
K = {}
HERE = os.path.dirname(__file__)

# ==================================================== 1. parametres d'une couche conv
def params_conv(k, cin, cout):
    return k * k * cin * cout + cout

def params_dense(nin, nout):
    return nin * nout + nout

K["p_conv_3_1_32"] = params_conv(3, 1, 32)
K["p_conv_3_3_64"] = params_conv(3, 3, 64)
K["p_conv_3_64_128"] = params_conv(3, 64, 128)
K["p_conv_5_16_120_vieux"] = params_conv(5, 16, 120)
K["p_dense_1024_1000"] = params_dense(1024, 1000)

k = np.arange(1, 8)
fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.5))
for cin, coul in [(1, C["blue"]), (3, C["cyan"]), (64, C["orange"])]:
    ax[0].plot(k, [params_conv(int(kk), cin, 64) for kk in k], "o-", color=coul, lw=2,
               label=f"$C_{{in}}={cin}$, $C_{{out}}=64$")
ax[0].set_xlabel("côté du noyau $K$"); ax[0].set_ylabel("paramètres")
ax[0].set_yscale("log"); ax[0].set_title("Paramètres d'une couche convolutive\n$K^2 C_{in} C_{out} + C_{out}$")
ax[0].legend(fontsize=8.5)

ent = np.array([256, 1024, 4096, 16384])
ax[1].plot(ent, [params_dense(e, 1000) for e in ent], "o-", color=C["red"], lw=2, label="couche dense (1000)")
ax[1].plot(ent, [params_conv(3, e, 64) for e in ent], "s-", color=C["green"], lw=2, label="couche conv 3×3 (64)")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("nombre d'entrées"); ax[1].set_ylabel("paramètres")
ax[1].set_title("Dense vs convolutif : la même donnée,\ndeux coûts très différents")
ax[1].legend(fontsize=8.5)
save(fig, "h01_parametres_couches.png")

# ==================================================== 2. ReLU et activations
x = np.linspace(-6, 6, 500)
fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.9))
acts = {
    "ReLU : $\\max(0,x)$": np.maximum(0, x),
    "Sigmoïde": 1 / (1 + np.exp(-x)),
    "Tanh": np.tanh(x),
    "LeakyReLU ($\\alpha$=0,1)": np.where(x > 0, x, .1 * x),
}
for ax, (nom, y), coul in zip(axes, acts.items(),
                              [C["blue"], C["orange"], C["purple"], C["cyan"]]):
    ax.plot(x, y, color=coul, lw=2.2)
    ax.axhline(0, color=C["grey"], lw=.7)
    ax.axvline(0, color=C["grey"], lw=.7)
    ax.set_title(nom, fontsize=10)
    ax.set_ylim(-1.6, 6.2)
    ax.grid(alpha=.25)
fig.suptitle("Fonctions d'activation : ReLU est la plus utilisée dans les CNN",
             fontsize=11.5, fontweight="bold", y=1.04)
save(fig, "h02_activations.png")

# ==================================================== 3. pooling
M = np.array([[1, 2, 5, 3], [4, 9, 2, 1], [7, 3, 8, 6], [2, 1, 4, 5]], float)
K["pooling_matrice"] = M.astype(int).tolist()

def pool(M, mode, k=2, s=2):
    n = (M.shape[0] - k) // s + 1
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            blk = M[i * s:i * s + k, j * s:j * s + k]
            out[i, j] = blk.max() if mode == "max" else blk.mean()
    return out

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
for ax, (mode, t, cm) in zip(axes[:2], [("max", "Max pooling 2×2", "Greens"),
                                        ("avg", "Moyenne 2×2", "Purples")]):
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j, 3 - i), 1, 1, facecolor="white", edgecolor=C["grey"], lw=.6))
            ax.text(j + .5, 3.5 - i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j * 2, 2 - i * 2), 2, 2, fill=False,
                                   edgecolor=C[mode == "max" and "blue" or "purple"], lw=2))
    out = pool(M, mode)
    ax.set_title(t, fontsize=10); ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.text(2, -.75, "→  " + "  ".join("  ".join(f"{v:.1f}" for v in r) for r in out),
            ha="center", fontsize=9.5, color=C["deep"], fontweight="bold")
    K[f"pooling_{mode}"] = out.tolist()

ax = axes[2]; ax.axis("off"); ax.set_title("Ce que le pooling apporte", fontsize=10)
ax.text(.5, .5,
        "• divise la résolution par $S$\n  (ici 4×4 → 2×2)\n"
        "• réduit les paramètres des couches suivantes\n"
        "• rend la détection moins sensible\n  à la position exacte du motif\n"
        "• aucun poids à apprendre",
        ha="center", va="center", fontsize=9.4, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=C["light"], edgecolor=C["grey"]))
save(fig, "h03_pooling.png")

# ==================================================== 4. pipeline complet
fig, ax = plt.subplots(figsize=(12.6, 3.1))
ax.axis("off")
etapes = [("Entrée\n32×32×1", C["grey"]), ("Conv 5×5\n6 filtres", C["blue"]),
          ("ReLU", C["cyan"]), ("Pool 2×2", C["orange"]), ("Conv 5×5\n16 filtres", C["blue"]),
          ("ReLU", C["cyan"]), ("Pool 2×2", C["orange"]), ("Flatten\n400", C["purple"]),
          ("Dense\n120", C["purple"]), ("Softmax\n10 classes", C["red"])]
w, gap = .085, .0175
for i, (t, coul) in enumerate(etapes):
    x0 = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x0, .35), w, .34, boxstyle="round,pad=0.01",
                                facecolor=coul, alpha=.16, edgecolor=coul, lw=1.5))
    ax.text(x0 + w / 2, .52, t, ha="center", va="center", fontsize=7.9, color=C["deep"],
            fontweight="bold")
    if i < len(etapes) - 1:
        ax.annotate("", xy=(x0 + w + gap, .52), xytext=(x0 + w, .52),
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.2))
ax.text(.5, .95, "Le pipeline d'un CNN : alternance convolution / activation / pooling,\n"
                 "puis aplatissement et couches denses",
        ha="center", fontsize=11.5, fontweight="bold", transform=ax.transAxes, color=C["deep"])
ax.text(.5, .06, "extraction automatique des caractéristiques  →  décision",
        ha="center", fontsize=9.2, style="italic", color=C["grey"], transform=ax.transAxes)
ax.set_xlim(-.01, 1.01); ax.set_ylim(0, 1.02)
save(fig, "h04_pipeline.png")

# ==================================================== 5. LeNet-5 exact
lenet = [("Entrée", 32, 32, 1, None, 0),
         ("C1 conv 5×5, 6", 28, 28, 6, ("conv", 5, 1, 6), params_conv(5, 1, 6)),
         ("S2 pool 2×2", 14, 14, 6, None, 0),
         ("C3 conv 5×5, 16", 10, 10, 16, ("conv", 5, 6, 16), params_conv(5, 6, 16)),
         ("S4 pool 2×2", 5, 5, 16, None, 0),
         ("C5 conv 5×5, 120", 1, 1, 120, ("conv", 5, 16, 120), params_conv(5, 16, 120)),
         ("F6 dense 84", 1, 1, 84, ("dense", 120, 84), params_dense(120, 84)),
         ("Sortie softmax 10", 1, 1, 10, ("dense", 84, 10), params_dense(84, 10))]
K["lenet_couches"] = [{"nom": n, "h": h, "w": w_, "c": cc, "params": p}
                      for (n, h, w_, cc, _, p) in lenet]
K["lenet_params_total"] = int(sum(p for (_, _, _, _, _, p) in lenet))
K["lenet_entrees"] = 32 * 32
K["lenet_flatten"] = 5 * 5 * 16

fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.6),
                         gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
labels = [n.replace(" ", "\n", 1) for (n, *_ ) in lenet]
vals = [p for (_, _, _, _, _, p) in lenet]
cols = [C["grey"] if v == 0 else C["blue"] for v in vals]
b = ax.bar(range(len(vals)), [max(v, 1) for v in vals], color=cols, alpha=.85)
ax.set_yscale("log"); ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7.2, rotation=0)
ax.set_ylabel("paramètres (log)")
ax.set_title(f"LeNet-5 (LeCun, 1998) : {K['lenet_params_total']:,} paramètres"
             .replace(",", " "), fontsize=11)
for i, v in enumerate(vals):
    if v > 0:
        ax.text(i, v * 1.4, f"{v:,}".replace(",", " "), ha="center", fontsize=7.4,
                color=C["deep"], fontweight="bold")

ax = axes[1]; ax.axis("off")
ax.set_title("Pourquoi si peu de paramètres ?", fontsize=11)
ax.text(.5, .5,
        f"LeNet-5 complet : {K['lenet_params_total']:,} paramètres\n\n"
        f"Un MLP sur les mêmes {K['lenet_entrees']} entrées\navec 1000 neurones en 1ʳᵉ couche :\n"
        f"{params_dense(K['lenet_entrees'], 1000):,} paramètres\n\n"
        "→ le partage de poids et la localité\n   font tout le travail"
        .replace(",", " "),
        ha="center", va="center", fontsize=9.8, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=C["light"], edgecolor=C["grey"]))
save(fig, "h05_lenet.png")

# ==================================================== 6. VGG-16 calcule exactement
def vgg16_params():
    cfg = [(64, 2), ("M",), (128, 2), ("M",), (256, 3), ("M",), (512, 3), ("M",), (512, 3), ("M",)]
    cin, tot, couches = 3, 0, []
    for e in cfg:
        if e == ("M",):
            couches.append(("pool", 0)); continue
        n, rep = e
        for _ in range(rep):
            p = params_conv(3, cin, n); tot += p
            couches.append((f"conv3-{n}", p)); cin = n
    # tete dense classique VGG-16 : 512*7*7 -> 4096 -> 4096 -> 1000
    d1 = params_dense(512 * 7 * 7, 4096); d2 = params_dense(4096, 4096); d3 = params_dense(4096, 1000)
    return tot, d1, d2, d3, couches

vc, d1, d2, d3, couches = vgg16_params()
K["vgg16_conv_params"] = int(vc)
K["vgg16_dense_params"] = int(d1 + d2 + d3)
K["vgg16_total"] = int(vc + d1 + d2 + d3)
K["vgg16_part_dense_pct"] = round(100 * (d1 + d2 + d3) / K["vgg16_total"], 1)

arch = {
    "LeNet-5\n(1998)": (K["lenet_params_total"], 0.95),
    "AlexNet\n(2012)": (62_300_000, 15.3),
    "VGG-16\n(2014)": (K["vgg16_total"], 7.3),
    "ResNet-50\n(2015)": (25_600_000, 3.6),
}
fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.7), gridspec_kw={"width_ratios": [1, 1]})
ax = axes[0]
noms = list(arch); vals = [arch[n][0] for n in noms]
b = ax.bar(noms, vals, color=[C["grey"], C["orange"], C["blue"], C["green"]], alpha=.88)
ax.set_yscale("log"); ax.set_ylabel("paramètres (log)")
ax.set_title("Taille des architectures historiques", fontsize=11)
for bb, v in zip(b, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v * 1.35, f"{v/1e6:.1f} M", ha="center",
            fontsize=8.6, fontweight="bold", color=C["deep"])
ax.tick_params(axis="x", labelsize=9)

ax = axes[1]
parts = [K["vgg16_conv_params"], K["vgg16_dense_params"]]
ax.pie(parts, labels=["couches conv", "couches denses (tête)"], autopct="%1.1f %%",
       colors=[C["blue"], C["red"]], startangle=90, textprops={"fontsize": 9.5},
       wedgeprops={"alpha": .85})
ax.set_title(f"VGG-16 : où sont les {K['vgg16_total']/1e6:.0f} M de paramètres ?", fontsize=11)
save(fig, "h06_architectures.png")

# ==================================================== 7. feature maps hierarchie
img = np.load(os.path.join(HERE, "..", "data", "face_gray.npy")).astype(float)[::4, ::4]
from scipy import signal
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
blur = np.ones((5, 5)) / 25

f1 = np.abs(signal.convolve2d(img, sobel, mode="same", boundary="symm"))
f2 = np.abs(signal.convolve2d(signal.convolve2d(img, blur, mode="same", boundary="symm"),
                              lap, mode="same", boundary="symm"))
f3 = signal.convolve2d(signal.convolve2d(img, blur, mode="same", boundary="symm"),
                       blur, mode="same", boundary="symm")

fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.1))
for ax, (im, t) in zip(axes, [(img, "Entrée"), (f1, "Couche 1 : contours"),
                              (f2, "Couche 2 : textures / formes"), (f3, "Couche 3 : objets")]):
    ax.imshow(im, cmap="gray")
    ax.set_title(t, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Hiérarchie des caractéristiques : des bords aux objets",
             fontsize=11.5, fontweight="bold", y=1.04)
save(fig, "h07_hierarchie_features.png")

with open(os.path.join(FIGDIR, "key_numbers_ch3.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch3.json -- {len(K)} cles")
