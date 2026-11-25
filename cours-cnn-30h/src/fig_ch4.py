"""Figures du chapitre 4 -- Entrainement, regularisation, transfert."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
K = {}
HERE = os.path.dirname(__file__)

# ==================================================== 1. courbes d'apprentissage
def courbe(n, biais, bruit, pente, cap=None):
    t = np.arange(1, n + 1)
    y = biais + pente * np.exp(-t / (n * .35)) + bruit * RNG.normal(size=n) * .3
    if cap is not None:
        y = np.maximum(y, cap)
    return y

n = 40
c_under = (courbe(n, .42, .02, .10), courbe(n, .40, .02, .10))
c_good = (courbe(n, .05, .02, .42), courbe(n, .08, .04, .40))
c_over = (courbe(n, .01, .015, .49), courbe(n, .35, .05, .02) + np.linspace(0, .10, n))
K["courbes_finales"] = {
    "sous_apprentissage": [round(float(c_under[0][-1]), 3), round(float(c_under[1][-1]), 3)],
    "bon_compromis": [round(float(c_good[0][-1]), 3), round(float(c_good[1][-1]), 3)],
    "surapprentissage": [round(float(c_over[0][-1]), 3), round(float(c_over[1][-1]), 3)],
}

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), sharey=True)
for ax, (tr, va, t, coul) in zip(axes,
                                 [(c_under[0], c_under[1], "Sous-apprentissage", C["blue"]),
                                  (c_good[0], c_good[1], "Bon compromis", C["green"]),
                                  (c_over[0], c_over[1], "Surapprentissage", C["red"])]):
    ax.plot(tr, color=coul, lw=2, label="entraînement")
    ax.plot(va, color=coul, lw=2, ls="--", alpha=.75, label="validation")
    if t == "Surapprentissage":
        i = int(np.argmin(va))
        ax.axvline(i, color=C["orange"], ls=":", lw=1.6)
        ax.annotate("arrêt optimal", (i, va[i]), xytext=(i + 4, va[i] + .12),
                    fontsize=8.4, color=C["orange"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.2))
        K["epoque_arret_optimal"] = i
    ax.set_title(t, fontsize=10.5); ax.set_xlabel("époque")
    ax.set_ylim(0, .58); ax.legend(fontsize=8.4, loc="center right")
axes[0].set_ylabel("perte")
fig.suptitle("Lire les courbes : entraînement vs validation", fontsize=11.5,
             fontweight="bold", y=1.04)
save(fig, "i01_courbes_apprentissage.png")

# ==================================================== 2. dropout
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
def reseau(ax, masque, titre, tri=False):
    L = [5, 6, 6, 3]
    xs = np.linspace(.12, .88, len(L))
    for l in range(len(L) - 1):
        for i in range(L[l]):
            for j in range(L[l + 1]):
                actif = True
                if tri and l + 1 == 1 and masque is not None:
                    actif = masque[j]
                ax.plot([xs[l], xs[l + 1]], [1 - i / (L[l] - 1) if L[l] > 1 else .5,
                                             1 - j / (L[l + 1] - 1) if L[l + 1] > 1 else .5],
                        lw=.5, color=C["blue"] if actif else "#DDDDDD", alpha=.55, zorder=1)
    for l in range(len(L)):
        for i in range(L[l]):
            y = 1 - i / (L[l] - 1) if L[l] > 1 else .5
            off = False
            if tri and l == 1 and masque is not None:
                off = not masque[i]
            ax.add_patch(plt.Circle((xs[l], y), .035,
                                    color="#DDDDDD" if off else
                                    (C["cyan"] if l == 1 else C["blue"]),
                                    zorder=3, alpha=.9 if not off else .5))
    ax.set_title(titre, fontsize=10); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(-.05, 1.06)

reseau(axes[0], None, "Réseau complet")
m1 = np.array([True, False, True, True, False, True])
m2 = np.array([False, True, True, False, True, False])
reseau(axes[1], m1, "Dropout : neurones désactivés (p = 0,5)", tri=True)
reseau(axes[2], m2, "… un autre tirage à chaque itération", tri=True)
K["dropout_p"] = 0.5
K["dropout_actifs_sur_6"] = int(m1.sum())
save(fig, "i02_dropout.png")

# ==================================================== 3. normalisation par lot
avant = RNG.normal(2.0, 1.9, 4000)
apres = (avant - avant.mean()) / avant.std()
K["bn_moyenne_avant"] = round(float(avant.mean()), 3)
K["bn_ecart_type_avant"] = round(float(avant.std()), 3)
K["bn_moyenne_apres"] = round(float(apres.mean()), 3)
K["bn_ecart_type_apres"] = round(float(apres.std()), 3)

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.3), sharey=True)
for ax, (d, t, coul) in zip(axes, [(avant, "Avant normalisation\n(moyenne ≠ 0, variance libre)", C["red"]),
                                   (apres, "Après normalisation par lot\n(moyenne 0, écart-type 1)", C["green"])]):
    ax.hist(d, bins=45, color=coul, alpha=.75, density=True)
    ax.axvline(d.mean(), color=C["deep"], ls="--", lw=1.5,
               label=f"moyenne = {d.mean():.2f}")
    ax.set_title(t, fontsize=10); ax.set_xlabel("activation"); ax.legend(fontsize=8.8)
axes[0].set_ylabel("densité")
fig.suptitle("Batch normalization : recaler les activations à chaque couche",
             fontsize=11.5, fontweight="bold", y=1.04)
save(fig, "i03_batchnorm.png")

# ==================================================== 4. augmentation de donnees
img = np.load(os.path.join(HERE, "..", "data", "face_gray.npy")).astype(float)[::6, ::6]
def rot(im, ang):
    from scipy import ndimage
    return ndimage.rotate(im, ang, reshape=False, mode="nearest")

aug = [("originale", img), ("rotation +15°", rot(img, 15)), ("miroir horizontal", img[:, ::-1]),
       ("translation", np.roll(img, 18, axis=1)), ("zoom (recadrage)", img[12:-12, 12:-12]),
       ("bruit gaussien", np.clip(img + RNG.normal(0, 18, img.shape), 0, 255))]
fig, axes = plt.subplots(1, 6, figsize=(13.4, 2.7))
for ax, (t, im) in zip(axes, aug):
    ax.imshow(im, cmap="gray")
    ax.set_title(t, fontsize=8.6); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Augmentation de données : une image source, six variantes d'entraînement",
             fontsize=11.5, fontweight="bold", y=1.05)
K["augmentation_variantes"] = [t for t, _ in aug]
K["augmentation_facteur"] = len(aug)
save(fig, "i04_augmentation.png")

# ==================================================== 5. transfert learning
fig, ax = plt.subplots(figsize=(11.6, 3.3))
ax.axis("off")
def bloc(x, y, w, h, txt, coul, fs=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=coul, alpha=.17, edgecolor=coul, lw=1.5))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=C["deep"], fontweight="bold")
bloc(.02, .55, .40, .30, "Modèle pré-entraîné\n(ImageNet, 1,2 M images, 1000 classes)", C["grey"])
bloc(.46, .55, .24, .30, "Couches convolutives\nGELÉES", C["blue"])
bloc(.74, .55, .24, .30, "Nouvelle tête dense\nENTRAÎNÉE (43 classes)", C["green"])
bloc(.46, .10, .52, .28, "Fine-tuning : on dégèle aussi les dernières couches\n"
                         "convolutives avec un learning rate réduit", C["orange"])
ax.annotate("", xy=(.46, .70), xytext=(.42, .70), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C["deep"]))
ax.annotate("", xy=(.74, .70), xytext=(.70, .70), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C["deep"]))
ax.text(.5, .95, "Transfert learning : réutiliser ce qu'un autre a appris",
        ha="center", fontsize=11.5, fontweight="bold", transform=ax.transAxes, color=C["deep"])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
K["imagenet_images"] = "1,2 M"
K["imagenet_classes"] = 1000
save(fig, "i05_transfert.png")

# ==================================================== 6. learning rate
epoques = np.arange(1, 61)
fig, ax = plt.subplots(figsize=(8.2, 3.5))
for lr, coul, lbl in [(.001, C["blue"], "trop petit (0,001)"),
                      (.05, C["green"], "adapté (0,05)"),
                      (.9, C["red"], "trop grand (0,9)")]:
    if lr == .001:
        y = .5 * np.exp(-epoques / 90) + .06
    elif lr == .05:
        y = .5 * np.exp(-epoques / 12) + .02
    else:
        y = .25 + .22 * np.sin(epoques / 1.7) * np.exp(-epoques / 200) + .12
    ax.plot(epoques, np.clip(y, 0, 1), lw=2, color=coul, label=lbl)
ax.set_xlabel("époque"); ax.set_ylabel("perte d'entraînement")
ax.set_title("L'effet du pas d'apprentissage (learning rate)")
ax.legend(fontsize=9)
K["lr_cas"] = [0.001, 0.05, 0.9]
save(fig, "i06_learning_rate.png")

with open(os.path.join(FIGDIR, "key_numbers_ch4.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch4.json -- {len(K)} cles")
