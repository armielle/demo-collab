"""Figures du chapitre 5 -- CNN au-dela de la classification."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
K = {}
HERE = os.path.dirname(__file__)

# ==================================================== 1. detection d'objets
img = np.load(os.path.join(HERE, "..", "data", "face_gray.npy")).astype(float)[::3, ::3]
H, W = img.shape
K["detection_image"] = [H, W]

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("Image", fontsize=10)
axes[1].imshow(img, cmap="gray"); axes[1].set_title("Détection : boîtes englobantes", fontsize=10)
for (x, y, w, h, lab, sc) in [(.38, .18, .40, .62, "animal 0.97", C["red"]),
                              (.03, .55, .30, .40, "pelage 0.81", C["orange"])]:
    axes[1].add_patch(Rectangle((x * W, y * H), w * W, h * H, fill=False,
                                edgecolor=sc, lw=2.2))
    axes[1].text(x * W, y * H - 6, lab, fontsize=8, color=sc, fontweight="bold")
axes[2].imshow(img, cmap="gray"); axes[2].set_title("Segmentation : un masque par pixel", fontsize=10)
yy, xx = np.mgrid[0:H, 0:W]
masque = ((xx - .58 * W) ** 2 / (.30 * W) ** 2 + (yy - .45 * H) ** 2 / (.36 * H) ** 2) < 1
over = np.zeros((H, W, 4)); over[masque] = [0.07, 0.65, 0.58, .45]
axes[2].imshow(over)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Trois niveaux de compréhension d'une image", fontsize=11.5,
             fontweight="bold", y=1.03)
for ax in axes:
    ax.axis("off")
K["detection_seuil_iou"] = 0.5
save(fig, "j01_detection_segmentation.png")

# ==================================================== 2. trois segmentations
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))
base = img
for ax, (t, desc) in zip(axes, [("Sémantique", "chaque pixel reçoit une classe\n(les 3 voitures = même étiquette)"),
                                ("Par instance", "chaque objet est distingué\n(3 voitures = 3 étiquettes)"),
                                ("Panoptique", "objets + fond, tout est étiqueté")]):
    ax.imshow(base, cmap="gray")
    yy, xx = np.mgrid[0:H, 0:W]
    if t == "Sémantique":
        m = (xx > .35 * W) & (yy > .3 * H)
        ov = np.zeros((H, W, 4)); ov[m] = [0.12, 0.44, 0.92, .45]; ax.imshow(ov)
    elif t == "Par instance":
        for (cx, cy, r) in [(.55, .55, .18), (.75, .62, .15), (.90, .45, .12)]:
            m = ((xx - cx * W) ** 2 / (r * W) ** 2 + (yy - cy * H) ** 2 / (r * H) ** 2) < 1
            ov = np.zeros((H, W, 4)); ov[m] = [RNG.random(), .55, .35, .5]; ax.imshow(ov)
    else:
        m = np.zeros((H, W), bool); m[:int(.18 * H), :] = True
        ov = np.zeros((H, W, 4)); ov[m] = [.55, .55, .55, .5]; ax.imshow(ov)
        m2 = (xx > .35 * W)
        ov = np.zeros((H, W, 4)); ov[m2 & ~m] = [0.12, 0.44, 0.92, .45]; ax.imshow(ov)
    ax.set_title(f"{t}\n", fontsize=10.5)
    ax.text(.5, -.10, desc, ha="center", va="top", fontsize=8.2, color=C["grey"],
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Trois familles de segmentation", fontsize=11.5, fontweight="bold", y=1.05)
save(fig, "j02_trois_segmentations.png")

# ==================================================== 3. une etape vs deux etapes
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.4))
for ax, (t, etapes, coul) in zip(axes, [
        ("En deux étapes (R-CNN, Faster R-CNN)\nprécision forte, plus lent",
         ["propositions\nde régions", "classification\npar région", "affinage\ndes boîtes"], C["blue"]),
        ("En une étape (YOLO, SSD)\nrapide, temps réel",
         ["grille sur\nl'image", "chaque cellule :\nclasse + boîte", "suppression\ndes doublons"], C["green"])]):
    ax.axis("off"); ax.set_title(t, fontsize=10)
    for i, e in enumerate(etapes):
        x = .04 + i * .325
        ax.add_patch(FancyBboxPatch((x, .35), .28, .32, boxstyle="round,pad=0.012",
                                    facecolor=coul, alpha=.17, edgecolor=coul, lw=1.5))
        ax.text(x + .14, .51, e, ha="center", va="center", fontsize=8.4,
                color=C["deep"], fontweight="bold")
        if i < 2:
            ax.annotate("", xy=(x + .325, .51), xytext=(x + .28, .51),
                        arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
K["nms_iou"] = 0.45
save(fig, "j03_une_deux_etapes.png")

# ==================================================== 4. U-Net
fig, ax = plt.subplots(figsize=(11.8, 4.0))
ax.axis("off")
enc = [(.04, .60, .12, .26), (.18, .68, .11, .20), (.31, .74, .10, .15), (.43, .79, .09, .11)]
for (x, y, w, h) in enc:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                facecolor=C["blue"], alpha=.22, edgecolor=C["blue"], lw=1.4))
ax.add_patch(FancyBboxPatch((.55, .80), .07, .09, boxstyle="round,pad=0.008",
                            facecolor=C["purple"], alpha=.28, edgecolor=C["purple"], lw=1.4))
dec = [(.66, .79, .09, .11), (.77, .74, .10, .15), (.88, .68, .11, .20)]
for (x, y, w, h) in dec:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                facecolor=C["green"], alpha=.22, edgecolor=C["green"], lw=1.4))
ax.add_patch(FancyBboxPatch((.55, .30), .44, .22, boxstyle="round,pad=0.01",
                            facecolor=C["cyan"], alpha=.16, edgecolor=C["cyan"], lw=1.5))
ax.text(.77, .41, "carte de segmentation\n(1 classe par pixel)", ha="center", va="center",
        fontsize=9, color=C["deep"], fontweight="bold")
for (x1, y1, x2, y2) in [(.16, .73, .66, .845), (.29, .815, .77, .815), (.41, .845, .88, .785)]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.5,
                                connectionstyle="arc3,rad=-0.15"))
ax.text(.30, .97, "Encodeur\n(réduction)", ha="center", fontsize=9, color=C["blue"], fontweight="bold")
ax.text(.78, .97, "Décodeur\n(agrandissement)", ha="center", fontsize=9, color=C["green"], fontweight="bold")
ax.text(.50, .90, "goulot", ha="center", fontsize=7.5, color=C["purple"], fontweight="bold")
ax.text(.50, .08, "Les connexions sautées (orange) transmettent les détails fins perdus par la réduction",
        ha="center", fontsize=9, color=C["deep"], style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
K["unet_connexions_sautees"] = 3
save(fig, "j04_unet.png")

# ==================================================== 5. CNN 1D : ECG
def ecg(n=1200, bpm=72):
    t = np.linspace(0, 10, n)
    sig = np.zeros(n)
    for k in range(int(10 * bpm / 60)):
        c = k * 60 / bpm
        idx = np.argmin(np.abs(t - c))
        for dt, amp in [(-.05, .08), (0, -0.12), (0.012, 1.0), (0.03, -0.22), (0.16, 0.22)]:
            j = idx + int(dt * n / 10)
            if 0 <= j < n:
                w = max(2, int(.008 * n))
                sig[max(0, j - w):j + w] += amp * np.exp(
                    -((np.arange(max(0, j - w), j + w) - j) ** 2) / (2 * (.004 * n) ** 2))
    return t, sig + RNG.normal(0, .012, n)

t, s = ecg()
K["ecg_frequence_bpm"] = 72
K["ecg_duree_s"] = 10
K["ecg_points"] = 1200

fig, axes = plt.subplots(2, 1, figsize=(11.0, 4.6), sharex=True)
axes[0].plot(t, s, color=C["grey"], lw=.9)
axes[0].set_title("Signal ECG (72 battements/min) — une série temporelle 1D", fontsize=10.5)
axes[0].set_ylabel("amplitude (mV)")
noyau_rythme = np.exp(-np.linspace(-1, 1, 41) ** 2 * 6)
det = np.convolve(np.maximum(s, 0) ** 2, noyau_rythme / noyau_rythme.sum(), mode="same")
seuil = det.mean() + 2.2 * det.std()
pic = (det > seuil) & (np.roll(det, 1) <= seuil)
axes[1].plot(t, det, color=C["blue"], lw=1.4, label="activation du filtre 1D (détecteur de pics)")
axes[1].axhline(seuil, color=C["red"], ls="--", lw=1.3,
                label=f"seuil = moyenne + 2,2 σ")
axes[1].plot(t[pic], det[pic], "v", color=C["orange"], ms=7, label="battements détectés")
axes[1].set_xlabel("temps (s)"); axes[1].legend(fontsize=8.4, loc="upper right")
axes[1].set_title("Sortie du filtre convolutif 1D : les QRS ressortent nettement", fontsize=10.5)
K["ecg_battements_detectes"] = int(pic.sum())
K["ecg_seuil_sigma"] = 2.2
save(fig, "j05_cnn1d_ecg.png")

# ==================================================== 6. CNN 1D : trafic
n = 24 * 7
t2 = np.arange(n)
trafic = (1200 + 500 * np.sin(2 * np.pi * t2 / 24 - 1.4)
          + 300 * np.sin(2 * np.pi * t2 / (24 * 7))
          + RNG.normal(0, 90, n))
fig, ax = plt.subplots(figsize=(11.0, 3.2))
ax.plot(t2[:96], trafic[:96], color=C["grey"], lw=1.1, label="mesure")
ax.plot(t2[:96], np.convolve(trafic, np.ones(6) / 6, mode="same")[:96],
        color=C["cyan"], lw=2, label="prédiction du CNN 1D (moyenne mobile apprise)")
for d in range(0, 96, 24):
    ax.axvline(d, color=C["blue"], ls=":", lw=1, alpha=.6)
    ax.text(d + 1, trafic[:96].max() * .97, f"J{d//24 + 1}", fontsize=7.5, color=C["blue"])
ax.set_xlabel("heure"); ax.set_ylabel("véhicules / heure")
ax.set_title("CNN 1D sur trafic routier : cycles journaliers et hebdomadaires")
ax.legend(fontsize=8.6)
K["trafic_periode_nuit_h"] = 24
K["trafic_horizon_jours"] = 4
save(fig, "j06_cnn1d_trafic.png")

# ==================================================== 7. autres modalites
fig, ax = plt.subplots(figsize=(11.8, 3.0))
ax.axis("off")
cases = [("Séries temporelles 1D", "ECG, trafic, météo, capteurs", C["blue"]),
         ("Audio", "spectrogrammes = images\n(reconnaissance de parole)", C["cyan"]),
         ("Texte", "séquences 1D de mots\n(avant les Transformers)", C["orange"]),
         ("Imagerie 3D", "scanner, IRM (convolution\nvolumétrique)", C["purple"]),
         ("Landmarking", "pose de points anatomiques\n(morphométrie)", C["green"]),
         ("Comptage dense", "foules, cellules\n(cartes de densité)", C["red"])]
for i, (t, d, coul) in enumerate(cases):
    x = .02 + (i % 3) * .33; y = .55 - (i // 3) * .45
    ax.add_patch(FancyBboxPatch((x, y), .30, .38, boxstyle="round,pad=0.012",
                                facecolor=coul, alpha=.14, edgecolor=coul, lw=1.5))
    ax.text(x + .15, y + .26, t, ha="center", fontsize=9.4, color=C["deep"], fontweight="bold")
    ax.text(x + .15, y + .10, d, ha="center", fontsize=8, color=C["grey"])
ax.text(.5, .98, "Le CNN n'est pas réservé aux photos", ha="center", fontsize=11.5,
        fontweight="bold", color=C["deep"], transform=ax.transAxes)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "j07_autres_modalites.png")

with open(os.path.join(FIGDIR, "key_numbers_ch5.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch5.json -- {len(K)} cles")
