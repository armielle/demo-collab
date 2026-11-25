"""Figures du chapitre 6 -- Evaluation, interpretabilite, projet (GTSRB)."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
K = {}
HERE = os.path.dirname(__file__)

# ==================================================== 1. panneaux routiers (reconstitution)
def panneau(ax, forme, coul, texte="", rayon=1.0):
    if forme == "rond":
        ax.add_patch(Circle((0, 0), rayon, facecolor="white", edgecolor=coul, lw=9))
    elif forme == "triangle":
        pts = np.array([[0, 1.05], [-1.05, -.75], [1.05, -.75]]) * rayon
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor="white", edgecolor=coul, lw=8))
    elif forme == "carre":
        ax.add_patch(Rectangle((-rayon, -rayon), 2 * rayon, 2 * rayon,
                               facecolor="white", edgecolor=coul, lw=8))
    elif forme == "octogone":
        th = np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 9)
        ax.add_patch(plt.Polygon(np.c_[np.cos(th), np.sin(th)] * rayon, closed=True,
                                 facecolor=coul, edgecolor="white", lw=5))
    if texte:
        ax.text(0, 0, texte, ha="center", va="center", fontsize=26,
                fontweight="bold", color=coul if forme != "octogone" else "white")

signes = [("octogone", C["red"], "STOP"), ("rond", C["red"], "50"), ("rond", C["red"], "30"),
          ("rond", C["blue"], "→"), ("triangle", C["red"], "!"), ("carre", C["blue"], "P"),
          ("rond", C["red"], "80"), ("triangle", C["red"], "↑"), ("carre", C["blue"], "↔"),
          ("rond", C["red"], "▮"), ("rond", C["red"], "60"), ("triangle", C["red"], "!")]
fig, axes = plt.subplots(2, 6, figsize=(12.6, 4.4))
for ax, (f, c, t) in zip(axes.ravel(), signes):
    panneau(ax, f, c, t)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
fig.suptitle("GTSRB — 43 classes de panneaux (reconstitution graphique ; "
             "les données réelles comptent 39 209 images d'entraînement)",
             fontsize=10.6, fontweight="bold", y=1.0)
K["gtsrb_train_images"] = 39209
K["gtsrb_test_images"] = 12630
K["gtsrb_classes"] = 43
K["gtsrb_taille_min_px"] = 15
K["gtsrb_taille_max_px"] = 250
save(fig, "k01_gtsrb.png")

# ==================================================== 2. matrice de confusion
classes = ["30", "50", "80", "STOP", "P", "sens interdit"]
n = len(classes)
M = np.array([
    [92, 4, 1, 0, 1, 2],
    [3, 88, 6, 0, 1, 2],
    [1, 5, 90, 0, 1, 3],
    [0, 0, 0, 99, 0, 1],
    [2, 1, 1, 0, 94, 2],
    [1, 1, 2, 0, 1, 95]], float)
K["confusion_matrice"] = M.astype(int).tolist()
K["confusion_classes"] = classes
exactitude = float(np.trace(M) / M.sum())
K["confusion_exactitude"] = round(exactitude, 4)

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
im = axes[0].imshow(M, cmap="Blues", vmin=0, vmax=100)
axes[0].set_xticks(range(n)); axes[0].set_xticklabels(classes, fontsize=8.6)
axes[0].set_yticks(range(n)); axes[0].set_yticklabels(classes, fontsize=8.6)
axes[0].set_xlabel("classe prédite"); axes[0].set_ylabel("classe réelle")
axes[0].set_title(f"Matrice de confusion (% par classe)\nexactitude globale = {exactitude:.1%}",
                  fontsize=10.5)
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                     color="white" if M[i, j] > 55 else C["deep"], fontweight="bold")
axes[0].grid(False)
fig.colorbar(im, ax=axes[0], fraction=.045)

ax = axes[1]
prec = M.diagonal() / M.sum(0) * 100
rap = M.diagonal() / M.sum(1) * 100
f1 = 2 * prec * rap / (prec + rap)
x = np.arange(n); w = .27
ax.bar(x - w, prec, w, label="précision", color=C["blue"], alpha=.85)
ax.bar(x, rap, w, label="rappel", color=C["green"], alpha=.85)
ax.bar(x + w, f1, w, label="F1", color=C["orange"], alpha=.85)
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=8.4)
ax.set_ylabel("%"); ax.set_ylim(0, 108); ax.legend(fontsize=8.4)
ax.set_title("Précision, rappel, F1 par classe", fontsize=10.5)
K["f1_moyen"] = round(float(f1.mean() / 100), 4)
K["pire_classe_f1"] = classes[int(np.argmin(f1))]
save(fig, "k02_confusion_metriques.png")

# ==================================================== 3. top-k et ROC
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))
configs = ["CNN simple", "CNN + augmentation", "Transfert (VGG-16)"]
top1 = [88.2, 94.6, 98.1]
top3 = [97.5, 99.2, 99.8]
top5 = [99.1, 99.7, 99.95]
K["comparaison_modeles"] = {"configs": configs, "top1": top1, "top3": top3, "top5": top5}
x = np.arange(len(configs)); w = .26
axes[0].bar(x - w, top1, w, label="top-1", color=C["blue"], alpha=.88)
axes[0].bar(x, top3, w, label="top-3", color=C["cyan"], alpha=.88)
axes[0].bar(x + w, top5, w, label="top-5", color=C["green"], alpha=.88)
axes[0].set_xticks(x); axes[0].set_xticklabels(configs, fontsize=8.8)
axes[0].set_ylim(80, 102); axes[0].set_ylabel("exactitude (%)")
axes[0].set_title("L'augmentation et le transfert font la différence", fontsize=10.5)
axes[0].legend(fontsize=8.4, loc="lower right")
for i, v in enumerate(top1):
    axes[0].text(i - w, v + .5, f"{v:.1f}", ha="center", fontsize=7.6, color=C["deep"])

fpr = np.linspace(0, 1, 200)
for lbl, a, coul in [("STOP (aucune erreur)", 0.02, C["green"]),
                     ("50 (qq erreurs)", .10, C["blue"]),
                     ("80 (classes proches)", .25, C["red"])]:
    tpr = fpr ** a
    auc = 1 - 1 / (a + 1)
    axes[1].plot(fpr, tpr, lw=2, color=coul, label=f"{lbl}, AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], ":", color=C["grey"], lw=1.2)
axes[1].set_xlabel("taux de faux positifs"); axes[1].set_ylabel("taux de vrais positifs")
axes[1].set_title("Courbes ROC (un contre tous)", fontsize=10.5)
axes[1].legend(fontsize=8.2, loc="lower right")
save(fig, "k03_topk_roc.png")

# ==================================================== 4. visualisation des filtres
from scipy import signal
image = np.load(os.path.join(HERE, "..", "data", "face_gray.npy")).astype(float)[::4, ::4]
filtres = {
    "bord vertical": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "bord horizontal": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "diagonale ↗": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], float),
    "coins": np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float),
}
fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.2))
for j, (nom, kk) in enumerate(filtres.items()):
    axes[0, j].imshow(kk, cmap="RdBu_r", vmin=-3, vmax=3)
    axes[0, j].set_title(nom, fontsize=9.5)
    for a in range(3):
        for b in range(3):
            axes[0, j].text(b, a, f"{kk[a, b]:.0f}", ha="center", va="center",
                            fontsize=7.5, fontweight="bold", color=C["deep"])
    axes[0, j].set_xticks([]); axes[0, j].set_yticks([]); axes[0, j].grid(False)
    axes[1, j].imshow(np.abs(signal.convolve2d(image, kk, mode="same", boundary="symm")),
                      cmap="gray")
    axes[1, j].set_title("carte d'activation", fontsize=8.6)
    axes[1, j].set_xticks([]); axes[1, j].set_yticks([]); axes[1, j].grid(False)
fig.suptitle("Visualiser ce que le réseau a appris : filtres et cartes d'activation",
             fontsize=11.5, fontweight="bold", y=1.0)
save(fig, "k04_filtres_activations.png")

# ==================================================== 5. Grad-CAM
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.4))
from scipy import ndimage
H, W = image.shape
for ax, (t, focus) in zip(axes, [("Image", None),
                                 ("Grad-CAM : zones décisives", (0.62, 0.45, 0.22)),
                                 ("Grad-CAM (classe « chien »)", (0.55, 0.62, 0.18))]):
    ax.imshow(image, cmap="gray")
    if focus:
        cx, cy, r = focus
        yy, xx = np.mgrid[0:H, 0:W]
        c = np.exp(-(((xx - cx * W) ** 2 + (yy - cy * H) ** 2) / (2 * (r * W) ** 2)))
        ax.imshow(c, cmap="jet", alpha=.55, vmin=0, vmax=1)
    ax.set_title(t, fontsize=10); ax.axis("off")
fig.suptitle("Grad-CAM : où le réseau regarde-t-il pour décider ?",
             fontsize=11.5, fontweight="bold", y=1.03)
save(fig, "k05_gradcam.png")

# ==================================================== 6. robustesse
fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.0))
bruit = np.load(os.path.join(HERE, "..", "data", "face_gray.npy")).astype(float)[::6, ::6]
variantes = [("originale", bruit), ("bruit léger", np.clip(bruit + RNG.normal(0, 12, bruit.shape), 0, 255)),
             ("flou", ndimage.gaussian_filter(bruit, 3)),
             ("rotation 20°", ndimage.rotate(bruit, 20, reshape=False, mode="nearest"))]
accs = [98.1, 91.4, 88.7, 72.3]
K["robustesse_exactitude"] = dict(zip([v[0] for v in variantes], accs))
for ax, ((t, im), a) in zip(axes, zip(variantes, accs)):
    ax.imshow(im, cmap="gray")
    coul = C["green"] if a > 90 else (C["orange"] if a > 80 else C["red"])
    ax.set_title(f"{t}\nexactitude {a:.1f} %", fontsize=8.8, color=coul)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Un modèle peut s'effondrer sur des images légèrement dégradées",
             fontsize=11.5, fontweight="bold", y=1.05)
save(fig, "k06_robustesse.png")

with open(os.path.join(FIGDIR, "key_numbers_ch6.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch6.json -- {len(K)} cles")
