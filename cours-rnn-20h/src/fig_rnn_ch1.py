"""Figures du chapitre 1 -- Donnees sequentielles."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
K = {}

# ==================================================== 1. types de sequences
t = np.linspace(0, 6, 300)
serie = np.sin(2 * np.pi * t / 2) + .18 * RNG.normal(size=t.size)
ecg = np.zeros(300)
for k in range(4):
    i = 40 + k * 75
    ecg[i:i + 5] = 1.0
    ecg[i + 8:i + 11] = -.35
traj = np.cumsum(RNG.normal(.35, .18, (120, 2)), axis=0)

fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.9))
axes[0].axis("off")
mots = ["Le", "chat", "dort", "sur", "le", "tapis"]
for i, m in enumerate(mots):
    axes[0].add_patch(FancyBboxPatch((.08 + .15 * i, .55), .13, .22,
                                     boxstyle="round,pad=.01",
                                     facecolor=C["blue"], alpha=.16,
                                     edgecolor=C["blue"]))
    axes[0].text(.145 + .15 * i, .66, m, ha="center", va="center", fontsize=7.6,
                 color=C["deep"], fontweight="bold", rotation=0)
axes[0].set_title("Texte : une suite de mots", fontsize=10)
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

axes[1].plot(t, serie, color=C["cyan"], lw=1.4)
axes[1].set_title("Série temporelle", fontsize=10)
axes[1].set_xlabel("temps")

axes[2].plot(ecg, color=C["red"], lw=1.1)
axes[2].set_title("Signal physiologique (ECG)", fontsize=10)
axes[2].set_xlabel("temps")

axes[3].plot(traj[:, 0], traj[:, 1], color=C["purple"], lw=1.3)
axes[3].plot(traj[0, 0], traj[0, 1], "o", color=C["green"], ms=8)
axes[3].plot(traj[-1, 0], traj[-1, 1], "s", color=C["red"], ms=8)
axes[3].set_title("Trajectoire 2D", fontsize=10)
axes[3].set_aspect("equal")
fig.suptitle("Quatre données de nature très différente — une même question : "
             "l'ordre compte", fontsize=11.5, fontweight="bold", y=1.05)
save(fig, "r01_types_sequences.png")

# ==================================================== 2. embeddings
mots_vec = {
    "roi":    (2.4, 3.1), "reine": (2.9, 3.6),
    "homme":  (1.4, 1.7), "femme": (1.9, 2.2),
    "chat":   (-2.2, 2.6), "chien": (-1.7, 2.1),
    "tapis":  (-2.6, -1.4), "table": (-1.9, -2.0),
}
fig, ax = plt.subplots(figsize=(6.8, 4.6))
for m, (x, y) in mots_vec.items():
    ax.scatter(x, y, s=90, color=C["blue"], alpha=.85, zorder=3)
    ax.text(x, y + .22, m, ha="center", fontsize=10, fontweight="bold", color=C["deep"])
ax.annotate("", xy=(2.9, 3.6), xytext=(2.4, 3.1),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2))
ax.annotate("", xy=(1.9, 2.2), xytext=(1.4, 1.7),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2))
ax.text(3.35, 3.25, "roi → reine", fontsize=8.6, color=C["orange"], fontweight="bold")
ax.text(.95, 2.35, "homme → femme", fontsize=8.6, color=C["orange"], fontweight="bold")
ax.text(0, -3.4, "même translation = même relation grammaticale",
        ha="center", fontsize=9, color=C["grey"], style="italic")
ax.axhline(0, color=C["grey"], lw=.5, alpha=.4)
ax.axvline(0, color=C["grey"], lw=.5, alpha=.4)
ax.set_title("Embeddings : chaque mot devient un vecteur", fontsize=11.5)
ax.set_xlabel("dimension 1"); ax.set_ylabel("dimension 2")
ax.set_xlim(-4, 4.6); ax.set_ylim(-4, 4.6)
K["embedding_exemple"] = {m: [x, y] for m, (x, y) in mots_vec.items()}
save(fig, "r02_embeddings.png")

# ==================================================== 3. ordre vs sac de mots
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.1))
for ax, (ph, jud) in zip(axes, [
        ("Le chat mange la souris.", "sens : le chat mange"),
        ("La souris mange le chat.", "sens : le chat est mangé")]):
    ax.axis("off")
    ax.text(.5, .72, ph, ha="center", fontsize=15, fontweight="bold", color=C["deep"])
    ax.text(.5, .40, "sac de mots : {le, chat, mange, la, souris}",
            ha="center", fontsize=11, color=C["grey"])
    ax.text(.5, .18, jud, ha="center", fontsize=12, color=C["red"], fontweight="bold")
fig.suptitle("Mêmes mots, sens opposés : l'ordre porte l'information",
             fontsize=11.5, fontweight="bold", y=1.02)
save(fig, "r03_ordre_compte.png")
K["mots_communs"] = 5

# ==================================================== 4. longueurs variables
fig, ax = plt.subplots(figsize=(10.6, 3.2))
ax.axis("off")
seqs = [3, 7, 5, 9, 4]
for r, n in enumerate(seqs):
    for i in range(9):
        x = .18 + .085 * i
        coul = C["blue"] if i < n else "#DDDDDD"
        op = .85 if i < n else .5
        ax.add_patch(FancyBboxPatch((x, .18 + .15 * r), .07, .10,
                                    boxstyle="round,pad=.008",
                                    facecolor=coul, alpha=op,
                                    edgecolor=C["grey"], lw=.6))
        if i >= n:
            ax.text(x + .035, .23 + .15 * r, "PAD", ha="center", va="center",
                    fontsize=6.4, color=C["grey"])
    ax.text(.09, .23 + .15 * r, f"{n}", ha="center", fontsize=10,
            color=C["deep"], fontweight="bold")
ax.text(.5, .02, "les lots (batchs) imposent une longueur commune : "
                 "on complète par du padding et on masque",
        ha="center", fontsize=9.4, color=C["deep"], style="italic")
ax.set_title("Séquences de longueurs différentes", fontsize=11.5)
K["longueurs"] = seqs
save(fig, "r04_padding.png")

# ==================================================== 5. dependance a distance
fig, ax = plt.subplots(figsize=(11.6, 2.9))
ax.axis("off")
phrase = ["Le", "professeur", "qui", "enseigne", "la", "statistique", "bayésienne",
          "arrive", "demain", "à", "Douala."]
for i, m in enumerate(phrase):
    ax.add_patch(FancyBboxPatch((.03 + .086 * i, .42), .078, .24,
                                boxstyle="round,pad=.008",
                                facecolor=C["blue"] if i in (1, 7) else "white",
                                alpha=.30 if i in (1, 7) else 1,
                                edgecolor=C["grey"], lw=.7))
    ax.text(.069 + .086 * i, .54, m, ha="center", va="center", fontsize=7.8,
            color=C["deep"], fontweight="bold")
ax.annotate("", xy=(.03 + .086 * 7 + .04, .40), xytext=(.03 + .086 * 1 + .04, .40),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2.2,
                            connectionstyle="arc3,rad=.25"))
ax.text(.42, .16, "la forme du verbe dépend d'un mot situé 6 places avant",
        ha="center", fontsize=10, color=C["orange"], fontweight="bold")
ax.set_title("Dépendance à longue distance", fontsize=11.5)
K["distance_dependance"] = 6
save(fig, "r05_dependance_distance.png")

# ==================================================== 6. fenetre glissante (le CNN)
sig = np.sin(2 * np.pi * np.linspace(0, 5, 400)) + .15 * RNG.normal(size=400)
fig, ax = plt.subplots(figsize=(10.8, 3.2))
ax.plot(sig, color=C["grey"], lw=1.0)
for i, (a, b) in enumerate([(50, 130), (150, 230), (250, 330)]):
    ax.axvspan(a, b, color=C["cyan"], alpha=.18)
    ax.annotate(f"fenêtre {i+1}", ((a + b) / 2, 1.35), ha="center",
                fontsize=8.6, color=C["cyan"], fontweight="bold")
ax.annotate("", xy=(330, -1.7), xytext=(50, -1.7),
            arrowprops=dict(arrowstyle="<|-|>", color=C["red"], lw=1.6))
ax.text(190, -1.95, "la dépendance traverse toute la séquence : une fenêtre la coupe",
        ha="center", fontsize=9.6, color=C["red"], fontweight="bold")
ax.set_title("Ce qu'une fenêtre (convolution) voit — et ce qu'elle rate",
             fontsize=11.5)
ax.set_xlabel("temps")
ax.set_ylim(-2.2, 1.6)
save(fig, "r06_fenetre_insuffisante.png")

# ==================================================== 7. pipeline d'une sequence
fig, ax = plt.subplots(figsize=(12.4, 2.7))
ax.axis("off")
etapes = [("texte\n« Le chat dort »", C["grey"]), ("tokens\n5 identifiants", C["blue"]),
          ("embeddings\n5 vecteurs", C["cyan"]), ("modèle\nséquentiel", C["purple"]),
          ("décision", C["red"])]
w, gap = .165, .045
for i, (tx, coul) in enumerate(etapes):
    x0 = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x0, .32), w, .40, boxstyle="round,pad=0.012",
                                facecolor=coul, alpha=.16, edgecolor=coul, lw=1.5))
    ax.text(x0 + w / 2, .52, tx, ha="center", va="center", fontsize=9,
            color=C["deep"], fontweight="bold")
    if i < len(etapes) - 1:
        ax.annotate("", xy=(x0 + w + gap, .52), xytext=(x0 + w, .52),
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.3))
ax.text(.5, .95, "Le trajet d'une phrase dans le réseau", ha="center",
        fontsize=11.5, fontweight="bold", transform=ax.transAxes, color=C["deep"])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "r07_pipeline_sequence.png")

# ==================================================== 8. la coccinelle (fil rouge)
fig, ax = plt.subplots(figsize=(6.6, 4.4))
passe = traj[:80]
futur = traj[80:120]
ax.plot(passe[:, 0], passe[:, 1], color=C["blue"], lw=1.8, label="observé")
ax.plot(futur[:, 0], futur[:, 1], color=C["red"], lw=1.8, ls="--", label="à prédire")
ax.plot(passe[-1, 0], passe[-1, 1], "o", color=C["deep"], ms=9)
ax.annotate("maintenant", passe[-1], xytext=(passe[-1][0] + .8, passe[-1][1] - .9),
            fontsize=9.5, color=C["deep"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["deep"]))
ax.set_title("Fil rouge du module : prédire la trajectoire d'une coccinelle",
             fontsize=11.5)
ax.legend(fontsize=9)
ax.set_xlabel("x"); ax.set_ylabel("y")
K["trajectoire_observe"] = len(passe)
K["trajectoire_predire"] = len(futur)
save(fig, "r08_coccinelle.png")

with open(os.path.join(FIGDIR, "key_numbers_ch1.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch1.json -- {len(K)} cles")
