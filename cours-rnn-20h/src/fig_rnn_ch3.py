"""Figures du chapitre 3 -- LSTM et GRU."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
GRIS = C["grey"]
K = {}


def boite(ax, x, y, w, h, txt, coul, fs=8.5, alpha=.18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=coul, alpha=alpha, edgecolor=coul, lw=1.5))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=C["deep"], fontweight="bold")


# ==================================================== 1. cellule LSTM (schema)
fig, ax = plt.subplots(figsize=(11.6, 5.4))
ax.axis("off")
# etats
ax.annotate("", xy=(.93, .52), xytext=(.05, .52),
            arrowprops=dict(arrowstyle="-|>", color=C["deep"], lw=2.6))
ax.text(.5, .56, "état de cellule  C_t   (le convoyeur)", ha="center", fontsize=10.5,
        color=C["deep"], fontweight="bold")
ax.annotate("", xy=(.93, .30), xytext=(.05, .30),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2.2))
ax.text(.5, .26, "état caché  h_t", ha="center", fontsize=10.5,
        color=C["orange"], fontweight="bold")
# portes
portes = [(.18, "oubli\nf_t", C["red"]), (.42, "entrée\ni_t", C["green"]),
          (.68, "sortie\no_t", C["blue"])]
for x, t_, coul in portes:
    boite(ax, x - .06, .60, .12, .16, t_, coul, fs=9)
    ax.annotate("", xy=(x, .545), xytext=(x, .60),
                arrowprops=dict(arrowstyle="-|>", color=coul, lw=1.8))
ax.annotate("", xy=(.10, .70), xytext=(.04, .78),
            arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.4))
ax.text(.035, .83, "h_(t-1)", fontsize=10, color=C["orange"], fontweight="bold")
ax.annotate("", xy=(.50, .10), xytext=(.50, .04),
            arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=1.6))
ax.text(.52, .05, "x_t", fontsize=10, color=C["blue"], fontweight="bold")
# operations sur l'etat
ax.text(.30, .47, "×", ha="center", fontsize=15, color=C["red"], fontweight="bold")
ax.text(.54, .47, "+", ha="center", fontsize=16, color=C["green"], fontweight="bold")
boite(ax, .46, .66, .10, .10, "tanh", C["purple"], fs=8.5)
ax.text(.80, .47, "×", ha="center", fontsize=15, color=C["blue"], fontweight="bold")
ax.text(.5, .02, "C_t = f_t ⊙ C_(t-1) + i_t ⊙ g_t        h_t = o_t ⊙ tanh(C_t)",
        ha="center", fontsize=11, color=C["deep"], fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "t01_lstm_cellule.png")

# ==================================================== 2. valeurs des portes
n = 80
t = np.linspace(0, 8, n)
x = np.sin(t) + .2 * RNG.normal(size=n)
f = 1 / (1 + np.exp(-(2.2 * np.ones(n))))
i = 1 / (1 + np.exp(-(3.0 * x)))
o = 1 / (1 + np.exp(-(1.6 * x)))
K["portes_moyennes"] = {"oubli": round(float(f.mean()), 3),
                        "entree": round(float(i.mean()), 3),
                        "sortie": round(float(o.mean()), 3)}
fig, axes = plt.subplots(3, 1, figsize=(10.8, 5.0), sharex=True)
axes[0].plot(x, color=C["grey"], lw=1.0)
axes[0].set_title("Entrée x_t", fontsize=10)
axes[1].plot(i, color=C["green"], lw=1.4, label="porte d'entrée i_t (ouvrir ?)")
axes[1].plot(f, color=C["red"], lw=1.4, label="porte d'oubli f_t (garder ?)")
axes[1].legend(fontsize=8.5, loc="lower left")
axes[1].set_title("Deux portes, deux décisions", fontsize=10)
axes[2].plot(o, color=C["blue"], lw=1.4, label="porte de sortie o_t")
axes[2].legend(fontsize=8.5, loc="lower left")
axes[2].set_title("Combien révéler ?", fontsize=10)
axes[2].set_xlabel("pas de temps")
for a in axes:
    a.set_ylim(-.05, 1.15)
plt.tight_layout()
save(fig, "t02_portes.png")

# ==================================================== 3. convoyeur
fig, ax = plt.subplots(figsize=(11.4, 3.0))
ax.axis("off")
n = 7
xs = np.linspace(.08, .92, n)
ax.annotate("", xy=(.97, .68), xytext=(.03, .68),
            arrowprops=dict(arrowstyle="-|>", color=C["deep"], lw=2.4))
for i, x in enumerate(xs):
    ax.add_patch(Circle((x, .68), .045, facecolor=C["red"], alpha=.55,
                        edgecolor=C["red"], lw=1.2))
    ax.text(x, .68, "×", ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")
ax.text(.5, .80, "f_t proche de 1 : l'information TRAVERSE sans être écrasée",
        ha="center", fontsize=10.5, color=C["deep"], fontweight="bold")
ax.annotate("", xy=(.97, .38), xytext=(.03, .38),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2.2))
for i, x in enumerate(xs):
    v = (0.5 ** (i + 1))
    ax.add_patch(Circle((x, .38), .045, facecolor=C["orange"], alpha=.15 + .75 * v,
                        edgecolor=C["orange"], lw=1.2))
ax.text(.5, .24, "dans un RNN simple, chaque pas écrase : après quelques pas, "
                 "il ne reste presque rien", ha="center", fontsize=10.5,
        color=C["orange"], fontweight="bold")
ax.text(.5, .10, "0,99^n ≈ 1     contre     0,55^n → 0   très vite",
        ha="center", fontsize=11, color=C["deep"], fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, .92)
K["f_proche_un"] = .99
K["w_rnn_simple"] = .55
save(fig, "t03_conveyor.png")

# ==================================================== 4. survival des gradients (calcule)
pas = np.arange(0, 41)
K["survie"] = {"rnn_0.55": [.55 ** k for k in pas],
               "lstm_f0.99": [.99 ** k for k in pas],
               "lstm_f0.999": [.999 ** k for k in pas]}
K["rnn_t30"] = round(float(.55 ** 30), 8)
K["lstm_t30"] = round(float(.99 ** 30), 4)
fig, ax = plt.subplots(figsize=(8.8, 4.2))
for lbl, coul in [("rnn_0.55", C["red"]), ("lstm_f0.99", C["green"]),
                  ("lstm_f0.999", C["cyan"])]:
    ax.semilogy(pas, K["survie"][lbl], lw=2, color=coul,
                label=lbl.replace("_", " = "))
ax.axhline(1e-2, color=GRIS, ls=":", lw=1.2)
ax.text(20, 1.3e-2, "seuil utile", fontsize=8.5, color=GRIS)
ax.set_xlabel("pas de temps"); ax.set_ylabel("fraction du gradient conservée (log)")
ax.set_title("Le gradient à travers le temps : calcul exact", fontsize=11.5)
ax.legend(fontsize=9)
ax.set_ylim(1e-8, 3)
save(fig, "t04_survie_gradient.png")

# ==================================================== 5. GRU
fig, ax = plt.subplots(figsize=(10.4, 4.6))
ax.axis("off")
boite(ax, .38, .44, .24, .22, "candidat\nh~_t = tanh(...)", C["purple"], fs=9)
boite(ax, .10, .60, .16, .14, "update\nz_t", C["green"], fs=9)
boite(ax, .10, .20, .16, .14, "reset\nr_t", C["red"], fs=9)
ax.annotate("", xy=(.50, .44), xytext=(.18, .67),
            arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.6))
ax.annotate("", xy=(.40, .55), xytext=(.26, .27),
            arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.6))
ax.annotate("", xy=(.74, .55), xytext=(.62, .55),
            arrowprops=dict(arrowstyle="-|>", color=C["deep"], lw=2.2))
ax.text(.80, .62, "h_t", fontsize=13, color=C["deep"], fontweight="bold")
ax.annotate("", xy=(.10, .27), xytext=(.80, .22),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.6,
                            connectionstyle="arc3,rad=.25"))
ax.text(.46, .13, "h_(t-1)", fontsize=10, color=C["orange"], fontweight="bold")
ax.text(.5, .02, "h_t = (1 - z_t) ⊙ h_(t-1) + z_t ⊙ h~_t     —  deux portes au lieu de trois",
        ha="center", fontsize=11, color=C["deep"], fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "t05_gru.png")

# ==================================================== 6. parametres RNN / LSTM / GRU
def params(cel, E, H):
    if cel == "RNN":
        return E * H + H * H + H
    n_portes = {"LSTM": 4, "GRU": 3}[cel]
    return n_portes * (E * H + H * H) + n_portes * H

K["params"] = {f"{cel}_{E}_{H}": params(cel, E, H)
               for cel in ["RNN", "LSTM", "GRU"] for E, H in [(64, 128), (300, 256)]}
cfg = [("RNN", C["grey"]), ("GRU", C["green"]), ("LSTM", C["blue"])]
vals = [params(c, 300, 256) for c, _ in cfg]
fig, ax = plt.subplots(figsize=(7.8, 3.6))
b = ax.bar([c for c, _ in cfg], vals, color=[c for _, c in cfg], alpha=.85)
ax.set_ylabel("paramètres (E=300, H=256)")
ax.set_title("Le prix de la mémoire : 3 ou 4 jeux de poids", fontsize=11.5)
for bb, v in zip(b, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v * 1.04, f"{v/1e3:.0f} k",
            ha="center", fontsize=9.5, fontweight="bold", color=C["deep"])
ax.set_ylim(0, max(vals) * 1.22)
save(fig, "t06_parametres.png")

# ==================================================== 7. bidirectionnel
fig, ax = plt.subplots(figsize=(11.2, 3.0))
ax.axis("off")
n = 6
xs = np.linspace(.12, .88, n)
ax.annotate("", xy=(.95, .62), xytext=(.06, .62),
            arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2.2))
ax.annotate("", xy=(.06, .38), xytext=(.95, .38),
            arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=2.2))
for i, x in enumerate(xs):
    ax.add_patch(Circle((x, .62), .05, facecolor=C["blue"], alpha=.35, edgecolor=C["blue"]))
    ax.add_patch(Circle((x, .38), .05, facecolor=C["green"], alpha=.35, edgecolor=C["green"]))
    ax.annotate("", xy=(x, .24), xytext=(x, .30),
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.2))
ax.text(.5, .78, "couche avant : lit de gauche à droite", ha="center", fontsize=10,
        color=C["blue"], fontweight="bold")
ax.text(.5, .14, "couche arrière : lit de droite à gauche", ha="center", fontsize=10,
        color=C["green"], fontweight="bold")
ax.text(.5, .03, "chaque pas voit le passé ET le futur — impossible en temps réel",
        ha="center", fontsize=10, color=C["deep"], style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, .9)
save(fig, "t07_bidirectionnel.png")

# ==================================================== 8. empilement
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.axis("off")
for r, (nom, coul) in enumerate([("couche 2", C["purple"]), ("couche 1", C["blue"])]):
    y = .62 - r * .30
    ax.add_patch(FancyBboxPatch((.18, y), .64, .16, boxstyle="round,pad=.012",
                                facecolor=coul, alpha=.16, edgecolor=coul, lw=1.5))
    ax.text(.5, y + .08, nom + " :  H neurones, séquence de pas de temps",
            ha="center", va="center", fontsize=9.5, color=C["deep"], fontweight="bold")
for i, x in enumerate(np.linspace(.20, .80, 6)):
    ax.annotate("", xy=(x, .62), xytext=(x, .48),
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.2))
ax.text(.5, .90, "Empiler des couches : la couche supérieure voit des suites "
                 "d'états, pas des mots", ha="center", fontsize=10,
        color=C["deep"], fontweight="bold")
ax.text(.5, .12, "entrée x_t", ha="center", fontsize=10, color=C["deep"])
ax.annotate("", xy=(.5, .18), xytext=(.5, .10),
            arrowprops=dict(arrowstyle="-|>", color=C["deep"], lw=1.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "t08_empilement.png")

# ==================================================== 9. attention
mots = ["Le", "chat", "s'est", "assis", "sur", "le", "tapis"]
poids = np.array([.04, .32, .06, .18, .06, .04, .30])
poids = poids / poids.sum()
K["attention_mots"] = mots
K["attention_poids"] = poids.round(3).tolist()
fig, ax = plt.subplots(figsize=(9.6, 2.9))
ax.imshow(poids[None, :], cmap="Blues", aspect="auto", vmin=0, vmax=.35)
ax.set_xticks(range(len(mots)))
ax.set_xticklabels(mots, fontsize=11, fontweight="bold", color=C["deep"])
ax.set_yticks([])
ax.set_title("Poids d'attention : sur quels mots porter la décision ?",
             fontsize=11.5)
for i, p in enumerate(poids):
    ax.text(i, 0, f"{p:.2f}", ha="center", va="center", fontsize=9,
            color="white" if p > .18 else C["deep"], fontweight="bold")
ax.grid(False)
save(fig, "t09_attention.png")

with open(os.path.join(FIGDIR, "key_numbers_ch3.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch3.json -- {len(K)} cles")
