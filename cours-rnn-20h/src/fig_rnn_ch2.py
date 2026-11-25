"""Figures du chapitre 2 -- Architecture RNN, propagation, gradients."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
K = {}

# ==================================================== 1. deroulement temporel
fig, ax = plt.subplots(figsize=(12.0, 3.4))
ax.axis("off")
xs = np.linspace(.10, .90, 4)
for i, x in enumerate(xs):
    ax.add_patch(FancyBboxPatch((x - .055, .45), .11, .22,
                                boxstyle="round,pad=.012",
                                facecolor=C["purple"], alpha=.20,
                                edgecolor=C["purple"], lw=1.6))
    ax.text(x, .56, f"RNN", ha="center", va="center", fontsize=10.5,
            color=C["deep"], fontweight="bold")
    ax.text(x, .80, f"x_{i+1}", ha="center", fontsize=11, color=C["blue"],
            fontweight="bold")
    ax.text(x, .38, f"h_{i+1}", ha="center", fontsize=11, color=C["orange"],
            fontweight="bold")
    ax.text(x, .20, f"y_{i+1}", ha="center", fontsize=11, color=C["red"],
            fontweight="bold")
    ax.annotate("", xy=(x, .74), xytext=(x, .66),
                arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=1.2))
    ax.annotate("", xy=(x, .25), xytext=(x, .33),
                arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.2))
    if i < 3:
        ax.annotate("", xy=(xs[i + 1] - .055, .56), xytext=(x + .055, .56),
                    arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.8))
        ax.text((x + xs[i + 1]) / 2, .60, "h transmis", ha="center", fontsize=8,
                color=C["orange"], style="italic")
GRIS = C["grey"]
ax.annotate("", xy=(.10, .07), xytext=(.90, .07),
            arrowprops=dict(arrowstyle="<|-|>", color=GRIS, lw=1.2))
ax.text(.5, .03, "le MÊME jeu de poids (W, U, b) est réutilisé à chaque pas de temps",
        ha="center", fontsize=10.5, color=C["deep"], fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, .95)
save(fig, "s01_deroulement.png")
K["pas_de_temps"] = 4

# ==================================================== 2. la cellule
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.axis("off")
ax.add_patch(FancyBboxPatch((.38, .40), .26, .26, boxstyle="round,pad=.02",
                            facecolor=C["purple"], alpha=.22,
                            edgecolor=C["purple"], lw=1.8))
ax.text(.51, .53, "tanh", ha="center", va="center", fontsize=13,
        color=C["deep"], fontweight="bold")
ax.text(.51, .44, "$W_x x_t + W_h h_{t-1} + b$", ha="center", fontsize=9,
        color=C["deep"])
ax.annotate("", xy=(.38, .60), xytext=(.10, .60),
            arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2))
ax.text(.22, .65, "$x_t$", fontsize=13, color=C["blue"], fontweight="bold")
ax.annotate("", xy=(.38, .46), xytext=(.10, .24),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2))
ax.text(.20, .30, "$h_{t-1}$", fontsize=13, color=C["orange"], fontweight="bold")
ax.annotate("", xy=(.90, .60), xytext=(.64, .60),
            arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=2))
ax.text(.78, .66, "$h_t$", fontsize=13, color=C["red"], fontweight="bold")
# boucle de recurrence
ax.annotate("", xy=(.51, .24), xytext=(.80, .24),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.8,
                            connectionstyle="arc3,rad=-.45"))
ax.text(.68, .13, "boucle", fontsize=9, color=C["orange"], style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(.5, .02, "h_t = tanh(W_x · x_t + W_h · h_(t-1) + b)",
        ha="center", fontsize=12, color=C["deep"], fontweight="bold")
save(fig, "s02_cellule.png")

# ==================================================== 3. evolution de l'etat cache
n = 60
sig = np.sin(2 * np.pi * np.linspace(0, 3, n))
h = np.zeros(n)
for i in range(1, n):
    h[i] = np.tanh(.85 * h[i - 1] + 1.4 * sig[i])
K["etat_cache_amplitude"] = [round(float(h.min()), 3), round(float(h.max()), 3)]
K["w_rec"] = .85
fig, axes = plt.subplots(2, 1, figsize=(10.8, 4.2), sharex=True)
axes[0].plot(sig, color=C["grey"], lw=1.1)
axes[0].set_title("Entrée x_t", fontsize=10.5)
axes[1].plot(h, color=C["orange"], lw=1.6)
axes[1].set_title("État caché h_t : le réseau maintient une mémoire du passé",
                  fontsize=10.5)
axes[1].set_xlabel("pas de temps")
plt.tight_layout()
save(fig, "s03_etat_cache.png")

# ==================================================== 4. BPTT
fig, ax = plt.subplots(figsize=(11.4, 3.0))
ax.axis("off")
xs = np.linspace(.10, .90, 5)
for i, x in enumerate(xs):
    ax.add_patch(FancyBboxPatch((x - .05, .50), .10, .20,
                                boxstyle="round,pad=.012",
                                facecolor=C["purple"], alpha=.20,
                                edgecolor=C["purple"], lw=1.5))
    ax.text(x, .60, f"t={i+1}", ha="center", va="center", fontsize=9.5,
            color=C["deep"], fontweight="bold")
    if i < 4:
        ax.annotate("", xy=(xs[i + 1] - .05, .60), xytext=(x + .05, .60),
                    arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=1.6))
        ax.annotate("", xy=(x + .05, .52), xytext=(xs[i + 1] - .05, .52),
                    arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.6))
ax.text(.5, .78, "propagation avant (bleu)   ·   rétropropagation à travers le temps "
                 "(rouge)", ha="center", fontsize=10.5, color=C["deep"],
        fontweight="bold")
ax.text(.5, .30, "le gradient remonte la chaîne entière : c'est un produit de "
                 "dérivées successives", ha="center", fontsize=10,
        color=C["red"], style="italic")
ax.set_xlim(0, 1); ax.set_ylim(.15, .90)
save(fig, "s04_bptt.png")

# ==================================================== 5. gradients evanescents (calcule)
rayons = [.55, .85, 1.0, 1.15, 1.35]
pas = np.arange(1, 41)
K["rayons_testes"] = rayons
K["norme_t20"] = {str(r): round(float(r ** 20), 6) for r in rayons}
fig, ax = plt.subplots(figsize=(8.6, 4.2))
for r, coul in zip(rayons, [C["red"], C["orange"], C["grey"], C["cyan"], C["purple"]]):
    ax.semilogy(pas, r ** pas, lw=1.9, color=coul,
                label=f"rayon spectral = {r}")
ax.axhline(1, color=C["deep"], ls="--", lw=1.2)
ax.set_xlabel("pas de temps (t)")
ax.set_ylabel("|d h_t / d h_0|  (échelle log)")
ax.set_title("Gradients qui s'évanouissent ou explosent — calcul exact",
             fontsize=11.5)
ax.legend(fontsize=9)
ax.set_ylim(1e-8, 1e8)
save(fig, "s05_gradients.png")

# ==================================================== 6. clipping
avant = np.abs(RNG.normal(0, 4.2, 400)) * RNG.choice([1, 6], p=[.8, .2], size=400)
normes = np.sort(avant)
K["gradient_norme_mediane"] = round(float(np.median(normes)), 2)
K["gradient_norme_p95"] = round(float(np.percentile(normes, 95)), 2)
K["clip_max"] = 5.0
fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.3), sharey=True)
for ax, (d, ttl, coul) in zip(axes, [
        (avant, "Avant clipping : quelques normes explosent", C["red"]),
        (np.clip(avant, None, 5.0), "Après clipping (max = 5)", C["green"])]):
    ax.hist(d, bins=40, color=coul, alpha=.8, density=True)
    ax.set_title(ttl, fontsize=10.5)
    ax.set_xlabel("norme du gradient")
axes[0].set_ylabel("densité")
fig.suptitle("Le clipping borne la norme du gradient : l'explosion devient gérable",
             fontsize=11.5, fontweight="bold", y=1.04)
save(fig, "s06_clipping.png")

# ==================================================== 7. formes d'entree/sortie
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.0))
for ax, (ttl, n_in, n_out, coul) in zip(axes, [
        ("many-to-one\n(sentiment : une décision)", 5, 1, C["blue"]),
        ("one-to-many\n(génération : une amorce)", 1, 5, C["orange"]),
        ("many-to-many\n(alignement, prédiction)", 5, 5, C["green"])]):
    ax.axis("off")
    ax.set_title(ttl, fontsize=10)
    xs = np.linspace(.18, .82, max(n_in, n_out))
    ax.scatter(xs[:n_in], [.72] * n_in, s=70, color=C["blue"], zorder=3)
    ax.scatter(xs[:n_out], [.30] * n_out, s=70, color=coul, zorder=3)
    for i in range(n_in):
        ax.annotate("", xy=(xs[min(i, n_out - 1)], .36), xytext=(xs[i], .66),
                    arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=.8, alpha=.5))
    ax.text(.5, .06, "", ha="center")
ax.text(.5, .02, "une même cellule, plusieurs usages", ha="center", fontsize=9,
        color=C["grey"], style="italic", transform=fig.transFigure)
save(fig, "s07_formes_es.png")

# ==================================================== 8. seq2seq
fig, ax = plt.subplots(figsize=(11.0, 3.2))
ax.axis("off")
enc = np.linspace(.08, .34, 4)
dec = np.linspace(.60, .86, 4)
for i, x in enumerate(enc):
    ax.add_patch(FancyBboxPatch((x - .045, .45), .09, .20, boxstyle="round,pad=.012",
                                facecolor=C["blue"], alpha=.18, edgecolor=C["blue"]))
    ax.text(x, .55, "enc", ha="center", va="center", fontsize=8.5, color=C["deep"])
for i, x in enumerate(dec):
    ax.add_patch(FancyBboxPatch((x - .045, .45), .09, .20, boxstyle="round,pad=.012",
                                facecolor=C["green"], alpha=.18, edgecolor=C["green"]))
    ax.text(x, .55, "dec", ha="center", va="center", fontsize=8.5, color=C["deep"])
ctx = FancyBboxPatch((.43, .48), .14, .14, boxstyle="round,pad=.012",
                     facecolor=C["orange"], alpha=.28, edgecolor=C["orange"], lw=1.6)
ax.add_patch(ctx)
ax.text(.50, .55, "vecteur\nde contexte", ha="center", va="center", fontsize=7.6,
        color=C["deep"], fontweight="bold")
for x in enc:
    ax.annotate("", xy=(.43, .55), xytext=(x + .045, .55),
                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=.9))
for x in dec:
    ax.annotate("", xy=(x - .045, .55), xytext=(.57, .55),
                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=.9))
ax.text(.22, .78, "encodeur : lit la source", ha="center", fontsize=10,
        color=C["blue"], fontweight="bold")
ax.text(.74, .78, "décodeur : produit la cible", ha="center", fontsize=10,
        color=C["green"], fontweight="bold")
ax.text(.5, .20, "traduction, résumé, transcription : tout le seq2seq part de là",
        ha="center", fontsize=10, color=C["deep"], style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "s08_seq2seq.png")

# ==================================================== 9. nombre de parametres
def params_rnn(n_entree, n_cache):
    return n_entree * n_cache + n_cache * n_cache + n_cache

K["params_rnn_10_32"] = params_rnn(10, 32)
K["params_rnn_300_128"] = params_rnn(300, 128)
K["params_dense_300_128"] = 300 * 128 + 128
K["gain_param_sequence"] = round(K["params_dense_300_128"] / K["params_rnn_300_128"], 1)

fig, ax = plt.subplots(figsize=(8.6, 3.6))
configs = [("embedding 10\n→ 32", params_rnn(10, 32)),
           ("embedding 300\n→ 128", params_rnn(300, 128)),
           ("embedding 300\n→ 512", params_rnn(300, 512))]
b = ax.bar([c[0] for c in configs], [c[1] for c in configs],
           color=[C["blue"], C["cyan"], C["purple"]], alpha=.85)
ax.set_ylabel("paramètres")
ax.set_title("Une couche RNN : (E×H + H×H + H) — le même poids à chaque pas",
             fontsize=11)
for bb, (_, v) in zip(b, configs):
    ax.text(bb.get_x() + bb.get_width() / 2, v * 1.05, f"{v:,}".replace(",", " "),
            ha="center", fontsize=9, fontweight="bold", color=C["deep"])
ax.set_ylim(0, max(c[1] for c in configs) * 1.25)
save(fig, "s09_parametres.png")

with open(os.path.join(FIGDIR, "key_numbers_ch2.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch2.json -- {len(K)} cles")
