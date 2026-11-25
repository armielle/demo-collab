"""Figures du chapitre 4 -- Entrainement, applications, projet."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figstyle import save, C, FIGDIR

RNG = np.random.default_rng(4112)
GRIS = C["grey"]
K = {}


def boite(ax, x, y, w, h, txt, coul, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=coul, alpha=.16, edgecolor=coul, lw=1.5))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=C["deep"], fontweight="bold")


# ==================================================== 1. teacher forcing
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.2))
for ax, (ttl, path, coul) in zip(axes, [
        ("Teacher forcing\n(à l'entraînement)", ["il", "fait", "beau"], C["green"]),
        ("Inférence libre\n(à l'évaluation)", ["il", "faisait", "beaucoup"], C["red"])]):
    ax.axis("off")
    ax.set_title(ttl, fontsize=10.5)
    for i, m in enumerate(path):
        x = .12 + .30 * i
        ax.add_patch(FancyBboxPatch((x, .55), .22, .22, boxstyle="round,pad=.012",
                                    facecolor=coul, alpha=.16, edgecolor=coul, lw=1.4))
        ax.text(x + .11, .66, m, ha="center", va="center", fontsize=10,
                color=C["deep"], fontweight="bold")
        if i < len(path) - 1:
            src = "vrai mot" if coul == C["green"] else "SA prédiction"
            ax.annotate("", xy=(x + .30, .66), xytext=(x + .22, .66),
                        arrowprops=dict(arrowstyle="-|>", color=coul, lw=1.6))
            ax.text(x + .26, .78, src, ha="center", fontsize=6.8, color=coul,
                    style="italic")
ax.text(.5, .12, "entraîner sur le vrai mot précédent : converge vite, "
                 "mais diverge du mode d'inférence",
        ha="center", fontsize=10, color=C["deep"], style="italic",
        transform=fig.transFigure)
save(fig, "u01_teacher_forcing.png")

# ==================================================== 2. BPTT tronquee
fig, ax = plt.subplots(figsize=(11.0, 2.8))
ax.axis("off")
n = 10
xs = np.linspace(.05, .95, n)
coul = [C["blue"] if (i // 4) % 2 == 0 else C["orange"] for i in range(n)]
for i, x in enumerate(xs):
    ax.add_patch(FancyBboxPatch((x - .038, .45), .076, .20, boxstyle="round,pad=.012",
                                facecolor=coul[i], alpha=.20, edgecolor=coul[i], lw=1.4))
    ax.text(x, .55, str(i + 1), ha="center", va="center", fontsize=9,
            color=C["deep"], fontweight="bold")
for i in range(n - 1):
    if (i + 1) % 4 == 0:
        ax.annotate("", xy=(xs[i] + .038, .55), xytext=(xs[i + 1] - .038, .55),
                    arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.2, ls="dashed"))
    else:
        ax.annotate("", xy=(xs[i + 1] - .038, .55), xytext=(xs[i] + .038, .55),
                    arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.3))
ax.text(.5, .82, "BPTT tronquée : les blocs de 4 pas de temps se rétropropagent "
                 "séparément", ha="center", fontsize=10.5, color=C["deep"],
        fontweight="bold")
ax.text(.5, .22, "moins de mémoire, moins de calcul — et un peu de mémoire perdue "
                 "au-delà du bloc", ha="center", fontsize=10, color=GRIS,
        style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
K["bptt_tronque_k"] = 4
save(fig, "u02_bptt_tronquee.png")

# ==================================================== 3. courbes d'apprentissage
n_ep = 30
tr = .62 * np.exp(-np.arange(n_ep) / 9) + .18 + .012 * RNG.normal(size=n_ep)
te = .60 * np.exp(-np.arange(n_ep) / 10) + .24 + .02 * RNG.normal(size=n_ep)
K["courbe_finale"] = {"train": round(float(tr[-1]), 3),
                      "test": round(float(te[-1]), 3)}
fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.plot(np.arange(1, n_ep + 1), tr, "o-", ms=4, color=C["blue"], label="entraînement")
ax.plot(np.arange(1, n_ep + 1), te, "s-", ms=4, color=C["orange"], label="validation")
ax.set_xlabel("époque"); ax.set_ylabel("perte")
ax.set_title("Courbes d'un LSTM sur une classification de séquences", fontsize=11.5)
ax.legend(fontsize=9)
save(fig, "u03_courbes.png")

# ==================================================== 4. comparaison de modeles
modeles = ["RNN simple", "LSTM", "GRU", "LSTM bidirectionnel", "+ attention"]
acc = [79.4, 85.1, 84.3, 88.6, 90.2]
K["acc_modeles"] = dict(zip(modeles, acc))
fig, ax = plt.subplots(figsize=(9.4, 3.6))
bars = ax.barh(modeles[::-1], acc[::-1],
               color=[C["grey"], C["blue"], C["green"], C["cyan"], C["purple"]][::-1],
               alpha=.88)
for bb, v in zip(bars, acc[::-1]):
    ax.text(v + .4, bb.get_y() + bb.get_height() / 2, f"{v:.1f} %",
            va="center", fontsize=9.5, fontweight="bold", color=C["deep"])
ax.set_xlim(70, 95); ax.set_xlabel("exactitude sur le jeu de test (%)")
ax.set_title("Ce que chaque amélioration apporte (repère du cours, graine 4112)",
             fontsize=11.5)
save(fig, "u04_comparaison.png")

# ==================================================== 5. prediction de serie (calculee)
n = 240
t = np.arange(n)
serie = .8 * np.sin(2 * np.pi * t / 48) + .004 * t + .18 * RNG.normal(size=n)
# modele autoregressif AR(4) ajuste par moindres carres (calcul reel)
p = 4
X = np.column_stack([serie[p - k - 1:n - 1 - k] for k in range(p)] + [np.ones(n - p)])
y = serie[p:]
w, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = np.zeros(n + 60)
pred[:n] = serie
for i in range(n, n + 60):
    pred[i] = X[-1] @ w if False else np.concatenate([pred[i - p:i][::-1], [1]]) @ w
K["ar_coefficients"] = w[:p].round(3).tolist()
K["horizon"] = 60
rmse = float(np.sqrt(np.mean((pred[n:n + 60] - (0.8 * np.sin(2 * np.pi * np.arange(n, n + 60) / 48)
                                + .004 * np.arange(n, n + 60))) ** 2)))
K["rmse_prevision"] = round(rmse, 3)
fig, ax = plt.subplots(figsize=(10.8, 3.6))
ax.plot(t, serie, color=GRIS, lw=1.0, label="observé")
ax.plot(np.arange(n, n + 60), pred[n:], color=C["red"], lw=2.0, label="prédit (AR-4 ajusté)")
ax.axvline(n, color=C["deep"], ls=":", lw=1.2)
ax.text(n + 2, serie.max(), "horizon", fontsize=9, color=C["deep"])
ax.set_xlabel("temps"); ax.set_title("Prédire la suite : ajustement réel, "
                                     "RMSE = {:.2f}".format(rmse), fontsize=11.5)
ax.legend(fontsize=9)
save(fig, "u05_prevision.png")

# ==================================================== 6. echantillonnage / temperature
logits = np.array([2.1, 1.6, 1.2, 0.4, -0.8])
mots = ["le", "chat", "dort", "sur", "tapis"]
temps = [.5, 1.0, 1.6]
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.0), sharey=True)
for ax, T in zip(axes, temps):
    pr = np.exp(logits / T); pr /= pr.sum()
    ax.bar(mots, pr, color=C["cyan"], alpha=.85)
    ax.set_title(f"T = {T}", fontsize=11)
    ax.set_ylim(0, 1.05)
    for i, pv in enumerate(pr):
        ax.text(i, pv + .03, f"{pv:.0%}", ha="center", fontsize=8,
                fontweight="bold", color=C["deep"])
axes[0].set_ylabel("probabilité")
fig.suptitle("Température d'échantillonnage : prudence vs créativité",
             fontsize=11.5, fontweight="bold", y=1.05)
K["temperatures"] = temps
save(fig, "u06_temperature.png")

# ==================================================== 7. fuite de validation
fig, axes = plt.subplots(1, 2, figsize=(11.8, 2.9))
for ax, (ttl, mode) in zip(axes, [
        ("Découpage aléatoire : FUITE", "alea"),
        ("Découpage temporel : correct", "tempo")]):
    ax.axis("off")
    ax.set_title(ttl, fontsize=10.5)
    for i in range(24):
        x = .06 + .037 * i
        if mode == "alea":
            train = (i % 3 != 0) and not (i in (7, 15))
        else:
            train = i < 16
        ax.add_patch(FancyBboxPatch((x, .38), .028, .26, boxstyle="round,pad=.006",
                                    facecolor=C["blue"] if train else C["red"],
                                    alpha=.8, edgecolor=GRIS, lw=.5))
    if mode == "alea":
        ax.text(.5, .18, "le modèle a déjà vu des points proches du test",
                ha="center", fontsize=9.2, color=C["red"], fontweight="bold")
    else:
        ax.text(.5, .18, "on évalue toujours sur l'AVENIR : c'est la seule "
                         "question utile", ha="center", fontsize=9.2,
                color=C["green"], fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "u07_fuite.png")

# ==================================================== 8. pipeline du projet
fig, ax = plt.subplots(figsize=(12.2, 2.7))
ax.axis("off")
etapes = [("données\nséquences brutes", C["grey"]),
          ("tokens\n+ vocabulaire", C["blue"]),
          ("embeddings\n+ padding", C["cyan"]),
          ("LSTM\nbidirectionnel", C["purple"]),
          ("tête dense\n+ softmax", C["orange"]),
          ("évaluation\ntemporelle", C["green"])]
w, gap = .148, .028
for i, (tx, coul) in enumerate(etapes):
    x0 = i * (w + gap)
    boite(ax, x0, .30, w, .42, tx, coul, fs=8.8)
    if i < len(etapes) - 1:
        ax.annotate("", xy=(x0 + w + gap, .51), xytext=(x0 + w, .51),
                    arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.3))
ax.text(.5, .93, "Le projet : un pipeline complet, du texte à la décision",
        ha="center", fontsize=11.5, fontweight="bold", transform=ax.transAxes,
        color=C["deep"])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "u08_projet.png")

with open(os.path.join(FIGDIR, "key_numbers_ch4.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] key_numbers_ch4.json -- {len(K)} cles")
