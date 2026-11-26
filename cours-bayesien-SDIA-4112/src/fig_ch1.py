"""Chapitre 1 -- figures generees par du code reellement execute.
Sorties : figures/f01..f09*.png|gif + figures/key_numbers_ch1.json
"""
import json, os
import numpy as np
from scipy import stats
from figstyle import plt, C, save, FIGDIR

rng = np.random.default_rng(4112)
K = {}

# =====================================================================
# f01 : loi des grands nombres -- la frequence converge vers p, pas vers
#       la valeur observee. Montre ce que le frequentiste peut/ne peut pas dire.
# =====================================================================
p_true = 0.5
n_max = 4000
x = (rng.random(n_max) < p_true).astype(int)
freq = np.cumsum(x) / np.arange(1, n_max + 1)
ns = np.arange(1, n_max + 1)
band = 1.96 * np.sqrt(p_true * (1 - p_true) / ns)
fig, ax = plt.subplots(figsize=(7.4, 3.5))
ax.fill_between(ns, p_true - band, p_true + band, color=C["blue"], alpha=.14,
                label="zone à 95 % sous $H_0$ (\\,$p=0{,}5$\\,)")
ax.plot(ns, freq, color=C["blue"], lw=1.5, label="fréquence observée $\\hat p_n$")
ax.axhline(p_true, color=C["red"], ls="--", lw=1.4, label="$p$ vrai = 0,5")
ax.axhline(0.7, color=C["orange"], ls=":", lw=1.4, label="0,7 = estimation après 10 lancers")
for n in (10, 100, 1000, 4000):
    ax.plot(n, freq[n - 1], "o", color=C["deep"], ms=4.5)
    ax.annotate(f"{freq[n-1]:.3f}", (n, freq[n - 1]), textcoords="offset points",
                xytext=(6, -12), fontsize=8.5, color=C["deep"])
ax.set_xscale("log")
ax.set_xlabel("nombre de lancers $n$ (échelle log)")
ax.set_ylabel("proportion de faces")
ax.set_title("f01 — Convergence de la fréquence empirique (simulation, $p=0{,}5$)")
ax.set_ylim(0.15, 0.95)
ax.legend(loc="upper right", ncol=2)
save(fig, "f01_convergence_frequence.png")
K["f01_freq_n10"], K["f01_freq_n4000"] = float(freq[9]), float(freq[-1])
K["f01_freq_n100"] = float(freq[99])

# =====================================================================
# f02 : mise a jour bayesienne sequentielle Beta-Binomial (figure statique)
# =====================================================================
obs = (rng.random(200) < 0.65).astype(int)          # ~65 % de faces
steps = [0, 1, 2, 5, 20, 200]
fig, axes = plt.subplots(2, 3, figsize=(9.6, 4.4), sharex=True, sharey=True)
th = np.linspace(0, 1, 500)
for ax, n in zip(axes.ravel(), steps):
    a, b = 1 + obs[:n].sum(), 1 + (n - obs[:n].sum())
    ax.fill_between(th, 0, stats.beta.pdf(th, a, b), color=C["blue"], alpha=.28)
    ax.plot(th, stats.beta.pdf(th, a, b), color=C["blue"], lw=1.8)
    m = a / (a + b)
    ax.axvline(m, color=C["red"], lw=1.2, ls="--")
    ax.set_title(f"n = {n}   →   Beta({a}, {b})", fontsize=9.5)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
for ax in axes[1]:
    ax.set_xlabel("$\\theta$")
axes[0, 0].set_ylabel("densité a posteriori")
fig.suptitle("f02 — Le posterior d'aujourd'hui est le prior de demain (mise à jour séquentielle)",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "f02_beta_updating.png")

# =====================================================================
# f03 : piece truquee -- prior, vraisemblance, posterior, IC
# =====================================================================
a0, b0, n, k = 1, 1, 10, 7
a1, b1 = a0 + k, b0 + n - k
post = stats.beta(a1, b1)
mean, mode = a1 / (a1 + b1), (a1 - 1) / (a1 + b1 - 2)
lo, hi = post.ppf([.025, .975])
grid = np.linspace(0, 1, 600)
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.fill_between(grid, 0, post.pdf(grid), where=(grid >= lo) & (grid <= hi),
                color=C["blue"], alpha=.20, label=f"IC 95 % = [{lo:.3f}, {hi:.3f}]")
ax.plot(grid, post.pdf(grid), color=C["blue"], lw=2.2, label=f"posterior Beta({a1}, {b1})")
ax.plot(grid, stats.beta.pdf(grid, a0, b0), color=C["grey"], lw=1.4, ls=":",
        label=f"prior Beta({a0}, {b0})")
ax.plot(grid, stats.beta.pdf(grid, k + 1, n - k + 1) / 2.2, color=C["orange"], lw=1.4, ls="--",
        label="vraisemblance (recalibrée)")
for v, lab, col in [(mean, f"moyenne = {mean:.3f}", C["red"]),
                    (0.7, "$\\hat p_{MLE}$ = 0,700", C["orange"])]:
    ax.axvline(v, color=col, lw=1.3, ls="--")
    ax.annotate(lab, (v, ax.get_ylim()[1] * .93), color=col, fontsize=8.6,
                rotation=90, ha="right", va="top", fontweight="bold")
ax.set_xlabel("$\\theta$ (probabilité de face)")
ax.set_ylabel("densité")
ax.set_title("f03 — Prior + vraisemblance → posterior : 7 faces sur 10 lancers")
ax.legend(loc="upper left", fontsize=8.4)
save(fig, "f03_piece_posterior.png")
K["f03_post_mean"], K["f03_post_mode"] = float(mean), float(mode)
K["f03_hdi95"] = [float(lo), float(hi)]
K["f03_prob_gt_half"] = float(1 - post.cdf(0.5))

# =====================================================================
# f04 : mise a jour sequentielle -- GIF anime
# =====================================================================
import matplotlib.animation as anim
fig, ax = plt.subplots(figsize=(6.0, 3.3))
seq_n = list(range(0, 61)) + list(range(65, 205, 5))
def draw(i):
    ax.clear()
    n = seq_n[i]
    a, b = 1 + obs[:n].sum(), 1 + (n - obs[:n].sum())
    ax.fill_between(th, 0, stats.beta.pdf(th, a, b), color=C["blue"], alpha=.30)
    ax.plot(th, stats.beta.pdf(th, a, b), color=C["blue"], lw=2.0)
    ax.axvline(0.65, color=C["red"], ls="--", lw=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 9.5); ax.set_yticks([])
    ax.set_xlabel("$\\theta$")
    ax.set_title(f"Posterior après n = {n} observations   →   Beta({a}, {b})",
                 fontsize=10, fontweight="bold")
    ax.text(.02, .9, f"moyenne = {a/(a+b):.3f}", transform=ax.transAxes,
            fontsize=9, color=C["deep"])
anim.ArtistAnimation
an = anim.FuncAnimation(fig, draw, frames=len(seq_n), interval=70)
path = os.path.join(FIGDIR, "f04_mise_a_jour.gif")
an.save(path, writer=anim.PillowWriter(fps=14))
plt.close(fig)
print(f"[fig] f04_mise_a_jour.gif                   {os.path.getsize(path)/1024:7.1f} Ko")

# =====================================================================
# f05 : sondage electoral
# =====================================================================
ns_, ks_ = 600, 324
a2, b2 = 1 + ks_, 1 + ns_ - ks_
post2 = stats.beta(a2, b2)
lo2, hi2 = post2.ppf([.025, .975])
p_gt = 1 - post2.cdf(0.5)
grid2 = np.linspace(.44, .65, 600)
fig, ax = plt.subplots(figsize=(7.6, 3.4))
ax.fill_between(grid2, 0, post2.pdf(grid2), where=(grid2 >= lo2) & (grid2 <= hi2),
                color=C["cyan"], alpha=.25, label=f"crédibilité 95 % = [{lo2:.3f} ; {hi2:.3f}]")
ax.plot(grid2, post2.pdf(grid2), color=C["cyan"], lw=2.4, label=f"posterior Beta({a2}, {b2})")
ax.axvline(0.5, color=C["red"], lw=1.4, ls="--")
ax.annotate("seuil 50 %", (0.5, ax.get_ylim()[1]), xytext=(6, -4),
            textcoords="offset points", color=C["red"], fontsize=9, fontweight="bold", va="top")
ax.axvline(ks_ / ns_, color=C["orange"], lw=1.2, ls=":")
ax.set_xlabel("$p$ (intention de vote pour A)")
ax.set_ylabel("densité")
ax.set_title("f05 — Sondage : 324 intentions sur 600  →  "
             f"$P(p > 0,5 \\,|\\, y) = {p_gt:.4f}$")
ax.legend(loc="upper left", fontsize=8.6)
save(fig, "f05_sondage_posterior.png")
K["f05_n"], K["f05_k"], K["f05_ci95"] = ns_, ks_, [float(lo2), float(hi2)]
K["f05_p_gt_half"] = float(p_gt)
K["f05_mean"] = float(a2 / (a2 + b2))

# =====================================================================
# f06 : spam -- mise a jour et effet de repetitions
# =====================================================================
P_spam, p_g_spam, p_g_ham = 0.20, 0.60, 0.05
def post_spam(times=1):
    lr = (p_g_spam / p_g_ham) ** times
    o0 = P_spam / (1 - P_spam)
    return o0 * lr / (1 + o0 * lr)
vals = [post_spam(t) for t in range(0, 6)]
fig, ax = plt.subplots(figsize=(6.9, 3.3))
bars = ax.bar(range(6), vals, color=[C["grey"]] + [C["orange"]] * 2 + [C["red"]] * 3,
              width=.62, edgecolor="white")
for i, v in enumerate(vals):
    ax.annotate(f"{v:.1%}", (i, v), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=9.5, fontweight="bold", color=C["deep"])
ax.axhline(P_spam, color=C["grey"], ls=":", lw=1.3)
ax.annotate("prior 20 %", (5.35, P_spam), fontsize=8.5, color=C["grey"], va="center")
ax.set_xticks(range(6)); ax.set_xticklabels([f"{t}×" for t in range(6)])
ax.set_xlabel("nombre d'occurrences du mot « gain »")
ax.set_ylabel("$P(\\mathrm{spam}\\,|\, \\mathrm{donn\\acute ees})$")
ax.set_title("f06 — Détection de spam : chaque indice multiplie les cotes par 12")
ax.set_ylim(0, 1.05)
save(fig, "f06_spam.png")
K["f06_spam"] = [float(v) for v in vals]
K["f06_lr"] = float(p_g_spam / p_g_ham)

# =====================================================================
# f07 : test COVID -- P(malade|+) en fonction de la prevalence  (LE slide choc)
# =====================================================================
sens, spec = 0.95, 0.98
prev = np.linspace(0.0005, 0.5, 500)
ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.plot(prev * 100, ppv * 100, color=C["blue"], lw=2.4)
ax.fill_between(prev * 100, 0, ppv * 100, color=C["blue"], alpha=.10)
pts = [0.001, 0.005, 0.01, 0.05, 0.10, 0.30]
for pv in pts:
    v = sens * pv / (sens * pv + (1 - spec) * (1 - pv))
    ax.plot(pv * 100, v * 100, "o", color=C["red"], ms=5.5, zorder=5)
    ax.annotate(f"{v:.1%}", (pv * 100, v * 100), textcoords="offset points",
                xytext=(7, -3), fontsize=9, fontweight="bold", color=C["red"])
    K[f"f07_ppv_prev_{pv}"] = float(v)
v5 = sens * 0.05 / (sens * 0.05 + (1 - spec) * 0.95)
ax.axvline(5, color=C["orange"], ls="--", lw=1.3)
ax.annotate("prévalence 5 %", (5, 4), xytext=(8, 0), textcoords="offset points",
            color=C["orange"], fontsize=9, fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("prévalence de la maladie (%, échelle log)")
ax.set_ylabel("$P(\\mathrm{malade}\\,|\, \\mathrm{test}^+)$  (%)")
ax.set_title("f07 — Sensibilité 95 %, spécificité 98 % : le même test, deux vérités")
ax.set_ylim(0, 100)
save(fig, "f07_covid_ppv.png")

# =====================================================================
# f08 : p-value vs proba a posteriori (calibration de Jeffreys/Sellke)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
pvals = np.linspace(0.001, 0.999, 400)
z = stats.norm.ppf(1 - pvals / 2)
bf01 = -1 / (np.e * pvals * np.log(pvals))          # BF approx (Sellke et al. 2001)
prior_odds = np.array([1, 1 / 3.0, 1 / 10.0])       # H0 prior : 0.5, 0.25, 0.091
for po, col, lab in zip(prior_odds, [C["green"], C["orange"], C["red"]],
                        ["prior 50 % (uniforme)", "prior 25 %", "prior 10 %"]):
    p0 = po / (1 + po)
    post0 = 1 / (1 + (1 - p0) / p0 * (1 / bf01))
    axes[0].plot(pvals, post0, color=col, lw=2.0, label=lab)
axes[0].axvline(0.05, color=C["grey"], ls="--", lw=1.2)
axes[0].annotate("seuil magique 0,05", (0.05, 0.98), xytext=(8, -6),
                 textcoords="offset points", fontsize=8.6, rotation=90,
                 color=C["grey"], va="top", fontweight="bold")
axes[0].set_xlabel("p-value observée")
axes[0].set_ylabel("$P(H_0\\,|\, \\mathrm{donn\\acute ees})$")
axes[0].set_title("Même p-value, verdicts opposés")
axes[0].legend(loc="lower right", fontsize=8.4)
axes[0].set_ylim(0, 1)

nsim = 20000
pvals_top = np.array([stats.ttest_1samp(rng.normal(0, 1, 12), 0).pvalue for _ in range(3000)])
axes[1].hist(pvals_top, bins=30, color=C["blue"], alpha=.75, edgecolor="white")
axes[1].axhline(3000 / 30, color=C["red"], ls="--", lw=1.5)
axes[1].set_xlabel("p-value sous $H_0$ réellement vraie")
axes[1].set_ylabel("effectif")
axes[1].set_title("Sous $H_0$, la p-value est uniforme")
save(fig, "f08_pvalue_vs_posterior.png")
K["f08_post0_p05_unif"] = float(1 / (1 + (1 - .5) / .5 * (1 / (-1 / (np.e * .05 * np.log(.05))))))
K["f08_bf01_at_p05"] = float(-1 / (np.e * .05 * np.log(.05)))

# =====================================================================
# f09 : budget de donnees -- taille d'echantillon pour trancher
# =====================================================================
n_grid = np.arange(10, 3001, 10)
res = {}
for lbl, d in [("effet faible (d = 0,2)", .2), ("effet moyen (d = 0,5)", .5),
               ("effet fort (d = 0,8)", .8)]:
    pw = stats.norm.cdf(d * np.sqrt(n_grid / 4) - 1.96) 
    res[lbl] = pw
fig, ax = plt.subplots(figsize=(7.4, 3.3))
for (lbl, _), col in zip(res.items(), [C["red"], C["orange"], C["green"]]):
    ax.plot(n_grid, res[lbl], lw=2.2, color=col, label=lbl)
ax.axhline(.8, color=C["grey"], ls="--", lw=1.2)
ax.annotate("puissance 80 %", (2900, .82), fontsize=8.6, color=C["grey"], ha="right")
for d, col in [(.2, C["red"]), (.5, C["orange"]), (.8, C["green"])]:
    nneed = (1.96 + 0.8416) ** 2 * 4 / d ** 2
    K[f"f09_n80_d{d}"] = int(np.ceil(nneed))
    ax.plot(nneed, .8, "o", color=col, ms=6)
    ax.annotate(f"n ≈ {int(np.ceil(nneed))}", (nneed, .8), textcoords="offset points",
                xytext=(4, -14), fontsize=9, fontweight="bold", color=col)
ax.set_xlabel("taille d'échantillon par groupe $n$")
ax.set_ylabel("puissance")
ax.set_title("f09 — Ce que le fréquentiste doit planifier d'avance")
ax.legend(loc="lower right")
ax.set_ylim(0, 1.02)
save(fig, "f09_puissance.png")

with open(os.path.join(FIGDIR, "key_numbers_ch1.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print("\n=== CHIFFRES CLES CH1 ===")
print(json.dumps(K, indent=1, ensure_ascii=False))
