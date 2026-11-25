"""Chapitre 2 -- figures issues de calculs REELS (analytique + simulation).

Sorties : figures/g01..g14*.png  +  figures/key_numbers_ch2.json
Toutes les valeurs numeriques citees dans slides/parts/ch2_*.tex proviennent
de ce script ou de fig_ch2b_pymc.py (parties MCMC).

Jeux de donnees reellement utilises :
  * O-rings de la navette Challenger (Commission Rogers, 1986) : 7 incidents / 23 vols
  * Rutherford & Geiger (1910) : comptage de particules alpha, 2609 intervalles de 7,5 s
  * Rubin (1981) / Gelman et al. : 8 ecoles, effets estimes + ecarts-types
  * Mendel (1865) : 556 pois a 4 phenotypes (315, 108, 101, 32)
Contraintes machine : CPU uniquement, threads limites, calculs legers.
"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
import numpy as np
from scipy import stats, optimize
from figstyle import plt, C, save, FIGDIR

rng = np.random.default_rng(4112)
K = {}
t0 = time.time()


def fr(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


# =====================================================================
# g01 : les familles de priors Beta, Gamma, Normale -- avec leurs moments
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1))
tab = []

betas = [("Beta(1, 1)", 1, 1, C["grey"]), ("Beta(2, 2)", 2, 2, C["blue"]),
         ("Beta(5, 2)", 5, 2, C["orange"]), ("Beta(50, 50)", 50, 50, C["red"])]
x = np.linspace(1e-4, 1 - 1e-4, 800)
for lab, a, b, col in betas:
    d = stats.beta(a, b)
    axes[0].plot(x, d.pdf(x), color=col, lw=1.9,
                 label=f"{lab} : moy. {fr(d.mean())}, é.-t. {fr(d.std())}")
    tab.append({"famille": "Beta", "loi": lab, "moyenne": round(float(d.mean()), 4),
                "sd": round(float(d.std()), 4)})
axes[0].set_xlabel(r"$p$ (probabilité)"); axes[0].set_ylabel("densité")
axes[0].set_title("Famille Beta — pour $p \\in [0,1]$", fontsize=10.4)
axes[0].set_xlim(0, 1); axes[0].legend(fontsize=7.2, loc="upper center")

gams = [("Gamma(1, 1)", 1, 1, C["grey"]), ("Gamma(2, 0,5)", 2, .5, C["blue"]),
        ("Gamma(5, 1)", 5, 1, C["orange"]), ("Gamma(25, 5)", 25, 5, C["red"])]
xg = np.linspace(1e-4, 20, 800)
for lab, a, r, col in gams:
    d = stats.gamma(a, scale=1 / r)
    axes[1].plot(xg, d.pdf(xg), color=col, lw=1.9,
                 label=f"{lab} : moy. {fr(d.mean())}, é.-t. {fr(d.std())}")
    tab.append({"famille": "Gamma", "loi": lab, "moyenne": round(float(d.mean()), 4),
                "sd": round(float(d.std()), 4)})
axes[1].set_xlabel(r"$\lambda$ (taux)"); axes[1].set_ylabel("densité")
axes[1].set_title("Famille Gamma — pour $\\lambda > 0$", fontsize=10.4)
axes[1].legend(fontsize=7.2, loc="upper right")

nors = [("N(0, 1)", 0, 1, C["grey"]), ("N(0, 2)", 0, 2, C["blue"]),
        ("N(5, 1)", 5, 1, C["orange"]), ("N(0, 10)", 0, 10, C["red"])]
xn = np.linspace(-25, 25, 800)
for lab, m, s, col in nors:
    d = stats.norm(m, s)
    axes[2].plot(xn, d.pdf(xn), color=col, lw=1.9,
                 label=f"{lab} : é.-t. {fr(d.std())} (masse dans ±2 é.-t. : {fr(2*d.cdf(2*s)-1, 4)})")
    tab.append({"famille": "Normale", "loi": lab, "moyenne": round(float(d.mean()), 4),
                "sd": round(float(d.std()), 4)})
axes[2].set_xlabel(r"$\mu$"); axes[2].set_ylabel("densité")
axes[2].set_title("Famille Normale — pour $\\mu \\in \\mathbb{R}$", fontsize=10.4)
axes[2].legend(fontsize=7.2, loc="upper right")
fig.suptitle("g01 — Trois familles conjuguées et leurs moments (calcul exact par SciPy)",
             fontsize=10.8, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "g01_familles_priors.png")
K["g01_tab"] = tab

# =====================================================================
# g02 : conjugaison Beta-Binomial -- Challenger, 7 incidents sur 23 vols
# =====================================================================
n_ch, k_ch = 23, 7                       # donnees reelles (Commission Rogers, 1986)
a0, b0 = 1, 1                            # prior uniforme
a1, b1 = a0 + k_ch, b0 + n_ch - k_ch     # posterior conjugue
post = stats.beta(a1, b1)
K["g02_data"] = {"n": n_ch, "k": k_ch, "k_obs_ratio": round(k_ch / n_ch, 4)}
K["g02_post_ab"] = [a1, b1]
K["g02_post_mean"] = round(float(post.mean()), 4)
K["g02_post_sd"] = round(float(post.std()), 4)
K["g02_post_ic95"] = [round(float(post.ppf(.025)), 4), round(float(post.ppf(.975)), 4)]
K["g02_post_mode"] = round(float((a1 - 1) / (a1 + b1 - 2)), 4)
K["g02_P_gt_020"] = round(float(post.sf(0.20)), 4)
K["g02_P_gt_050"] = round(float(post.sf(0.50)), 4)
K["g02_mle"] = round(k_ch / n_ch, 4)
# version informative : avis d'expert = taux de defaillance attendu 1 sur 40
ae, be = 1.3, 50.0
post_e = stats.beta(ae + k_ch, be + n_ch - k_ch)
K["g02_expert_prior"] = [round(ae, 2), round(be, 2)]
K["g02_expert_mean"] = round(float(post_e.mean()), 4)
K["g02_expert_ic95"] = [round(float(post_e.ppf(.025)), 4), round(float(post_e.ppf(.975)), 4)]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.3))
gp = np.linspace(1e-4, 1 - 1e-4, 900)
axes[0].fill_between(gp, 0, stats.beta(a0, b0).pdf(gp), color=C["grey"], alpha=.25)
axes[0].plot(gp, stats.beta(a0, b0).pdf(gp), color=C["grey"], lw=1.8, label="a priori Beta(1, 1)")
lik = stats.binom.pmf(k_ch, n_ch, gp)
axes[0].plot(gp, lik / lik.max() * stats.beta(a1, b1).pdf(gp).max() * .95, color=C["orange"],
             lw=1.7, ls="--", label="vraisemblance (recalibrée) : $7/23 = $ " + fr(k_ch / n_ch))
axes[0].fill_between(gp, 0, post.pdf(gp), color=C["blue"], alpha=.28)
axes[0].plot(gp, post.pdf(gp), color=C["blue"], lw=2.2, label=f"a posteriori Beta({a1}, {b1})")
lo, hi = post.ppf(.025), post.ppf(.975)
axes[0].axvline(lo, color=C["deep"], ls=":", lw=1.0)
axes[0].axvline(hi, color=C["deep"], ls=":", lw=1.0)
axes[0].set_xlim(0, 1); axes[0].set_xlabel("$p$ = probabilité d'incident"); axes[0].set_ylabel("densité")
axes[0].set_title("Beta-Binomial : le prior s'ajoute aux données", fontsize=10.2)
axes[0].legend(fontsize=7.8, loc="upper right")

axes[1].fill_between(gp, 0, post.pdf(gp), where=gp >= .20, color=C["green"], alpha=.35,
                     label=f"$P(p > 0,20 \\,|\\, y) = ${fr(post.sf(.20))}")
axes[1].plot(gp, post.pdf(gp), color=C["blue"], lw=2.0)
for v, lab, col in [(post.ppf(.025), fr(post.ppf(.025)), C["deep"]),
                    (post.mean(), fr(post.mean()), C["red"]),
                    (post.ppf(.975), fr(post.ppf(.975)), C["deep"])]:
    axes[1].axvline(v, color=col, ls="--", lw=1.3)
axes[1].annotate(f"moy. {fr(post.mean())}", (post.mean(), post.pdf(post.mean()) * 1.02),
                 ha="center", fontsize=8.4, color=C["red"], fontweight="bold")
axes[1].plot([], [], " ", label=f"IC 95 % = [{fr(lo)} ; {fr(hi)}]")
axes[1].set_xlim(0, 1); axes[1].set_xlabel("$p$"); axes[1].set_ylabel("densité")
axes[1].set_title("Décision : le risque d'incident dépasse-t-il 20 % ?", fontsize=10.2)
axes[1].legend(fontsize=7.8, loc="upper right")
fig.suptitle("g02 — Conjugaison Beta-Binomial sur les joints O-ring de Challenger (7 incidents / 23 vols, données réelles)",
             fontsize=10.2, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g02_conjugue_beta_binomial.png")

# =====================================================================
# g03 : conjugaison Gamma-Poisson -- radioactivite (Rutherford & Geiger, 1910)
# =====================================================================
part = np.arange(0, 15)
freq = np.array([57, 203, 383, 525, 532, 408, 273, 139, 45, 27, 10, 4, 2, 0, 1])
n_obs = int(freq.sum())
tot_ev = int((part * freq).sum())
lam_hat = tot_ev / n_obs
K["g03_table"] = {"nb_intervalles": n_obs, "nb_particules": tot_ev, "moyenne": round(lam_hat, 4)}
# prior Gamma(1,1) puis mises a jour sequentielles (Poisson conjugue)
steps = [(0, 0), (1, 60), (2, 300), (n_obs, tot_ev)]
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
axes[0].bar(part, freq / n_obs, color=C["blue"], alpha=.65, edgecolor="white",
            label="fréquences observées")
xg2 = np.linspace(0, 15, 400)
axes[0].plot(xg2, stats.poisson.pmf(np.round(xg2), lam_hat), color=C["red"], lw=1.9,
             label=f"Poisson($\\lambda$ = {fr(lam_hat)})")
axes[0].set_xlabel("nombre de particules par intervalle de 7,5 s")
axes[0].set_ylabel("fréquence relative")
axes[0].set_title(f"Données : {n_obs} intervalles, {tot_ev} particules", fontsize=10.2)
axes[0].legend(fontsize=8.0)

xl = np.linspace(1e-4, 10, 800)
for (nn, ss), col in zip(steps, [C["grey"], C["green"], C["orange"], C["blue"]]):
    aa, bb = 1 + ss, 1 + nn
    axes[1].plot(xl, stats.gamma(aa, scale=1 / bb).pdf(xl), color=col, lw=1.9,
                 label=f"n = {nn}, $\\sum y$ = {ss} → Gamma({aa}, {bb})")
axes[1].set_xlabel(r"$\lambda$ (particules / 7,5 s)"); axes[1].set_ylabel("densité")
axes[1].set_title("Gamma-Poisson : mise à jour séquentielle", fontsize=10.2)
axes[1].legend(fontsize=7.4)
post_lam = stats.gamma(1 + tot_ev, scale=1 / (1 + n_obs))
K["g03_post_ab"] = [1 + tot_ev, 1 + n_obs]
K["g03_post_mean"] = round(float(post_lam.mean()), 4)
K["g03_post_ic95"] = [round(float(post_lam.ppf(.025)), 4), round(float(post_lam.ppf(.975)), 4)]
K["g03_post_sd"] = round(float(post_lam.std()), 4)

ns = np.arange(1, n_obs + 1)
cum = np.cumsum(np.repeat(part, freq))
means = np.array([(1 + cum[i]) / (1 + ns[i]) for i in range(n_obs)])
cis = np.array([stats.gamma(1 + cum[i], scale=1 / (1 + ns[i])).ppf([.025, .975])
                for i in range(n_obs)])
axes[2].fill_between(ns, cis[:, 0], cis[:, 1], color=C["blue"], alpha=.18, label="IC 95 %")
axes[2].plot(ns, means, color=C["blue"], lw=1.6, label="moyenne a posteriori")
axes[2].axhline(lam_hat, color=C["red"], ls="--", lw=1.4, label=f"moyenne des données = {fr(lam_hat)}")
axes[2].set_xscale("log"); axes[2].set_xlabel("nombre d'intervalles $n$ (log)")
axes[2].set_ylabel(r"$\lambda$")
axes[2].set_title("Le prior Gamma(1,1) est vite dominé", fontsize=10.2)
axes[2].set_ylim(1.5, 7.5); axes[2].legend(fontsize=8.0)
fig.suptitle("g03 — Conjugaison Gamma-Poisson sur le comptage de particules alpha (données réelles, 1910)",
             fontsize=10.2, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g03_conjugue_gamma_poisson.png")

# =====================================================================
# g04 : conjugaison Normal-Normal -- ecoles du SAT (Rubin, 1981)
# =====================================================================
yA, seA = 28.39, 14.9                    # ecole A, ecart-type suppose connu
mu0, tau0 = 5.0, 10.0                    # prior faiblement informatif (points SAT)
prec = 1 / tau0 ** 2 + 1 / seA ** 2
mu1 = (mu0 / tau0 ** 2 + yA / seA ** 2) / prec
sd1 = float(np.sqrt(1 / prec))
K["g04_data"] = {"y": yA, "sigma": seA, "prior": [mu0, tau0]}
K["g04_post_mean"] = round(float(mu1), 4)
K["g04_post_sd"] = round(sd1, 4)
K["g04_post_ic95"] = [round(float(stats.norm(mu1, sd1).ppf(.025)), 4),
                      round(float(stats.norm(mu1, sd1).ppf(.975)), 4)]
K["g04_shrink"] = round(float(mu1 / yA), 4)      # facteur de contraction vers le prior
K["g04_mle"] = yA

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.3))
xm = np.linspace(-50, 90, 900)
axes[0].plot(xm, stats.norm(mu0, tau0).pdf(xm), color=C["grey"], lw=1.8, label=f"a priori N({fr(mu0, 1)}, {fr(tau0, 1)}²)")
lh = stats.norm(yA, seA).pdf(xm)
axes[0].plot(xm, lh / lh.max() * stats.norm(mu1, sd1).pdf(mu1) * .95, color=C["orange"],
             ls="--", lw=1.7, label=f"vraisemblance ($\\hat y$ = {fr(yA, 2)}, $\\sigma$ = {fr(seA, 1)})")
axes[0].plot(xm, stats.norm(mu1, sd1).pdf(xm), color=C["blue"], lw=2.2,
             label=f"a posteriori N({fr(mu1, 2)}, {fr(sd1, 2)}²)")
axes[0].axvline(yA, color=C["orange"], ls=":", lw=1.1)
axes[0].axvline(mu1, color=C["blue"], ls="--", lw=1.1)
axes[0].set_xlabel(r"$\mu$ (gain moyen en points SAT)"); axes[0].set_ylabel("densité")
axes[0].set_title("Normal-Normal : la précision s'additionne", fontsize=10.2)
axes[0].legend(fontsize=7.6)

taus = np.array([1, 2, 5, 10, 20, 50, 100, 1e4])
mm = np.array([(mu0 / t ** 2 + yA / seA ** 2) / (1 / t ** 2 + 1 / seA ** 2) for t in taus])
axes[1].semilogx(taus, mm, "o-", color=C["blue"], lw=1.8, ms=4.5, label="moyenne a posteriori")
axes[1].axhline(yA, color=C["orange"], ls="--", lw=1.4, label=f"maximum de vraisemblance = {fr(yA, 2)}")
axes[1].axhline(mu0, color=C["grey"], ls=":", lw=1.4, label=f"prior : moyenne {fr(mu0, 1)}")
for t, m in zip(taus, mm):
    axes[1].annotate(fr(m, 2), (t, m), textcoords="offset points", xytext=(2, 8), fontsize=7.0)
K["g04_tau_table"] = [{"tau": float(t), "post_mean": round(float(m), 3)} for t, m in zip(taus, mm)]
axes[1].set_xlabel(r"écart-type du prior $\tau$ (échelle log)")
axes[1].set_ylabel(r"moyenne a posteriori de $\mu$")
axes[1].set_title("Sensibilité à l'échelle du prior", fontsize=10.2)
axes[1].legend(fontsize=7.6)
fig.suptitle("g04 — Conjugaison Normal-Normal : école A du jeu de données des 8 écoles (Rubin, 1981)",
             fontsize=10.2, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g04_conjugue_normal_normal.png")

# =====================================================================
# g05 : conjugaison Dirichlet-Multinomial -- les 556 pois de Mendel (1865)
# =====================================================================
mend = np.array([315, 108, 101, 32])          # lisse-jaune, lisse-vert, ride-jaune, ride-vert
n_mend = int(mend.sum())
alpha0 = np.array([9, 3, 3, 1], dtype=float)  # theorie de Mendel comme a priori
post_d = alpha0 + mend
tab_m = []
for i, nom in enumerate(["lisse-jaune", "lisse-vert", "ridé-jaune", "ridé-vert"]):
    m_post = stats.beta(post_d[i], post_d.sum() - post_d[i])
    th = alpha0[i] / alpha0.sum()
    tab_m.append({"categorie": nom, "obs": int(mend[i]), "prop_obs": round(float(mend[i] / n_mend), 4),
                  "theorie_9_3_3_1": round(float(th), 4), "post_mean": round(float(m_post.mean()), 4),
                  "post_ic95": [round(float(m_post.ppf(.025)), 4), round(float(m_post.ppf(.975)), 4)]})
K["g05_data"] = {"n": n_mend, "counts": [int(v) for v in mend]}
K["g05_alpha_post"] = [round(float(v), 2) for v in post_d]
K["g05_tab"] = tab_m
K["g05_KM"] = round(float(np.sum(mend ** 2) / (n_mend ** 2)), 4)      # concentration empirique

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
w = .27
idx = np.arange(4)
axes[0].bar(idx - w, alpha0 / alpha0.sum(), w, color=C["grey"], alpha=.75, label="théorie 9:3:3:1 (prior)")
axes[0].bar(idx, mend / n_mend, w, color=C["orange"], alpha=.75, label="fréquences observées")
axes[0].bar(idx + w, post_d / post_d.sum(), w, color=C["blue"], alpha=.75, label="moyennes a posteriori")
axes[0].set_xticks(idx)
axes[0].set_xticklabels(["lisse\njaune", "lisse\nvert", "ridé\njaune", "ridé\nvert"], fontsize=8.2)
axes[0].set_ylabel("proportion"); axes[0].set_ylim(0, .75)
axes[0].set_title(f"Dirichlet-Multinomial : {n_mend} pois de Mendel", fontsize=10.2)
axes[0].legend(fontsize=7.6)

for i, col in enumerate([C["blue"], C["green"], C["orange"], C["purple"]]):
    m_post = stats.beta(post_d[i], post_d.sum() - post_d[i])
    xs = np.linspace(max(0, m_post.mean() - 5 * m_post.std()), min(1, m_post.mean() + 5 * m_post.std()), 500)
    axes[1].plot(xs, m_post.pdf(xs), color=col, lw=1.9,
                 label=f"$p_{i+1}$ : moy. {fr(m_post.mean())}, IC [{fr(m_post.ppf(.025))} ; {fr(m_post.ppf(.975))}]")
    axes[1].axvline(alpha0[i] / alpha0.sum(), color=col, ls=":", lw=1.0)
axes[1].set_xlabel("proportions du posterior (marginaux Beta)"); axes[1].set_ylabel("densité")
axes[1].set_title("Le posterior resserre la théorie sans la contredire", fontsize=10.2)
axes[1].legend(fontsize=6.8)
fig.suptitle("g05 — Conjugaison Dirichlet–Multinomial sur les données historiques de Mendel",
             fontsize=10.2, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g05_conjugue_dirichlet_multinomial.png")

# =====================================================================
# g06 : vue d'ensemble des quatre trios conjugues
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.0))
xs = np.linspace(1e-4, 1 - 1e-4, 600)
pairs = [("Beta(1, 1)", stats.beta(1, 1), f"Beta({a1}, {b1})", C["blue"], "Beta-$\\theta$"),
         (f"Gamma(1, 1)", stats.gamma(1, scale=1), f"Gamma({1+tot_ev}, {1+n_obs})", C["orange"], "Gamma-$\\lambda$"),
         ("N(5, 10²)", stats.norm(mu0, tau0), f"N({fr(mu1, 1)}, {fr(sd1, 1)}²)", C["green"], "Normale-$\\mu$")]
for ax, (lp, dp, lq, col, tit) in zip(axes.ravel()[:3], pairs):
    if "Beta" in tit:
        grid = xs
    elif "Gamma" in tit:
        grid = np.linspace(1e-4, 12, 600)
    else:
        grid = np.linspace(-40, 80, 600)
    ax.plot(grid, dp.pdf(grid) / dp.pdf(grid).max(), color=C["grey"], ls="--", lw=1.7, label=f"prior {lp}")
    dq = {"Beta-$\\theta$": stats.beta(a1, b1), "Gamma-$\\lambda$": stats.gamma(1 + tot_ev, scale=1 / (1 + n_obs)),
          "Normale-$\\mu$": stats.norm(mu1, sd1)}[tit]
    ax.fill_between(grid, 0, dq.pdf(grid) / dq.pdf(grid).max(), color=col, alpha=.22)
    ax.plot(grid, dq.pdf(grid) / dq.pdf(grid).max(), color=col, lw=2.0, label=f"posterior {lq}")
    ax.set_title(tit + "  (densités normalisées)", fontsize=9.8)
    ax.set_yticks([]); ax.legend(fontsize=7.4)
ax = axes.ravel()[3]
names = ["lisse-jaune", "lisse-vert", "ridé-jaune", "ridé-vert"]
ax.bar(np.arange(4) - .2, alpha0 / alpha0.sum(), .4, color=C["grey"], alpha=.8, label="prior Dirichlet(9,3,3,1)")
ax.bar(np.arange(4) + .2, post_d / post_d.sum(), .4, color=C["purple"], alpha=.8,
       label=f"posterior Dirichlet({int(post_d[0])}, {int(post_d[1])}, {int(post_d[2])}, {int(post_d[3])})")
ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=7.6)
ax.set_title("Dirichlet-$p$  (moyennes)", fontsize=9.8); ax.legend(fontsize=7.0)
fig.suptitle("g06 — Quatre couples conjugués : le posterior garde la famille du prior",
             fontsize=10.6, fontweight="bold", y=1.0)
fig.tight_layout()
save(fig, "g06_quatre_trios.png")

# =====================================================================
# g08 : prior de Jeffreys -- effet sur petits echantillons et couverture
# =====================================================================
priors_j = [("Beta(1, 1) — uniforme (non informatif naïf)", 1, 1, C["grey"]),
            ("Beta(0,5, 0,5) — Jeffreys", .5, .5, C["blue"]),
            ("Beta(0, 0) — Haldane (impropre)", 1e-3, 1e-3, C["red"]),
            ("Beta(2, 2) — faiblement informatif", 2, 2, C["green"])]
data_small = (0, 3)     # 0 succes sur 3 essais : le cas qui fait mal
grid = np.linspace(1e-3, 1 - 1e-3, 900)
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.1))
for lab, aa, bb, col in priors_j:
    axes[0].plot(grid, stats.beta(aa, bb).pdf(grid), color=col, lw=1.8, label=lab)
axes[0].set_ylim(0, 4); axes[0].set_xlabel("$\\theta$"); axes[0].set_ylabel("densité")
axes[0].set_title("Les priors candidats", fontsize=10.2); axes[0].legend(fontsize=7.0)
kj, nj = data_small
K["g08_jeffreys_small"] = {}
for lab, aa, bb, col in priors_j:
    p = stats.beta(aa + kj, bb + nj - kj)
    axes[1].plot(grid, p.pdf(grid), color=col, lw=1.8,
                 label=f"{lab.split(' — ')[1]} : moy. {fr(p.mean())}")
    K["g08_jeffreys_small"][lab.split(" — ")[1]] = {"mean": round(float(p.mean()), 4),
                                                    "ic95": [round(float(p.ppf(.025)), 4),
                                                             round(float(p.ppf(.975)), 4)]}
axes[1].set_xlabel("$\\theta$"); axes[1].set_ylabel("densité")
axes[1].set_title(f"Posterior après {kj} succès sur {nj} essais", fontsize=10.2)
axes[1].legend(fontsize=7.0)

# couverture frequentiste exacte des IC a 95 % pour n = 10 et n = 100
def couverture(aa, bb, n, pv):
    ks = np.arange(n + 1)
    lo = stats.beta.ppf(.025, aa + ks, bb + n - ks)
    hi = stats.beta.ppf(.975, aa + ks, bb + n - ks)
    inside = ((lo <= pv) & (pv <= hi)).astype(float)
    return float((stats.binom.pmf(ks, n, pv) * inside).sum())


pv = np.linspace(.05, .95, 19)
for (lab, aa, bb, col), ls in zip(priors_j, ["--", "-", ":", "-."]):
    cov10 = [couverture(aa, bb, 10, p) for p in pv]
    axes[2].plot(pv, cov10, ls, color=col, lw=1.7, label=lab.split(" — ")[1])
    K[f"g08_couverture_n10_{lab.split(' — ')[1]}"] = [round(float(min(cov10)), 4), round(float(max(cov10)), 4)]
axes[2].axhline(.95, color=C["deep"], lw=1.2)
axes[2].set_ylim(.6, 1.01); axes[2].set_xlabel("vraie valeur de $p$")
axes[2].set_ylabel("couverture réelle de l'IC 95 %")
axes[2].set_title("$n = 10$ : l'uniforme sous-couvre, Jeffreys non", fontsize=10.2)
axes[2].legend(fontsize=7.0)
fig.suptitle("g08 — A priori de Jeffreys $\\propto \\sqrt{\\det I(\\theta)}$ : le remède aux petits échantillons",
             fontsize=10.4, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g08_jeffreys.png")

# =====================================================================
# g09 : piege du prior uniforme sur un parametre d'echelle
# =====================================================================
nS = 200000
sig_unif = rng.uniform(0, 100, nS)                       # sigma ~ U(0,100)
sig_log = np.exp(rng.uniform(np.log(0.5), np.log(50), nS))  # log-uniforme
sig_hn = np.abs(rng.normal(0, 5, nS))                    # demi-normale faiblement informative
y_unif = rng.normal(0, sig_unif)
K["g09_P_sigma_gt_50"] = round(float((sig_unif > 50).mean()), 4)
K["g09_P_absy_gt_50"] = round(float((np.abs(y_unif) > 50).mean()), 4)
K["g09_P_absy_gt_100"] = round(float((np.abs(y_unif) > 100).mean()), 4)
K["g09_med_sigma_unif"] = round(float(np.median(sig_unif)), 2)
K["g09_P_sigma_gt_50_log"] = round(float((sig_log > 50).mean()), 4)

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.1))
xg3 = np.linspace(1e-3, 120, 800)
axes[0].plot(xg3, stats.uniform(0, 100).pdf(xg3), color=C["red"], lw=2.0, label="$\\sigma \\sim U(0, 100)$")
axes[0].plot(xg3, stats.lognorm(s=1, scale=np.exp(np.log(0.5) * .5 + np.log(50) * .5)).pdf(xg3) / 12,
             color=C["blue"], lw=1.8, label="uniforme sur $\\log\\sigma$ (recalibré)")
axes[0].plot(xg3, stats.halfnorm(scale=5).pdf(xg3), color=C["green"], lw=1.8, label="demi-normale $\\sigma \\sim |N(0,5)|$")
axes[0].set_xlabel("$\\sigma$"); axes[0].set_ylabel("densité du prior")
axes[0].set_title("Priors sur un paramètre d'échelle", fontsize=10.2); axes[0].legend(fontsize=7.2)
axes[0].annotate(f"$P(\\sigma > 50) = ${fr(K['g09_P_sigma_gt_50'])}", (55, .0085),
                 fontsize=8.4, color=C["red"], fontweight="bold")

xs8 = np.linspace(-200, 200, 900)
for s, col, lab in [(stats.uniform(0, 100), C["red"], "$\\sigma \\sim U(0,100)$"),
                    (stats.halfnorm(scale=5), C["green"], "$\\sigma \\sim |N(0,5)|$")]:
    # predictive = melange de normales centrees, densite = moyenne de N(0,sigma) sur le prior
    if s.dist.name == "uniform":
        grid_s = np.linspace(1e-3, 100, 500)
    else:
        grid_s = np.linspace(1e-3, 40, 500)
    w = s.pdf(grid_s)
    w = w / w.sum()
    dens = sum(wi * stats.norm(0, gsi).pdf(xs8) for wi, gsi in zip(w, grid_s))
    axes[1].plot(xs8, dens, color=col, lw=1.9, label=lab)
axes[1].set_xlabel("$y$ (observation prédite)"); axes[1].set_ylabel("densité prédictive a priori")
axes[1].set_title("Conséquence : des observations absurdes sont plausibles", fontsize=10.2)
axes[1].legend(fontsize=7.6)
axes[1].annotate(f"$P(|y| > 50) = ${fr(K['g09_P_absy_gt_50'])}\n$P(|y| > 100) = ${fr(K['g09_P_absy_gt_100'])}",
                 (60, .004), fontsize=8.0, color=C["red"], fontweight="bold")

axes[2].hist(np.log10(sig_unif), bins=60, density=True, color=C["red"], alpha=.5,
             label="$U(0,100)$ vu en $\\log_{10}\\sigma$")
axes[2].hist(np.log10(sig_log), bins=60, density=True, color=C["blue"], alpha=.5,
             label="uniforme en $\\log\\sigma$")
axes[2].set_xlabel("$\\log_{10}\\sigma$"); axes[2].set_ylabel("densité")
axes[2].set_title("« Non informatif » sur $\\sigma$ = informatif sur $\\log\\sigma$", fontsize=10.2)
axes[2].legend(fontsize=7.4)
fig.suptitle("g09 — La subtilité du prior uniforme sur un paramètre d'échelle (200 000 simulations)",
             fontsize=10.4, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g09_uniforme_echelle.png")

# =====================================================================
# g10 : controles predictifs a priori -- Challenger, n = 23
# =====================================================================
ppc = [("Beta(1, 1)", 1, 1, C["grey"]), ("Beta(2, 2)", 2, 2, C["green"]),
       ("Beta(5, 3)", 5, 3, C["orange"]), ("Beta(50, 50)", 50, 50, C["red"])]
NSIM = 20000
ns_tir = np.arange(0, n_ch + 1)
res = []
for lab, aa, bb, col in ppc:
    th = rng.beta(aa, bb, NSIM)
    kk = rng.binomial(n_ch, th)
    res.append((lab, aa, bb, col, kk))
K["g10_data"] = {"n": n_ch, "k_observe": k_ch, "nsim": NSIM}
K["g10_tab"] = [{"prior": lab, "moy_pred": round(float(k.mean()), 3),
                 "q05": float(np.percentile(k, 5)), "q95": float(np.percentile(k, 95)),
                 "P_k_le_7": round(float((k <= k_ch).mean()), 4),
                 "P_thetasup_0_5": round(float((rng.beta(aa, bb, 20000) > .5).mean()), 4)}
                for lab, aa, bb, col, k in res]

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
for lab, aa, bb, col, k in res:
    axes[0].step(ns_tir, np.array([(k == v).mean() for v in ns_tir]), where="mid",
                 color=col, lw=1.7, label=f"{lab} — moy. {fr(k.mean(), 1)}")
axes[0].axvline(k_ch, color=C["deep"], ls="--", lw=1.6)
axes[0].annotate(f"observé : {k_ch}", (k_ch, .085), xytext=(-46, 0), textcoords="offset points",
                 fontsize=8.2, color=C["deep"], fontweight="bold")
axes[0].set_xlabel("nombre d'incidents prédits sur 23 vols")
axes[0].set_ylabel("probabilité prédictive a priori")
axes[0].set_title("Contrôle prédictif a priori — que prédit chaque prior ?", fontsize=10.0)
axes[0].legend(fontsize=7.2)

for lab, aa, bb, col, k in res:
    q = np.percentile(k, [5, 50, 95])
    axes[1].errorbar(q[1], lab, xerr=[[q[1] - q[0]], [q[2] - q[1]]], color=col, capsize=4, lw=2.0)
axes[1].axvline(k_ch, color=C["deep"], ls="--", lw=1.6, label=f"observé = {k_ch} (sur 23)")
axes[1].set_xlabel("intervalle prédictif a priori à 90 %")
axes[1].set_title("Un prior informatif mal choisi exclut la réalité", fontsize=10.0)
axes[1].legend(fontsize=8.0)
fig.suptitle("g10 — Contrôle prédictif a priori : simuler AVANT d'avoir les données (20 000 tirages / prior)",
             fontsize=10.2, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g10_prior_predictive.png")

# =====================================================================
# g12 : sensibilite au prior selon n -- dominante en 1/sqrt(n)
# =====================================================================
p_vrai = 0.35
nmax = 3000
obs = (rng.random(nmax) < p_vrai).astype(int)
ns = np.arange(1, nmax + 1)
cum = np.cumsum(obs)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
rows = []
for lab, aa, bb, col in [("Beta(1, 1)", 1, 1, C["grey"]), ("Beta(2, 2)", 2, 2, C["green"]),
                         ("Beta(5, 3)", 5, 3, C["orange"]), ("Beta(50, 50)", 50, 50, C["red"])]:
    mm = (aa + cum) / (aa + bb + ns)
    axes[0].plot(ns, mm, color=col, lw=1.6, label=lab)
    axes[1].plot(ns, mm - p_vrai, color=col, lw=1.6, label=lab)
    for nn in (10, 100, 1000, 3000):
        rows.append({"prior": lab, "n": nn, "mean": round(float((aa + cum[nn - 1]) / (aa + bb + nn)), 4)})
axes[0].axhline(p_vrai, color=C["deep"], ls="--", lw=1.5, label=f"vérité $p$ = {fr(p_vrai, 2)}")
axes[0].set_xscale("log"); axes[0].set_xlabel("$n$ (log)"); axes[0].set_ylabel("moyenne a posteriori")
axes[0].set_title("Les quatre priors convergent vers la même valeur", fontsize=10.2)
axes[0].legend(fontsize=7.4)
sd_theo = np.sqrt(p_vrai * (1 - p_vrai) / ns)
axes[1].plot(ns, 1.96 * sd_theo, color=C["blue"], lw=1.6, ls=":", label="enveloppe $1{,}96\\sqrt{p(1-p)/n}$")
axes[1].plot(ns, -1.96 * sd_theo, color=C["blue"], lw=1.6, ls=":")
axes[1].axhline(0, color=C["deep"], ls="--", lw=1.3)
axes[1].set_xscale("log"); axes[1].set_xlabel("$n$ (log)"); axes[1].set_ylabel("biais du posterior")
axes[1].set_title("Décalage du prior : décroît en $1/\\sqrt{n}$", fontsize=10.2)
axes[1].legend(fontsize=7.6)
K["g12_table"] = rows
K["g12_p_vrai"] = p_vrai
K["g12_biais_theo_Beta50_50"] = [{"n": int(nn), "biais": round(float(15 / (100 + nn)), 4)} for nn in (10, 100, 1000, 3000, 1400)]
K["g12_n_biais_1pct"] = 1400
K["g12_sd_n10"] = round(float(stats.beta(50 + int(cum[9]), 50 + 10 - int(cum[9])).std()), 4)
K["g12_sd_n1000"] = round(float(stats.beta(50 + int(cum[999]), 50 + 1000 - int(cum[999])).std()), 4)
fig.suptitle("g12 — Robustesse : le prior est dominé par les données, à la vitesse $1/\\sqrt{n}$",
             fontsize=10.4, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g12_sensibilite_n.png")

# =====================================================================
# g14 : power prior -- conflit entre donnees historiques et actuelles
# =====================================================================
s0, n0 = 20, 100        # historique : 20 % de succes sur 100
s1, n1 = 80, 200        # actuel : 40 % de succes sur 200  -> CONFLIT
a0grid = np.linspace(0, 1, 201)
means = (a0grid * s0 + s1 + 1) / (a0grid * n0 + n1 + 2)
K["g14_hist"] = {"s0": s0, "n0": n0, "p0": round(s0 / n0, 3)}
K["g14_courant"] = {"s1": s1, "n1": n1, "p1": round(s1 / n1, 3)}
K["g14_means"] = {f"a0={v:.2f}": round(float(means[np.argmin(np.abs(a0grid - v))]), 4)
                  for v in (0, .25, .5, .75, 1)}
K["g14_ic_a0_1"] = [round(float(v), 4) for v in stats.beta(s0 + s1 + 1, (n0 - s0) + (n1 - s1) + 1).ppf([.025, .975])]
K["g14_ic_a0_0"] = [round(float(v), 4) for v in stats.beta(s1 + 1, n1 - s1 + 1).ppf([.025, .975])]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.3))
axes[0].plot(a0grid, means, color=C["blue"], lw=2.2)
for v, col in [(0, C["grey"]), (.25, C["green"]), (.5, C["orange"]), (1, C["red"])]:
    i = np.argmin(np.abs(a0grid - v))
    axes[0].plot(a0grid[i], means[i], "o", color=col, ms=6)
    axes[0].annotate(f"$a_0$ = {fr(v, 2)} → {fr(means[i])}", (a0grid[i], means[i]),
                     textcoords="offset points", xytext=(6, -12), fontsize=8.0, color=col,
                     fontweight="bold")
axes[0].axhline(s1 / n1, color=C["deep"], ls="--", lw=1.3, label="données seules (40 %)")
axes[0].axhline(s0 / n0, color=C["grey"], ls=":", lw=1.3, label="historique (20 %)")
axes[0].set_xlabel("poids du prior $a_0$ (0 = ignoré, 1 = vraisemblance historique entière)")
axes[0].set_ylabel("moyenne a posteriori")
axes[0].set_title("Power prior : régler le poids du passé", fontsize=10.2)
axes[0].legend(fontsize=8.0)
g = np.linspace(1e-4, 1 - 1e-4, 700)
for v, col, lab in [(0, C["grey"], "$a_0 = 0$ (pas d'historique)"), (.25, C["green"], "$a_0 = 0{,}25$"),
                    (.5, C["orange"], "$a_0 = 0{,}50$"), (1, C["red"], "$a_0 = 1$ (historique entier)")]:
    axes[1].plot(g, stats.beta(v * s0 + s1 + 1, v * (n0 - s0) + (n1 - s1) + 1).pdf(g), color=col,
                 lw=1.9, label=lab)
axes[1].axvline(s1 / n1, color=C["deep"], ls="--", lw=1.2)
axes[1].set_xlabel("$p$"); axes[1].set_ylabel("densité a posteriori")
axes[1].set_title("Le pouvoir du prior est un curseur, pas un dogme", fontsize=10.2)
axes[1].legend(fontsize=7.6)
fig.suptitle("g14 — Power prior : conflit entre un historique (20 %) et les données (40 %)",
             fontsize=10.4, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g14_power_prior.png")

out = os.path.join(FIGDIR, "key_numbers_ch2.json")
json.dump(K, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n=== CH2 (analytique) ===")
print(json.dumps(K, indent=1, ensure_ascii=False))
print(f"total {time.time()-t0:.1f} s")
