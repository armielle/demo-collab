"""Chapitre 3 -- figures generees par du code reellement execute (partie 1).
Sorties : figures/h01..h06, h08, h10..h12 + figures/key_numbers_ch3.json

Sections couvertes : 3.1 estimateurs ponctuels, 3.2 intervalles,
3.3 prediction a posteriori, 3.4 cas appliques, 3.5 synthese.
Contraintes machine : CPU, calculs legers, threads limites par figstyle.
"""
import json
import os
import numpy as np
from scipy import stats
from scipy.integrate import simpson
from figstyle import plt, C, save, FIGDIR

rng = np.random.default_rng(4112)
K = {}

# =====================================================================
# h01 -- les trois estimateurs ponctuels et la perte que chacun minimise
# =====================================================================
a_p, b_p = 3.0, 7.0                      # posterior Beta(3,7) : nettement asymetrique
post = stats.beta(a_p, b_p)
th = np.linspace(0.002, 0.998, 1500)
dens = post.pdf(th)
mean_p, med_p, mode_p = post.mean(), post.median(), (a_p - 1) / (a_p + b_p - 2)

eps = 0.05
grid = np.linspace(0.02, 0.96, 470)
E2 = lambda v: stats.beta.moment(2, a_p, b_p) - 2 * v * mean_p + v ** 2          # perte quadratique
L2 = np.array([E2(v) for v in grid])
L1 = np.array([simpson(np.abs(th - v) * dens, th) for v in grid])                # perte absolue
L0 = np.array([1 - simpson(np.where(np.abs(th - v) <= eps, dens, 0.0), th) for v in grid])
arg2, arg1, arg0 = grid[L2.argmin()], grid[L1.argmin()], grid[L0.argmin()]

fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.9))
ax = axes[0]
ax.fill_between(th, 0, dens, color=C["blue"], alpha=.22)
ax.plot(th, dens, color=C["blue"], lw=2.0)
for v, lab, col in ((mean_p, f"moyenne {mean_p:.3f}", C["red"]),
                    (med_p, f"médiane {med_p:.3f}", C["orange"]),
                    (mode_p, f"MAP {mode_p:.3f}", C["green"])):
    ax.axvline(v, color=col, lw=1.6, ls="--", label=lab)
ax.set_title("posterior Beta(3, 7) et ses 3 estimateurs", fontsize=9.6)
ax.set_xlabel("$\\theta$")
ax.set_ylabel("densité a posteriori")
ax.legend(fontsize=7.4, loc="upper right")
for ax, L, arg, name, col in ((axes[1], L2, arg2, "perte quadratique $(\\theta-a)^2$", C["red"]),
                              (axes[2], L1, arg1, "perte absolue $|\\theta-a|$", C["orange"]),
                              (axes[3], L0, arg0, "perte 0--1 (tolérance 0,05)", C["green"])):
    Ln = (L - L.min()) / (L.max() - L.min())
    ax.plot(grid, Ln, color=col, lw=2.0)
    ax.axvline(arg, color=C["deep"], ls="--", lw=1.3)
    ax.plot(arg, 0, "o", color=C["deep"], ms=6)
    ax.set_title(name, fontsize=9.6)
    ax.set_xlabel("valeur proposée $a$")
    ax.annotate(f"min en {arg:.3f}", (arg, 0), textcoords="offset points",
                xytext=(6, 22), fontsize=8.4, color=C["deep"], fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
axes[1].set_ylabel("perte attendue (normalisée)")
fig.suptitle("h01 — Chaque perte a son estimateur : quadratique $\\to$ moyenne, absolue $\\to$ médiane, 0--1 $\\to$ MAP",
             fontsize=11, fontweight="bold", y=1.05)
fig.tight_layout()
save(fig, "h01_estimateurs_pertes.png")
K["h01_mean"], K["h01_median"], K["h01_mode"] = round(float(mean_p), 4), round(float(med_p), 4), round(float(mode_p), 4)
K["h01_argmin_L2"], K["h01_argmin_L1"], K["h01_argmin_L0"] = round(float(arg2), 3), round(float(arg1), 3), round(float(arg0), 3)
K["h01_ecart_mode_moyenne"] = round(float(mean_p - mode_p), 4)

# =====================================================================
# h02 -- MAP contre maximum de vraisemblance : le shrinkage en fonction
#        de la force du prior (loi a priori conjuguee Beta)
# =====================================================================
n_obs, k_obs = 20, 14                    # 14 succes sur 20 -> MLE = 0,700
m0, mle = 0.40, k_obs / n_obs            # prior centre sur 0,40 (volontairement trop bas)
strength = np.logspace(np.log10(1.5), np.log10(400.0), 260)
a0, b0 = m0 * strength, (1 - m0) * strength
p_mean = (k_obs + a0) / (n_obs + strength)
p_mode = (k_obs + a0 - 1) / (n_obs + strength - 2)
w_data = n_obs / (n_obs + strength)      # poids accorde aux donnees

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
ax = axes[0]
ax.plot(strength, p_mean, color=C["blue"], lw=2.2, label="moyenne a posteriori (Bayes)")
ax.plot(strength, p_mode, color=C["purple"], lw=2.0, ls="-.", label="MAP (mode a posteriori)")
ax.axhline(mle, color=C["red"], lw=1.8, ls="--", label=f"MLE = {mle:.3f} ($k/n$, prior ignoré)")
ax.axhline(m0, color=C["green"], lw=1.6, ls=":", label=f"moyenne du prior = {m0:.2f}")
for s in (2.0, 10.0, 100.0):
    pm = (k_obs + m0 * s) / (n_obs + s)
    ax.plot(s, pm, "o", color=C["deep"], ms=5)
    ax.annotate(f"n$_0$={s:.0f} : {pm:.3f}", (s, pm), textcoords="offset points",
                xytext=(-4, 10), fontsize=8.0, color=C["deep"], fontweight="bold")
    K[f"h02_mean_n0_{int(s)}"] = round(float(pm), 4)
    K[f"h02_mode_n0_{int(s)}"] = round(float((k_obs + m0 * s - 1) / (n_obs + s - 2)), 4)
ax.set_xscale("log")
ax.set_xlabel("force du prior  $n_0 = \\alpha_0 + \\beta_0$  (échelle log)")
ax.set_ylabel("estimation de $\\theta$")
ax.set_title("Le MAP/MLE est tiré vers le prior : c'est le shrinkage", fontsize=10.2)
ax.legend(fontsize=7.8, loc="center right")

ax = axes[1]
ax.plot(strength, w_data, color=C["blue"], lw=2.2)
ax.fill_between(strength, 0, w_data, color=C["blue"], alpha=.13)
ax.axvline(n_obs, color=C["orange"], ls="--", lw=1.4)
ax.annotate(f"$n_0 = n = {n_obs}$\nprior et données\nde même poids",
            (n_obs, .55), textcoords="offset points", xytext=(10, -6),
            fontsize=8.4, color=C["orange"], fontweight="bold")
for s in (2.0, 10.0, 100.0):
    ax.plot(s, n_obs / (n_obs + s), "o", color=C["deep"], ms=5)
    ax.annotate(f"{n_obs/(n_obs+s):.2f}", (s, n_obs / (n_obs + s)),
                textcoords="offset points", xytext=(-16, 8), fontsize=8.2, color=C["deep"])
    K[f"h02_poids_donnees_n0_{int(s)}"] = round(float(n_obs / (n_obs + s)), 4)
ax.set_xscale("log")
ax.set_xlabel("force du prior  $n_0$")
ax.set_ylabel("poids des données  $n/(n+n_0)$")
ax.set_title("Poids relatif données / prior (posterior Beta)", fontsize=10.2)
ax.set_ylim(0, 1.02)
fig.suptitle("h02 — Même échantillon ($k = 14$, $n = 20$) : plus le prior est fort, plus l'estimation s'éloigne du MLE",
             fontsize=10.8, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h02_shrinkage_prior.png")
K["h02_mle"] = round(float(mle), 4)

# =====================================================================
# h03 -- HDI contre intervalle a queues egales, sur une loi asymetrique
# =====================================================================
def hdi_grid(dist, level=0.95, xmax=None, m=200001):
    """HDI exact sur grille (recherche vectorisee du plus court intervalle
    de masse >= level ; la densite est egale aux deux bornes)."""
    hi_x = xmax if xmax else dist.ppf(1 - 1e-7)
    x = np.linspace(0.0, float(hi_x), m)
    dens = dist.pdf(x)
    c = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(x))])
    c /= c[-1]
    j = np.searchsorted(c, c + level, side="left")
    jj = np.clip(j, 0, m - 1)
    w = np.where(j < m, x[jj] - x, np.inf)
    i = int(np.argmin(w))
    return float(x[i]), float(x[jj[i]])


g = stats.gamma(a=2.0, scale=2.0)                       # moyenne 4, queue droite epaisse
lo_e, hi_e = g.ppf(.025), g.ppf(.975)
lo_h, hi_h = hdi_grid(g, .95)
xs = np.linspace(1e-6, g.ppf(.9995), 900)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
ax = axes[0]
ax.plot(xs, g.pdf(xs), color=C["deep"], lw=2.0)
ax.fill_between(np.linspace(lo_e, hi_e, 600), 0, g.pdf(np.linspace(lo_e, hi_e, 600)),
                color=C["orange"], alpha=.35, label=f"queues égales : [{lo_e:.2f} ; {hi_e:.2f}]  larg. {hi_e-lo_e:.3f}")
ax.fill_between(np.linspace(lo_h, hi_h, 600), 0, g.pdf(np.linspace(lo_h, hi_h, 600)),
                color=C["blue"], alpha=.30, label=f"HDI 95 % : [{lo_h:.2f} ; {hi_h:.2f}]  larg. {hi_h-lo_h:.3f}")
ax.plot(xs, g.pdf(xs), color=C["deep"], lw=2.0)
ax.axvline(lo_e, color=C["orange"], ls="--", lw=1.2)
ax.axvline(hi_e, color=C["orange"], ls="--", lw=1.2)
ax.axvline(lo_h, color=C["blue"], ls=":", lw=1.6)
ax.axvline(hi_h, color=C["blue"], ls=":", lw=1.6)
ax.annotate(f"1,1 % de masse\nperdue dans la queue\n(trop de masse inutile)",
            (hi_e, g.pdf(hi_e)), textcoords="offset points", xytext=(-72, 44),
            fontsize=8.0, color=C["orange"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.0))
ax.set_xlabel("$\\theta$")
ax.set_ylabel("densité a posteriori")
ax.set_title("Posterior $\\mathrm{Gamma}(2\\,;\\,0{,}5)$, asymétrique à droite", fontsize=10.2)
ax.legend(fontsize=7.8, loc="upper right")

dists = [("Gamma(2 ; 0,5)", stats.gamma(a=2.0, scale=2.0)),
         ("Beta(2, 9)", stats.beta(2, 9)),
         ("$\\chi^2_3$", stats.chi2(3))]
wid_e, wid_h = [], []
for (lab, d), key in zip(dists, ["gamma", "beta", "chi2"]):
    e = float(d.ppf(.975) - d.ppf(.025))
    hh = hdi_grid(d, .95)
    h = float(hh[1] - hh[0])
    wid_e.append(e)
    wid_h.append(h)
    K[f"h03_{key}_queue"] = round(e, 4)
    K[f"h03_{key}_hdi"] = round(h, 4)
    K[f"h03_{key}_gain_pct"] = round(100 * (1 - h / e), 2)
xpos = np.arange(len(dists))
axes[1].bar(xpos - .19, wid_e, .36, color=C["orange"], label="queues égales")
axes[1].bar(xpos + .19, wid_h, .36, color=C["blue"], label="HDI (le plus court)")
for xp, we, wh in zip(xpos, wid_e, wid_h):
    axes[1].annotate(f"{we:.2f}", (xp - .19, we), ha="center", va="bottom", fontsize=8.2, color=C["deep"])
    axes[1].annotate(f"{wh:.2f}", (xp + .19, wh), ha="center", va="bottom", fontsize=8.2, color=C["deep"])
    axes[1].annotate(f"−{100*(1-wh/we):.0f} %", (xp, max(we, wh) * 1.12), ha="center", fontsize=8.4,
                     color=C["red"], fontweight="bold")
axes[1].set_xticks(xpos)
axes[1].set_xticklabels([d[0] for d in dists])
axes[1].set_ylabel("largeur de l'intervalle à 95 %")
axes[1].set_title("Le HDI est toujours le plus court", fontsize=10.2)
axes[1].legend(fontsize=8.2)
fig.suptitle("h03 — HDI contre intervalle à queues égales : même masse, longueur minimale",
             fontsize=10.8, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h03_hdi_vs_queue.png")
K["h03_gamma_queue"] = [round(float(lo_e), 4), round(float(hi_e), 4)]
K["h03_gamma_hdi"] = [round(float(lo_h), 4), round(float(hi_h), 4)]
K["h03_gain_hdi_gamma"] = round(float(100 * (1 - (hi_h - lo_h) / (hi_e - lo_e))), 2)

# =====================================================================
# h04 -- couverture empirique mesuree : 2000 echantillons simules
# =====================================================================
NSIM, n_obs2, th_true, z = 2000, 20, 0.30, 1.959963985
k = rng.binomial(n_obs2, th_true, NSIM)
ph = k / n_obs2
half = z * np.sqrt(np.maximum(ph * (1 - ph), 1e-12) / n_obs2)
wald = np.vstack([ph - half, ph + half])

meth = {}
# (1) intervalle de confiance de Wald (frequentiste)
meth["IC Wald (fréquentiste)"] = (wald[0], wald[1], C["orange"])
# (2) posterior Beta(1,1) -> intervalle a queues egales
pa, pb = 1 + k, 1 + (n_obs2 - k)
meth["crédible, prior Beta(1,1)"] = (stats.beta.ppf(.025, pa, pb), stats.beta.ppf(.975, pa, pb), C["blue"])
# (3) prior informatif centre sur la verite Beta(6,14)
pa2, pb2 = 6 + k, 14 + (n_obs2 - k)
meth["crédible, prior Beta(6,14)"] = (stats.beta.ppf(.025, pa2, pb2), stats.beta.ppf(.975, pa2, pb2), C["green"])
# (4) prior informatif faux Beta(12,4) (centre sur 0,75)
pa3, pb3 = 12 + k, 4 + (n_obs2 - k)
meth["crédible, prior Beta(12,4)"] = (stats.beta.ppf(.025, pa3, pb3), stats.beta.ppf(.975, pa3, pb3), C["red"])

cov, wid, labs, cols = [], [], [], []
for lab, (lo, hi, col) in meth.items():
    c = float(np.mean((lo <= th_true) & (th_true <= hi)))
    cov.append(100 * c)
    wid.append(float(np.mean(hi - lo)))
    labs.append(lab)
    cols.append(col)
    K[f"h04_couverture_{lab}"] = round(100 * c, 2)
    K[f"h04_largeur_{lab}"] = round(float(np.mean(hi - lo)), 4)

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
xp = np.arange(len(labs))
axes[0].bar(xp, cov, .62, color=cols)
axes[0].axhline(95, color=C["deep"], ls="--", lw=1.6, label="niveau nominal 95 %")
for x_, v in zip(xp, cov):
    axes[0].annotate(f"{v:.1f} %", (x_, v), ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color=C["deep"])
axes[0].set_xticks(xp)
axes[0].set_xticklabels(["IC de Wald", "crédible\nBeta(1,1)", "crédible\nBeta(6,14)", "crédible\nBeta(12,4)"], fontsize=8.6)
axes[0].set_ylabel("couverture empirique (%)")
axes[0].set_ylim(0, 108)
axes[0].set_title(f"Couverture sur {NSIM:,} échantillons ($n={n_obs2}$, $\\theta={th_true}$)".replace(",", " "),
                  fontsize=10.0)
axes[0].legend(fontsize=8.2, loc="lower right")
axes[1].bar(xp, wid, .62, color=cols)
for x_, v in zip(xp, wid):
    axes[1].annotate(f"{v:.3f}", (x_, v), ha="center", va="bottom", fontsize=9, fontweight="bold", color=C["deep"])
axes[1].set_xticks(xp)
axes[1].set_xticklabels(["IC de Wald", "crédible\nBeta(1,1)", "crédible\nBeta(6,14)", "crédible\nBeta(12,4)"], fontsize=8.6)
axes[1].set_ylabel("largeur moyenne")
axes[1].set_ylim(0, max(wid) * 1.25)
axes[1].set_title("Un prior plus fort rétrécit l'intervalle", fontsize=10.0)
fig.suptitle("h04 — Couverture empirique mesurée par simulation : la garantie fréquentiste se paie en largeur",
             fontsize=10.6, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h04_couverture.png")
K["h04_nsim"], K["h04_n"], K["h04_theta_vrai"] = NSIM, n_obs2, th_true

# =====================================================================
# h05 -- loi predictive a posteriori : densite predite contre nouvelles donnees
# =====================================================================
obs_counts = np.array([4, 2, 5, 3, 6])          # accidents par an, 5 ans d'historique
a0_g, b0_g = 2.5, 1.0
post_g = stats.gamma(a0_g + obs_counts.sum(), scale=1.0 / (b0_g + len(obs_counts)))
lam = np.linspace(1e-4, post_g.ppf(.9995), 800)
_gl = np.linspace(1e-3, post_g.ppf(.999), 40000)
_pd = post_g.pdf(_gl)
lam_draw = rng.choice(_gl, 6000, p=_pd / _pd.sum())
y_new = rng.poisson(lam_draw)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
ax = axes[0]
ax.fill_between(lam, 0, post_g.pdf(lam), color=C["blue"], alpha=.25)
ax.plot(lam, post_g.pdf(lam), color=C["blue"], lw=2.1)
lo_c, hi_c = post_g.ppf(.025), post_g.ppf(.975)
ax.axvline(lo_c, color=C["red"], ls="--", lw=1.3)
ax.axvline(hi_c, color=C["red"], ls="--", lw=1.3, label=f"crédibilité 95 % = [{lo_c:.2f} ; {hi_c:.2f}]")
ax.axvline(post_g.mean(), color=C["orange"], lw=1.8, label=f"moyenne = {post_g.mean():.3f}")
ax.set_xlabel("taux d'accidents $\\lambda$ (par an)")
ax.set_ylabel("densité a posteriori")
ax.set_title(f"Posterior $\\mathrm{{Gamma}}({(a0_g + obs_counts.sum()):.1f},\\,{b0_g + len(obs_counts):.0f})$ "
             f"après {obs_counts.sum()} accidents en {len(obs_counts)} ans", fontsize=9.8)
ax.legend(fontsize=8.0, loc="upper right")
ax = axes[1]
kmax = 14
bins = np.arange(-.5, kmax + 1.5, 1)
ax.hist(y_new, bins=bins, density=True, color=C["grey"], alpha=.55, edgecolor="white",
        label="6 000 nouvelles années simulées")
# densite predictive theorique (Negative Binomial)
r_, p_ = a0_g + obs_counts.sum(), (b0_g + len(obs_counts)) / (b0_g + len(obs_counts) + 1)
nb = stats.nbinom(r_, p_)
kk = np.arange(0, kmax + 1)
ax.plot(kk, nb.pmf(kk), "o-", color=C["red"], lw=1.8, ms=5,
        label="prédictive : NegBin($r$, $p$) théorique")
ax.set_xlabel("nombre d'accidents l'année prochaine $\\tilde y$")
ax.set_ylabel("probabilité")
ax.set_title("Prédiction a posteriori des années futures", fontsize=9.8)
ax.legend(fontsize=8.0)
fig.suptitle("h05 — La prédictive a posteriori intègre l'incertitude sur $\\lambda$ (elle est plus large que Poisson)",
             fontsize=10.5, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h05_predictive.png")
K["h05_somme_accidents"] = int(obs_counts.sum())
K["h05_post_mean_lambda"] = round(float(post_g.mean()), 4)
K["h05_post_cri95"] = [round(float(lo_c), 4), round(float(hi_c), 4)]
K["h05_P_lambda_sup_4"] = round(float(1 - post_g.cdf(4.0)), 4)
K["h05_P_ynew_sup_6"] = round(float(nb.sf(6)), 4)
K["h05_frac_zero_simule"] = round(float(np.mean(y_new == 0)), 4)

# =====================================================================
# h06 -- posterior predictive check avec statistique de test
# =====================================================================
y_obs = rng.negative_binomial(2, 0.4, 30).astype(float)   # surdispersé : moyenne 3, variance ~7,5
T = lambda v: float(np.var(v, ddof=1) / np.mean(v))
T_obs = T(y_obs)
# posterior : lambda ~ Gamma(1,1) ; y_i ~ Poisson(lambda)  -> Gamma(1+somme, 1+n)
post_l = stats.gamma(1 + y_obs.sum(), scale=1.0 / (1 + len(y_obs)))
lam_s = post_l.rvs(4000, random_state=3)
T_rep = np.array([T(rng.poisson(l_ * np.ones_like(y_obs))) for l_ in lam_s])
T_rep = np.clip(T_rep, 0, 8)
p_bayes = float(np.mean(T_rep >= T_obs))

# intervalles predictifs par observation
lo_i = stats.poisson.ppf(.025, lam_s[:, None])
hi_i = stats.poisson.ppf(.975, lam_s[:, None])
lo_o = np.percentile(lo_i, 95, axis=0)   # borne "95 %" (melange du posterior)
hi_o = np.percentile(hi_i, 5, axis=0)
cover = np.mean((y_obs >= lo_o) & (y_obs <= hi_o))

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
ax = axes[0]
xpos = np.arange(1, len(y_obs) + 1)
ax.vlines(xpos, lo_o, hi_o, color=C["blue"], lw=2.4, alpha=.65,
          label="intervalle prédictif à 95 % (mélange)")
ax.plot(xpos, y_obs, "o", color=C["red"], ms=5.5, label="données observées")
outside = (y_obs < lo_o) | (y_obs > hi_o)
ax.plot(xpos[outside], y_obs[outside], "x", color=C["deep"], ms=9, mew=2.2,
        label=f"dépassements : {int(outside.sum())}/{len(y_obs)}")
ax.set_xlabel("observation $i$")
ax.set_ylabel("nombre d'événements")
ax.set_title("Dépassements des intervalles prédictifs", fontsize=10.0)
ax.legend(fontsize=7.8)

ax = axes[1]
ax.hist(T_rep, bins=45, color=C["blue"], alpha=.45, edgecolor="white",
        label=f"$T(\\tilde y)$ sous la prédictive ({len(T_rep)} réplications)")
ax.axvline(T_obs, color=C["red"], lw=2.4, label=f"$T(y)$ observé = {T_obs:.2f}")
ax.annotate(f"$p_B = P(T(\\tilde y) \\geq T(y)) = {p_bayes:.4f}$\n"
            f"le modèle Poisson est rejeté",
            (T_obs, ax.get_ylim()[1] * .72), textcoords="offset points", xytext=(14, 0),
            fontsize=8.6, color=C["blue"], fontweight="bold")
ax.set_xlabel("statistique de test  $T(y) = \\mathrm{Var}(y)/\\bar y$")
ax.set_ylabel("fréquence")
ax.set_title("Posterior predictive check (PCC)", fontsize=10.0)
ax.legend(fontsize=7.8, loc="upper right")
fig.suptitle(f"h06 — Le modèle Poisson ne sait pas reproduire la surdispersion des données (moyenne {y_obs.mean():.2f}, variance {y_obs.var(ddof=1):.2f})",
             fontsize=10.2, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h06_ppc.png")
K["h06_T_obs"] = round(float(T_obs), 4)
K["h06_p_bayes"] = round(float(p_bayes), 4)
K["h06_depassements"] = f"{int(outside.sum())}/{len(y_obs)}"
K["h06_couverture_pred"] = round(float(100 * cover), 2)
K["h06_moyenne_obs"] = round(float(y_obs.mean()), 3)
K["h06_variance_obs"] = round(float(y_obs.var(ddof=1)), 3)

# =====================================================================
# h08 -- risque MSE : estimateur bayesien contre maximum de vraisemblance
# =====================================================================
NS_MSE, th_true2 = 500, 0.35
ns_mse = [5, 10, 20, 50, 100, 400]
mse_b, mse_m = [], []
for n_ in ns_mse:
    kk = rng.binomial(n_, th_true2, NS_MSE)
    mle_ = kk / n_
    bay_ = (kk + 2) / (n_ + 4)          # prior Beta(2,2), 500 simulations
    mse_m.append(float(np.mean((mle_ - th_true2) ** 2)))
    mse_b.append(float(np.mean((bay_ - th_true2) ** 2)))
k5 = rng.binomial(20, th_true2, NS_MSE)
d5 = (k5 / 20 - th_true2) ** 2 - ((k5 + 2) / 24 - th_true2) ** 2   # gain bayesien
K["h08_mse_n5_mle"], K["h08_mse_n5_bayes"] = round(mse_m[0], 6), round(mse_b[0], 6)
K["h08_mse_n400_mle"], K["h08_mse_n400_bayes"] = round(mse_m[-1], 6), round(mse_b[-1], 6)
K["h08_ratio_n5"] = round(mse_b[0] / mse_m[0], 4)
K["h08_ratio_n20"] = round(mse_b[2] / mse_m[2], 4)
K["h08_gain_moyen_n20"] = round(float(np.mean(d5)), 6)
K["h08_nsim_mse"], K["h08_theta"] = NS_MSE, th_true2

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
ax = axes[0]
ax.plot(ns_mse, mse_m, "o-", color=C["red"], lw=2.0, ms=5, label="MLE  $\\hat\\theta = k/n$")
ax.plot(ns_mse, mse_b, "s-", color=C["blue"], lw=2.0, ms=5, label="moyenne a posteriori, prior Beta(2,2)")
ax.axhline(th_true2 * 1, color=C["grey"], ls=":", lw=1.0)
ax.set_xscale("log")
ax.set_xlabel("taille d'échantillon $n$ (échelle log)")
ax.set_ylabel("MSE empirique (500 simulations)")
ax.set_title(f"Risque mesuré, $\\theta$ vrai = {th_true2}", fontsize=10.0)
ax.legend(fontsize=8.2)
ax.annotate(f"$n=5$ : gain $\\times${mse_m[0]/mse_b[0]:.2f}", (5, mse_b[0]),
            textcoords="offset points", xytext=(8, 10), fontsize=8.4, color=C["blue"], fontweight="bold")
ax = axes[1]
ax.hist(d5, bins=42, color=C["blue"], alpha=.45, edgecolor="white")
ax.axvline(0, color=C["grey"], ls="--", lw=1.4)
ax.axvline(float(np.mean(d5)), color=C["red"], lw=2.2, label=f"gain moyen = {np.mean(d5):.4f}")
ax.set_xlabel("$L_{\\mathrm{MLE}} - L_{\\mathrm{Bayes}}$ par simulation ($n=20$)")
ax.set_ylabel("fréquence")
ax.set_title(f"Le bayésien gagne dans {100*np.mean(d5>0):.1f} % des {NS_MSE} simulations", fontsize=10.0)
ax.legend(fontsize=8.2)
fig.suptitle("h08 — Risque MSE mesuré par simulation : avantage bayésien net aux petits échantillons",
             fontsize=10.6, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h08_mse_bayes_vs_mle.png")
K["h08_frac_gain_positif"] = round(float(100 * np.mean(d5 > 0)), 2)

# =====================================================================
# h10 -- application Gamma-Poisson : accidents et choix du prior
# =====================================================================
obs2 = np.array([12, 9, 14, 11, 13, 8])
n2, s2 = len(obs2), obs2.sum()
priors = [("vague", "Gamma(0,01 ; 0,01) — vague", .01, .01, C["grey"]),
          ("faible", "Gamma(3 ; 1) faiblement informatif", 3., 1., C["blue"]),
          ("fort", "Gamma(40 ; 4) centre sur 10", 40., 4., C["red"])]
xs2 = np.linspace(1e-3, 25, 800)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
for key, lab, ap, bp, col in priors:
    d = stats.gamma(ap + s2, scale=1.0 / (bp + n2))
    axes[0].plot(xs2, d.pdf(xs2), color=col, lw=2.0, label=lab.split(' — ')[0])
    axes[1].plot(xs2, d.pdf(xs2), color=col, lw=2.0,
                 label=f"moy. {d.mean():.3f}, IC95 [{d.ppf(.025):.2f} ; {d.ppf(.975):.2f}]")
    K[f"h10_moy_{key}"] = round(float(d.mean()), 4)
    K[f"h10_largeur_{key}"] = round(float(d.ppf(.975) - d.ppf(.025)), 4)
    K[f"h10_ic95_{key}"] = [round(float(d.ppf(.025)), 3), round(float(d.ppf(.975)), 3)]
axes[0].axvline(s2 / n2, color=C["orange"], ls="--", lw=1.8, label=f"MLE = {s2/n2:.3f}")
axes[0].set_xlabel("taux d'accidents mensuel $\\lambda$")
axes[0].set_ylabel("densité a priori")
axes[0].set_title("Trois priors candidats", fontsize=10.0)
axes[0].legend(fontsize=7.8)
axes[1].axvline(s2 / n2, color=C["orange"], ls="--", lw=1.8, label=f"MLE = {s2/n2:.3f}")
axes[1].set_xlabel("taux d'accidents mensuel $\\lambda$")
axes[1].set_ylabel("densité a posteriori")
axes[1].set_title(f"Posterior après {s2} accidents en {n2} mois", fontsize=10.0)
axes[1].legend(fontsize=7.6)
fig.suptitle("h10 — Cas Gamma-Poisson : le prior vague et le faiblement informatif donnent le même posterior",
             fontsize=10.4, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h10_gamma_poisson.png")
K["h10_somme_accidents"], K["h10_n_mois"], K["h10_mle"] = int(s2), n2, round(float(s2 / n2), 4)

# =====================================================================
# h11 -- application Normal-Normal : mesures repetees
# =====================================================================
mu0, tau0, sigma = 100.0, 15.0, 8.0        # prior N(100, 15^2), ecart-type de mesure connu 8
y_mes = np.array([112, 121, 108, 117, 126, 115, 119])
n3, ybar = len(y_mes), y_mes.mean()
prec = 1 / tau0 ** 2 + n3 / sigma ** 2
mu_post = (mu0 / tau0 ** 2 + n3 * ybar / sigma ** 2) / prec
sd_post = 1 / np.sqrt(prec)
post_n = stats.norm(mu_post, sd_post)
grid_m = np.linspace(96, 132, 700)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
ax = axes[0]
ax.plot(grid_m, stats.norm(mu0, tau0).pdf(grid_m), color=C["green"], lw=2.0,
        label=f"prior $\\mathcal{{N}}({mu0:.0f},\\,{tau0:.0f}^2)$")
ax.plot(grid_m, post_n.pdf(grid_m), color=C["blue"], lw=2.4,
        label=f"posterior $\\mathcal{{N}}({mu_post:.2f},\\,{sd_post:.2f}^2)$")
ax.axvline(ybar, color=C["orange"], ls="--", lw=1.8, label=f"$\\bar y$ = {ybar:.3f} (MLE)")
lo_n, hi_n = post_n.ppf(.025), post_n.ppf(.975)
ax.fill_between(grid_m, 0, post_n.pdf(grid_m), where=(grid_m >= lo_n) & (grid_m <= hi_n),
                color=C["blue"], alpha=.18)
ax.set_xlabel("masse moyenne $\\mu$ (g)")
ax.set_ylabel("densité")
ax.set_title("Mise à jour conjuguée Normal-Normal", fontsize=10.0)
ax.legend(fontsize=7.6)
ax = axes[1]
y_grid = np.linspace(80, 150, 700)
pred_sd = np.sqrt(sigma ** 2 + sd_post ** 2)
ax.hist(y_mes, bins=np.arange(100, 132, 4), density=True, color=C["grey"], alpha=.5,
        edgecolor="white", label="7 mesures observées")
ax.plot(y_grid, stats.norm(mu_post, pred_sd).pdf(y_grid), color=C["purple"], lw=2.2,
        label="prédictive : $\\mathcal{N}(\\mu_{post}, \\sigma^2+\\sigma_{post}^2)$")
ax.plot(y_grid, stats.norm(ybar, sigma).pdf(y_grid), color=C["orange"], lw=1.6, ls="--",
        label="prédictive naïve $\\mathcal{N}(\\bar y, \\sigma^2)$")
ax.set_xlabel("prochaine mesure $\\tilde y$ (g)")
ax.set_ylabel("densité")
ax.set_title("La prédiction garde deux sources d'incertitude", fontsize=10.0)
ax.legend(fontsize=7.6)
fig.suptitle("h11 — Cas Normal-Normal : le posterior resserre le prior, la prédictive reste large",
             fontsize=10.5, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h11_normal_normal.png")
K["h11_ybar"], K["h11_mu_post"], K["h11_sd_post"] = round(float(ybar), 4), round(float(mu_post), 4), round(float(sd_post), 4)
K["h11_cri95_mu"] = [round(float(lo_n), 3), round(float(hi_n), 3)]
K["h11_reduction_sd"] = round(float(100 * (1 - sd_post / (sigma / np.sqrt(n3)))), 2)

# =====================================================================
# h12 -- credibilite unilaterale et lecture par quantiles
# =====================================================================
post_u = stats.beta(20, 10)              # posterior du chapitre : 20 succes, 10 echecs
xu = np.linspace(0.001, .999, 900)
du = post_u.pdf(xu)
q50, q90, q95 = post_u.ppf(.5), post_u.ppf(.90), post_u.ppf(.95)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
ax = axes[0]
ax.plot(xu, du, color=C["blue"], lw=2.2)
ax.fill_between(xu, 0, du, where=(xu >= q95), color=C["red"], alpha=.30,
                label=f"$P(\\theta > {q95:.3f} \\mid y) = 0,05$")
ax.fill_between(xu, 0, du, where=(xu <= q50), color=C["grey"], alpha=.22,
                label=f"médiane du posterior = {q50:.3f}")
for q, col in ((q90, C["orange"]), (q95, C["red"])):
    ax.axvline(q, color=col, ls="--", lw=1.5)
ax.set_xlabel("$\\theta$")
ax.set_ylabel("densité a posteriori")
ax.set_title("Bornes unilatérales à 90 % et 95 %", fontsize=10.0)
ax.legend(fontsize=7.8)
ax = axes[1]
ax.plot(xu, du, color=C["blue"], lw=2.2, label="posterior Beta(20, 10)")
ax.axvline(.5, color=C["green"], ls=":", lw=1.8, label="seuil $\\theta_0 = 0{,}5$")
for q, lab in ((q50, "médiane"), (q90, "q90"), (q95, "q95")):
    ax.axvline(q, color=C["deep"], ls="--", lw=1.0, alpha=.6)
    ax.annotate(f"{lab} = {q:.3f}", (q, ax.get_ylim()[1] * .8), rotation=90, fontsize=7.8,
                color=C["deep"], ha="right")
ax.set_xlabel("$\\theta$")
ax.set_ylabel("densité a posteriori")
ax.set_title(f"$P(\\theta > 0,5 \\mid y) = {1 - post_u.cdf(.5):.4f}$", fontsize=10.0)
ax.legend(fontsize=7.8, loc="upper left")
fig.suptitle("h12 — Crédibilité unilatérale : la question « le taux dépasse-t-il le seuil ? » reçoit une probabilité",
             fontsize=10.4, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "h12_credibilite_unilaterale.png")
K["h12_q50"], K["h12_q90"], K["h12_q95"] = round(float(q50), 4), round(float(q90), 4), round(float(q95), 4)
K["h12_P_sup_0_5"] = round(float(1 - post_u.cdf(.5)), 4)

# =====================================================================
out = os.path.join(FIGDIR, "key_numbers_ch3.json")
json.dump(K, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n=== CH3 (statique) ===")
print(json.dumps(K, indent=1, ensure_ascii=False))
print(f"total {len(K)} chiffres cles -> {out}")
