"""Chapitre 3 -- figures issues d'un echantillonnage MCMC REEL avec PyMC.

Sorties : figures/h07_regression_bayesienne.png, h09_pymc_trace.png,
          figures/h13_regression_marginales.png
Complete figures/key_numbers_ch3.json (chiffres produits par le script statique).
Contraintes machine : CPU seulement, 2 chaines, cores=1, pas de compilateur C.
"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
import numpy as np
import pymc as pm
import arviz as az
from scipy import stats
from figstyle import plt, C, save, FIGDIR

t0 = time.time()
rng = np.random.default_rng(3)
K = {}

# =====================================================================
# Donnees : regression lineaire simple, 25 points, bruit connu en loi
# =====================================================================
N_CHAINS, DRAWS, TUNE = 2, 1000, 600
n_r = 25
a_true, b_true, s_true = 2.0, 1.5, 0.9
x = np.sort(rng.uniform(0.2, 5.0, n_r))
y = a_true + b_true * x + rng.normal(0.0, s_true, n_r)
lin = stats.linregress(x, y)
a_ols, b_ols = float(lin.intercept), float(lin.slope)
K["h07_n"], K["h07_a_vrai"], K["h07_b_vrai"], K["h07_sigma_vrai"] = n_r, a_true, b_true, s_true
K["h07_a_ols"], K["h07_b_ols"] = round(a_ols, 4), round(b_ols, 4)

# =====================================================================
# Modele bayesien complet (PyMC -- NUTS reellement execute)
# =====================================================================
with pm.Model() as mreg:
    alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
    beta = pm.Normal("beta", mu=0.0, sigma=5.0)
    sigma = pm.HalfNormal("sigma", sigma=2.0)
    mu = alpha + beta * x
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=N_CHAINS, cores=1,
                      progressbar=False, random_seed=4112,
                      compute_convergence_checks=True)
el = time.time() - t0
with mreg:
    ppc = pm.sample_posterior_predictive(idata, progressbar=False, random_seed=7)

d_a = idata.posterior["alpha"].values.ravel()
d_b = idata.posterior["beta"].values.ravel()
d_s = idata.posterior["sigma"].values.ravel()
K["h07_alpha_post"] = round(float(d_a.mean()), 4)
K["h07_beta_post"] = round(float(d_b.mean()), 4)
K["h07_sigma_post"] = round(float(d_s.mean()), 4)
K["h07_beta_hdi95"] = [round(float(v), 4) for v in az.hdi(d_b, hdi_prob=.95)]
K["h07_alpha_hdi95"] = [round(float(v), 4) for v in az.hdi(d_a, hdi_prob=.95)]
K["h07_P_beta_sup_1"] = round(float(np.mean(d_b > 1.0)), 4)
K["h07_P_beta_sup_0"] = round(float(np.mean(d_b > 0.0)), 4)
K["h07_couv_vrai_beta"] = bool(K["h07_beta_hdi95"][0] <= b_true <= K["h07_beta_hdi95"][1])
K["h07_draws"], K["h07_chains"], K["h07_seconds"] = int(N_CHAINS * DRAWS), N_CHAINS, round(el, 1)
K["h07_ecart_bayes_mle"] = round(float(d_b.mean() - b_ols), 4)

rhat = az.rhat(idata)
ess = az.ess(idata)
K["h09_rhat"] = {v: round(float(rhat[v].values), 4) for v in ("alpha", "beta", "sigma")}
K["h09_ess"] = {v: round(float(ess[v].values), 1) for v in ("alpha", "beta", "sigma")}
K["h09_rhat_max"] = round(max(K["h09_rhat"].values()), 4)
K["h09_ess_min"] = round(min(K["h09_ess"].values()), 1)
K["h09_divergences"] = int(idata.sample_stats["diverging"].values.sum())

# =====================================================================
# h07 -- nuage + bande de credibilite + posterior joint des coefficients
# =====================================================================
xg = np.linspace(0.0, 5.4, 220)
band = d_a[:, None] + d_b[:, None] * xg[None, :]          # (draws, len(xg))
lo95, hi95 = np.percentile(band, [2.5, 97.5], axis=0)
lo50, hi50 = np.percentile(band, [25, 75], axis=0)
pp_band = ppc.posterior_predictive["y"].values.reshape(-1, n_r)
plo, phi = np.percentile(pp_band, [2.5, 97.5], axis=0)
couv = float(np.mean((y >= plo) & (y <= phi)))

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5),
                         gridspec_kw={"width_ratios": [1.25, 1.0]})
ax = axes[0]
ax.fill_between(xg, lo95, hi95, color=C["blue"], alpha=.20,
                label="bande de crédibilité 95 % de $\\mu(x)$")
ax.fill_between(xg, lo50, hi50, color=C["blue"], alpha=.35, label="bande 50 %")
ax.plot(xg, d_a.mean() + d_b.mean() * xg, color=C["blue"], lw=2.4,
        label=f"droite moyenne : {d_a.mean():.3f} + {d_b.mean():.3f}$x$")
ax.plot(xg, a_ols + b_ols * xg, color=C["red"], lw=1.6, ls="--",
        label=f"moindres carrés : {a_ols:.3f} + {b_ols:.3f}$x$")
ax.plot(x, y, "o", color=C["deep"], ms=5.5, label=f"{n_r} observations simulées")
ax.set_xlabel("$x$ (variable explicative)")
ax.set_ylabel("$y$ (réponse)")
ax.set_title("Nuage, droite a posteriori et bandes de crédibilité", fontsize=10.0)
ax.legend(fontsize=7.4, loc="upper left")

ax = axes[1]
H, xe, ye = np.histogram2d(d_a, d_b, bins=70)
xm, ym = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
lev = np.sort(H.ravel())[::-1]
cum = np.cumsum(lev) / lev.sum()
l50 = lev[np.searchsorted(cum, 0.50)]
l95 = lev[np.searchsorted(cum, 0.95)]
ax.contourf(xm, ym, H.T, levels=[l95, l50, H.max()], colors=[C["blue"], C["cyan"]],
            alpha=.32)
ax.contour(xm, ym, H.T, levels=[l95, l50], colors=[C["blue"], C["cyan"]], linewidths=1.3)
ax.plot(a_ols, b_ols, "*", color=C["red"], ms=14, label=f"MCO ({a_ols:.2f} ; {b_ols:.2f})")
ax.plot(a_true, b_true, "x", color=C["green"], ms=9, mew=2.4,
        label=f"vérité ({a_true:.2f} ; {b_true:.2f})")
ax.set_xlabel("$\\alpha$ (ordonnée à l'origine)")
ax.set_ylabel("$\\beta$ (pente)")
ax.set_title("Posterior joint $p(\\alpha, \\beta \\mid y)$ (50 % et 95 %)", fontsize=10.0)
ax.legend(fontsize=7.6, loc="upper right")
fig.suptitle(f"h07 — {N_CHAINS * DRAWS} tirages NUTS réels (PyMC, {el:.0f} s sur CPU) : "
             f"pente a posteriori {d_b.mean():.3f} contre {b_ols:.3f} (MCO)",
             fontsize=10.4, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "h07_regression_bayesienne.png")
K["h07_couverture_pred_obs"] = round(100 * couv, 2)

# =====================================================================
# h09 -- tracés et diagnostics MCMC reellement produits par PyMC
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0),
                         gridspec_kw={"width_ratios": [1.5, 1.2, 1.2]})
ax = axes[0]
for ch, col in zip(range(N_CHAINS), [C["blue"], C["orange"]]):
    ax.plot(idata.posterior["beta"].values[ch], lw=.6, color=col, label=f"chaîne {ch + 1}")
ax.axhline(d_b.mean(), color=C["deep"], ls="--", lw=1.2)
ax.set_xlabel("itération (après échauffement)")
ax.set_ylabel("$\\beta$")
ax.set_title(f"Tracé des {N_CHAINS} chaînes — $\\hat R$ = {K['h09_rhat_max']:.4f}", fontsize=9.6)
ax.legend(fontsize=7.6)
ax = axes[1]
az.plot_rank(idata, var_names=["beta"], ax=ax, colors=[C["blue"], C["orange"]])
ax.set_title("Rank plot : profils plats = chaînes mélangées", fontsize=9.6)
ax = axes[2]
az.plot_autocorr(idata, var_names=["beta"], ax=ax)
for ln in ax.lines:
    ln.set_color(C["blue"])
ax.set_title(f"Autocorrélation — ESS = {K['h09_ess']['beta']:.0f}", fontsize=9.6)
fig.suptitle(f"h09 — Diagnostics réels PyMC : $\\hat R$ max = {K['h09_rhat_max']:.4f}, "
             f"ESS min = {K['h09_ess_min']:.0f}, divergences = {K['h09_divergences']}",
             fontsize=10.4, fontweight="bold", y=1.05)
fig.tight_layout()
save(fig, "h09_pymc_trace.png")

# =====================================================================
# h13 -- densites marginales des coefficients + predictive a posteriori
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
ax = axes[0]
gb = np.linspace(d_b.min(), d_b.max(), 500)
ga = np.linspace(d_a.min(), d_a.max(), 500)
ax.plot(gb, stats.gaussian_kde(d_b)(gb), color=C["blue"], lw=2.3,
        label=f"$\\beta$ : moyenne {d_b.mean():.3f}")
ax.plot(ga, stats.gaussian_kde(d_a)(ga), color=C["cyan"], lw=2.3,
        label=f"$\\alpha$ : moyenne {d_a.mean():.3f}")
hb, ha = az.hdi(d_b, hdi_prob=.95), az.hdi(d_a, hdi_prob=.95)
ax.axvline(hb[0], color=C["blue"], ls="--", lw=1.1)
ax.axvline(hb[1], color=C["blue"], ls="--", lw=1.1,
           label=f"HDI 95 % de $\\beta$ = [{hb[0]:.2f} ; {hb[1]:.2f}]")
ax.axvline(b_ols, color=C["red"], lw=1.6, ls=":", label=f"$\\hat\\beta$ MCO = {b_ols:.3f}")
ax.axvline(b_true, color=C["green"], lw=1.6, ls="-.", label=f"valeur vraie = {b_true:.2f}")
ax.set_xlabel("valeur du paramètre")
ax.set_ylabel("densité marginale a posteriori")
ax.set_title("Densités marginales de $\\alpha$ et $\\beta$", fontsize=10.0)
ax.legend(fontsize=7.2, loc="upper right")
ax = axes[1]
ytil = band + rng.standard_normal(band.shape) * d_s[:, None]   # predictive a posteriori
plo2, phi2 = np.percentile(ytil, [2.5, 97.5], axis=0)
ax.fill_between(xg, plo2, phi2, color=C["purple"], alpha=.14,
                label="intervalle prédictif 95 % (un point futur)")
ax.fill_between(xg, lo95, hi95, color=C["blue"], alpha=.24,
                label="bande de crédibilité 95 % de $\\mu(x)$")
ax.plot(x, y, "o", color=C["deep"], ms=5, label="données")
ax.plot(xg, d_a.mean() + d_b.mean() * xg, color=C["blue"], lw=2.2, label="droite moyenne")
K["h13_largeur_pred_x35"] = round(float(np.interp(3.5, xg, phi2 - plo2)), 3)
K["h13_largeur_cred_x35"] = round(float(np.interp(3.5, xg, hi95 - lo95)), 3)
ax.set_xlabel("$x$")
ax.set_ylabel("$\\tilde y$ : réponse future")
ax.set_title("Crédibilité de la moyenne contre prédiction d'un point futur", fontsize=10.0)
ax.legend(fontsize=7.4, loc="upper left")
fig.suptitle(f"h13 — Pente : HDI 95 % [{hb[0]:.3f} ; {hb[1]:.3f}], "
             f"$P(\\beta > 1 \\mid y)$ = {K['h07_P_beta_sup_1']:.4f}, "
             f"couverture des données = {100 * couv:.1f} %",
             fontsize=10.2, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "h13_regression_marginales.png")

K["h07_seconds_total"] = round(time.time() - t0, 1)

out = os.path.join(FIGDIR, "key_numbers_ch3.json")
old = {}
if os.path.exists(out):
    old = json.load(open(out, encoding="utf-8"))
old.update(K)
json.dump(old, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n=== CH3 (PyMC) ===")
print(json.dumps(K, indent=1, ensure_ascii=False))
print(f"total {len(old)} chiffres cles -> {out}  ({time.time()-t0:.1f} s)")
