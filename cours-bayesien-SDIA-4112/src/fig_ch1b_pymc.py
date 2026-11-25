"""Chapitre 1 -- figures issues d'un echantillonnage MCMC REEL avec PyMC.
Contraintes machine : CPU uniquement, 2 chaines, threads limites, priorite basse.
"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
import numpy as np
import pymc as pm
import arviz as az
from figstyle import plt, C, save, FIGDIR
from scipy import stats

rng = np.random.default_rng(7)
K = {}
t0 = time.time()

# =====================================================================
# Modele 1 : piece truquee -- 7 faces sur 10
# =====================================================================
N_CHAINS, DRAWS, TUNE = 2, 2000, 1000
with pm.Model() as m1:
    theta = pm.Beta("theta", alpha=2, beta=2)
    pm.Binomial("y", n=10, p=theta, observed=7)
    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=N_CHAINS, cores=1,
                      progressbar=False, random_seed=42, compute_convergence_checks=True)
el = time.time() - t0
post = idata.posterior["theta"].values.ravel()
K["pymc_coin_mean"] = float(post.mean())
K["pymc_coin_ci"] = [float(v) for v in np.percentile(post, [2.5, 97.5])]
K["pymc_coin_rhat"] = float(az.rhat(idata)["theta"].values)
K["pymc_coin_ess"] = float(az.ess(idata)["theta"].values)
K["pymc_coin_seconds"] = round(el, 2)
K["pymc_coin_draws"] = int(N_CHAINS * DRAWS)

# figure : posterior + trace
fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9),
                         gridspec_kw={"width_ratios": [1.5, 1.3, 1.3]})
xs = np.linspace(0.05, 0.99, 500)
axes[0].hist(post, bins=60, density=True, color=C["blue"], alpha=.55, edgecolor="white")
axes[0].plot(xs, stats.beta.pdf(xs, 9, 5), color=C["red"], lw=1.8,
             label="Beta(9, 5) analytique")
lo, hi = np.percentile(post, [2.5, 97.5])
axes[0].axvline(lo, color=C["deep"], ls="--", lw=1.1)
axes[0].axvline(hi, color=C["deep"], ls="--", lw=1.1, label=f"IC 95 % = [{lo:.3f} ; {hi:.3f}]")
axes[0].set_xlabel("$\\theta$"); axes[0].set_ylabel("densité")
axes[0].set_title("Posterior échantillonné (PyMC)", fontsize=10)
axes[0].legend(fontsize=7.6, loc="upper left")
for ax, ch in zip(axes[1:], range(N_CHAINS)):
    ax.plot(idata.posterior["theta"].values[ch], lw=.5, color=[C["blue"], C["orange"]][ch])
    ax.set_title(f"Trace chaîne {ch + 1}", fontsize=10)
    ax.set_xlabel("itération"); ax.set_yticks([])
fig.suptitle("f10 — 4 000 tirages MCMC en %.1f s sur CPU : le posterior analytique est retrouvé"
             % el, fontsize=10.6, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "f10_pymc_coin.png")

# figure : diagnostics (rank plot + autocorrelation)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.8))
az.plot_rank(idata, var_names=["theta"], ax=axes[0], colors=[C["blue"], C["orange"]])
axes[0].set_title("Rank plot — homogénéité des chaînes (bon signe : plat)", fontsize=9.6)
az.plot_autocorr(idata, var_names=["theta"], ax=axes[1])
for ln in axes[1].lines:
    ln.set_color(C["blue"])
axes[1].set_title(f"Autocorrélation — ESS = {K['pymc_coin_ess']:.0f}", fontsize=9.6)
fig.suptitle(f"f11 — Diagnostics de convergence : $\\hat R$ = {K['pymc_coin_rhat']:.4f}",
             fontsize=10.6, fontweight="bold", y=1.05)
fig.tight_layout()
save(fig, "f11_pymc_diagnostics.png")

# =====================================================================
# Modele 2 : sensibilite au prior -- plusieurs priors, memes donnees
# =====================================================================
priors = [("Beta(1, 1) — uniforme", 1, 1, C["green"]),
          ("Beta(2, 2) — faiblement informatif", 2, 2, C["blue"]),
          ("Beta(5, 3) — informatif (biais)", 5, 3, C["orange"]),
          ("Beta(50, 50) — très informatif", 50, 50, C["red"])]
grid = np.linspace(0.001, .999, 600)
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
tab = []
for lab, a0, b0, col in priors:
    axes[0].plot(grid, stats.beta.pdf(grid, a0, b0), color=col, lw=1.9, label=lab)
for lab, a0, b0, col in priors:
    p = stats.beta(a0 + 7, b0 + 3)
    axes[1].plot(grid, p.pdf(grid), color=col, lw=1.9,
                 label=f"$\\alpha$={a0}, $\\beta$={b0} → moy. {p.mean():.3f}")
    tab.append({"prior": lab, "posterior": f"Beta({a0+7}, {b0+3})",
                "moyenne": round(float(p.mean()), 3),
                "IC95": [round(float(p.ppf(.025)), 3), round(float(p.ppf(.975)), 3)]})
axes[0].set_title("Quatre a priori candidats", fontsize=10)
axes[1].set_title("Posterior après 7 succès sur 10", fontsize=10)
for ax in axes:
    ax.set_xlabel("$\\theta$"); ax.set_xlim(0, 1)
axes[0].set_ylabel("densité")
axes[0].legend(fontsize=7.8); axes[1].legend(fontsize=7.8)
fig.suptitle("f12 — Le prior s'efface quand les données arrivent (mais pas toujours assez vite)",
             fontsize=10.6, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "f12_prior_sensitivity.png")
K["f12_table"] = tab

# =====================================================================
# Modele 3 : combien de donnees pour effacer un prior informatif ?
# =====================================================================
y_data = rng.random(500) < 0.35          # vraie valeur 0,35
ns = np.arange(1, 501)
frac = []
grid_short = np.array([.25, .30, .35, .40, .45, .50])
for n in ns:
    k = int(y_data[:n].sum())
    p = stats.beta(50 + k, 50 + n - k)          # prior Beta(50,50) centre sur 0,5 -> biais fort
    frac.append(float(p.pdf(0.35) / max(p.pdf(grid_short).max(), 1e-300)))
frac = np.array(frac)
fig, ax = plt.subplots(figsize=(7.6, 3.3))
ax.plot(ns, frac, color=C["blue"], lw=2.2)
ax.axhline(1.0, color=C["red"], ls="--", lw=1.3, label="le posterior est centré sur la vérité")
for n in (100, 250, 500):
    k = int(y_data[:n].sum())
    p = stats.beta(50 + k, 50 + n - k)
    ax.plot(n, frac[n - 1], "o", color=C["deep"], ms=5)
    ax.annotate(f"n={n} : moy. {p.mean():.3f}", (n, frac[n - 1]), textcoords="offset points",
                xytext=(-10, 12), fontsize=8.4, color=C["deep"], fontweight="bold")
    K[f"f13_mean_n{n}"] = round(float(p.mean()), 3)
ax.set_xlabel("nombre d'observations")
ax.set_ylabel("densité du posterior en $\\theta_{vrai}$ / densité max")
ax.set_title("f13 — Prior très informatif Beta(50,50) contre données réelles ($\\theta$ = 0,35)")
ax.legend(fontsize=8.6)
save(fig, "f13_prior_vs_data.png")

out = os.path.join(FIGDIR, "key_numbers_ch1b.json")
json.dump(K, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n=== CH1 (PyMC) ===")
print(json.dumps(K, indent=1, ensure_ascii=False))
print(f"total {time.time()-t0:.1f} s")
