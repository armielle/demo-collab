"""Chapitre 2 -- figures issues d'un echantillonnage MCMC REEL avec PyMC.

Sorties : figures/g07_analytique_vs_pymc.png, g11_prior_predictive_pymc.png,
          g13_shrinkage_8groupes.png   (+ fusion dans figures/key_numbers_ch2.json)
Contraintes machine : CPU uniquement, 2 chaines, 1 coeur, draws <= 2000,
priorite basse (OMP/OPENBLAS/MKL = 2 threads), PyMC en Python pur (pas de g++).
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

rng = np.random.default_rng(4112)
t0 = time.time()
K = json.load(open(os.path.join(FIGDIR, "key_numbers_ch2.json"), encoding="utf-8"))


def fr(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


# =====================================================================
# g07 : analytique vs MCMC -- Challenger, 7 incidents sur 23 vols
# =====================================================================
N_CHAINS, DRAWS, TUNE = 2, 2000, 1000
n_ch, k_ch = 23, 7
with pm.Model() as m_bb:
    p = pm.Beta("p", alpha=1, beta=1)
    pm.Binomial("y", n=n_ch, p=p, observed=k_ch)
    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=N_CHAINS, cores=1,
                      progressbar=False, random_seed=42)
el1 = time.time() - t0
ps = idata.posterior["p"].values.ravel()
ana = stats.beta(8, 17)
K["mcmc_bb"] = {"mean": round(float(ps.mean()), 4), "sd": round(float(ps.std()), 4),
                "ic95": [round(float(v), 4) for v in np.percentile(ps, [2.5, 97.5])],
                "analytique_mean": round(float(ana.mean()), 4),
                "analytique_ic95": [round(float(v), 4) for v in ana.ppf([.025, .975])],
                "rhat": round(float(az.rhat(idata)["p"].values), 5),
                "ess": round(float(az.ess(idata)["p"].values), 1),
                "seconds": round(el1, 1), "draws": DRAWS * N_CHAINS}

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0),
                         gridspec_kw={"width_ratios": [1.5, 1.1, 1.1]})
g = np.linspace(1e-3, 1 - 1e-3, 500)
axes[0].hist(ps, bins=70, density=True, color=C["blue"], alpha=.5, edgecolor="white",
             label=f"échantillons MCMC ({K['mcmc_bb']['draws']} tirages)")
axes[0].plot(g, ana.pdf(g), color=C["red"], lw=2.0, label="Beta(8, 17) analytique")
axes[0].set_xlabel("$p$"); axes[0].set_ylabel("densité")
axes[0].set_title("Le posterior conjugué est retrouvé par NUTS", fontsize=9.8)
axes[0].legend(fontsize=7.4)
axes[1].hist(ps, bins=30, density=True, cumulative=True, color=C["blue"], alpha=.5,
             edgecolor="white", histtype="stepfilled", label="MCMC")
axes[1].plot(g, ana.cdf(g), color=C["red"], lw=2.0, label="analytique")
axes[1].set_xlabel("$p$"); axes[1].set_ylabel("fonction de répartition")
axes[1].set_title("Même loi, deux chemins", fontsize=9.8); axes[1].legend(fontsize=7.4)
for ch_i in range(N_CHAINS):
    axes[2].plot(idata.posterior["p"].values[ch_i], lw=.5,
                 color=[C["blue"], C["orange"]][ch_i], label=f"chaîne {ch_i + 1}")
axes[2].set_xlabel("itération"); axes[2].set_yticks([])
axes[2].set_title(f"Traces — $\\hat R$ = {K['mcmc_bb']['rhat']}", fontsize=9.8)
axes[2].legend(fontsize=7.4)
fig.suptitle(f"g07 — Analytique vs échantillonnage PyMC : {el1:.1f} s de calcul, ESS = {K['mcmc_bb']['ess']:.0f}",
             fontsize=10.4, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g07_analytique_vs_pymc.png")
json.dump(K, open(os.path.join(FIGDIR, "key_numbers_ch2.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)   # sauvegarde intermediaire
print(f"[t] modele Beta-Binomial termine en {time.time()-t0:.1f} s", flush=True)

# =====================================================================
# g11 : controle predictif a priori en PyMC -- modele normal (8 ecoles)
# =====================================================================
y_schools = np.array([28.39, 7.94, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16])
se_schools = np.array([14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6])
K["g11_ecoles"] = {"y": [float(v) for v in y_schools], "se": [float(v) for v in se_schools],
                   "moyenne_brute": round(float(y_schools.mean()), 3)}

NDRAW = 4000
predictifs = {}
for tag, prior_mu, prior_sig in [
        ("large (mu ~ N(50, 100²), sigma ~ U(0, 100))", (50, 100), "uniform"),
        ("calibre (mu ~ N(0, 20²), sigma ~ HN(10))", (0, 20), "halfnorm")]:
    with pm.Model() as m_pp:
        mu = pm.Normal("mu", mu=prior_mu[0], sigma=prior_mu[1])
        if prior_sig == "uniform":
            sigma = pm.Uniform("sigma", 0, 100)
        else:
            sigma = pm.HalfNormal("sigma", 10)
        pm.Normal("y", mu=mu, sigma=sigma, shape=8)
        pr = pm.sample_prior_predictive(draws=NDRAW, random_seed=42)
    ypred = pr.prior["y"].values.reshape(-1, 8)
    predictifs[tag] = ypred
    K[f"prior_predictive_{prior_sig}"] = {
        "q05": round(float(np.percentile(ypred, 5)), 2),
        "q95": round(float(np.percentile(ypred, 95)), 2),
        "P_abs_gt_100": round(float((np.abs(ypred) > 100).mean()), 4),
        "P_abs_gt_200": round(float((np.abs(ypred) > 200).mean()), 4),
        "amplitude_p99": round(float(np.percentile(np.abs(ypred), 99)), 1)}

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
for (tag, yp), col in zip(predictifs.items(), [C["red"], C["blue"]]):
    axes[0].hist(yp.ravel(), bins=90, density=True, histtype="step", lw=1.9, color=col,
                 label=tag.split(" (")[0] + f" — 90 % dans [{fr(np.percentile(yp, 5), 0)} ; {fr(np.percentile(yp, 95), 0)}]")
axes[0].axvspan(y_schools.min(), y_schools.max(), color=C["green"], alpha=.18,
                label=f"plage des 8 valeurs observées [{fr(y_schools.min(), 1)} ; {fr(y_schools.max(), 1)}]")
axes[0].set_xlim(-350, 350)
axes[0].set_xlabel("effet d'un programme de préparation, en points SAT")
axes[0].set_ylabel("densité prédictive a priori")
axes[0].set_title("Deux priors, deux mondes prédits", fontsize=10.0)
axes[0].legend(fontsize=7.0)
for (tag, yp), col in zip(predictifs.items(), [C["red"], C["blue"]]):
    ks = np.sort(np.abs(yp).ravel())
    axes[1].plot(np.linspace(0, 1, ks.size), ks, color=col, lw=1.9,
                 label=tag.split(" (")[0])
axes[1].set_yscale("log"); axes[1].set_xlabel("quantile prédictif")
axes[1].set_ylabel("$|y|$ prédit (log)")
axes[1].set_title("Le prior uniforme sur $\\sigma$ prédit des effets absurdes", fontsize=10.0)
axes[1].legend(fontsize=7.6)
fig.suptitle(f"g11 — Contrôle prédictif a priori exécuté en PyMC ({NDRAW} tirages par prior, sample_prior_predictive)",
             fontsize=10.2, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g11_prior_predictive_pymc.png")
json.dump(K, open(os.path.join(FIGDIR, "key_numbers_ch2.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)   # sauvegarde intermediaire
print(f"[t] prior predictive termine en {time.time()-t0:.1f} s", flush=True)

# =====================================================================
# g13 : prior hierarchique -- shrinkage de 8 groupes vers la moyenne
# =====================================================================
with pm.Model() as m_h:
    mu = pm.Normal("mu", 0, 25)
    tau = pm.HalfNormal("tau", 10)
    z = pm.Normal("z", 0, 1, shape=8)
    theta = pm.Deterministic("theta", mu + tau * z)
    pm.Normal("y", mu=theta, sigma=se_schools, observed=y_schools)
    id_h = pm.sample(draws=2000, tune=1500, chains=2, cores=1, target_accept=.9,
                     progressbar=False, random_seed=42)
el3 = time.time() - t0
th = id_h.posterior["theta"].values.reshape(-1, 8)
mu_s = id_h.posterior["mu"].values.ravel()
tau_s = id_h.posterior["tau"].values.ravel()
post_mean = th.mean(0)
post_lo, post_hi = np.percentile(th, [2.5, 97.5], axis=0)
K["hier"] = {"mu_mean": round(float(mu_s.mean()), 3), "mu_ic95": [round(float(v), 3) for v in np.percentile(mu_s, [2.5, 97.5])],
             "tau_mean": round(float(tau_s.mean()), 3), "tau_ic95": [round(float(v), 3) for v in np.percentile(tau_s, [2.5, 97.5])],
             "post_mean": [round(float(v), 3) for v in post_mean],
             "post_ic95": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(post_lo, post_hi)],
             "rhat_mu": round(float(az.rhat(id_h)["mu"].values), 5),
             "rhat_tau": round(float(az.rhat(id_h)["tau"].values), 5),
             "ess_min": round(float(az.ess(id_h)[["mu", "tau"]].to_array().values.min()), 1),
             "epaisseur_brute": round(float((2 * 1.96 * se_schools).mean()), 2),
             "epaisseur_post": round(float((post_hi - post_lo).mean()), 2),
             "shrinkage": [round(float(1 - (post_mean[j] - mu_s.mean()) / (y_schools[j] - mu_s.mean())), 3)
                           for j in range(8)],
             "seconds": round(el3, 1)}

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5))
xs = np.arange(8)
axes[0].errorbar(xs - .13, y_schools, yerr=1.96 * se_schools, fmt="o", color=C["grey"],
                 capsize=3, ms=5, lw=1.6, label="estimation brute $y_j \\pm 1{,}96\\,se_j$")
axes[0].errorbar(xs + .13, post_mean, yerr=[post_mean - post_lo, post_hi - post_mean], fmt="o",
                 color=C["blue"], capsize=3, ms=5, lw=1.6, label="moyenne a posteriori (pooling partiel)")
axes[0].axhline(mu_s.mean(), color=C["red"], ls="--", lw=1.4,
                label=f"$\\mu$ = {fr(mu_s.mean(), 2)} (population)")
axes[0].set_xticks(xs); axes[0].set_xticklabels([f"école {chr(65+j)}" for j in range(8)], fontsize=8)
axes[0].set_ylabel("effet estimé (points SAT)")
axes[0].set_title("Les estimations extrêmes sont ramenées vers la moyenne", fontsize=10.0)
axes[0].legend(fontsize=7.2)
axes[1].scatter(se_schools, 1 - (post_mean - mu_s.mean()) / (y_schools - mu_s.mean()),
                color=C["blue"], s=42, zorder=3)
for j in range(8):
    axes[1].annotate(chr(65 + j), (se_schools[j], 1 - (post_mean[j] - mu_s.mean()) / (y_schools[j] - mu_s.mean())),
                     textcoords="offset points", xytext=(6, -3), fontsize=8.5, fontweight="bold")
tt = np.linspace(8, 20, 100)
axes[1].plot(tt, tt ** 2 / (tt ** 2 + tau_s.mean() ** 2), color=C["red"], lw=1.8, ls="--",
             label=f"théorie $se^2/(se^2+\\tau^2)$, $\\tau$ = {fr(tau_s.mean(), 2)}")
axes[1].set_xlabel("erreur standard $se_j$")
axes[1].set_ylabel("facteur de contraction vers $\\mu$")
axes[1].set_title("Plus un groupe est bruité, plus il est contracté", fontsize=10.0)
axes[1].legend(fontsize=7.6)
fig.suptitle(f"g13 — Prior hiérarchique et shrinkage (8 écoles, NUTS : {el3:.0f} s, $\\hat R$ max = {max(K['hier']['rhat_mu'], K['hier']['rhat_tau'])})",
             fontsize=10.2, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "g13_shrinkage_8groupes.png")

out = os.path.join(FIGDIR, "key_numbers_ch2.json")
json.dump(K, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n=== CH2 (PyMC) ===")
print(json.dumps({k: v for k, v in K.items() if k in ("mcmc_bb", "g11_ecoles", "hier",
                                                      "prior_predictive_uniform", "prior_predictive_halfnorm")},
                 indent=1, ensure_ascii=False))
print(f"total {time.time()-t0:.1f} s")
