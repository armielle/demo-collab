"""Chapitre 5 -- partie PyMC reelle : diagnostics, posterior predictive,
forest plot et prior predictive sur les donnees radon du Minnesota.

Produit figures/j10_*.png ... j14_*.png et complete figures/key_numbers_ch5.json.
Branche les 6 etapes du mini-projet guide (section 5.5).
"""
import os, json, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import pymc as pm
import arviz as az
from figstyle import plt, C, save, FIGDIR

t0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KF = os.path.join(FIGDIR, "key_numbers_ch5.json")
K = json.load(open(KF, encoding="utf-8")) if os.path.exists(KF) else {"chapitre": 5}
# cout de calcul reglable par variable d'environnement (machine sans compilateur C)
DRAWS = int(os.environ.get("CH5_DRAWS", "1000"))
TUNE = int(os.environ.get("CH5_TUNE", "1000"))
NCOUNTY = int(os.environ.get("CH5_NCOUNTY", "12"))

# ------------------------------------------------------------------ donnees
df = pd.read_csv(os.path.join(ROOT, "src", "data", "srrs2.dat"), skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
df["state"] = df["state"].astype(str).str.strip()
df["county"] = df["county"].astype(str).str.strip()
df["activity"] = pd.to_numeric(df["activity"], errors="coerce")
df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
mn = df[(df.state == "MN") & df.activity.notna()].copy()
top = list(mn.county.value_counts().head(NCOUNTY).index)
sub = mn[mn.county.isin(top)].copy()
sub["logradon"] = np.log(sub.activity.clip(lower=0.1))
sub["x"] = (sub.floor == 1).astype(float)
counties = np.array(sorted(top))
idx = pd.Categorical(sub.county, categories=list(counties)).codes
y = sub.logradon.values
x = sub.x.values
NC = len(counties)
K["radon_donnees"] = {
    "fichier": "src/data/srrs2.dat", "etat": "MN", "n_comtes_total": int(mn.county.nunique()),
    "n_obs_total": int(len(mn)), "n_comtes_retenus": NC, "n_obs_retenus": int(len(sub)),
    "comptes": [str(c) for c in counties],
    "n_premier_etage": int(x.sum()), "n_sous_sol": int((1 - x).sum()),
    "log_radon_moyen": round(float(y.mean()), 4),
    "moyenne_brute_premier_etage": round(float(y[x == 1].mean()), 4),
    "moyenne_brute_sous_sol": round(float(y[x == 0].mean()), 4)}

# ------------------------------------------------- etape 2-3 : priors + modele
NOM = "radon_hiérarchique"
with pm.Model() as modele:
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=5.0)
    sigma_a = pm.HalfNormal("sigma_a", sigma=2.0)
    b = pm.Normal("b", mu=0.0, sigma=3.0)
    a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=NC)
    sigma_y = pm.HalfNormal("sigma_y", sigma=2.0)
    mu = a[idx] + b * x
    pm.Normal("y", mu=mu, sigma=sigma_y, observed=y)
    prior = pm.sample_prior_predictive(draws=500, random_seed=4)

prior_y = prior.prior_predictive["y"].values.ravel()
K["radon_priors"] = {
    "mu_a": "Normal(0, 5)", "sigma_a": "HalfNormal(2)", "b": "Normal(0, 3)",
    "sigma_y": "HalfNormal(2)", "n_simulations_prior": 500,
    "log_radon_prior_min": round(float(np.percentile(prior_y, 1)), 2),
    "log_radon_prior_max": round(float(np.percentile(prior_y, 99)), 2),
    "pct_prior_au_dessus_de_10_pCi": round(100 * float((np.exp(prior_y) > 10).mean()), 1)}

fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.1))
axs[0].hist(y, bins=45, color=C["blue"], alpha=.72, density=True, edgecolor="white",
            label=f"donnees observees ($n$ = {len(y)})")
axs[1].hist(prior_y, bins=60, color=C["orange"], alpha=.72, density=True, edgecolor="white",
            label="predictive a priori (500 tirages)")
axs[1].axvline(np.log(10), color=C["red"], ls="--", lw=1.6, label="10 pCi/L (seuil d'action)")
axs[0].axvline(np.log(10), color=C["red"], ls="--", lw=1.6, label="10 pCi/L (seuil d'action)")
axs[0].set_xlim(-4, 6); axs[1].set_xlim(-14, 14)
for a_, t in zip(axs, ["Predictive a priori du prior choisi", "Log-radon simule par le prior"]):
    a_.set_xlabel("log(radon) [pCi/L]"); a_.set_ylabel("densite")
    a_.set_title(t, fontsize=10); a_.legend(fontsize=7.4)
fig.suptitle("Etape 3 -- verification predictive a priori : le prior est-il plausible ?",
             fontsize=11, y=1.03)
save(fig, "j13_prior_predictive.png")

# ---------------------------------------------- etape 4 : echantillonnage NUTS
t1 = time.time()
with modele:
    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=2, cores=1, target_accept=0.9,
                      random_seed=7, progressbar=False)
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=9)
t_sample = time.time() - t1
post = idata.posterior
A = post["a"].values.reshape(-1, NC)
B = post["b"].values.ravel()
MUA = post["mu_a"].values.ravel()
SIGA = post["sigma_a"].values.ravel()
SIGY = post["sigma_y"].values.ravel()

rv = ["mu_a", "sigma_a", "b", "sigma_y"]
rhat = az.rhat(idata)
ess = az.ess(idata)
K["radon_posterior"] = {
    "n_draws_par_chaine": DRAWS, "n_draws_total": 2 * DRAWS, "n_chains": 2,
    "tune": TUNE, "target_accept": 0.9,
    "secondes_echantillonnage": round(t_sample, 1),
    "divergences": int(idata.sample_stats["diverging"].values.sum()),
    "acc_par_chaine": [round(float(v), 3) for v in idata.sample_stats["acceptance_rate"].values.mean(axis=1)],
    "n_evaluations_fonction": int(idata.sample_stats["n_steps"].values.sum()),
    "mu_a_moy": round(float(MUA.mean()), 4),
    "mu_a_ic": [round(float(v), 4) for v in np.percentile(MUA, [3, 97])],
    "sigma_a_moy": round(float(SIGA.mean()), 4),
    "sigma_a_ic": [round(float(v), 4) for v in np.percentile(SIGA, [3, 97])],
    "b_moy": round(float(B.mean()), 4),
    "b_ic": [round(float(v), 4) for v in np.percentile(B, [3, 97])],
    "rapport_etage": round(float(np.exp(B.mean())), 4),
    "sigma_y_moy": round(float(SIGY.mean()), 4),
    "rhat_max": round(float(max(rhat[v].values.max() for v in rv)), 5),
    "rhat_max_a": round(float(rhat["a"].values.max()), 5),
    "ess_min": round(float(min(ess[v].values.min() for v in rv)), 1),
    "ess_min_a": round(float(ess["a"].values.min()), 1)}

# ---------------------------------------------------- etape 5 : diagnostics
fig = plt.figure(figsize=(10.4, 4.6))
gs = fig.add_gridspec(2, 3, hspace=.62, wspace=.30)
traces = [("mu_a", MUA.reshape(2, -1)), ("sigma_a", SIGA.reshape(2, -1)), ("b", B.reshape(2, -1))]
for j, (nm, arr) in enumerate(traces):
    a_ = fig.add_subplot(gs[0, j])
    for ch, col in zip(range(2), [C["blue"], C["orange"]]):
        a_.plot(arr[ch], color=col, lw=.7, alpha=.85, label=f"chaine {ch+1}")
    a_.set_title(f"Trace de {nm}", fontsize=9.5)
    a_.set_xlabel("iteration apres burn-in", fontsize=8)
    if j == 0:
        a_.legend(fontsize=7)
pooled = np.concatenate([MUA.reshape(2, -1)[0], MUA.reshape(2, -1)[1]])
ranks = stats.rankdata(pooled)
a_ = fig.add_subplot(gs[1, 0])
for ch, col in zip(range(2), [C["blue"], C["orange"]]):
    a_.hist(ranks[ch * 1000:(ch + 1) * 1000], bins=np.linspace(0, 2000, 26), color=col,
            alpha=.55, density=True, label=f"chaine {ch+1}")
a_.axhline(1 / 2000, color=C["red"], ls="--", lw=1.2, label="uniforme attendu")
a_.set_title("Rank plot de $\\mu_a$", fontsize=9.5); a_.set_xlabel("rang du tirage", fontsize=8)
a_.legend(fontsize=6.6)
a_ = fig.add_subplot(gs[1, 1])
vals = [float(rhat[v].values.max()) for v in rv] + [float(rhat["a"].values.max())]
a_.bar(range(len(vals)), vals, color=C["purple"], width=.6)
a_.axhline(1.01, color=C["red"], ls="--", lw=1.6, label="seuil 1,01")
a_.set_xticks(range(len(vals))); a_.set_xticklabels(rv + ["a (comtes)"], fontsize=7.5)
a_.set_ylim(.99, max(vals) * 1.0006)
a_.set_title(f"$\\hat R$ maximal = {max(vals):.5f}", fontsize=9.5)
a_.set_ylabel("$\\hat R$", fontsize=8); a_.legend(fontsize=7)
a_ = fig.add_subplot(gs[1, 2])
ev = [float(ess[v].values.min()) for v in rv] + [float(ess["a"].values.min())]
a_.bar(range(len(ev)), ev, color=C["cyan"], width=.6)
a_.axhline(200, color=C["red"], ls="--", lw=1.4, label="seuil usuel 200")
a_.set_xticks(range(len(ev))); a_.set_xticklabels(rv + ["a (comtes)"], fontsize=7.5)
a_.set_title("ESS minimal par parametre", fontsize=9.5)
a_.set_ylabel("ESS (bulk)", fontsize=8); a_.legend(fontsize=7)
fig.suptitle(f"Etape 5 -- diagnostics de convergence PyMC sur le modele radon "
             f"({t_sample:.0f} s, {K['radon_posterior']['divergences']} divergence)",
             fontsize=11, y=.99)
save(fig, "j10_pymc_diagnostics.png")

# --------------------------------------------- etape 6 : posterior predictive
pp = idata.posterior_predictive["y"].values.reshape(-1, len(y))
lo, hi = np.percentile(pp, [3, 97], axis=0)
med = np.median(pp, axis=0)
cover = float(((y >= lo) & (y <= hi)).mean())
K["radon_ppc"] = {"n_tirages_predictifs": int(pp.shape[0]),
                  "couverture_94pct": round(cover, 4),
                  "moyenne_observee": round(float(y.mean()), 4),
                  "moyenne_predite": round(float(pp.mean()), 4),
                  "sd_observee": round(float(y.std()), 4),
                  "sd_predite": round(float(pp.std()), 4)}

fig = plt.figure(figsize=(10.2, 3.4))
gs = fig.add_gridspec(1, 3, wspace=.30)
a_ = fig.add_subplot(gs[0])
a_.hist(pp.ravel()[:120_000], bins=60, density=True, color=C["cyan"], alpha=.55,
        label="log-radon predit")
a_.hist(y, bins=60, density=True, histtype="step", color=C["deep"], lw=2,
        label="log-radon observe")
a_.set_xlabel("log(radon)"); a_.set_ylabel("densite")
a_.set_title("Distributions : predit contre observe", fontsize=9.5); a_.legend(fontsize=7.4)
a_ = fig.add_subplot(gs[1])
o = np.argsort(y)
a_.errorbar(np.arange(len(y)), med[o], yerr=[med[o] - lo[o], hi[o] - med[o]],
            fmt="none", ecolor=C["grey"], elinewidth=.6, alpha=.7, label="intervalle 94 %")
a_.plot(np.arange(len(y)), y[o], "o", ms=2.6, color=C["red"], label="observe")
a_.set_xlabel("observations triees"); a_.set_ylabel("log(radon)")
a_.set_title(f"{cover*100:.1f} % des points dans l'intervalle", fontsize=9.5)
a_.legend(fontsize=7.4)
a_ = fig.add_subplot(gs[2])
a_.scatter(pp.mean(axis=0), y, s=6, alpha=.5, color=C["blue"])
lim = [min(y.min(), pp.mean()), max(y.max(), pp.mean())]
a_.plot(lim, lim, "--", color=C["red"], lw=1.4, label="y = $\\hat y$")
a_.set_xlabel("moyenne predite"); a_.set_ylabel("observe")
a_.set_title("Calibration du modele", fontsize=9.5); a_.legend(fontsize=7.4)
fig.suptitle("Etape 6 -- controle predictif a posteriori (posterior predictive check)",
             fontsize=11, y=1.03)
save(fig, "j11_ppc_radon.png")

# ------------------------------------------------------- forest plot comtes
nopool = np.array([y[idx == j].mean() for j in range(NC)])
fullpool = y.mean()
order = np.argsort(nopool)
moy = A.mean(axis=0)
ic = np.percentile(A, [3, 97], axis=0)
K["radon_comtes"] = {
    "comte_plus_eleve": str(counties[order[-1]]), "comte_plus_bas": str(counties[order[0]]),
    "shrinkage_moyen": round(float(np.mean(1 - np.sqrt(A.var(axis=0)) / max(nopool.std(), 1e-9))), 3),
    "ecart_type_partiel_moyen": round(float(np.mean(A.std(axis=0))), 4),
    "ecart_type_sans_pooling_moyen": round(float(np.mean([y[idx == j].std() / np.sqrt((idx == j).sum())
                                                          for j in range(NC)])), 4),
    "mu_a_moyenne": round(float(y.mean()), 4),
    "intervalle_partiel_min": round(float(moy.min()), 3),
    "intervalle_partiel_max": round(float(moy.max()), 3),
    "intervalle_brut_min": round(float(nopool.min()), 3),
    "intervalle_brut_max": round(float(nopool.max()), 3)}

fig, ax = plt.subplots(figsize=(9.8, 3.8))
yy = np.arange(NC)
ax.errorbar(moy[order], yy, xerr=[moy[order] - ic[0][order], ic[1][order] - moy[order]],
            fmt="o", ms=5.5, color=C["blue"], ecolor=C["grey"], elinewidth=1.4,
            capsize=2.5, label="posterior hierarchique (intervalle 94 %)")
ax.plot(nopool[order], yy, "s", ms=4.6, color=C["orange"], alpha=.85,
        label="moyenne brute du comte (sans pooling)")
ax.axvline(fullpool, color=C["red"], ls="--", lw=1.8,
           label=f"pooling total : {fullpool:.2f}")
ax.axvline(MUA.mean(), color=C["green"], ls=":", lw=1.8,
           label=f"moyenne hierarchique : {MUA.mean():.2f}")
ax.set_yticks(yy)
ax.set_yticklabels([f"{counties[o]} ({np.sum(idx == o)})" for o in order], fontsize=8)
ax.set_xlabel("intercept de comte $a_j$ (log-radon, sous-sol)")
ax.set_title("Etape 6 -- effets de comte : le << shrinkage >> des petits comtes", fontsize=10.5)
ax.legend(fontsize=7.6, loc="lower right")
fig.suptitle("Modele hierarchique radon -- 12 comtes, "
             f"{K['radon_donnees']['n_obs_retenus']} mesures, 2 chaines x 1000 tirages",
             fontsize=11, y=1.02)
save(fig, "j12_forest_radon.png")

# ------------------------------------------- validation MCMC contre exact
n_obs, k_succ = 30, 22
rng = np.random.default_rng(3)
draws = rng.beta(1 + k_succ, 1 + n_obs - k_succ, 200_000)
with pm.Model() as petit:
    p = pm.Beta("p", alpha=1, beta=1)
    pm.Binomial("k", n=n_obs, p=p, observed=k_succ)
    idata2 = pm.sample(draws=1000, tune=1000, chains=2, cores=1, progressbar=False,
                       random_seed=12, target_accept=0.9)
pj = idata2.posterior["p"].values.ravel()
ex = dict(mean=(1 + k_succ) / (1 + n_obs + 1),
          q=[float(v) for v in stats.beta.ppf([.03, .5, .97], 1 + k_succ, 1 + n_obs - k_succ)])
mc = dict(mean=float(pj.mean()), q=[float(v) for v in np.percentile(pj, [3, 50, 97])])
K["validation_exacte"] = {"n": n_obs, "k": k_succ,
                          "posterior_exact": "Beta(23, 9)",
                          "moyenne_exacte": round(ex["mean"], 5),
                          "moyenne_mcmc": round(mc["mean"], 5),
                          "ecart_absolu_moyenne": round(abs(ex["mean"] - mc["mean"]), 5),
                          "quantiles_exacts": [round(v, 4) for v in ex["q"]],
                          "quantiles_mcmc": [round(v, 4) for v in mc["q"]],
                          "rhat_p": round(float(az.rhat(idata2)["p"].values.max()), 5),
                          "ess_p": round(float(az.ess(idata2)["p"].values.min()), 1)}

fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.2))
axs[0].hist(pj, bins=55, density=True, color=C["purple"], alpha=.6, edgecolor="white",
            label="tirages MCMC (PyMC, NUTS)")
g = np.linspace(.35, .95, 400)
axs[0].plot(g, stats.beta.pdf(g, 1 + k_succ, 1 + n_obs - k_succ), color=C["deep"], lw=2.2,
            label="posterior exact Beta(23, 9)")
axs[0].set_xlabel("$p$"); axs[0].set_ylabel("densite")
axs[0].set_title(f"MCMC contre calcul exact ({k_succ} succes / {n_obs})", fontsize=10)
axs[0].legend(fontsize=7.6)
lab = ["q3 %", "mediane", "q97 %"]
xx = np.arange(3)
axs[1].bar(xx - .18, ex["q"], width=.36, color=C["deep"], alpha=.85, label="exact")
axs[1].bar(xx + .18, mc["q"], width=.36, color=C["cyan"], alpha=.85, label="MCMC")
for i in range(3):
    axs[1].text(i - .18, ex["q"][i] + .005, f"{ex['q'][i]:.3f}", ha="center", fontsize=7.6)
    axs[1].text(i + .18, mc["q"][i] + .005, f"{mc['q'][i]:.3f}", ha="center", fontsize=7.6)
axs[1].set_xticks(xx); axs[1].set_xticklabels(lab)
axs[1].set_title("Quantiles : accord a la 3e decimale", fontsize=10)
axs[1].legend(fontsize=8)
fig.suptitle("Quand Monte-Carlo suffit : le posterior exact sert de reference a MCMC",
             fontsize=11, y=1.03)
save(fig, "j14_mcmc_vs_exact.png")

K["secondes_pymc"] = round(time.time() - t0, 1)
json.dump(K, open(KF, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[key] key_numbers_ch5.json complete -- total {K['secondes_pymc']} s")
