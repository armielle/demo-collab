"""Chapitre 1 -- regression lineaire bayesienne reelle (12 points) + prediction."""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
import numpy as np
import pymc as pm
import arviz as az
from figstyle import plt, C, save, FIGDIR

rng = np.random.default_rng(4112)
n = 12
x = np.linspace(0, 10, n)
y = 2.0 + 0.7 * x + rng.normal(0, 1.0, n)
K = {"data": {"x": [round(float(v), 2) for v in x], "y": [round(float(v), 2) for v in y]}}
ols = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]
K["ols"] = [round(float(v), 3) for v in ols]

t0 = time.time()
with pm.Model() as m:
    b0 = pm.Normal("b0", mu=0, sigma=5)          # faiblement informatif
    b1 = pm.Normal("b1", mu=0.5, sigma=1)        # faiblement informatif
    sigma = pm.HalfNormal("sigma", sigma=2)
    mu = b0 + b1 * x
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    idata = pm.sample(draws=2000, tune=1500, chains=2, cores=1,
                      target_accept=0.9, progressbar=False, random_seed=11)
el = time.time() - t0

p = idata.posterior
K["b0_mean"], K["b0_hdi"] = float(p["b0"].mean()), [round(float(v), 3) for v in
                                                    np.percentile(p["b0"].values, [2.5, 97.5])]
K["b1_mean"], K["b1_hdi"] = float(p["b1"].mean()), [round(float(v), 3) for v in
                                                    np.percentile(p["b1"].values, [2.5, 97.5])]
K["sigma_mean"] = float(p["sigma"].mean())
K["rhat_max"] = float(max(az.rhat(idata)[v].values for v in ["b0", "b1", "sigma"]))
K["seconds"] = round(el, 1)
K["draws"] = 4000

# prediction sur la grille
xs = np.linspace(-0.5, 11, 120)
b0s = p["b0"].values.ravel()
b1s = p["b1"].values.ravel()
pred = b0s[:, None] + b1s[:, None] * xs[None, :]
lo, hi = np.percentile(pred, [2.5, 97.5], axis=0)
mid = pred.mean(axis=0)

fig = plt.figure(figsize=(9.8, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1, 1], wspace=.30)
ax = fig.add_subplot(gs[0])
ax.fill_between(xs, lo, hi, color=C["blue"], alpha=.20, label="crédibilité 95 % de la droite")
ax.plot(xs, mid, color=C["blue"], lw=2.2, label=f"droite moyenne : {K['b0_mean']:.2f} + {K['b1_mean']:.3f} x")
ax.plot(x, y, "o", color=C["deep"], ms=6.5, label=f"{n} points observés")
ax.plot(xs, ols[0] + ols[1] * xs, color=C["orange"], lw=1.3, ls="--", label="MCO (fréquentiste)")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.legend(fontsize=7.6, loc="upper left")
ax.set_title("Posterior de la droite de régression", fontsize=10)
for k, (nm, col) in enumerate([("b0", C["green"]), ("b1", C["orange"])]):
    a = fig.add_subplot(gs[k + 1])
    v = p[nm].values.ravel()
    a.hist(v, bins=50, color=col, alpha=.80, edgecolor="white", density=True)
    a.axvline(v.mean(), color=C["red"], lw=1.6)
    a.set_title(f"{nm} : {v.mean():.3f}\n[{np.percentile(v,2.5):.3f} ; {np.percentile(v,97.5):.3f}]",
                fontsize=9)
    a.set_yticks([])
    a.set_xlabel("$" + nm + "$")
fig.suptitle("f14 — Régression linéaire bayésienne (PyMC, 4 000 tirages, %.0f s CPU)"
             % el, fontsize=10.6, fontweight="bold", y=1.04)
save(fig, "f14_regression_bayesienne.png")

# posterior predictive : 30 nouvelles mesures a x = 8
x_new = 8.0
pred_p = (b0s + b1s * x_new)[:, None] + rng.normal(0, p["sigma"].values.ravel()[:, None], (len(b0s), 200))
K["pred_x8_mean"] = float(pred_p.mean())
K["pred_x8_ci"] = [round(float(v), 3) for v in np.percentile(pred_p.mean(axis=1), [2.5, 97.5])]
K["pred_x8_pi"] = [round(float(v), 3) for v in np.percentile(pred_p.ravel(), [2.5, 97.5])]

fig, ax = plt.subplots(figsize=(7.4, 3.1))
ax.hist(pred_p.ravel(), bins=70, color=C["cyan"], alpha=.75, edgecolor="white", density=True,
        label="prédictive a posteriori pour une nouvelle mesure")
ax.axvline(K["pred_x8_mean"], color=C["red"], lw=1.8, label=f"moyenne prédite {K['pred_x8_mean']:.2f}")
for v in K["pred_x8_pi"]:
    ax.axvline(v, color=C["deep"], ls="--", lw=1.2)
ax.annotate(f"intervalle prédictif 95 %\n[{K['pred_x8_pi'][0]:.2f} ; {K['pred_x8_pi'][1]:.2f}]",
            (K["pred_x8_pi"][1], ax.get_ylim()[1] * .72), xytext=(-8, 0),
            textcoords="offset points", ha="right", fontsize=8.6, fontweight="bold",
            color=C["deep"])
ax.set_xlabel("valeur prédite de $y$ en $x = 8$"); ax.set_ylabel("densité")
ax.set_title("f15 — Prédiction : l'incertitude du paramètre ET du bruit")
ax.legend(fontsize=8.4, loc="upper left")
save(fig, "f15_prediction.png")

json.dump(K, open(os.path.join(FIGDIR, "key_numbers_ch1d.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(json.dumps(K, indent=1, ensure_ascii=False))
