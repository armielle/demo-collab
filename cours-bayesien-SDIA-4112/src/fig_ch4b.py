"""Chapitre 4 -- figures generees par du code réellement execute (partie 2).

Sorties : figures/i07..i11*.png + fusion dans figures/key_numbers_ch4.json

Sections couvertes : 4.2 critères hors échantillon (log-score prédictif, DIC,
WAIC, LOO-PSIS, poids de modèles, moyennage), 4.3 decision bayésienne
(risque a posteriori, seuil optimal, ROPE), 4.4 A/B testing séquentiel.
PyMC 5.28 + ArviZ 0.23, CPU only, chains = 2, cores = 1, pas de compilateur C.
"""
import json
import os

import numpy as np
import pymc as pm
import arviz as az
from scipy import stats
from figstyle import plt, C, save, FIGDIR

rng = np.random.default_rng(4112)
KDIR = os.path.join(FIGDIR, "key_numbers_ch4.json")
K = json.load(open(KDIR, encoding="utf-8")) if os.path.exists(KDIR) else {}


def add(d, **kw):
    d.update(kw)
    json.dump(d, open(KDIR, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# =====================================================================
# i07 / i08 -- trois modèles, WAIC / DIC / LOO réels, poids, moyennage
# =====================================================================
# données : 8 equipes, taux de succes heterogenes (genere, puis fige)
P_TRUE = np.array([0.42, 0.51, 0.55, 0.58, 0.63, 0.68, 0.72, 0.78])
N_OBS = np.array([35, 40, 30, 45, 40, 35, 30, 40])
Y_OBS = rng.binomial(N_OBS, P_TRUE)
p_hat = Y_OBS / N_OBS
print("[i07] données :", list(zip(N_OBS.tolist(), Y_OBS.tolist())))

NDRAW, NTUNE = 500, 500
SAMP = dict(draws=NDRAW, tune=NTUNE, chains=2, cores=1, progressbar=False,
            compute_convergence_checks=False, random_seed=42)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ch4b_cache")
os.makedirs(CACHE, exist_ok=True)
CACHE_PATHS = [os.path.join(CACHE, f"idata{i}.nc") for i in (1, 2, 3)]

if all(os.path.exists(p) for p in CACHE_PATHS):
    idata1, idata2, idata3 = (az.from_netcdf(p) for p in CACHE_PATHS)
    print("[cache] idata recharges depuis _ch4b_cache (échantillonnage non refait)")
else:
    with pm.Model() as m1:
        theta1 = pm.Beta("theta", 1.0, 1.0)
        pm.Binomial("y", n=N_OBS, p=theta1, observed=Y_OBS)
        idata1 = pm.sample(**SAMP)
        pm.compute_log_likelihood(idata1)

    with pm.Model() as m2:
        theta2 = pm.Beta("theta", 1.0, 1.0, shape=len(N_OBS))
        pm.Binomial("y", n=N_OBS, p=theta2, observed=Y_OBS)
        idata2 = pm.sample(**SAMP)
        pm.compute_log_likelihood(idata2)

    with pm.Model() as m3:
        mu_h = pm.Beta("mu", 2.0, 2.0)
        kappa_h = pm.HalfNormal("kappa", 5.0)
        theta3 = pm.Beta("theta", mu_h * kappa_h, (1 - mu_h) * kappa_h, shape=len(N_OBS))
        pm.Binomial("y", n=N_OBS, p=theta3, observed=Y_OBS)
        idata3 = pm.sample(**SAMP)
        pm.compute_log_likelihood(idata3)

    for _p, _id in zip(CACHE_PATHS, (idata1, idata2, idata3)):
        az.to_netcdf(_id, _p)
    print("[cache] idata sauvegardes dans _ch4b_cache")

MODELS = {"M1 regroupé": idata1, "M2 indépendant": idata2, "M3 hiérarchique": idata3}
cmp_loo = az.compare(MODELS, ic="loo")
cmp_waic = az.compare(MODELS, ic="waic")
cmp_stack = az.compare(MODELS, ic="loo", method="stacking")
print(cmp_loo[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"]].to_string())
print(cmp_waic[["rank", "elpd_waic", "p_waic", "elpd_diff", "dse", "weight"]].to_string())
print("stacking :", cmp_stack["weight"].to_dict())
pareto_k = {}
for k, v in MODELS.items():
    loo_k = az.loo(v)
    pk = np.asarray(loo_k.pareto_k.values).ravel()
    pareto_k[k] = dict(max_pareto_k=float(pk.max()),
                       n_sup_067=int((pk > 0.67).sum()), n_obs=int(pk.size),
                       good_k=float(loo_k.good_k))
    print(f"  pareto_k {k}: max = {pk.max():.3f}, n>0.67 = {(pk>0.67).sum()}/{pk.size}, "
          f"good_k = {loo_k.good_k:.2f}")
top = cmp_loo.index[0]
second = cmp_loo.index[1]
ep_diff = float(cmp_loo.loc[second, "elpd_diff"])
se_diff = float(cmp_loo.loc[second, "dse"])


def dic_of(idata, name):
    """DIC calcule à la main : DIC = -2 log p(y|thetabar) + 2 p_DIC."""
    ll = idata.log_likelihood["y"].values.reshape(-1, len(N_OBS))   # (draws, 8)
    mean_ll = float(ll.sum(axis=1).mean())
    th = idata.posterior["theta"].values.reshape(-1, len(N_OBS)).mean(axis=0)
    ll_bar = float(stats.binom.logpmf(Y_OBS, N_OBS, th).sum())
    p_dic = 2 * (ll_bar - mean_ll)
    return dict(name=name, loglik_at_mean=ll_bar, mean_loglik=mean_ll,
                p_dic=p_dic, dic=-2 * ll_bar + 2 * p_dic, p_eff=p_dic)


dic = {k: dic_of(v, k) for k, v in MODELS.items()}
print("DIC :", {k: round(v["dic"], 2) for k, v in dic.items()})

names = list(MODELS.keys())
fig = plt.figure(figsize=(12.2, 3.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.28], wspace=.32)

ax = fig.add_subplot(gs[0])
elp = [float(cmp_loo.loc[n, "elpd_loo"]) for n in names]
se = [float(cmp_loo.loc[n, "se"]) for n in names]
cols = [C["grey"], C["orange"], C["green"]]
ax.bar(range(3), elp, color=cols, alpha=.85)
ax.errorbar(range(3), elp, yerr=se, fmt="none", ecolor=C["deep"], capsize=5, lw=1.4)
for i, (v, s) in enumerate(zip(elp, se)):
    ax.text(i, v - abs(v) * .18, f"{v:.1f}\n$\\pm${s:.1f}", ha="center", fontsize=8.4,
            fontweight="bold", color="white")
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=8.0, rotation=12)
ax.set_ylabel("$elpd_{LOO}$ (log-score cumule)")
ax.set_title("LOO-PSIS : mieux vaut grand")
ax.axhline(0, color=C["deep"], lw=.8)

ax = fig.add_subplot(gs[1])
waic_v = [float(cmp_waic.loc[n, "elpd_waic"]) for n in names]
dic_v = [dic[n]["dic"] for n in names]
w = 0.36
ax.bar(np.arange(3) - w / 2, waic_v, w, color=C["blue"], alpha=.85, label="WAIC (elpd)")
ax.bar(np.arange(3) + w / 2, [-d / 4 for d in dic_v], w, color=C["red"], alpha=.85,
       label="$-DIC/4$ (mise a l'echelle)")
for i, (a, b) in enumerate(zip(waic_v, dic_v)):
    ax.text(i - w / 2, a - abs(a) * .22, f"{a:.1f}", ha="center", fontsize=8.0,
            fontweight="bold", color="white")
    ax.text(i + w / 2, -b / 4 - abs(b) / 4 * .22, f"{-b:.1f}", ha="center", fontsize=8.0,
            fontweight="bold", color="white")
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=8.0, rotation=12)
ax.set_title("WAIC et DIC : le même classement")
ax.legend(fontsize=7.8, loc="lower right")

ax = fig.add_subplot(gs[2])
rows = [["modèle", "log $p$", "$p_{DIC}$", "DIC", "$p_{LOO}$", "$elpd_{LOO}$", "poids"]]
for n in names:
    rows.append([n, f"{dic[n]['loglik_at_mean']:.2f}", f"{dic[n]['p_dic']:.2f}",
                 f"{dic[n]['dic']:.2f}", f"{float(cmp_loo.loc[n, 'p_loo']):.1f}",
                 f"{float(cmp_loo.loc[n, 'elpd_loo']):.1f}",
                 f"{float(cmp_loo.loc[n, 'weight'])*100:.1f}"])
ax.axis("off")
tab = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(7.6)
tab.scale(1, 1.55)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if r == 0:
        cell.set_facecolor(C["deep"])
        cell.set_text_props(color="white", fontweight="bold")
    elif r == names.index(top) + 1:
        cell.set_facecolor("#E8F5EE")
    else:
        cell.set_facecolor("white")
ax.set_title("Sortie réelle de az.compare() (+ DIC recalculé)", fontsize=9.8,
             fontweight="bold")
fig.suptitle("i07 -- Les trois critères ne désignent pas le même vainqueur : "
             "$\\Delta elpd_{LOO}(M_2,M_3) = %.2f$ pour un écart-type de %.2f"
             % (ep_diff, se_diff), fontsize=9.3, y=1.10)
save(fig, "i07_waic_dic_loo.png")

# --- i08 : poids et prédiction moyennée pour un nouveau groupe -------------
n_new = 40
th_grid = np.linspace(0.01, 0.99, 500)
th1 = idata1.posterior["theta"].values.ravel()
mu_d = idata3.posterior["mu"].values.ravel()
ka_d = idata3.posterior["kappa"].values.ravel()
a3 = np.clip(mu_d * ka_d, 1e-3, None)
b3 = np.clip((1 - mu_d) * ka_d, 1e-3, None)
NMIX = 1000
idx = rng.choice(len(th1), NMIX, replace=len(th1) < NMIX)
# M1 : un seul theta pour tous les groupes -> predictive concentree (dirac lisse)
th1s = th1[idx]
dens1 = stats.norm.pdf(th_grid[None, :], th1s[:, None], 0.012)
# M2 : theta indépendants -> predictive = prior Beta(1,1) (aucun partage d'information)
dens2 = np.repeat(stats.beta.pdf(th_grid, 1.0, 1.0)[None, :], 8, axis=0)
# M3 : theta nouveaux tires de Beta(mu*kappa, (1-mu)*kappa)
th3 = rng.beta(a3[idx], b3[idx])
dens3 = stats.beta.pdf(th_grid[None, :], a3[idx][:, None], b3[idx][:, None])
wts = np.array([float(cmp_loo.loc[n, "weight"]) for n in names])
wts_stack = np.array([float(cmp_stack.loc[n, "weight"]) for n in names])
avg = np.vstack([dens1.mean(axis=0), dens2.mean(axis=0), dens3.mean(axis=0)])
mix = (wts[:, None] * avg).sum(axis=0)
mix_stack = (wts_stack[:, None] * avg).sum(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))
ax = axes[0]
for d, n, col in zip(avg, names, [C["grey"], C["orange"], C["green"]]):
    ax.plot(th_grid, d, color=col, lw=2.0, label=f"{n} (poids {wts[names.index(n)]*100:.1f} %)")
ax.plot(th_grid, mix, color=C["blue"], lw=2.6, ls="-",
        label="mélange (poids LOO)")
ax.plot(th_grid, mix_stack, color=C["purple"], lw=2.0, ls="--",
        label="mélange (poids de stacking)")
ax.set_xlabel("$\\theta_{nouveau}$ (groupe non observe)")
ax.set_ylabel("densité predictive")
ax.set_title("Prédiction moyennée : le mélange n'est aucun des modèles")
ax.legend(fontsize=7.6, loc="upper center")

ax = axes[1]
ax.bar(np.arange(3) - .19, wts, .38, color=C["blue"], alpha=.85, label="poids pseudo-BMA")
ax.bar(np.arange(3) + .19, wts_stack, .38, color=C["purple"], alpha=.85,
       label="poids de stacking")
for i, (a, b) in enumerate(zip(wts, wts_stack)):
    ax.text(i - .19, a + .015, f"{a*100:.1f} %", ha="center", fontsize=8.6,
            fontweight="bold", color=C["blue"])
    ax.text(i + .19, b + .015, f"{b*100:.1f} %", ha="center", fontsize=8.6,
            fontweight="bold", color=C["purple"])
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=8.2)
ax.set_ylim(0, 1.12)
ax.set_ylabel("poids du modèle")
ax.set_title("Les poids dépendent de la méthode de calcul")
ax.legend(fontsize=8.0, loc="upper left")
fig.suptitle("i08 -- Moyennage de modèles : on predit par mélange, pas en choisissant "
             "un seul modèle", fontsize=9.3, y=1.06)
save(fig, "i08_model_averaging.png")

add(K, i07=dict(y_obs=Y_OBS.tolist(), n_obs=N_OBS.tolist(), p_hat=p_hat.tolist(),
                p_true=P_TRUE.tolist(), draw=NDRAW, tune=NTUNE,
                loo={n: dict(elpd=float(cmp_loo.loc[n, "elpd_loo"]),
                             se=float(cmp_loo.loc[n, "se"]),
                             p_loo=float(cmp_loo.loc[n, "p_loo"]),
                             elpd_diff=float(cmp_loo.loc[n, "elpd_diff"]),
                             weight=float(cmp_loo.loc[n, "weight"]),
                             rank=int(cmp_loo.loc[n, "rank"])) for n in names},
                waic={n: dict(elpd=float(cmp_waic.loc[n, "elpd_waic"]),
                              p_waic=float(cmp_waic.loc[n, "p_waic"]),
                              weight=float(cmp_waic.loc[n, "weight"]),
                              rank=int(cmp_waic.loc[n, "rank"])) for n in names},
                dic={n: dict(dic=v["dic"], p_dic=v["p_dic"],
                             loglik_at_mean=v["loglik_at_mean"]) for n, v in dic.items()},
                pareto_k=pareto_k,
                vainqueur_loo=top, second_loo=second,
                elpd_diff_loo_m2_m3=ep_diff, se_diff_loo_m2_m3=se_diff,
                stacking_weights={n: float(cmp_stack.loc[n, "weight"]) for n in names}))
add(K, i08=dict(weights_loo={n: float(v) for n, v in zip(names, wts)},
                weights_stacking={n: float(v) for n, v in zip(names, wts_stack)},
                n_new=int(n_new),
                overlap_m3_mix=float(np.minimum(avg[2], mix).sum() *
                                     (th_grid[1] - th_grid[0]))))

# =====================================================================
# i09 -- risque a posteriori, perte 0-1 / quadratique / asymétrique
# et seuil optimal derive analytiquement (dépistage)
# =====================================================================
a_p, b_p = 3.0, 7.0
post = stats.beta(a_p, b_p)
grid = np.linspace(0.005, 0.995, 900)
w = grid[1] - grid[0]
dens = post.pdf(grid)
mean_p = post.mean()
R2 = (stats.beta.moment(2, a_p, b_p) - 2 * grid * mean_p + grid ** 2)   # (theta - a)^2
Rabs = np.array([np.sum(np.abs(grid - v) * dens) * w for v in grid])   # |theta - a|


def risk_vec(v, c_fp, c_fn):
    """Risque de la regle 'traiter si theta > v' : manquer un malade coute c_fn,
    traiter un bien portant coute c_fp."""
    v = np.asarray(v, dtype=float)
    e_low = post.mean() * stats.beta.cdf(v, a_p + 1, b_p)      # E[theta 1(theta<=v)]
    e_high = post.mean() - e_low                               # E[theta 1(theta>v)]
    p_high = 1.0 - post.cdf(v)                                 # P(theta > v)
    return c_fn * e_low + c_fp * (p_high - e_high)


def risk_asym(v, c_fp, c_fn):
    return float(risk_vec(v, c_fp, c_fn))


COSTS = [(1.0, 1.0), (1.0, 4.0), (1.0, 9.0), (1.0, 19.0)]
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))
ax = axes[0]
verif = {}
for (c_fp, c_fn), col, lab in zip(
        COSTS, [C["grey"], C["orange"], C["red"], C["purple"]],
        ["perte 0--1 symetrique ($c_{fp}=c_{fn}$)",
         "coût asymétrique $4{:}1$", "coût asymétrique $9{:}1$", "coût asymétrique $19{:}1$"]):
    r = risk_vec(grid, c_fp, c_fn)
    q_thr = c_fp / (c_fp + c_fn)                 # seuil analytique sur theta
    v_num = float(grid[np.argmin(r)])            # verification numerique du seuil
    verif[f"{c_fn/c_fp:g}"] = dict(q_analytique=q_thr, argmin_numerique=v_num,
                                   risque_min=float(r.min()),
                                   P_theta_sup_q=float(1 - post.cdf(q_thr)))
    ax.plot(grid, r / r.max(), color=col, lw=2.0, label=f"{lab} : seuil $q={q_thr:.2f}$")
    ax.axvline(q_thr, color=col, lw=1.0, ls="--", alpha=.7)
ax.plot(grid, R2 / R2.max(), color=C["blue"], lw=1.6, ls=":",
        label="perte quadratique (min en $\\mathbb{E}[\\theta]$)")
ax.plot(grid, Rabs / Rabs.max(), color=C["green"], lw=1.6, ls="-.",
        label="perte absolue (min en médiane)")
ax.annotate("seuil $q=c_{fp}/(c_{fp}+c_{fn})$", (0.30, 0.90),
            xycoords="axes fraction", fontsize=8.6, color=C["deep"], fontweight="bold")
ax.set_xlabel("seuil de décision $a$ (on traite si $\\theta > a$)")
ax.set_ylabel("risque a posteriori (normalisé)")
ax.set_title("Le seuil de coût déplace la frontière")
ax.legend(fontsize=7.4, loc="upper right")

# application dépistage : posterior Beta(3,7) = P(malade | test positif)
ax = axes[1]
ax.fill_between(grid, 0, dens, color=C["blue"], alpha=.20)
ax.plot(grid, dens, color=C["blue"], lw=2.2, label="posterior $\\theta$ = P(malade | $T^+$)")
for (c_fp, c_fn), col in zip(COSTS, [C["grey"], C["orange"], C["red"], C["purple"]]):
    q_thr = c_fp / (c_fp + c_fn)
    ax.axvline(q_thr, color=col, lw=1.8,
               label=f"$c_{{fn}}/c_{{fp}}={c_fn/c_fp:g}$ $\\to$ seuil $\\theta^*={q_thr:.2f}$")
ax.set_xlabel("$\\theta$ = P(malade | test positif)")
ax.set_ylabel("densité a posteriori")
ax.set_title("Dépistage : seuil selon le coût")
ax.legend(fontsize=7.6, loc="upper right")
fig.suptitle("i09 -- Risque a posteriori : la décision optimale dépend du rapport de coûts, "
             "pas de la $p$-value", fontsize=9.3, y=1.06)
save(fig, "i09_risk_threshold.png")

scen_dep = {}
for c_fn in (1, 4, 9, 19):
    scen_dep[f"c_fn={c_fn}"] = verif[f"{float(c_fn):g}"]
add(K, i09=dict(posterior=f"Beta({a_p:g},{b_p:g})", moyenne=float(mean_p),
                seuil_perte_01_symetrique=0.5,
                risque_min_01_symetrique=float(risk_asym(0.5, 1.0, 1.0)),
                argmin_perte_quad=float(grid[np.argmin(R2)]),
                argmin_perte_abs=float(grid[np.argmin(Rabs)]),
                scenarios=scen_dep,
                P_traite=float(1 - post.cdf(0.5))))

# =====================================================================
# i10 -- A/B test bayésien avec arrêt séquentiel
# =====================================================================
P_A, P_B = 0.10, 0.13
N_MAX = 20000
LOOKS = np.arange(400, N_MAX + 1, 400)
ND_DRAW = 1000


def prob_sup(a_b, b_b, a_a, b_a, nd=ND_DRAW, rr=None):
    rr = rr or np.random.default_rng(7)
    return float((rr.beta(a_b, b_b, nd) > rr.beta(a_a, b_a, nd)).mean())


# --- une experience réelle, suivie pas a pas
rr = np.random.default_rng(4112)
conv_a = rr.random(N_MAX) < P_A
conv_b = rr.random(N_MAX) < P_B
cum_a = np.cumsum(conv_a)
cum_b = np.cumsum(conv_b)
psup, gain = [], []
for n in LOOKS:
    aA, bA = 1 + cum_a[n - 1], 1 + (n - cum_a[n - 1])
    aB, bB = 1 + cum_b[n - 1], 1 + (n - cum_b[n - 1])
    p = prob_sup(aB, bB, aA, bA)
    psup.append(p)
    draw_b = rr.beta(aB, bB, 20000)
    draw_a = rr.beta(aA, bA, 20000)
    gain.append(float(np.maximum(draw_b - draw_a, 0).mean()))
psup = np.array(psup)
gain = np.array(gain)
stop_idx = int(np.argmax(psup >= 0.95)) if (psup >= 0.95).any() else -1
n_stop = int(LOOKS[stop_idx]) if stop_idx >= 0 else int(N_MAX)
p_stop = float(psup[stop_idx]) if stop_idx >= 0 else float(psup[-1])
gain_stop = float(gain[stop_idx]) if stop_idx >= 0 else float(gain[-1])
# valeur esperee de la decision : gain par visiteur multiplie par l'audience restante
VALUE_PER_CONV = 2.0
AUDIENCE = 200000
value_at_stop = gain_stop * VALUE_PER_CONV * AUDIENCE
p_a_final = float((1 + cum_a[n_stop - 1]) / (2 + n_stop))
p_b_final = float((1 + cum_b[n_stop - 1]) / (2 + n_stop))

# --- sous H0 : gonflement du risque par le "peeking" fréquentiste vs règle bayésienne
NREP = 400
looks_h0 = np.arange(200, N_MAX + 1, 200)
freq_rej, bayes_rej, bayes99_rej = 0, 0, 0
for r in range(NREP):
    rr = np.random.default_rng(50000 + r)
    ca = np.cumsum(rr.random(N_MAX) < P_A)
    cb = np.cumsum(rr.random(N_MAX) < P_A)          # H0 : mêmes taux
    rej_f = False
    rej_b = False
    rej_b99 = False
    for n in looks_h0:
        pa, pb = ca[n - 1] / n, cb[n - 1] / n
        pp = (ca[n - 1] + cb[n - 1]) / (2 * n)
        se = np.sqrt(pp * (1 - pp) * 2 / n)
        if se > 0 and abs(pb - pa) / se > 1.96:
            rej_f = True
        if not (rej_b and rej_b99):
            aA, bA = 1 + ca[n - 1], 1 + (n - ca[n - 1])
            aB, bB = 1 + cb[n - 1], 1 + (n - cb[n - 1])
            p = prob_sup(aB, bB, aA, bA, nd=400, rr=rr)
            if p >= 0.95:
                rej_b = True
            if p >= 0.99:
                rej_b99 = True
    freq_rej += rej_f
    bayes_rej += rej_b
    bayes99_rej += rej_b99
freq_rate = freq_rej / NREP
bayes_rate = bayes_rej / NREP
bayes99_rate = bayes99_rej / NREP
# taux de 1re espece d'un test unique en fin d'experience (pas de peeking)
rr = np.random.default_rng(999)
single = 0
for r in range(NREP):
    ca = rr.binomial(N_MAX, P_A)
    cb = rr.binomial(N_MAX, P_A)
    pa, pb = ca / N_MAX, cb / N_MAX
    pp = (ca + cb) / (2 * N_MAX)
    se = np.sqrt(pp * (1 - pp) * 2 / N_MAX)
    if se > 0 and abs(pb - pa) / se > 1.96:
        single += 1
single_rate = single / NREP
print(f"[i10] H0 : fréquentiste avec peeking {freq_rate:.3f} | bayésien seq. 0,95 "
      f"{bayes_rate:.3f} | bayésien seq. 0,99 {bayes99_rate:.3f} | test unique "
      f"{single_rate:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))
ax = axes[0]
ax.plot(LOOKS, psup, color=C["blue"], lw=2.2, label="$P(p_B > p_A \\,|\\, données)$")
ax.axhline(0.95, color=C["red"], lw=1.6, ls="--", label="seuil de decision 0,95")
ax.axvline(n_stop, color=C["green"], lw=1.6, ls=":")
ax.annotate(f"arrêt a $n={n_stop}$\\n$P={p_stop:.3f}$", (n_stop, p_stop),
            textcoords="offset points", xytext=(-92, -34), fontsize=8.6,
            color=C["green"], fontweight="bold")
ax.axhline(0.5, color=C["grey"], lw=.9, ls=":")
ax.set_xlabel("visiteurs par bras")
ax.set_ylabel("probabilité de supériorité")
ax.set_title(f"A/B bayésien : arrêt des $n={n_stop}$ visiteurs")
ax.legend(fontsize=8.0, loc="lower right")
ax.set_ylim(0, 1.05)

ax = axes[1]
bars = [("test unique\nsans peeking\n$\\alpha = 5\\,\\%$", single_rate, C["green"]),
        ("règle bayésienne\nséquentielle\nseuil 0,99", bayes99_rate, C["purple"]),
        ("règle bayésienne\nséquentielle\nseuil 0,95", bayes_rate, C["blue"]),
        ("z-test repete\na chaque regard\n(peeking)", freq_rate, C["red"])]
ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars], alpha=.85)
ax.axhline(0.05, color=C["deep"], lw=1.6, ls="--")
ax.text(3.45, 0.056, "$\\alpha = 5\\,\\%$", fontsize=8.4, color=C["deep"], fontweight="bold",
        ha="right")
for i, b in enumerate(bars):
    ax.text(i, b[1] + .008, f"{b[1]*100:.1f} %", ha="center", fontsize=9.0,
            fontweight="bold", color=C["deep"])
ax.set_ylim(0, max(freq_rate, bayes_rate) * 1.30)
ax.set_ylabel("faux positif simule sous $H_0$ (400 exp.)")
ax.set_title("Le peeking est le vrai coupable, pas le bayésien")
ax.tick_params(axis="x", labelsize=7.0)
fig.suptitle("i10 -- A/B séquentiel : la probabilité de supériorité reste valide a chaque "
             "regard, mais le risque d'une règle d'arrêt se simule", fontsize=9.3, y=1.06)
save(fig, "i10_ab_test.png")

add(K, i10=dict(p_a=P_A, p_b=P_B, n_max=N_MAX, n_stop=n_stop, p_sup_stop=p_stop,
                p_sup_final=float(psup[-1]), gain_attendu=gain_stop,
                valeur_attendue=value_at_stop, audience=AUDIENCE, valeur_par_conversion=VALUE_PER_CONV,
                nrep_h0=NREP, taux_faux_positif_unique=single_rate,
                taux_faux_positif_bayésien_seq=bayes_rate,
                taux_faux_positif_bayésien_seq_99=bayes99_rate,
                taux_faux_positif_peeking=freq_rate,
                p_a_estimee_arret=p_a_final, p_b_estimee_arret=p_b_final,
                n_looks=int(len(LOOKS))))

# --- comparaison frequentiste chiffree sur la MEME experience arretee ------
ca_stop = int(cum_a[n_stop - 1])
cb_stop = int(cum_b[n_stop - 1])
pa_hat, pb_hat = ca_stop / n_stop, cb_stop / n_stop
pp_hat = (ca_stop + cb_stop) / (2 * n_stop)
se_pool = float(np.sqrt(pp_hat * (1 - pp_hat) * 2 / n_stop))
z_val = float((pb_hat - pa_hat) / se_pool)
p_val = float(2 * stats.norm.sf(abs(z_val)))
se_wald = float(np.sqrt(pa_hat * (1 - pa_hat) / n_stop + pb_hat * (1 - pb_hat) / n_stop))
ci = [float(pb_hat - pa_hat - 1.96 * se_wald), float(pb_hat - pa_hat + 1.96 * se_wald)]
n_power = float((1.959964 + 0.841621) ** 2 *
                (pa_hat * (1 - pa_hat) + pb_hat * (1 - pb_hat)) / (pb_hat - pa_hat) ** 2)
print(f"[i10] experience arretee : {ca_stop}/{n_stop} vs {cb_stop}/{n_stop} "
      f"({pa_hat:.4f} vs {pb_hat:.4f}) ; z = {z_val:.3f}, p = {p_val:.4f}, "
      f"IC 95 % = [{ci[0]:.4f} ; {ci[1]:.4f}] ; n par bras (80 % puissance) = {n_power:.0f}")
add(K, i10_frequentiste=dict(ca=ca_stop, cb=cb_stop, n=n_stop, pa=pa_hat, pb=pb_hat,
                             z=z_val, p_value=p_val, ci95=ci, se_wald=se_wald,
                             n_par_bras_puissance_80=n_power,
                             delta_observe=float(pb_hat - pa_hat)))

# =====================================================================
# i11 -- ROPE : decision a trois issues
# =====================================================================
ROPE = 0.005
rr = np.random.default_rng(4242)
sizes = np.array([500, 1000, 2000, 5000, 10000, 20000, 50000])
a_true, b_true = 0.10, 0.108                                  # effet réel minuscule
# une seule experience, suivie a des tailles croissantes (données imbriquees)
ca_all = np.cumsum(rr.random(int(sizes[-1])) < a_true)
cb_all = np.cumsum(rr.random(int(sizes[-1])) < b_true)
in_rope, below, above = [], [], []
for n in sizes:
    ca, cb = int(ca_all[n - 1]), int(cb_all[n - 1])
    da = (rr.beta(1 + cb, 1 + n - cb, 40000) - rr.beta(1 + ca, 1 + n - ca, 40000))
    in_rope.append(float((np.abs(da) <= ROPE).mean()))
    below.append(float((da < -ROPE).mean()))
    above.append(float((da > ROPE).mean()))
in_rope, below, above = map(np.array, (in_rope, below, above))

# cas d'ecole (illustration) : n = 5000 par bras, difference observee quasi nulle
n_case = 5000
ca_case, cb_case = 500, 502                     # (+0,04 point observe)
da_case = (rr.beta(1 + cb_case, 1 + n_case - cb_case, 200000)
           - rr.beta(1 + ca_case, 1 + n_case - ca_case, 200000))
p_below = float((da_case < -ROPE).mean())
p_in = float((np.abs(da_case) <= ROPE).mean())
p_above = float((da_case > ROPE).mean())
print(f"[i11] cas n={n_case} ({ca_case} vs {cb_case}) : P(below)={p_below:.4f}, "
      f"P(in ROPE)={p_in:.4f}, P(above)={p_above:.4f}")
print(f"[i11] experience imbriquee : in_rope={[round(v,3) for v in in_rope]}, "
      f"above={[round(v,3) for v in above]}")

fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))
ax = axes[0]
ax.hist(da_case, bins=120, density=True, color=C["blue"], alpha=.55)
ax.axvspan(-ROPE, ROPE, color=C["green"], alpha=.30,
           label=f"ROPE $[-{ROPE},{ROPE}]$ : equivalence pratique")
ax.axvline(0, color=C["deep"], lw=1.2, ls=":")
ax.annotate(f"$P(\\Delta < -{ROPE}) = {p_below:.3f}$\\n"
            f"$P(\\Delta \\in ROPE) = {p_in:.3f}$\\n"
            f"$P(\\Delta > {ROPE}) = {p_above:.3f}$",
            (0.02, 0.60), xycoords="axes fraction", fontsize=8.8,
            color=C["deep"], fontweight="bold")
ax.set_xlabel("$\\Delta = p_B - p_A$ (difference des taux)")
ax.set_ylabel("densité a posteriori")
ax.set_title(f"$n={n_case}$ par bras : la donnée ne tranche pas")
ax.legend(fontsize=8.2, loc="upper left")

ax = axes[1]
ax.plot(sizes, in_rope, "o-", color=C["green"], lw=2.2,
        label="$P(\\Delta \\in ROPE)$ : indécis")
ax.plot(sizes, above, "s-", color=C["blue"], lw=2.0, label="$P(\\Delta > ROPE)$ : B gagne")
ax.plot(sizes, below, "^-", color=C["red"], lw=2.0, label="$P(\\Delta < -ROPE)$ : A gagne")
ax.axhline(0.95, color=C["deep"], lw=1.2, ls="--")
ax.text(600, 0.965, "decision ferme a 0,95", fontsize=8.2, color=C["deep"], fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("$n$ par bras")
ax.set_ylabel("probabilité a posteriori")
ax.set_title("Avec un effet réel de $+0{,}8$ point : il faut beaucoup de données")
ax.legend(fontsize=8.0, loc="center left")
fig.suptitle("i11 -- ROPE : trois issues au lieu de deux (gagner, perdre, ne pas conclure)",
             fontsize=9.3, y=1.06)
save(fig, "i11_rope.png")

add(K, i11=dict(rope=ROPE, n_case=n_case, ca_case=ca_case, cb_case=cb_case, p_below=p_below, p_in_rope=p_in,
                p_above=p_above, effort_réel=a_true, effet_b=True,
                sizes=[int(s) for s in sizes],
                in_rope=[float(v) for v in in_rope],
                above=[float(v) for v in above], below=[float(v) for v in below],
                n_pour_95=(float(sizes[np.argmax((above + below) >= 0.95)])
                           if (above + below).max() >= 0.95 else None)))
print("[key_numbers_ch4.json] cles :", sorted(K.keys()))
