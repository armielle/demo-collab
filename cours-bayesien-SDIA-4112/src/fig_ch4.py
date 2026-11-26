"""Chapitre 4 -- figures generees par du code réellement execute (partie 1 : facteur de Bayes).

Sorties : figures/i01..i06*.png + fusion dans figures/key_numbers_ch4.json

Section 4.1 : definition du BF, cotes, echelle de Jeffreys, calibration de
Sellke-Bayarri-Berger (SBB 2001) et Johnson (2013), Savage-Dickey,
calcul par échantillonnage, sensibilite au prior.
Contraintes machine : CPU only, pas de g++, threads limites par figstyle.
"""
import json
import os
from math import comb, log

import numpy as np
from scipy import stats, special
from scipy.optimize import minimize_scalar
from figstyle import plt, C, save, FIGDIR

rng = np.random.default_rng(4112)
KDIR = os.path.join(FIGDIR, "key_numbers_ch4.json")
K = json.load(open(KDIR, encoding="utf-8")) if os.path.exists(KDIR) else {}


def add(d, **kw):
    d.update(kw)
    json.dump(d, open(KDIR, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# =====================================================================
# i01 -- deux posteriors concurrents, aire chevauchante et vrai BF
# =====================================================================
n_flip, y_head = 20, 12
M0 = 0.5                                      # modèle nul : theta = 0,5 exactement
a1, b1 = 1.0, 1.0                             # prior Beta(1,1) sous M1
post_a, post_b = a1 + y_head, b1 + n_flip - y_head   # Beta(13,9)

log_marg0 = log(comb(n_flip, y_head)) + n_flip * np.log(M0)
log_marg1 = log(comb(n_flip, y_head)) + (special.betaln(post_a, post_b) - special.betaln(a1, b1))
bf01 = float(np.exp(log_marg0 - log_marg1))
post_odds, p_h0 = bf01, bf01 / (1 + bf01)

# "aire chevauchante" : on approche M0 par une loi largeur de tolerance 0,05
tol = 0.05
xx = np.linspace(0.001, 0.999, 1200)
d1 = stats.beta.pdf(xx, post_a, post_b)
d0 = stats.norm.pdf(xx, M0, tol)              # posterior de M0 etale par la tolerance
w = xx[1] - xx[0]
overlap = float(np.minimum(d0, d1).sum() * w)
mass0 = float(d0.sum() * w)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5))
ax = axes[0]
ax.fill_between(xx, 0, np.minimum(d0, d1), color=C["purple"], alpha=.35,
                label=f"aire chevauchante = {overlap:.3f}")
ax.plot(xx, d1, color=C["blue"], lw=2.2, label=f"M1 : posterior Beta({post_a:.0f},{post_b:.0f})")
ax.plot(xx, d0, color=C["red"], lw=1.8, ls="--",
        label=f"M0 : theta = {M0} (tolerance {tol})")
ax.axvline(M0, color=C["red"], lw=1.0, alpha=.5)
ax.set_title("Deux modèles concurrents et leur recouvrement")
ax.set_xlabel("$\\theta$ = P(face)")
ax.set_ylabel("densité a posteriori")
ax.legend(fontsize=8.0, loc="upper right")
ax.annotate(f"données : {y_head} faces / {n_flip} lancers", (0.02, 0.94),
            xycoords="axes fraction", fontsize=8.4, color=C["deep"], fontweight="bold")

ax = axes[1]
bars = [("BF$_{01}$", bf01, C["red"]), ("BF$_{10}$", 1 / bf01, C["blue"]),
        ("cotes post.\nP(H0)/P(H1)", post_odds, C["purple"]),
        ("P(H0|y)", p_h0, C["orange"]), ("aire chev.", overlap, C["green"])]
ax.barh([b[0] for b in bars], [b[1] for b in bars],
        color=[b[2] for b in bars], alpha=.85)
for i, b in enumerate(bars):
    ax.text(b[1] * 1.04, i, f"{b[1]:.3f}", va="center", fontsize=9, fontweight="bold",
            color=C["deep"])
ax.set_xlim(0, max(b[1] for b in bars) * 1.35)
ax.set_title("Le BF n'est pas une aire : c'est un rapport de vraisemblances")
ax.set_xlabel("valeur")
ax.invert_yaxis()
fig.suptitle("i01 -- Numérateur = vraisemblance sous M0 (theta = 0,5) ; dénominateur = "
             "vraisemblance marginale sous M1 (integrale sur tout theta)",
             fontsize=9.4, y=1.06)
save(fig, "i01_bf_posteriors.png")

add(K, i01=dict(n=n_flip, heads=y_head, bf01=bf01, bf10=1 / bf01, post_odds=post_odds,
                p_h0=p_h0, overlap_area=overlap, mass_m0=mass0,
                log_marg0=log_marg0, log_marg1=log_marg1))

# =====================================================================
# i02 -- le BF evolue avec n : sous H0 vraie et sous H1 vraie
# =====================================================================
ns = np.array([10, 20, 30, 50, 75, 100, 150, 200, 300, 400])
nrep = 24


def bf01_coin(y, n, prior=(1.0, 1.0)):
    lg0 = log(comb(n, y)) + n * np.log(0.5)
    lg1 = log(comb(n, y)) + special.betaln(prior[0] + y, prior[1] + n - y) - special.betaln(*prior)
    return lg0 - lg1


scen = {}
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6), sharey=True)
for ax, (name, p_true, col) in zip(axes, [("H0 vraie ($\\theta=0{,}5$)", 0.5, C["red"]),
                                          ("H1 vraie ($\\theta=0{,}65$)", 0.65, C["blue"])]):
    curves = []
    for r in range(nrep):
        rr = np.random.default_rng(1000 + r)
        vals = []
        for n in ns:
            y = rr.binomial(n, p_true)
            vals.append(bf01_coin(y, n) / np.log(10))
        curves.append(vals)
    curves = np.array(curves)
    med = np.median(curves, axis=0)
    lo, hi = np.percentile(curves, [10, 90], axis=0)
    ax.fill_between(ns, lo, hi, color=col, alpha=.18, label="intervalle 10--90 %")
    ax.plot(ns, med, color=col, lw=2.4, label="médiane de $\\log_{10} BF_{01}$")
    scen[name] = dict(median_final=float(med[-1]), p_true=p_true,
                      slope_per_100=float((med[-1] - med[0]) / (ns[-1] - ns[0]) * 100))
    ax.axhline(0, color=C["deep"], lw=1.0, ls=":")
    for thr, lab in ((0.5, "substantiel"), (1.0, "fort"), (2.0, "decisif")):
        ax.axhline(thr, color=C["grey"], lw=.7, ls="--", alpha=.6)
        ax.text(ns[-1] * 0.99, thr, lab, fontsize=7.4, color=C["grey"], ha="right", va="bottom")
    ax.set_title(name)
    ax.set_xlabel("taille d'echantillon $n$")
    ax.legend(fontsize=8.2, loc="lower left" if p_true == 0.5 else "upper left")
axes[0].set_ylabel("$\\log_{10} BF_{01}$   ($>0$ : faveur a H0)")
axes[0].annotate("le BF s'accumule lineairement\nen $n$ : pas de seuil magique",
                 (0.40, 0.72), xycoords="axes fraction", fontsize=8.4,
                 color=C["deep"], fontweight="bold")
axes[1].annotate("$\\theta = 0{,}65$ : le BF\ns'enfonce vers H1", (0.26, 0.60),
                 xycoords="axes fraction", fontsize=8.4, color=C["deep"], fontweight="bold")
fig.suptitle("i02 -- Simulation réelle : $\\log_{10} BF_{01}$ (prior Beta(1,1) sous M1, "
             f"{nrep} réplications par point)", fontsize=9.4, y=1.06)
save(fig, "i02_bf_vs_n.png")

# nombre de lancers necessaires, en médiane, pour atteindre log10 BF01 = -1 sous H1
med_h1 = np.median(np.array([
    [bf01_coin(np.random.default_rng(2000 + r).binomial(n, 0.65), n) / np.log(10) for n in ns]
    for r in range(40)]), axis=0)
n_grid = np.linspace(ns.min(), ns.max(), 2000)
med_h1_interp = np.interp(n_grid, ns, med_h1)
cross = np.flatnonzero(med_h1_interp <= -1.0)
n_strong_h1 = float(n_grid[cross[0]]) if cross.size else float("nan")
add(K, i02=dict(scen=scen, nrep=nrep, ns=[int(v) for v in ns],
                med_lg10bf_h0=[float(v) for v in np.median(np.array([
                    [bf01_coin(np.random.default_rng(1000 + r).binomial(n, 0.5), n) / np.log(10)
                     for n in ns] for r in range(nrep)]), axis=0)],
                med_lg10bf_h1=[float(v) for v in med_h1],
                n_pour_log10BF_moins1_sous_H1=n_strong_h1,
                slope_per_100_h1=float(scen["H1 vraie ($\\theta=0{,}65$)"]["slope_per_100"])))

# =====================================================================
# i03 -- BF minimum atteignable pour une p-value donnee (SBB / Johnson)
# =====================================================================
p_grid = np.logspace(np.log10(0.2), np.log10(0.0005), 500)
p_grid = p_grid[p_grid < 1 / np.e]
B_sbb = -np.e * p_grid * np.log(p_grid)                  # borne inferieure de BF01
P_h0_min = B_sbb / (1 + B_sbb)                           # posterior minimum de H0 (prior 0,5)


def bf01_gauss2sided(p, prior_var):
    """BF01 exact (loi normale, statistique z, prior N(0,tau^2) sous H1)."""
    z = stats.norm.isf(p / 2)
    t = prior_var
    return np.sqrt(1 + t) * np.exp(-z ** 2 * t / (2 * (1 + t)))


# minimum sur la famille gaussienne a p fixee : tau^2 = z^2 - 1
def bf_min_gauss(p):
    z = stats.norm.isf(p / 2)
    t = max(z ** 2 - 1, 1e-9)
    return float(np.sqrt(1 + t) * np.exp(-z ** 2 * t / (2 * (1 + t))))


p_pts = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
b_pts = -np.e * p_pts * np.log(p_pts)
ph0_pts = b_pts / (1 + b_pts)
gauss_pts = np.array([bf_min_gauss(p) for p in p_pts])

grid_curve = np.array([bf_min_gauss(p) for p in p_grid])
# verification numérique : minimisation directe en 1-D pour p = 0,05
res = minimize_scalar(lambda lv: bf01_gauss2sided(0.05, np.exp(lv)),
                      bounds=(np.log(1e-4), np.log(1e3)), method="bounded")
tau2_opt_checked = float(np.exp(res.x))
bf_min_checked = float(res.fun)

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))
ax = axes[0]
ax.loglog(p_grid, B_sbb, color=C["red"], lw=2.4, label="borne SBB  $-e\\,p\\ln p$")
ax.loglog(p_grid, grid_curve, color=C["blue"], lw=1.8, ls="--",
          label="min. famille gaussienne")
ax.plot(p_pts, b_pts, "o", color=C["red"], ms=6)
ax.plot(0.05, -np.e * 0.05 * np.log(0.05), "*", color=C["deep"], ms=14)
ax.annotate("$p=0{,}05$\n$BF_{01}=0{,}407$", (0.05, 0.407), textcoords="offset points",
            xytext=(16, -18), fontsize=8.6, fontweight="bold", color=C["deep"])
ax.set_xlabel("p-value (test bilateral)")
ax.set_ylabel("$BF_{01}$ minimum atteignable")
ax.set_title("BF plancher d'une p-value")
ax.legend(fontsize=8.0, loc="lower right")

ax = axes[1]
ax.semilogx(p_pts, ph0_pts, "o-", color=C["purple"], lw=2.0,
            label="P(H0|données) min. (SBB)")
ax.semilogx(p_pts, 1 / (1 + b_pts), "s--", color=C["orange"], lw=1.6,
            label="croyance max. en H1")
ax.axhline(0.289, color=C["grey"], lw=.8, ls=":")
ax.annotate("0,289 au niveau $p=0{,}05$", (0.0012, 0.30), fontsize=8.4, color=C["grey"])
for p, v in zip(p_pts, ph0_pts):
    ax.annotate(f"{v:.3f}", (p, v), textcoords="offset points", xytext=(2, -13),
                fontsize=7.6, color=C["deep"])
ax.set_xlabel("p-value")
ax.set_ylabel("probabilite a posteriori")
ax.set_ylim(0, 1.02)
ax.set_title("Posterior minimal de $H_0$")
ax.legend(fontsize=8.0, loc="upper left")

ax = axes[2]
taus = np.logspace(-1.5, 2.0, 300)
vals = np.array([bf01_gauss2sided(0.05, t) for t in taus ** 2])
ax.loglog(taus, vals, color=C["blue"], lw=2.2)
ax.axhline(-np.e * 0.05 * np.log(0.05), color=C["red"], lw=1.8, ls="--",
           label="borne SBB $=0{,}407$")
ax.plot(tau2_opt_checked ** .5, bf_min_checked, "o", color=C["deep"], ms=7)
ax.annotate(f"min $= {bf_min_checked:.3f}$\n" + "a $\\tau = %.2f$" % (tau2_opt_checked ** .5),
            (tau2_opt_checked ** .5, bf_min_checked), textcoords="offset points",
            xytext=(10, 18), fontsize=8.4, fontweight="bold", color=C["deep"])
ax.set_xlabel("largeur du prior $\\tau$ (ecart-type)")
ax.set_ylabel("$BF_{01}$")
ax.set_title("Profil en $\tau$ pour $p=0{,}05$")
ax.legend(fontsize=8.0, loc="upper left")
fig.suptitle("i03 -- Calibration de Sellke--Bayarri--Berger (2001) et Johnson (2013) : "
             "$B(p) = -e\\,p\\ln p$ est la borne INFERIEURE des cotes de H0 contre H1",
             fontsize=9.3, y=1.06)
save(fig, "i03_bf_min_pvalue.png")

add(K, i03=dict(p_points=[float(v) for v in p_pts],
                B_sbb=[float(v) for v in b_pts],
                P_H0_min=[float(v) for v in ph0_pts],
                BF10_max=[float(1 / v) for v in b_pts],
                gauss_min=[float(v) for v in gauss_pts],
                bf01_min_gauss_p05=bf_min_checked, tau_opt_checked=tau2_opt_checked,
                p05=0.05, B_p05=float(-np.e * 0.05 * np.log(0.05)),
                P_H0_min_p05=float(((-np.e * 0.05 * np.log(0.05)) /
                                    (1 + -np.e * 0.05 * np.log(0.05)))),
                BF10_max_p05=float(1 / (-np.e * 0.05 * np.log(0.05)))))

# =====================================================================
# i04 -- 2000 simulations sous H0 : p-value contre BF (n petit vs n grand)
# =====================================================================
N_SIM = 2000
xs_ = np.array([20, 50, 100])
tab_sim = {}
for n_s in (20, 50, 100):
    ys = rng.binomial(n_s, 0.5, N_SIM)
    zs_ = (ys / n_s - 0.5) / np.sqrt(0.25 / n_s)
    ps = 2 * stats.norm.sf(np.abs(zs_))
    bs = np.array([bf01_coin(int(yy), n_s) for yy in ys])
    sg = ps < 0.05
    tab_sim[n_s] = dict(
        n=n_s, frac_sig=float(sg.mean()),
        frac_bf01_sup_1_si_sig=float((bs[sg] > 0).mean()) if sg.sum() else 0.0,
        n_sig=int(sg.sum()),
        med_bf01_si_sig=float(np.median(np.exp(bs[sg]))) if sg.sum() else float("nan"),
        min_p=float(ps.min()), p=ps, bf=np.exp(bs), sig=sg)

n_sim = 20
S = tab_sim[n_sim]
p_sim, bf_sim, sig = S["p"], S["bf"], S["sig"]
log10bf_sim = bf_sim / np.log(10)
frac_sig, frac_bf10 = S["frac_sig"], float((bf_sim < 1).mean())
frac_contradict = S["frac_bf01_sup_1_si_sig"]
med_bf_sig = S["med_bf01_si_sig"]
bf01_at_p05 = float(-np.e * 0.05 * np.log(0.05))

fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))
ax = axes[0]
s100 = tab_sim[100]
ax.scatter(s100["p"], np.log10(s100["bf"]), s=10, color=C["blue"],
           alpha=.35, label="2000 simulations sous $H_0$ ($n=100$)")
ax.scatter(s100["p"][s100["sig"]], np.log10(s100["bf"][s100["sig"]]), s=14,
           color=C["red"], alpha=.75, label="%d cas où $p<0,05$" % s100["n_sig"])
ax.axhline(0, color=C["deep"], lw=1.1, ls=":")
ax.axhline(np.log10(bf01_at_p05), color=C["orange"], lw=1.5, ls="--")
ax.text(0.002, np.log10(bf01_at_p05) - 0.16, "plancher SBB : $BF_{01}=0{,}407$",
        fontsize=8.0, color=C["orange"], fontweight="bold")
ax.axvline(0.05, color=C["grey"], lw=1.4, ls="--")
ax.set_xlim(0, 0.55)
ax.set_ylim(-1.5, 1.1)
ax.set_xlabel("p-value observée (test bilatéral)")
ax.set_ylabel("$\\log_{10} BF_{01}$   ($>0$ : faveur à $H_0$)")
ax.set_title(f"{N_SIM} expériences SIMULÉES sous $H_0$")
ax.legend(fontsize=8.4, loc="upper left")
bf01_edge = float(np.exp(bf01_coin(60, 100)))
bf10_edge = 1.0 / bf01_edge
ax.annotate("au bord du rejet ($y=60/100$, $p=0,045$)\n"
            "$BF_{01} = %.2f$ : la donnée favorise $H_0$ !\n"
            "médiane des rejets : $BF_{10} = %.2f$"
            % (bf01_edge, 1.0 / s100["med_bf01_si_sig"]),
            (0.28, 0.10), xycoords="axes fraction", fontsize=8.2,
            color=C["deep"], fontweight="bold")

ax = axes[1]
rng_fr, rng_bf = [], []
for n_ in xs_:
    s2 = tab_sim[n_]
    rng_fr.append(s2["n_sig"] / N_SIM)
    rng_bf.append(float(np.median(1.0 / s2["bf"][s2["sig"]])))
ax.bar(np.arange(3) - 0.19, rng_fr, 0.38, color=C["blue"], alpha=.85,
       label="P($p<0,05$ sous $H_0$) : risque de 1re espèce")
ax.bar(np.arange(3) + 0.19, rng_bf, 0.38, color=C["red"], alpha=.85,
       label="médiane de $BF_{10}$ parmi ces rejets")
for xx, v in zip(np.arange(3) - 0.19, rng_fr):
    ax.text(xx, v + .02, f"{v*100:.1f} %", ha="center", fontsize=8.6,
            fontweight="bold", color=C["blue"])
for xx, v in zip(np.arange(3) + 0.19, rng_bf):
    ax.text(xx, v + .02, f"{v:.2f}", ha="center", fontsize=8.6,
            fontweight="bold", color=C["red"])
ax.axhline(1.0, color=C["deep"], lw=1.0, ls=":")
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["$n=20$", "$n=50$", "$n=100$"])
ax.set_ylim(0, 4.0)
ax.set_ylabel("proportion / facteur de Bayes")
ax.set_title("Le 5 \% est garanti ; la preuve, non")
ax.legend(fontsize=7.6, loc="upper right")
fig.suptitle("i04 -- La $p$-value contrôle une fréquence d'erreur, le $BF$ mesure une "
             "force de preuve : les rejets a 5 \% restent faibles", fontsize=9.3, y=1.06)
save(fig, "i04_pvalue_vs_bf.png")

add(K, i04=dict(N_SIM=N_SIM, n_used=100, frac_sig=tab_sim[100]["frac_sig"],
                n_sig=tab_sim[100]["n_sig"],
                bf01_min_si_sig=float(tab_sim[100]["bf"][tab_sim[100]["sig"]].min()),
                bf01_max_si_sig=float(tab_sim[100]["bf"][tab_sim[100]["sig"]].max()),
                n_bf01_sup_1=int((tab_sim[100]["bf"][tab_sim[100]["sig"]] > 1).sum()),
                pct_rejets_favorables_a_H0=float((tab_sim[100]["bf"][tab_sim[100]["sig"]] > 1).mean()),
                median_bf10_si_sig=float(np.median(1.0 / tab_sim[100]["bf"][tab_sim[100]["sig"]])),
                bf01_au_bord_du_rejet_y60=bf01_edge, bf10_au_bord_du_rejet_y60=bf10_edge,
                table={str(n_): dict(frac_sig=tab_sim[n_]["frac_sig"],
                                     n_sig=tab_sim[n_]["n_sig"],
                                     median_bf10_si_sig=float(
                                         np.median(1.0 / tab_sim[n_]["bf"][tab_sim[n_]["sig"]])),
                                     min_bf01_si_sig=float(tab_sim[n_]["bf"][tab_sim[n_]["sig"]].min()),
                                     max_bf01_si_sig=float(tab_sim[n_]["bf"][tab_sim[n_]["sig"]].max()),
                                     n_bf01_sup_1=int((tab_sim[n_]["bf"][tab_sim[n_]["sig"]] > 1).sum()))
                       for n_ in xs_},
                p_min_sim=float(min(tab_sim[n_]["min_p"] for n_ in xs_))))

# =====================================================================
# i05 -- Savage-Dickey : BF = densité prior / densité posterior au point nul
# =====================================================================
a0, b0 = 1.0, 1.0
y_sd, n_sd = 12, 20
pa, pb = a0 + y_sd, b0 + n_sd - y_sd
prior, post = stats.beta(a0, b0), stats.beta(pa, pb)
sd_ratio = float(post.pdf(0.5) / prior.pdf(0.5))          # = BF01 (Savage-Dickey)
bf01_true = float(np.exp(log(comb(n_sd, y_sd)) + n_sd * np.log(0.5) -
                         (log(comb(n_sd, y_sd)) + special.betaln(pa, pb) - special.betaln(a0, b0))))
# verification indépendante par échantillonnage : BF10 = E_prior[ p(y|theta)/p(y|theta0) ]
NM = 400_000
th_mc = rng.random(NM)                                     # theta ~ Beta(1,1) = uniforme
lik_ratio = (th_mc / 0.5) ** y_sd * ((1 - th_mc) / 0.5) ** (n_sd - y_sd)
bf10_mc = float(lik_ratio.mean())
bf10_mc_se = float(lik_ratio.std(ddof=1) / np.sqrt(NM))
bf01_mc = 1.0 / bf10_mc
bf01_mc_se = bf10_mc_se / bf10_mc ** 2

fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5))
ax = axes[0]
xs = np.linspace(1e-4, 1 - 1e-4, 800)
ax.plot(xs, prior.pdf(xs), color=C["grey"], lw=2.0, ls="--", label="prior Beta(1,1)")
ax.plot(xs, post.pdf(xs), color=C["blue"], lw=2.4,
        label=f"posterior Beta({pa:.0f},{pb:.0f})")
ax.vlines(0.5, 0, prior.pdf(0.5), color=C["grey"], lw=1.6)
ax.vlines(0.5, 0, post.pdf(0.5), color=C["blue"], lw=1.6)
ax.plot([0.5, 0.5], [post.pdf(0.5), prior.pdf(0.5)], "o", color=C["deep"], ms=6)
ax.annotate(f"prior : {prior.pdf(0.5):.3f}", (0.5, prior.pdf(0.5)),
            textcoords="offset points", xytext=(8, 6), fontsize=8.6, color=C["grey"],
            fontweight="bold")
ax.annotate(f"posterior : {post.pdf(0.5):.3f}", (0.5, post.pdf(0.5)),
            textcoords="offset points", xytext=(8, -14), fontsize=8.6, color=C["blue"],
            fontweight="bold")
ax.set_xlabel("$\\theta$")
ax.set_ylabel("densité")
ax.set_title("Rapport de Savage--Dickey au point nul $\\theta_0 = 0{,}5$")
ax.legend(fontsize=8.4)

ax = axes[1]
labs = ["BF$_{01}$\nexact(marginale)", "BF$_{01}$Savage--Dickey$\\pi(\\theta_0|y)/\\pi(\\theta_0)$",
        "BF$_{01}$\néchantillonnage(400 000 tirages)"]
vals = [bf01_true, sd_ratio, bf01_mc]
err = [0.0, 0.0, bf01_mc_se]
ax.bar(labs, vals, color=[C["red"], C["purple"], C["green"]], alpha=.85,
       yerr=err, capsize=6, error_kw=dict(ecolor=C["deep"], lw=1.4))
for i, v in enumerate(vals):
    ax.text(i, v + bf01_mc_se + .05, f"{v:.4f}", ha="center", va="bottom", fontsize=9.6,
            fontweight="bold", color=C["deep"])
ax.set_ylim(0, max(vals) * 1.30)
ax.set_ylabel("$BF_{01}$")
ax.set_title("Trois chemins, le même nombre")
ax.tick_params(axis="x", labelsize=7.6)
fig.suptitle("i05 -- Savage--Dickey : $BF_{01} = \\pi(\\theta_0\\,|\\,y)\\,/\\,\\pi(\\theta_0)$ "
             "quand $H_0$ est un point de $H_1$", fontsize=9.4, y=1.06)
save(fig, "i05_savage_dickey.png")

add(K, i05=dict(bf01_exact=bf01_true, sd_ratio=sd_ratio, bf01_mc=bf01_mc,
                bf01_mc_se=bf01_mc_se, bf10_mc=bf10_mc, bf10_mc_se=bf10_mc_se, NM=NM,
                prior_dens_at_05=float(prior.pdf(0.5)), post_dens_at_05=float(post.pdf(0.5)),
                rel_err_sd=float(abs(sd_ratio - bf01_true) / bf01_true),
                rel_err_mc=float(abs(bf01_mc - bf01_true) / bf01_true)))

# =====================================================================
# i06 -- sensibilite au prior : le BF "indiscret" des priors vagues
# =====================================================================
zs = [1.96, 2.58, 3.29]                                  # p = 0,05 / 0,01 / 0,001
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.5))
ax = axes[0]
taus = np.logspace(-1.3, 2.2, 320)
tvals = {}
for z, col in zip(zs, [C["orange"], C["red"], C["purple"]]):
    v = np.array([np.sqrt(1 + t ** 2) * np.exp(-z ** 2 * t ** 2 / (2 * (1 + t ** 2)))
                  for t in taus])
    tvals[z] = v
    ax.loglog(taus, v, color=col, lw=2.2,
              label=f"$z={z:.2f}$ ($p={2*stats.norm.sf(z):.3g}$)")
ax.axhline(1, color=C["deep"], lw=1.0, ls=":")
ax.axhline(-np.e * 0.05 * np.log(0.05), color=C["grey"], lw=1.2, ls="--")
ax.annotate("plancher SBB ($p=0{,}05$) : 0,407", (0.06, 0.44), fontsize=8.2,
            color=C["grey"], fontweight="bold")
ax.annotate("$BF_{01} > 1$ : faveur a H0\nmalgre un $z$ significatif", (0.95, 20),
            fontsize=8.2, color=C["red"], fontweight="bold", ha="right")
ax.set_xlabel("largeur du prior $\\tau$ sous $H_1$")
ax.set_ylabel("$BF_{01}$")
ax.set_title("Le piège de Lindley : un prior vague fabrique de la preuve pour $H_0$")
ax.legend(fontsize=8.2, loc="upper left")

ax = axes[1]
# même donnee (z = 1,96), quatre priors déclarés sous H1
bftau = lambda tt, z=1.96: float(np.sqrt(1 + tt ** 2) *
                                 np.exp(-z ** 2 * tt ** 2 / (2 * (1 + tt ** 2))))
choix = [(0.10, "$\\tau=0{,}10$\\neffet minuscule", C["green"]),
         (0.50, "$\\tau=0{,}50$\\neffet moyen", C["blue"]),
         (2.00, "$\\tau=2{,}00$\\neffet large", C["orange"]),
         (50.0, "$\\tau=50$\\nprior vague : piège", C["red"])]
vals = [bftau(t) for t, _, _ in choix]
lbl = [c[1] for c in choix]
cols = [c[2] for c in choix]
bars = ax.barh(range(len(choix)), vals, color=cols, alpha=.85)
ax.set_yticks(range(len(choix)))
ax.set_yticklabels(lbl, fontsize=7.8)
ax.set_xscale("log")
scen6 = {}
for i, (v, (t, _, _)) in enumerate(zip(vals, choix)):
    verdict = ("aucune preuve" if 1 < v < 3 else
               "faveur à $H_1$" if v < 1 else "faveur à $H_0$")
    scen6[f"tau={t}"] = dict(bf01=v, log10=float(np.log10(v)), verdict=verdict)
    ax.text(v * 1.06, i, f"{v:.3f}  ({verdict})", va="center", fontsize=8.4,
            fontweight="bold", color=C["deep"])
ax.axvline(1, color=C["deep"], lw=1.2, ls=":")
ax.set_xlim(0.2, 60)
ax.set_xlabel("$BF_{01}$ a donnee IDENTIQUE ($z=1{,}96$, $p=0{,}05$)")
ax.set_title("Quatre priors, quatre verdicts")
fig.suptitle("i06 -- Le BF dépend du prior déclaré : changez $\\tau$, changez la conclusion "
             "(piège de Lindley)", fontsize=9.4, y=1.06)
save(fig, "i06_bf_prior_sensitivity.png")

add(K, i06=dict(z_values=zs, scenarios=scen6, bf01_by_tau=dict(
    (str(t), bftau(t)) for t, _, _ in choix),
    tau_curve=[float(t) for t in taus],
    bf01_z196_curve=[float(v) for v in tvals[1.96]]))
print("[key_numbers_ch4.json] cles :", sorted(K.keys()))
