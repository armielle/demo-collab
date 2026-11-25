"""Chapitre 5 -- Monte-Carlo, Metropolis-Hastings, Gibbs et HMC.

Tout est ecrit a la main en numpy (aucun compilateur C requis) et REELLEMENT
execute : les chiffres publies dans les slides sortent de ce script.
Ecrit figures/j01_*.png ... j09_*.png et figures/key_numbers_ch5.json.
"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
from math import lgamma
import numpy as np
from figstyle import plt, C, save, FIGDIR

t00 = time.time()
K = {"chapitre": 5}

# =====================================================================
# 5.1  MONTE-CARLO
# =====================================================================
# Cible reelle : X ~ Beta(2,5).  E[X] = 2/7, Var = 10/392
a_, b_ = 2.0, 5.0
E_exact = a_ / (a_ + b_)
SD_exact = float(np.sqrt(a_ * b_ / ((a_ + b_) ** 2 * (a_ + b_ + 1))))
K["mc"] = {"cible": "Beta(2,5)", "esperance_exacte": round(E_exact, 6),
           "ecart_type_exact": round(SD_exact, 6)}

NMAX = 200_000
x = np.random.default_rng(4112).beta(a_, b_, NMAX)
run = np.cumsum(x) / np.arange(1, NMAX + 1)

# --- convergence : erreur absolue et bande theorique 1.96 sigma/sqrt(N)
Ngrid = np.unique(np.round(np.logspace(1, np.log10(NMAX), 220)).astype(int))
err = np.abs(run[Ngrid - 1] - E_exact)
ban = 1.96 * SD_exact / np.sqrt(Ngrid)

# --- RMSE MESUREE sur 1000 repetitions independantes (loi des grands nombres)
NR = 1000
rep = np.random.default_rng(77).beta(a_, b_, (NR, 10_000))
cs = np.cumsum(rep, axis=1) / np.arange(1, rep.shape[1] + 1)
Ns_rmse = np.array([10, 30, 100, 300, 1000, 3000, 10_000])
est = cs[:, Ns_rmse - 1]
rmse = np.sqrt(((est - E_exact) ** 2).mean(axis=0))
cov = ((np.abs(est - E_exact) <= 1.96 * SD_exact / np.sqrt(Ns_rmse)).mean(axis=0))
slope = float(np.polyfit(np.log(Ns_rmse), np.log(rmse), 1)[0])
K["mc"].update({
    "N_max": NMAX, "repetitions_rmse": NR,
    "erreur_a_100": round(float(err[np.searchsorted(Ngrid, 100)]), 6),
    "erreur_a_10000": round(float(err[np.searchsorted(Ngrid, 10000)]), 6),
    "erreur_a_200000": round(float(err[-1]), 6),
    "rmse_N100": round(float(rmse[2]), 6), "rmse_N10000": round(float(rmse[-1]), 6),
    "pente_loglog_rmse": round(slope, 4),
    "couverture_IC_N100": round(float(cov[2]), 4),
    "couverture_IC_N10000": round(float(cov[-1]), 4),
    "moyenne_finale": round(float(run[-1]), 6)})

fig = plt.figure(figsize=(9.6, 3.4))
gs = fig.add_gridspec(1, 3, wspace=.34)
ax = fig.add_subplot(gs[0])
ax.fill_between(Ngrid, 0, ban, color=C["cyan"], alpha=.25,
                label=r"bande theorique $1{,}96\,\sigma/\sqrt{N}$")
ax.plot(Ngrid, err, color=C["blue"], lw=1.8, label="erreur absolue observee")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("nombre de simulations $N$"); ax.set_ylabel(r"$|\hat\mu_N - \mathbb{E}[X]|$")
ax.set_title("L'erreur decroit en $1/\\sqrt{N}$", fontsize=10)
ax.legend(fontsize=7.2, loc="lower left")
ax2 = fig.add_subplot(gs[1])
for n, col in zip([100, 1000, 10_000], [C["orange"], C["cyan"], C["purple"]]):
    ax2.hist(est[:, Ns_rmse.tolist().index(n)], bins=42, color=col, alpha=.55,
             density=True, label=f"$N={n}$ (sd = {cs[:, n-1].std():.4f})")
ax2.set_xlabel(r"estimation $\hat\mu_N$"); ax2.set_ylabel("densite")
ax2.set_title("Dispersion des 1000 estimateurs", fontsize=10)
ax2.legend(fontsize=7.4)
ax3 = fig.add_subplot(gs[2])
ax3.loglog(Ns_rmse, rmse, "o-", color=C["red"], lw=2, ms=6, label="RMSE mesuree")
ref = rmse[0] * np.sqrt(Ns_rmse[0] / Ns_rmse)
ax3.loglog(Ns_rmse, ref, "--", color=C["grey"], lw=1.3,
           label=r"pente $-1/2$ exacte")
ax3.set_xlabel("$N$"); ax3.set_ylabel("RMSE")
ax3.set_title(f"pente ajustee = {slope:.3f}", fontsize=10)
ax3.legend(fontsize=7.6)
fig.suptitle("Estimation Monte-Carlo : cible Beta(2,5) -- 200 000 simulations reelles",
             fontsize=11, y=1.03)
save(fig, "j01_mc_convergence.png")

fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.1))
axs[0].semilogx(Ns_rmse, cov * 100, "o-", color=C["green"], lw=2, ms=6)
for n, cc in zip(Ns_rmse, cov):
    axs[0].annotate(f"{cc*100:.1f}", (n, cc * 100), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=C["deep"])
axs[0].axhline(95, color=C["red"], ls="--", lw=1.2, label="niveau nominal 95 %")
axs[0].set_ylim(80, 100); axs[0].set_xlabel("$N$")
axs[0].set_ylabel("couverture de l'IC Monte-Carlo (%)")
axs[0].set_title("L'IC Monte-Carlo tient sa promesse", fontsize=10)
axs[0].legend(fontsize=8)
axs[1].hist(run[Ngrid - 1], bins=40, color=C["blue"], alpha=.7)
axs[1].axvline(E_exact, color=C["red"], lw=2, label=f"vraie valeur {E_exact:.4f}")
axs[1].set_title("Trajectoire de la moyenne courante", fontsize=10)
axs[1].set_xlabel(r"$\hat\mu_N$ sur toute la trajectoire"); axs[1].legend(fontsize=8)
fig.suptitle("Monte-Carlo : erreur, couverture, trajectoire", fontsize=11, y=1.02)
save(fig, "j02_mc_erreur_ic.png")

# =====================================================================
# 5.2  METROPOLIS-HASTINGS ECRIT A LA MAIN
# =====================================================================
TT = dict(a=2.5, b=7.0)          # cible Beta(2.5, 7) : asymetrique, a support borne
TT_mean = TT["a"] / (TT["a"] + TT["b"])


def logpdf_beta(v, a=TT["a"], b=TT["b"]):
    return (a - 1) * np.log(v) + (b - 1) * np.log1p(-v)


def mh_beta(N, scale, x0=0.5, seed=1, thin=1):
    """Metropolis a marche aleatoire normale : proposition N(x, s^2)."""
    r = np.random.default_rng(seed)
    z = r.standard_normal(N * thin)
    u = r.random(N * thin)
    out = np.empty(N * thin)
    x, lp = x0, logpdf_beta(x0)          # lp = log-densite non normalisee de l'etat courant
    acc = 0
    for i in range(N * thin):
        y = x + scale * z[i]
        if 0.0 < y < 1.0:
            lpy = logpdf_beta(y)
            if np.log(u[i]) < lpy - lp:
                x, lp, acc = y, lpy, acc + 1
        out[i] = x
    return out[::thin], acc / (N * thin)


def acf(v, kmax=200):
    v = v - v.mean()
    d = np.dot(v, v)
    return np.array([np.dot(v[:len(v) - k], v[k:]) / d for k in range(kmax + 1)])


def ess_ips(v, kmax=None):
    """ESS par la << initial positive sequence >> (Geyer), autocorrelations brutes."""
    n = len(v)
    if not np.isfinite(v).all() or np.var(v) <= 0:
        return 0.0
    kmax = kmax or min(1000, n // 4)
    r = acf(v, kmax)
    s, k = 0.0, 1
    while k < len(r) - 1:
        p = r[k] + r[k + 1]
        if p <= 0:
            break
        s += p
        k += 2
    return n / (1.0 + 2.0 * s)


chain, acc_main = mh_beta(30_000, 0.35, seed=2024)
ac = acf(chain, 120)
K["mh"] = {"cible": "Beta(2.5, 7)", "esperance_exacte": round(TT_mean, 5),
           "N": 30_000, "proposition": 0.35,
           "taux_acceptation": round(acc_main, 4),
           "moyenne_post": round(float(chain.mean()), 5),
           "ecart_moyenne": round(abs(float(chain.mean()) - TT_mean), 5),
           "acf_lag1": round(float(ac[1]), 4), "acf_lag10": round(float(ac[10]), 4),
           "acf_lag50": round(float(ac[50]), 4),
           "ess": round(float(ess_ips(chain)), 1),
           "ess_sur_N": round(float(ess_ips(chain)) / len(chain), 4),
           "lag_autocorr_nulle": int(np.argmax(ac < 0.05))}

fig = plt.figure(figsize=(9.8, 3.3))
gs = fig.add_gridspec(1, 3, wspace=.32)
ax = fig.add_subplot(gs[0])
ax.plot(np.arange(500), chain[:500], color=C["blue"], lw=.9)
ax.axhline(TT_mean, color=C["red"], ls="--", lw=1.4, label=f"E[X] = {TT_mean:.3f}")
ax.set_xlabel("iteration $t$ (500 premieres)"); ax.set_ylabel(r"$\theta^{(t)}$")
ax.set_title(f"Trace MC-MC (acc. = {acc_main*100:.1f} %)", fontsize=10)
ax.legend(fontsize=8)
ax2 = fig.add_subplot(gs[1])
ax2.hist(chain, bins=60, density=True, color=C["orange"], alpha=.65,
         edgecolor="white", label="echantillon MH")
g = np.linspace(1e-4, 1 - 1e-4, 500)
from math import lgamma as _lg
lg = _lg(TT["a"]) + _lg(TT["b"]) - _lg(TT["a"] + TT["b"])
ax2.plot(g, np.exp(logpdf_beta(g) - lg), color=C["deep"], lw=2,
         label=r"densite exacte $\mathrm{Beta}(2{,}5\,;\,7)$")
ax2.set_xlabel(r"$\theta$"); ax2.set_ylabel("densite")
ax2.set_title("Histogramme et densite exacte", fontsize=10)
ax2.legend(fontsize=7.6)
ax3 = fig.add_subplot(gs[2])
ax3.bar(np.arange(len(ac)), ac, color=C["cyan"], width=1.0)
ax3.axhline(.05, color=C["red"], ls="--", lw=1.1)
ax3.set_xlabel("decalage $k$"); ax3.set_ylabel(r"$\rho_k$")
ax3.set_title(f"Autocorrelation ($\\rho_1$ = {ac[1]:.2f})", fontsize=10)
fig.suptitle("Metropolis-Hastings maison sur une loi Beta -- 30 000 iterations",
             fontsize=11, y=1.06)
save(fig, "j03_mh_beta.png")

# --- taux d'acceptation et autocorrelation selon la largeur de proposition
scales = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.8, 3.0, 6.0])
accs, esss, ac1 = [], [], []
for s in scales:
    ch, aa = mh_beta(20_000, float(s), seed=5)
    accs.append(aa); esss.append(ess_ips(ch)); ac1.append(acf(ch, 5)[1])
accs, esss, ac1 = np.array(accs), np.array(esss), np.array(ac1)
best = int(np.argmax(esss))
K["mh"]["echelles_testees"] = [float(s) for s in scales]
K["mh"]["acc_min"], K["mh"]["acc_max"] = round(float(accs.min()), 4), round(float(accs.max()), 4)
K["mh"]["echelle_optimale"] = float(scales[best])
K["mh"]["ess_max"] = round(float(esss[best]), 1)
K["mh"]["acc_a_l_echelle_optimale"] = round(float(accs[best]), 4)
K["mh"]["ess_min"] = round(float(esss.min()), 1)

fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.2))
axs[0].semilogx(scales, accs * 100, "o-", color=C["blue"], lw=2, ms=6)
axs[0].axhline(23.4, color=C["red"], ls="--", lw=1.3,
               label="repere 0,234 (comportement grande dimension)")
axs[0].scatter([scales[best]], [accs[best] * 100], s=120, facecolor="none",
               edgecolor=C["green"], lw=2, zorder=5)
axs[0].set_xlabel("largeur de proposition $s$"); axs[0].set_ylabel("taux d'acceptation (%)")
axs[0].set_title("Taux d'acceptation", fontsize=10); axs[0].legend(fontsize=8)
axs[1].semilogx(scales, esss, "o-", color=C["purple"], lw=2, ms=6)
axs[1].scatter([scales[best]], [esss[best]], s=120, facecolor="none",
               edgecolor=C["green"], lw=2, zorder=5,
               label=f"meilleur : $s$ = {scales[best]}")
axs[1].set_xlabel("largeur de proposition $s$")
axs[1].set_ylabel("taille effective d'echantillon (ESS)")
axs[1].set_title("ESS sur 20 000 iterations", fontsize=10); axs[1].legend(fontsize=8)
fig.suptitle("Choix de la proposition : ni trop petit, ni trop grand "
             "(chiffres mesures sur 11 echelles)", fontsize=11, y=1.02)
save(fig, "j04_mh_echelle_ess.png")

# --- diagnostic de melange : trois largeurs de proposition
fig = plt.figure(figsize=(9.8, 3.6))
gs = fig.add_gridspec(2, 3, hspace=.55, wspace=.28)
melange = {}
for i, (s, col, nom, cle) in enumerate([(0.02, C["red"], "trop petit", "petit"),
                                        (0.35, C["green"], "adapte", "adapte"),
                                        (6.0, C["orange"], "trop grand", "grand")]):
    ch, aa = mh_beta(4_000, s, seed=9)
    e = ess_ips(ch)
    melange[cle] = {"s": s, "acc": round(float(aa), 4), "ess": round(float(e), 1)}
    axt = fig.add_subplot(gs[0, i])
    axt.plot(np.arange(1500), ch[:1500], color=col, lw=.8)
    axt.set_title(f"$s$ = {s} -- {nom} (acc. {aa*100:.0f} %)", fontsize=9)
    axt.set_ylabel(r"$\theta^{(t)}$", fontsize=8)
    axh = fig.add_subplot(gs[1, i])
    axh.hist(ch, bins=45, density=True, color=col, alpha=.65, edgecolor="white")
    gg = np.linspace(1e-4, 1 - 1e-4, 300)
    axh.plot(gg, np.exp(logpdf_beta(gg) - lg), color=C["deep"], lw=1.4)
    axh.set_title(f"acc. {aa*100:.0f} %  --  ESS {e:.0f} / 4000", fontsize=9)
    axh.set_xlabel(r"$\theta$", fontsize=8)
K["mh"]["melange"] = melange
fig.suptitle("Diagnostic de melange selon la largeur de la proposition "
             "(4 000 iterations chacune)", fontsize=11, y=.99)
save(fig, "j05_mh_melange.png")

# =====================================================================
# 5.2b  CHAINE DE MARKOV : noyau, stationnarite, ergodicite
# =====================================================================
P = np.array([[0.80, 0.15, 0.05],
              [0.20, 0.60, 0.20],
              [0.10, 0.30, 0.60]])
w, V = np.linalg.eig(P.T)
i1 = int(np.argmin(np.abs(w - 1)))
pi = np.real(V[:, i1]); pi = pi / pi.sum()
ev = np.sort(np.abs(np.real(w)))[::-1]
gap = 1 - ev[1]
detail_bal = float(np.abs(pi[:, None] * P - (pi[None, :] * P.T)).max())
err_stat = float(np.abs(pi @ P - pi).max())


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


traces = []
for start in range(3):
    d = np.zeros(3); d[start] = 1.0
    seq = [d.copy()]
    for _ in range(25):
        d = d @ P
        seq.append(d.copy())
    traces.append(np.array(seq))
traces = np.array(traces)
dist_tv = np.array([[tv(traces[s, t], pi) for s in range(3)] for t in range(26)])
tmix = int(np.argmax((dist_tv < 0.01).all(axis=1)))
K["chaine"] = {"P": P.tolist(), "pi": [round(float(v), 5) for v in pi],
               "erreur_stationnarite": round(err_stat, 10),
               "erreur_bilan_detaille": round(detail_bal, 10),
               "valeurs_propres": [round(float(v), 4) for v in ev],
               "gap_spectral": round(float(gap), 4),
               "t_melange_TV_0.01": tmix}

fig = plt.figure(figsize=(9.9, 3.3))
gs = fig.add_gridspec(1, 3, wspace=.34)
ax = fig.add_subplot(gs[0])
ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if P[i, j] > .5 else C["deep"])
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(["$E_1$", "$E_2$", "$E_3$"]); ax.set_yticklabels(["$E_1$", "$E_2$", "$E_3$"])
ax.set_xlabel("etat suivant $j$"); ax.set_ylabel("etat courant $i$")
ax.set_title("Noyau de transition $P$", fontsize=10); ax.grid(False)
ax2 = fig.add_subplot(gs[1])
for s, col in zip(range(3), [C["red"], C["green"], C["purple"]]):
    ax2.plot(range(26), traces[s, :, 0], "o-", ms=3, color=col, lw=1.4,
             label=f"depart en $E_{s+1}$")
ax2.axhline(pi[0], color=C["deep"], ls="--", lw=1.6, label=f"$\\pi_1$ = {pi[0]:.4f}")
ax2.set_xlabel("iteration $t$"); ax2.set_ylabel(r"$P(X_t = E_1)$")
ax2.set_title("Ergodicite : oubli de l'etat initial", fontsize=10)
ax2.legend(fontsize=7.4)
ax3 = fig.add_subplot(gs[2])
for s, col in zip(range(3), [C["red"], C["green"], C["purple"]]):
    ax3.semilogy(range(26), np.maximum(dist_tv[:, s], 1e-16), "o-", ms=3, color=col, lw=1.4)
ax3.axhline(.01, color=C["grey"], ls="--", lw=1.2, label="seuil 0,01")
ax3.axvline(tmix, color=C["orange"], lw=1.6, label=f"$t_{{mel}}$ = {tmix}")
ax3.set_xlabel("iteration $t$"); ax3.set_ylabel("distance en variation totale")
ax3.set_title(f"Vitesse : gap spectral {gap:.3f}", fontsize=10)
ax3.legend(fontsize=7.4)
fig.suptitle("Chaine de Markov a 3 etats : stationnarite et convergence vers $\\pi$",
             fontsize=11, y=1.04)
save(fig, "j06_chaine_stationnaire.png")

# =====================================================================
# 5.3  GIBBS PAR BLOCS (normal bivarie) et comparaison a MH
# =====================================================================
rho = 0.85
Sig = np.array([[1.0, rho], [rho, 1.0]])
Sig_inv = np.linalg.inv(Sig)


def gibbs_bivarie(N, rho, seed=0, x0=(0.0, 0.0)):
    r = np.random.default_rng(seed)
    z = r.standard_normal((N, 2))
    out = np.empty((N, 2))
    x1, x2 = x0
    for i in range(N):
        x1 = rho * x2 + np.sqrt(1 - rho ** 2) * z[i, 0]
        x2 = rho * x1 + np.sqrt(1 - rho ** 2) * z[i, 1]
        out[i] = (x1, x2)
    return out


g1 = gibbs_bivarie(3_000, rho, seed=1, x0=(-3.0, -3.0))
g2 = gibbs_bivarie(3_000, rho, seed=2, x0=(3.0, 3.0))
ess_g = ess_ips(g1[:, 0])


def logp2(v):
    return -0.5 * float(v @ Sig_inv @ v)


def mh2d(N, scale, seed=0, x0=(0.0, 0.0)):
    r = np.random.default_rng(seed)
    z = r.standard_normal((N, 2)) * scale
    u = r.random(N)
    out = np.empty((N, 2))
    x = np.array(x0, float); lp = logp2(x); acc = 0
    for i in range(N):
        y = x + z[i]
        lpy = logp2(y)
        if np.log(u[i]) < lpy - lp:
            x, lp, acc = y, lpy, acc + 1
        out[i] = x
    return out, acc / N


mh_chain, mh_acc = mh2d(3_000, 1.2, seed=3)
ess_m = ess_ips(mh_chain[:, 0])
K["gibbs"] = {"rho": rho, "N": 3000,
              "moyenne_x1": round(float(g1[:, 0].mean()), 4),
              "ecart_type_x1": round(float(g1[:, 0].std()), 4),
              "correlation_empirique": round(float(np.corrcoef(g1[:, 0], g1[:, 1])[0, 1]), 4),
              "ess_gibbs": round(float(ess_g), 1),
              "ess_mh_2d": round(float(ess_m), 1),
              "taux_acceptation_mh_2d": round(float(mh_acc), 4),
              "ess_sur_N_gibbs": round(float(ess_g) / 3000, 3)}

fig = plt.figure(figsize=(9.9, 3.4))
gs = fig.add_gridspec(1, 3, wspace=.32)
gg = np.linspace(-3.4, 3.4, 200)
X, Y = np.meshgrid(gg, gg)
Z = np.exp(-0.5 * (Sig_inv[0, 0] * X ** 2 + 2 * Sig_inv[0, 1] * X * Y + Sig_inv[1, 1] * Y ** 2))
ax = fig.add_subplot(gs[0])
ax.contour(X, Y, Z, levels=6, colors=C["grey"], linewidths=.8)
axt = np.arange(1, 61)
ax.plot(g1[:60, 0], g1[:60, 1], "-o", color=C["blue"], lw=.9, ms=2.6)
ax.plot(g1[0, 0], g1[0, 1], "*", color=C["red"], ms=13, label="depart $(-3,-3)$")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("Trajectoire de Gibbs (60 premiers pas)", fontsize=10); ax.legend(fontsize=7.6)
ax2 = fig.add_subplot(gs[1])
ax2.plot(g1[:, 0], color=C["blue"], lw=.7, alpha=.85, label="chaine A (depart $-3$)")
ax2.plot(g2[:, 0], color=C["orange"], lw=.7, alpha=.85, label="chaine B (depart $+3$)")
ax2.axhline(0, color=C["deep"], ls="--", lw=1.2)
ax2.set_xlabel("iteration $t$"); ax2.set_ylabel("$x_1^{(t)}$")
ax2.set_title("Deux chaines, meme loi cible", fontsize=10); ax2.legend(fontsize=7.4)
ax3 = fig.add_subplot(gs[2])
ax3.bar([0, 1], [ess_g, ess_m], color=[C["green"], C["purple"]], width=.55)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["Gibbs par blocs", f"MH $s$=1,2"])
ax3.set_ylabel("ESS (coordonnee $x_1$, 3000 iter.)")
ax3.set_title(f"Gibbs : ESS {ess_g:.0f}  vs  MH : {ess_m:.0f}", fontsize=10)
for i, v in enumerate([ess_g, ess_m]):
    ax3.text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=9, color=C["deep"])
fig.suptitle(r"Gibbs par blocs sur $\mathcal{N}_2(0,\Sigma)$, $\rho$ = 0,85 -- "
             "3 000 iterations par chaine", fontsize=11, y=1.03)
save(fig, "j07_gibbs_bivarie.png")

# =====================================================================
# 5.3b  HMC : sauts saut-de-grenouille et NUTS (arbre binaire)
# =====================================================================
rho_h = 0.95
SIGH = np.array([[1.0, rho_h], [rho_h, 1.0]])
SIGH_inv = np.linalg.inv(SIGH)
Ltr = np.linalg.cholesky(SIGH)
rtr = np.random.default_rng(413)


def logp_grad_gauss(q):
    return (-0.5 * float(q @ SIGH_inv @ q), -SIGH_inv @ q)


def leapfrog(q, p, eps, L, lg):
    """Integrateur saut-de-grenouille : dp/dt = grad log p(q), dq/dt = p."""
    qn, pn = q.copy(), p.copy()
    _, g = lg(qn)
    pn += 0.5 * eps * g
    for _ in range(L - 1):
        qn += eps * pn
        _, g = lg(qn)
        pn += eps * g
    qn += eps * pn
    _, g = lg(qn)
    pn += 0.5 * eps * g
    return qn, pn


def hmc(lg, N, eps, L, seed, q0):
    r = np.random.default_rng(seed)
    q = np.array(q0, float)
    out = np.empty((N, len(q))); acc = 0; nlf = 0
    for i in range(N):
        p = r.standard_normal(len(q))
        lp0, _ = lg(q)
        H0 = -lp0 + 0.5 * float(p @ p)
        qn, pn = leapfrog(q, p, eps, L, lg)
        lp1, _ = lg(qn)
        H1 = -lp1 + 0.5 * float(pn @ pn)
        nlf += L
        if np.log(r.random()) < H0 - H1:
            q, acc = qn, acc + 1
        out[i] = q
    return out, acc / N, nlf / N


hmc_chain, hmc_acc, nlf = hmc(logp_grad_gauss, 3_000, 0.30, 10, seed=7, q0=(0.0, 0.0))
ess_h = ess_ips(hmc_chain[:, 0])
K["hmc"] = {"cible": "N2(0, [[1,.95],[.95,1]])", "eps": 0.30, "L": 10,
            "N": 3000, "taux_acceptation": round(float(hmc_acc), 4),
            "sauts_par_iteration": nlf, "ess": round(float(ess_h), 1),
            "ess_sur_N": round(float(ess_h) / 3000, 3),
            "ess_mh_2d": round(float(ess_m), 1)}

# --- NUTS : doublement d'arbre avec variable-tranche et critere de demi-tour
def build_tree(q, p, logu, v, j, eps, lg, r):
    if j == 0:
        qn, pn = leapfrog(q, p, v * eps, 1, lg)
        H = -lg(qn)[0] + 0.5 * float(pn @ pn)
        in_slice = -H > logu
        return qn, pn, qn, pn, qn, pn, (1 if in_slice else 0), (1 if in_slice else 0), (1.0 if in_slice else 0.0)
    qm, pm, qp, pp, qc, pc, nc, sc, ac = build_tree(q, p, logu, v, j - 1, eps, lg, r)
    if sc == 1:
        if v == -1:
            qm, pm, _, _, qc2, pc2, nc2, sc2, ac2 = build_tree(qm, pm, logu, v, j - 1, eps, lg, r)
        else:
            _, _, qp, pp, qc2, pc2, nc2, sc2, ac2 = build_tree(qp, pp, logu, v, j - 1, eps, lg, r)
        if r.random() < nc2 / max(nc + nc2, 1):
            qc, pc = qc2, pc2
        nc, ac = nc + nc2, ac + ac2
        d1 = float(np.dot(qp - qm, pm)); d2 = float(np.dot(qp - qm, pp))
        sc = sc2 * (1 if (d1 >= 0 and d2 >= 0) else 0)      # critere de demi-tour
    return qm, pm, qp, pp, qc, pc, nc, sc, ac


def nuts(lg, N, eps, seed, q0, maxdepth=7):
    r = np.random.default_rng(seed)
    q = np.array(q0, float)
    out = np.empty((N, len(q)))
    nlf_hist = np.empty(N, dtype=int)
    ndiv = 0
    for i in range(N):
        p0 = r.standard_normal(len(q))
        H0 = -lg(q)[0] + 0.5 * float(p0 @ p0)
        logu = -H0 + np.log(r.random())
        qm, pm = q.copy(), p0.copy()
        qp, pp = q.copy(), p0.copy()
        qnew, pnew = q.copy(), p0.copy()
        n, s, j, nlf = 1, 1, 0, 0
        while s == 1 and j < maxdepth:
            v = -1 if r.random() < .5 else 1
            if v == -1:
                qm, pm, _, _, qc, pc, nc, sc, _ = build_tree(qm, pm, logu, v, j, eps, lg, r)
            else:
                _, _, qp, pp, qc, pc, nc, sc, _ = build_tree(qp, pp, logu, v, j, eps, lg, r)
            if sc == 1 and r.random() < min(1.0, nc / n):
                qnew, pnew = qc.copy(), pc.copy()
            n += nc
            nlf += 2 ** j
            d1 = float(np.dot(qp - qm, pm)); d2 = float(np.dot(qp - qm, pp))
            s = sc * (1 if (d1 >= 0 and d2 >= 0) else 0)
            j += 1
        Hn = -lg(qnew)[0] + 0.5 * float(pnew @ pnew)
        if abs(Hn - H0) > 100.0:
            ndiv += 1
        q = qnew
        out[i] = q
        nlf_hist[i] = nlf
    return out, nlf_hist, ndiv


nuts_chain, nuts_lf, nuts_div = nuts(logp_grad_gauss, 800, 0.30, seed=11, q0=(0.0, 0.0))
ess_n = ess_ips(nuts_chain[:, 0])
# --- cout en evaluations de gradient : HMC a L fixe contre NUTS
Lgrid = np.array([1, 2, 4, 8, 16, 32])
ess_L, ess_eval_L = [], []
for LL in Lgrid:
    chL, _, _ = hmc(logp_grad_gauss, 2_000, 0.30, int(LL), seed=31 + int(LL), q0=(0.0, 0.0))
    eL = ess_ips(chL[:, 0])
    ess_L.append(eL); ess_eval_L.append(eL / (2000 * LL))
ess_eval_nuts = ess_n / (800 * max(nuts_lf.mean(), 1))
K["nuts"] = {"N": 800, "eps": 0.30, "maxdepth": 7,
             "sauts_medians_par_iteration": float(np.median(nuts_lf)),
             "sauts_moyens": round(float(nuts_lf.mean()), 1),
             "sauts_max": int(nuts_lf.max()),
             "divergences": int(nuts_div),
             "ess": round(float(ess_n), 1),
             "ess_sur_N": round(float(ess_n) / 800, 3),
             "ess_par_eval_gradient": round(float(ess_eval_nuts), 4)}
K["hmc"]["ess_par_eval_gradient"] = round(float(ess_h) / (3000 * 10), 4)
K["hmc"]["ess_selon_L"] = {int(L): round(float(e), 1) for L, e in zip(Lgrid, ess_L)}

# --- trajectoire NUTS reelle : on enregistre chaque saut d'une iteration
pts, lvl = [], []
r0 = np.random.default_rng(99)
qq = Ltr @ r0.standard_normal(2)
pp = r0.standard_normal(2)
pp_ = pp.copy()
qm, pm = qq.copy(), pp.copy()
qp = qq.copy()
pts.append(qq.copy()); lvl.append(-1)
for depth in range(6):
    lft = r0.random() < .5
    if lft:
        for k in range(2 ** depth):
            qm, pm = leapfrog(qm, pm, 0.30, 1, logp_grad_gauss)
            pts.append(qm.copy()); lvl.append(depth)
    else:
        for k in range(2 ** depth):
            qp, pp_ = leapfrog(qp, pp_, 0.30, 1, logp_grad_gauss)
            pts.append(qp.copy()); lvl.append(depth)
pts = np.array(pts); lvl = np.array(lvl)

fig = plt.figure(figsize=(9.9, 3.4))
gs = fig.add_gridspec(1, 3, wspace=.32)
gg = np.linspace(-4, 4, 220)
X, Y = np.meshgrid(gg, gg)
Z = np.exp(-0.5 * (SIGH_inv[0, 0] * X ** 2 + 2 * SIGH_inv[0, 1] * X * Y + SIGH_inv[1, 1] * Y ** 2))
ax = fig.add_subplot(gs[0])
ax.contour(X, Y, Z, levels=6, colors=C["grey"], linewidths=.8)
for s, col in zip(range(4), [C["red"], C["orange"], C["green"], C["purple"]]):
    st = Ltr @ rtr.standard_normal(2)
    p0 = rtr.standard_normal(2)
    traj = [st.copy()]
    q = st.copy(); p = p0.copy()
    for k in range(12):
        q, p = leapfrog(q, p, 0.30, 1, logp_grad_gauss)
        traj.append(q.copy())
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], "-o", ms=2.6, lw=1.1, color=col)
    ax.plot(st[0], st[1], "*", ms=9, color=col)
ax.set_xlabel("$q_1$"); ax.set_ylabel("$q_2$")
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_title("Sauts de saut-de-grenouille", fontsize=10)
ax2 = fig.add_subplot(gs[1])
ax2.contour(X, Y, Z, levels=6, colors=C["grey"], linewidths=.8)
cols = [C["red"], C["orange"], C["green"], C["blue"], C["purple"], C["cyan"]]
for d in range(6):
    m = lvl == d
    ax2.plot(pts[m, 0], pts[m, 1], "-o", ms=2.4, lw=1.0, color=cols[d],
             label=f"doublement {d} ({2**d} pas)")
ax2.plot(qq[0], qq[1], "*", color=C["deep"], ms=13, label="etat courant")
ax2.set_xlabel("$q_1$"); ax2.set_xlim(-4, 4); ax2.set_ylim(-4, 4)
ax2.set_title("NUTS : arbre par doublement", fontsize=10)
ax2.legend(fontsize=6.2, loc="lower right")
ax3 = fig.add_subplot(gs[2])
ax3.semilogx(Lgrid, np.array(ess_L) / 2000, "o-", color=C["blue"], lw=2, ms=6,
             label="HMC, $L$ fixe")
ax3.scatter([max(nuts_lf.mean(), 1)], [ess_n / 800], marker="*", s=210,
            color=C["red"], zorder=5, label="NUTS (pas median 3)")
ax3.set_xlabel("nombre de pas par iteration")
ax3.set_ylabel("ESS / iteration")
ax3.set_title("Melange selon le nombre de pas", fontsize=10)
ax3.legend(fontsize=7.6)
fig.suptitle("Dynamique hamiltonienne et NUTS -- cible $\\mathcal{N}_2$ correlee "
             "($\\rho$ = 0,95)", fontsize=11, y=1.03)
save(fig, "j08_hmc_nuts.png")

# =====================================================================
# 5.3c  FUNNEL : centree contre non centree (HMC, pas fixe)
# =====================================================================
D = 9


def lg_funnel_centree(q):
    y = q[0]; xv = q[1:]
    e = np.exp(-y)
    s = float(xv @ xv)
    lp = -y * y / 18.0 - 0.5 * e * s - 0.5 * D * y
    g = np.empty_like(q)
    g[0] = -y / 9.0 + 0.5 * e * s - 0.5 * D
    g[1:] = -xv * e
    return lp, g


def lg_funnel_non_centree(q):
    y = q[0]; z = q[1:]
    lp = -y * y / 18.0 - 0.5 * float(z @ z)
    g = np.empty_like(q)
    g[0] = -y / 9.0
    g[1:] = -z
    return lp, g


def hmc_funnel(lg, N, eps, L, seed, q0, thr=100.0):
    r = np.random.default_rng(seed)
    q = np.array(q0, float)
    out = np.empty((N, len(q))); dH = np.empty(N); acc = 0; ndiv = 0
    for i in range(N):
        with np.errstate(over="ignore", invalid="ignore"):
            p = r.standard_normal(len(q))
            lp0, _ = lg(q)
            H0 = -lp0 + 0.5 * float(p @ p)
            qn, pn = leapfrog(q, p, eps, L, lg)
            lp1, _ = lg(qn)
            if not np.isfinite(lp1) or not np.isfinite(pn).all():
                dH[i] = np.inf
                ndiv += 1
                out[i] = q
                continue
            H1 = -lp1 + 0.5 * float(pn @ pn)
            dH[i] = H1 - H0
            if abs(dH[i]) > thr:
                ndiv += 1
            if np.log(r.random()) < H0 - H1:
                q, acc = qn, acc + 1
            out[i] = q
    return out, acc / N, ndiv, dH


Nf = 1_500
cen, acc_c, div_c, dH_c = hmc_funnel(lg_funnel_centree, Nf, 0.25, 10, 21,
                                     np.r_[0.0, np.zeros(D)], thr=100.0)
nce, acc_n, div_n, dH_n = hmc_funnel(lg_funnel_non_centree, Nf, 0.25, 10, 22,
                                     np.r_[0.0, np.zeros(D)], thr=100.0)
x_c = cen[:, 1:] * np.exp(cen[:, [0]] / 2.0)
x_n = nce[:, 1:] * np.exp(nce[:, [0]] / 2.0)
rr = np.random.default_rng(5)
n_geom = 20_000
y_g = rr.normal(0, 3.0, n_geom)
x_g = rr.normal(0, np.exp(y_g / 2.0)[:, None], size=(n_geom, D))
def finitemax(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return round(float(np.abs(a).max()), 1) if a.size else float("inf")


K["funnel"] = {"D": D, "eps": 0.25, "L": 10, "N": Nf,
               "acc_centree": round(float(acc_c), 4), "acc_non_centree": round(float(acc_n), 4),
               "divergences_centree": int(div_c), "divergences_non_centree": int(div_n),
               "pct_div_centree": round(100 * div_c / Nf, 2),
               "pct_div_non_centree": round(100 * div_n / Nf, 2),
               "dH_max_fini_centree": finitemax(dH_c),
               "dH_max_fini_non_centree": finitemax(dH_n),
               "ess_centree": round(float(ess_ips(cen[:, 0])), 1),
               "ess_non_centree": round(float(ess_ips(nce[:, 0])), 1)}

fig = plt.figure(figsize=(9.9, 3.4))
gs = fig.add_gridspec(1, 3, wspace=.32)
ax = fig.add_subplot(gs[0])
ax.scatter(y_g[:4000], x_g[:4000, 0], s=3, color=C["grey"], alpha=.35, label="echantillon exact")
ax.set_xlim(-7.5, 7.5)
yy = np.linspace(-8, 7, 100)
ax.plot(yy, 2 * np.exp(yy / 2), color=C["red"], lw=1.8, label=r"$\pm 2\sigma(y)$")
ax.plot(yy, -2 * np.exp(yy / 2), color=C["red"], lw=1.8)
ax.set_xlabel("$y$ (moyenne des comtes)"); ax.set_ylabel("$x_1$ (effet centre)")
ax.set_title("Geometrie du funnel", fontsize=10); ax.legend(fontsize=7.4, loc="upper left")
ax2 = fig.add_subplot(gs[1])
ax2.scatter(cen[:, 0], cen[:, 1], s=3.4, color=C["blue"], alpha=.5)
bad = np.abs(dH_c) > 100
ax2.scatter(cen[bad, 0], cen[bad, 1], s=9, color=C["red"], alpha=.9, label="divergence")
ax2.set_xlabel("$y$"); ax2.set_ylabel("$x_1$")
ax2.set_title(f"Centree : {div_c} div. ({100*div_c/Nf:.1f} %)",
              fontsize=10)
ax2.legend(fontsize=7.6)
ax3 = fig.add_subplot(gs[2])
ax3.scatter(nce[:, 0], x_n[:, 0], s=3.4, color=C["green"], alpha=.5)
badn = np.abs(dH_n) > 100
ax3.scatter(nce[badn, 0], x_n[badn, 0], s=9, color=C["red"], alpha=.9, label="divergence")
ax3.set_xlabel("$y$"); ax3.set_ylabel("$x_1$")
ax3.set_title(f"Non centree : {div_n} divergence", fontsize=10)
ax3.legend(fontsize=7.6)
fig.suptitle("Reparametrisation centree contre non centree : "
             f"$D$ = {D} effets, $\\varepsilon$ = 0,25, $L$ = 10", fontsize=11, y=1.03)
save(fig, "j09_funnel_centree.png")

K["secondes_numpy"] = round(time.time() - t00, 1)
json.dump(K, open(os.path.join(FIGDIR, "key_numbers_ch5.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"[key] key_numbers_ch5.json ecrit (partie numpy) -- {K['secondes_numpy']} s")
