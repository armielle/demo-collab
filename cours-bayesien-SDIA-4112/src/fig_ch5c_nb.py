"""Chapitre 5 -- verifications numeriques ponctuelles citees dans les slides.

Petit script complementaire (aucune figure) : il recalcule EXACTEMENT les nombres
publies dans ch5_a.tex et ch5_c.tex, et les consigne dans key_numbers_ch5.json.
"""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from scipy import stats
from scipy.special import digamma
from figstyle import FIGDIR

KF = os.path.join(FIGDIR, "key_numbers_ch5.json")
K = json.load(open(KF, encoding="utf-8")) if os.path.exists(KF) else {"chapitre": 5}
out = {}

# --- slide 7 : 10 succes sur 20 essais, posterior exact Beta(11,11) --------
N = 200_000
th = np.random.default_rng(4112).beta(11, 11, N)
out["beta11_11"] = {
    "N": N, "moyenne_simulee": round(float(th.mean()), 5),
    "erreur_mc": round(float(th.std() / np.sqrt(N)), 5),
    "mediane": round(float(np.median(th)), 5),
    "ic95": [round(float(v), 4) for v in np.percentile(th, [2.5, 97.5])],
    "ic95_exact": [round(float(v), 4) for v in stats.beta.ppf([.025, .975], 11, 11)],
    "P_gt_0.4": round(float((th > 0.4).mean()), 4),
    "P_gt_0.4_exact": round(float(1 - stats.beta.cdf(0.4, 11, 11)), 4)}

# --- slide 8 : 7 faces sur 10, posterior Beta(8,4) ------------------------
th8 = np.random.default_rng(4112).beta(8, 4, 200_000)
g = np.log(th8 / (1 - th8))
out["beta8_4"] = {
    "moyenne": round(float(th8.mean()), 5),
    "erreur_mc": round(float(th8.std() / np.sqrt(len(th8))), 5),
    "ic95": [round(float(v), 4) for v in np.percentile(th8, [2.5, 97.5])],
    "P_gt_0.5": round(float((th8 > 0.5).mean()), 5),
    "esperance_logodds": round(float(g.mean()), 4),
    "ic95_logodds": [round(float(v), 4) for v in np.percentile(g, [2.5, 97.5])],
    "esperance_logodds_exacte": round(float(digamma(8) - digamma(4)), 4)}

# --- exercice 5.1 --------------------------------------------------------
sd400 = 0.15 / np.sqrt(400)
out["exercice_5_1"] = {
    "erreur_mc_95": round(float(1.96 * sd400), 5),
    "N_pour_0.001": int(np.ceil((1.96 * 0.15 / 0.001) ** 2)),
    "se_pour_p042_N400": round(float(np.sqrt(0.42 * 0.58 / 400)), 5),
    "N_pour_0.005": int(np.ceil(0.42 * 0.58 / 0.005 ** 2)),
    "erreur_N400_sur_p": round(float(1.96 * np.sqrt(0.42 * 0.58 / 400)), 5)}

# --- Monte-Carlo sur le modele radon simple : MC contre PyMC -------------
# (verifie que 2 chaines x 1000 tirages suffisent pour une moyenne a 3 decimales)
rng = np.random.default_rng(77)
ref = rng.beta(23, 9, 2_000_000)
out["reference_mcmc"] = {
    "posterior_exact": "Beta(23, 9) pour 22 succes sur 30 essais",
    "moyenne_exacte": round(float(23 / 32), 5),
    "quantiles_exacts": [round(float(v), 4) for v in stats.beta.ppf([.03, .5, .97], 23, 9)],
    "queue_gauche_exacte": round(float(stats.beta.ppf(.03, 23, 9)), 4),
    "reference_MC_1e6": round(float(ref.mean()), 6)}

# --- Gibbs : effondrement de l'ESS quand la correlation augmente -----------
from figstyle import plt  # noqa: F401  (garantit la meme config numpy/threads)


def gibbs_ess(N, rho, seed=0):
    r = np.random.default_rng(seed)
    z = r.standard_normal((N, 2))
    out = np.empty(N)
    x1, x2 = 0.0, 0.0
    s = np.sqrt(1 - rho ** 2)
    for i in range(N):
        x1 = rho * x2 + s * z[i, 0]
        x2 = rho * x1 + s * z[i, 1]
        out[i] = x1
    v = out - out.mean()
    d = float(v @ v)
    kmax = min(400, N // 4)
    ac = np.array([float(v[:N - k] @ v[k:]) / d for k in range(kmax + 1)])
    s_, k = 0.0, 1
    while k < len(ac) - 1:
        p = ac[k] + ac[k + 1]
        if p <= 0:
            break
        s_ += p
        k += 2
    return N / (1 + 2 * s_), float(ac[1])


out["gibbs_correlation"] = {}
for rho in [0.5, 0.85, 0.99]:
    e, a1 = gibbs_ess(20_000, rho, seed=5)
    out["gibbs_correlation"][str(rho)] = {
        "ESS_sur_20000": round(float(e), 1), "ESS_relatif": round(float(e) / 20_000, 4),
        "autocorr_lag1": round(a1, 4)}

K["exemples_nb"] = out
json.dump(K, open(KF, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
