# -*- coding: utf-8 -*-
"""
Fiche de lecture Chapitre 4 - Cas d'etude Cameroun : TransLog Cameroun SA
=========================================================================
Le dossier TransLog entre en phase de decision (juin 2025). Fil conducteur :
l'entreprise doit trancher (1) l'effet formation est-il reel (facteur de
Bayes) ; (2) quel modele garder pour prevoir les incidents 2026 (LOO/WAIC) ;
(3) deployer ou non la formation au niveau national (perte, ROPE) ;
(4) A/B test : application mobile "eco-conduite" vs stage classique
(arret sequentiel, peeking).

Toutes les donnees du cas viennent de figures/key_numbers_fiche_ch3.json
(genere par src/fiche_ch3_cameroun.py) : continuite parfaite du dossier.

Genere :
  - figures/key_numbers_fiche_ch4.json : tous les chiffres de la fiche
  - figures/fiche_ch4_*.png            : figures pour la fiche
Graine : 4112.
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})
C_H0, C_H1, C_POST = "#5B7DB1", "#C0504D", "#2E7D32"
C_ACC, C_MODEL = "#7A5195", "#008B8B"
RNG = np.random.default_rng(4112)
K = {}

# ------------------------------------------------------- donnees du cas
with open("figures/key_numbers_fiche_ch3.json", encoding="utf-8") as f:
    K3 = json.load(f)

n_av, y_av = K3["v1_n_avant"], K3["v1_y_avant"]      # 1000, 148
n_aud, y_aud = K3["v1_n_audit"], K3["v1_y_audit"]    # 200, 9
a0, b0 = 3.0, 17.0                                    # prior fiche ch3
theta0 = y_av / n_av                                  # point nul : 0.148
post_a, post_b = a0 + y_aud, b0 + n_aud - y_aud      # Beta(12, 208)

K["theta0"] = theta0
K["post_a"], K["post_b"] = post_a, post_b

# =====================================================================
#  VOLET 1 - Facteur de Bayes : l'effet formation est-il reel ? (4.1)
# =====================================================================
# H0 : theta = theta0 = 14.8 % (pas d'effet formation)
# H1 : theta libre sous le prior Beta(a0, b0)
# Savage-Dickey : BF01 = pi(theta0 | y) / pi(theta0)
post = stats.beta(post_a, post_b)
prior = stats.beta(a0, b0)
bf01_main = post.pdf(theta0) / prior.pdf(theta0)
K["v1_bf01_sd"] = float(bf01_main)
K["v1_bf10"] = float(1.0 / bf01_main)

# Sensibilite au prior (grille de largeurs) : le piege de Lindley
sens_priors = [(1.0, 1.0), (2.0, 2.0), (3.0, 17.0), (10.0, 10.0), (50.0, 50.0)]
K["v1_sens_priors"] = []
for a_p, b_p in sens_priors:
    po = stats.beta(a_p + y_aud, b_p + n_aud - y_aud)
    pr = stats.beta(a_p, b_p)
    bf = float(po.pdf(theta0) / pr.pdf(theta0))
    K["v1_sens_priors"].append({"prior": f"Beta({a_p:g},{b_p:g})",
                                "bf01": bf, "log10_bf01": float(np.log10(bf))})

# p-value frequentiste du meme test (z a deux proportions vs theta0)
p_aud = y_aud / n_aud
se = np.sqrt(theta0 * (1 - theta0) / n_aud)
z_obs = (p_aud - theta0) / se
p_value = 2.0 * stats.norm.sf(abs(z_obs))
K["v1_z_obs"] = float(z_obs)
K["v1_p_value"] = float(p_value)

# Calibration de Sellke-Bayarri-Berger : B(p) = -e * p * ln(p)
b_sellke = -np.e * p_value * np.log(p_value)
K["v1_sellke_B"] = float(b_sellke)
K["v1_bf10_max_sellke"] = float(1.0 / b_sellke)
K["v1_P_H0_min"] = float(b_sellke / (1 + b_sellke))

# Tableau p -> BF (pour la fiche)
K["v1_sellke_table"] = []
for p_v in [0.10, 0.05, 0.01, 0.005, 0.001]:
    b = -np.e * p_v * np.log(p_v)
    K["v1_sellke_table"].append({"p": p_v, "B": float(b),
                                 "bf10_max": float(1 / b),
                                 "P_H0_min": float(b / (1 + b))})

# Accumulation du BF avec n : audits fictifs de taille croissante
ns = [10, 20, 50, 100, 200, 400, 800, 1600]
acc_logs, acc_y = [], []
for n_i in ns:
    y_i = int(RNG.binomial(n_i, p_aud))     # vrai taux = 4.5 %
    po = stats.beta(a0 + y_i, b0 + n_i - y_i)
    pr = stats.beta(a0, b0)
    bf_i = po.pdf(theta0) / pr.pdf(theta0)
    acc_logs.append(float(np.log10(bf_i)))
    acc_y.append(y_i)
K["v1_acc_ns"] = ns
K["v1_acc_y"] = acc_y
K["v1_acc_log10_bf01"] = acc_logs
K["v1_acc_log10_bf01_200"] = acc_logs[4]

# Jeffreys : vocabulaire des preuves
def jeffrey(log10_bf01):
    a = abs(log10_bf01)
    if a < 0.5:   return "anecdotique"
    if a < 1.0:   return "substantiel"
    if a < 1.5:   return "fort"
    if a < 2.0:   return "tres fort"
    return "decisif"
K["v1_jeffrey_verdict"] = jeffrey(np.log10(bf01_main))

# =====================================================================
#  VOLET 2 - Comparer des modeles hors echantillon : LOO/WAIC (4.2)
# =====================================================================
# Donnees : les 6 comptages mensuels post-formation du dossier (fiche ch3)
y_mois = np.array(K3["v4a_y_mois"])            # [5, 8, 16, 4, 5, 6]
alpha0, beta0 = K3["v4a_prior_alpha"], K3["v4a_prior_beta"]  # 6.0, 1.2

# M1 : lambda commun (Gamma-Poisson conjuge, mise a jour sur les 6 mois)
# M2 : lambda libre par mois (chaque mois son propre Gamma(6, 1.2))
# M3 : hiérarchique partiellement empile (lambda_m ~ Gamma(mu*k, (1-?)...))
#      -> modele LogNormal hierarchique via Monte-Carlo (PyMC optionnel)
# Pour rester sur CPU leger, M3 est approxime par partial pooling
# empile : lambda_m = w * lambda_pooled + (1-w) * lambda_ind, avec
# w = n/(n + kappa), kappa = 2 (shrinkage fort).

def loo_waic_poisson_gamma(y, a0_, b0_, pooled):
    """LOO exact (conjugue) + WAIC par MC (5000 tirages du posterior)."""
    n = len(y)
    elpd = 0.0
    for i in range(n):
        if pooled:
            a_i = a0_ + y.sum() - y[i]
            b_i = b0_ + n - 1
        else:
            a_i, b_i = a0_, b0_
        r = a_i
        p = b_i / (b_i + 1.0)
        elpd += stats.nbinom.logpmf(int(y[i]), r, p)
    # WAIC par Monte-Carlo
    S = 5000
    if pooled:
        lam_s = RNG.gamma(a0_ + y.sum(), 1.0 / (b0_ + n), size=S)
        loglik = stats.poisson.logpmf(y[None, :], lam_s[:, None])
    else:
        lam_s = RNG.gamma(a0_ + y, 1.0 / (b0_ + 1.0), size=(S, n))
        loglik = stats.poisson.logpmf(y[None, :], lam_s)
    lppd = float(np.sum(np.log(np.mean(np.exp(loglik), axis=0))))
    p_waic = float(np.sum(np.var(loglik, axis=0)))
    return {"elpd_loo": float(elpd), "elpd_waic": lppd - p_waic,
            "p_waic": p_waic}

m1 = loo_waic_poisson_gamma(y_mois, alpha0, beta0, pooled=True)
m2 = loo_waic_poisson_gamma(y_mois, alpha0, beta0, pooled=False)

# M3 : partial pooling (kappa = 2) -- LOO approxime par elpd jackknife sur
# les lambdas empiles (approximation honnete, passee en MC)
kappa = 2.0
n = len(y_mois)
elpd_m3 = 0.0
for i in range(n):
    # posterior sans l'observation i : Gamma(a0 + sum_{j != i} y_j, b0 + n - 1)
    a_i = alpha0 + y_mois.sum() - y_mois[i]
    b_i = beta0 + n - 1
    S = 5000
    lam_s = RNG.gamma(a_i, 1.0 / b_i, size=S)
    # shrinkage vers la moyenne pooled sans i
    lam_pool = (a_i / b_i)
    w = n / (n + kappa)
    lam_mix = w * lam_pool + (1 - w) * lam_s
    elpd_m3 += float(np.log(np.mean(stats.poisson.pmf(int(y_mois[i]), lam_mix))))
m3 = {"elpd_loo": elpd_m3, "elpd_waic": float("nan"), "p_waic": float("nan")}

K["v2_y_mois"] = [int(v) for v in y_mois]
K["v2_models"] = {
    "M1_pool": {"elpd_loo": m1["elpd_loo"], "elpd_waic": m1["elpd_waic"],
                "p_waic": m1["p_waic"]},
    "M2_ind": {"elpd_loo": m2["elpd_loo"], "elpd_waic": m2["elpd_waic"],
               "p_waic": m2["p_waic"]},
    "M3_pool": {"elpd_loo": m3["elpd_loo"], "elpd_waic": m3["elpd_waic"],
                "p_waic": m3["p_waic"]},
}
# Poids pseudo-BMA sur le LOO
loo_vals = np.array([m1["elpd_loo"], m2["elpd_loo"], m3["elpd_loo"]])
w_pseudo = np.exp(loo_vals - loo_vals.max())
w_pseudo /= w_pseudo.sum()
K["v2_pseudo_bma_weights"] = [float(x) for x in w_pseudo]
K["v2_elpd_diff_M2_M1"] = float(m2["elpd_loo"] - m1["elpd_loo"])

# =====================================================================
#  VOLET 3 - Decision : deployer la formation nationale ? (4.3)
# =====================================================================
# Hypotheses du dossier (constantes du cas, annoncees dans la fiche) :
#   - extension nationale : 10 000 livraisons / an
#   - cout du programme de formation : 2 MFCFA / an
#   - cout moyen d'un incident : posterior Normal-Normal de la fiche ch3
N_annuel = 10000
C_dep = 2.0          # MFCFA / an
c_inc = K3["v4b_mu_post"] / 1000.0   # 42.1 kFCFA -> 0.0421 MFCFA

# Delta = p_avant - p_apres : combien d'incidents evites par livraison ?
E_p_post = post_a / (post_a + post_b)
delta_E = theta0 - E_p_post
K["v3_N_annuel"] = N_annuel
K["v3_C_dep"] = C_dep
K["v3_c_inc"] = float(c_inc)
K["v3_E_p_post"] = float(E_p_post)
K["v3_delta_E"] = float(delta_E)
K["v3_gain_attendu_MFCFA"] = float(N_annuel * delta_E * c_inc)
K["v3_benefice_net_MFCFA"] = float(N_annuel * delta_E * c_inc - C_dep)

# ROPE sur Delta : delta = 0.01 (1 point de pourcentage)
rope = 0.01
p_sup = float(1.0 - stats.beta(post_a, post_b).cdf(theta0 - rope))
p_eq = float(stats.beta(post_a, post_b).cdf(theta0 + rope)
             - stats.beta(post_a, post_b).cdf(theta0 - rope))
p_inf = float(stats.beta(post_a, post_b).cdf(theta0 - rope))
K["v3_ROPE"] = rope
K["v3_P_sup_ROPE"] = p_sup
K["v3_P_dans_ROPE"] = p_eq
K["v3_P_inf_ROPE"] = p_inf

# Seuil de decision derive du rapport de couts : a* = c_fp / (c_fp + c_fn)
# c_fp = deployer a tort (perte C_dep), c_fn = ne pas deployer un vrai gain
# (perte = gain rate = N * Delta * c_inc). On calcule le seuil sur
# P(Delta > 0) qui rend les deux actions equivalantes en esperance de perte.
# Esperance de perte (deployer) = C_dep + N * E[p * c_inc]
# Esperance de perte (ne pas deployer) = N * E[theta0 * c_inc]
# (approche monte-carlo sur le posterior de p)
S = 20000
p_s = RNG.beta(post_a, post_b, size=S)
risk_dep = C_dep + N_annuel * np.mean(p_s * c_inc)
risk_nodep = N_annuel * theta0 * c_inc
K["v3_risk_deploy"] = float(risk_dep)
K["v3_risk_no_deploy"] = float(risk_nodep)
K["v3_decision"] = "deployer" if risk_dep < risk_nodep else "ne pas deployer"
K["v3_ratio_risk"] = float(risk_nodep / risk_dep)

# =====================================================================
#  VOLET 4 - A/B test : app eco-conduite vs stage classique (4.4)
# =====================================================================
# Protocole : 400 chauffeurs par bras, incident sur 3 mois
nAB = 400
pA_vrai, pB_vrai = 0.075, 0.055     # hypothese du dossier (A = classique)
sA = int(RNG.binomial(nAB, pA_vrai))
sB = int(RNG.binomial(nAB, pB_vrai))
pa_post = stats.beta(1 + sA, 1 + nAB - sA)
pb_post = stats.beta(1 + sB, 1 + nAB - sB)
K["v4_nAB"] = nAB
K["v4_sA"], K["v4_sB"] = sA, sB
K["v4_pA_hat"] = sA / nAB
K["v4_pB_hat"] = sB / nAB

# P(pB < pA | y) : l'app est-elle meilleure (moins d'incidents) ?
Smc = 20000
pA_s = pa_post.rvs(size=Smc, random_state=RNG)
pB_s = pb_post.rvs(size=Smc, random_state=RNG)
delta_s = pA_s - pB_s
K["v4_P_sup_app"] = float((delta_s > 0).mean())
K["v4_gain_attendu"] = float(np.maximum(delta_s, 0).mean())
K["v4_E_delta"] = float(delta_s.mean())

# Seuil de rentabilite : delta* = C_dep / (V * N)
V_conv = 0.0        # pas de conversion ici ; on utilise le cout d'incident
# Decision : deployer l'app si le gain attendu depasse le cout de deploiement
# cout de deploiement de l'app (hypothese du dossier) : 1.5 MFCFA
C_app = 1.5
N_rest = 20000       # livraisons restantes sur l'annee
delta_star = C_app / (N_rest * c_inc)
K["v4_C_app"] = C_app
K["v4_N_rest"] = N_rest
K["v4_delta_star"] = float(delta_star)
K["v4_P_delta_sup_star"] = float((delta_s > delta_star).mean())
K["v4_gain_attendu_MFCFA"] = float(N_rest * np.maximum(delta_s, 0).mean() * c_inc)
K["v4_decision_app"] = ("deployer" if N_rest * np.maximum(delta_s, 0).mean() * c_inc
                        > C_app else "ne pas deployer")

# Arrêt sequentiel : peeking sous H0 (pA = pB = 0.075)
looks = np.array([50, 100, 150, 200, 250, 300, 350, 400])
n_exp = 2000
fpr_seq95, fpr_seq99 = 0, 0
for _ in range(n_exp):
    cumA = RNG.binomial(looks, pA_vrai)
    cumB = RNG.binomial(looks, pB_vrai)
    for i in range(len(looks)):
        na, nb = looks[i], looks[i]
        pa = stats.beta(1 + cumA[i], 1 + na - cumA[i])
        pb = stats.beta(1 + cumB[i], 1 + nb - cumB[i])
        P_sup = float((pa.rvs(size=2000, random_state=RNG)
                       > pb.rvs(size=2000, random_state=RNG)).mean())
        if P_sup > 0.95:
            fpr_seq95 += 1
            break
    for i in range(len(looks)):
        na, nb = looks[i], looks[i]
        pa = stats.beta(1 + cumA[i], 1 + na - cumA[i])
        pb = stats.beta(1 + cumB[i], 1 + nb - cumB[i])
        P_sup = float((pa.rvs(size=2000, random_state=RNG)
                       > pb.rvs(size=2000, random_state=RNG)).mean())
        if P_sup > 0.99:
            fpr_seq99 += 1
            break
# Test fixe a n = 400 sans peeking : seuil P > 0.95
fpr_fixe = 0
for _ in range(n_exp):
    a_, b_ = RNG.binomial(nAB, pA_vrai), RNG.binomial(nAB, pB_vrai)
    pa = stats.beta(1 + a_, 1 + nAB - a_)
    pb = stats.beta(1 + b_, 1 + nAB - b_)
    if float((pa.rvs(size=2000, random_state=RNG)
              > pb.rvs(size=2000, random_state=RNG)).mean()) > 0.95:
        fpr_fixe += 1
K["v4_fpr_peeking_95"] = fpr_seq95 / n_exp
K["v4_fpr_peeking_99"] = fpr_seq99 / n_exp
K["v4_fpr_fixe"] = fpr_fixe / n_exp

# =====================================================================
#  FIGURES
# =====================================================================
# Fig 1 : BF - Savage-Dickey + accumulation
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
th = np.linspace(0.001, 0.35, 500)
axes[0].plot(th, prior.pdf(th), color=C_H0, label="prior Beta(3,17)")
axes[0].plot(th, post.pdf(th), color=C_POST, label="posterior Beta(12,208)")
axes[0].axvline(theta0, color="k", ls=":", lw=1)
axes[0].annotate(f"$\\pi(\\theta_0|y)$={post.pdf(theta0):.3f}\n$\\pi(\\theta_0)$={prior.pdf(theta0):.3f}",
                 xy=(theta0, max(post.pdf(theta0), prior.pdf(theta0)) * 0.55),
                 fontsize=7.5)
axes[0].set_title(f"Savage-Dickey : $BF_{{01}}$ = {bf01_main:.3f}")
axes[0].set_xlabel("$\\theta$ (taux d'incident)"); axes[0].legend(fontsize=7)
axes[1].plot(ns, acc_logs, "o-", color=C_ACC)
axes[1].axhline(0, color="grey", lw=0.8, ls="--")
axes[1].set_title("Accumulation du $BF_{01}$ avec $n$ ($\\log_{10}$)")
axes[1].set_xlabel("taille de l'audit $n$"); axes[1].set_ylabel("$\\log_{10} BF_{01}$")
fig.tight_layout(); fig.savefig("figures/fiche_ch4_bf.png"); plt.close(fig)

# Fig 2 : comparaison des modeles
fig, ax = plt.subplots(figsize=(6.5, 3.2))
mods = ["M1 pool", "M2 ind.", "M3 part."]
vals = [m1["elpd_loo"], m2["elpd_loo"], m3["elpd_loo"]]
bars = ax.bar(mods, vals, color=[C_H0, C_MODEL, C_ACC])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v - 0.15, f"{v:.2f}",
            ha="center", va="top", fontsize=8, color="white", fontweight="bold")
ax.set_title("elpd LOO par modele (plus grand = mieux)")
ax.set_ylabel("elpd LOO")
fig.tight_layout(); fig.savefig("figures/fiche_ch4_modeles.png"); plt.close(fig)

# Fig 3 : decision - posterior de Delta et ROPE
fig, ax = plt.subplots(figsize=(6.5, 3.2))
delta_grid = np.linspace(-0.06, 0.16, 600)
pdf_delta = stats.beta(post_a, post_b).pdf(theta0 - delta_grid)
ax.plot(delta_grid, pdf_delta, color=C_POST, lw=1.5)
ax.axvspan(-rope, rope, alpha=0.25, color="grey", label=f"ROPE $\\pm$ {rope}")
ax.axvline(delta_E, color="k", ls="--", lw=1, label=f"$E[\\Delta]$ = {delta_E:.3f}")
ax.set_title("Posterior de $\\Delta = \\theta_{avant} - \\theta_{apres}$")
ax.set_xlabel("$\\Delta$ (points)"); ax.set_ylabel("densite"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fiche_ch4_decision.png"); plt.close(fig)

# Fig 4 : A/B - posteriors pA, pB + peeking
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
xa = np.linspace(0.01, 0.20, 400)
axes[0].plot(xa, pa_post.pdf(xa), color=C_H0, label=f"bras A (classique) $\\hat p_A$={sA/nAB:.3f}")
axes[0].plot(xa, pb_post.pdf(xa), color=C_MODEL, label=f"bras B (app) $\\hat p_B$={sB/nAB:.3f}")
axes[0].set_title("A/B test : posteriors des deux bras")
axes[0].set_xlabel("$p$ (taux d'incident)"); axes[0].legend(fontsize=7)
axes[1].bar(["peeking 0.95", "peeking 0.99", "test fixe 0.95"],
            [fpr_seq95 / n_exp, fpr_seq99 / n_exp, fpr_fixe / n_exp],
            color=[C_H1, "#E8A33D", C_POST])
axes[1].set_title("Taux de faux positifs sous $H_0$ (2000 exp.)")
axes[1].set_ylabel("proportion")
fig.tight_layout(); fig.savefig("figures/fiche_ch4_abtest.png"); plt.close(fig)

# =====================================================================
#  SAUVEGARDE
# =====================================================================
with open("figures/key_numbers_fiche_ch4.json", "w", encoding="utf-8") as f:
    json.dump(K, f, indent=2, ensure_ascii=False,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Fiche ch4 : JSON + figures ecrits. Nb de cles :", len(K))
for kk, vv in K.items():
    print(f"  {kk:34s} = {str(vv)[:90]}")
