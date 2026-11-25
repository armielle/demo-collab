# -*- coding: utf-8 -*-
"""
Fiche de lecture Chapitre 3 - Cas d'etude Cameroun : TransLog Cameroun SA
=========================================================================
Fil conducteur : entreprise de transport routier de marchandises sur
l'axe Douala-Yaounde. Programme de formation des chauffeurs (janvier 2025).
La fiche d'enseignement reutilise TOUS ces nombres ; aucun chiffre
n'est saisi a la main dans le LaTeX : tout vient de ce script.

Genere :
  - data/translog_case.csv           : donnees brutes du cas
  - figures/key_numbers_fiche_ch3.json : tous les chiffres de la fiche
  - figures/fiche_ch3_*.png          : figures pour la fiche
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
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})
C_PRIOR, C_DONNEES, C_POST = "#5B7DB1", "#C0504D", "#2E7D32"
RNG = np.random.default_rng(4112)
K = {}   # dictionnaire des chiffres -> JSON final

# =====================================================================
#  VOLET 1 & 2 - Proportion d'incidents apres formation (Beta-Binomial)
# =====================================================================
# Historique interne (avant formation) : sur 1 000 livraisons, 148 incidents
n_avant, y_avant = 1000, 148
# Prior informatif construit sur cet historique (equivalent a 20 livraisons fictives)
a0, b0 = 3.0, 17.0
# Audit post-formation : 200 livraisons controles, 9 incidents
n_post, y_post = 200, 9

post_a, post_b = a0 + y_post, b0 + n_post - y_post
bd = stats.beta(post_a, post_b)

p_mle = y_post / n_post
p_moy = bd.mean()
p_mode = (post_a - 1) / (post_a + post_b - 2)
p_med = bd.median()

K["v1_prior_mean"] = a0 / (a0 + b0)
K["v1_prior_n"] = a0 + b0
K["v1_n_avant"] = n_avant
K["v1_y_avant"] = y_avant
K["v1_taux_avant_pct"] = 100 * y_avant / n_avant
K["v1_n_audit"] = n_post
K["v1_y_audit"] = y_post
K["v1_mle_pct"] = 100 * p_mle
K["v1_post_a"] = post_a
K["v1_post_b"] = post_b
K["v1_moyenne_pct"] = 100 * p_moy
K["v1_mode_pct"] = 100 * p_mode
K["v1_mediane_pct"] = 100 * p_med
K["v1_ecart_mode_moyenne_pt"] = 100 * (p_moy - p_mode)
K["v1_poids_donnees"] = n_post / (n_post + a0 + b0)   # part des donnees dans la moyenne
K["v1_post_params_effectifs"] = f"Beta({post_a:g}, {post_b:g})"

# ---- Intervalles de credibilite (Volet 2) ----
niveau = 0.95
q_lo, q_hi = bd.ppf(0.025), bd.ppf(0.975)
# HDI : recherche numerique de l'intervalle de largeur minimale a 95% de masse
def hdi_beta(a, b, prob=0.95, tol=1e-8):
    lo, hi = 0.0, 1.0
    f = lambda x: stats.beta.ppf(x, a, b)
    lo_ = 0.0
    hi_ = 1.0
    # optimisation grossiere puis affinage (bissection sur le point de depart)
    best = None
    for start in np.linspace(0.001, 0.5, 400):
        l, h = f(start), f(start + prob)
        if h - l < (best[1] - best[0] if best else np.inf):
            best = (l, h)
    return best[0], best[1]

hdi_lo, hdi_hi = hdi_beta(post_a, post_b)
seuil_reglementaire = 0.10
P_sous_seuil = bd.cdf(seuil_reglementaire)
# IC frequentiste de Wald pour comparaison
p_hat = p_mle
se_wald = np.sqrt(p_hat * (1 - p_hat) / n_post)
wald_lo, wald_hi = p_hat - 1.96 * se_wald, p_hat + 1.96 * se_wald

K["v2_IC95_queue_egale"] = [q_lo, q_hi]
K["v2_IC95_largeur_queue"] = q_hi - q_lo
K["v2_HDI95"] = [hdi_lo, hdi_hi]
K["v2_HDI95_largeur"] = hdi_hi - hdi_lo
K["v2_gain_largeur_HDI_pt"] = (q_hi - q_lo) - (hdi_hi - hdi_lo)
K["v2_mode_dans_HDI"] = bool(hdi_lo <= p_mode <= hdi_hi)
K["v2_P_p_sous_10pct"] = P_sous_seuil
K["v2_seuil_reglementaire_pct"] = 100 * seuil_reglementaire
K["v2_decision_sous_seuil"] = P_sous_seuil > 0.95
K["v2_IC95_Wald"] = [wald_lo, wald_hi]
K["v2_IC95_Wald_largeur"] = wald_hi - wald_lo
K["v2_Wald_borne_neg"] = wald_lo < 0

# ---- Volet 3 : prediction a posteriori ----
# Loi beta-binomiale predictive : 10 prochaines livraisons
n_next = 10
bb = stats.betabinom(n_next, post_a, post_b)
P_zero_incident = bb.pmf(0)
P_un_ou_moins = bb.cdf(1)
P_aucun_sur_20 = np.mean([stats.betabinom(20, post_a, post_b).pmf(0)])
# PPC : on simule 10 000 echantillons de taille 200 sous le posterior
S = RNG.beta(post_a, post_b, size=10_000)
S_y = RNG.binomial(n_post, S)
K["v3_n_next"] = n_next
K["v3_P_zero_incident_pct"] = 100 * P_zero_incident
K["v3_P_moins_2_pct"] = 100 * P_un_ou_moins
K["v3_P_zero_sur_20_pct"] = 100 * P_aucun_sur_20
K["v3_PPC_p25"] = float(np.percentile(S_y, 25))
K["v3_PPC_p50"] = float(np.percentile(S_y, 50))
K["v3_PPC_p975"] = float(np.percentile(S_y, 97.5))
K["v3_PPC_y_obs"] = y_post
K["v3_PPC_pct_obs_dans_95"] = bool(np.percentile(S_y, 2.5) <= y_post <= np.percentile(S_y, 97.5))

# =====================================================================
#  VOLET 4a - Nombre d'incidents par mois (Gamma-Poisson)
# =====================================================================
# Comptage mensuel des incidents (6 mois apres formation)
lam_vrai = 8.0
y_mois = RNG.poisson(lam_vrai, size=6).tolist()
# Prior Gamma informatif : on croit a ~5 incidents/mois avec une incertitude +/-2
alpha0, beta0 = 6.0, 1.2
post_alpha = alpha0 + sum(y_mois)
post_beta = beta0 + len(y_mois)
gd = stats.gamma(post_alpha, scale=1 / post_beta)
lam_moy = gd.mean()
lam_hdi = hdi_beta(post_alpha, 1 / post_beta)   # attention : hdi sur gamma -> adapter
def hdi_gamma(shape, scale, prob=0.95):
    # recherche de l'intervalle HDI numerique
    xs = np.linspace(1e-6, stats.gamma.ppf(0.9999, shape, scale=scale), 2000)
    pdf = stats.gamma.pdf(xs, shape, scale=scale)
    cdf = stats.gamma.cdf(xs, shape, scale=scale)
    # chercher l'interval le plus court contenant prob
    best = None
    for i in range(len(xs)):
        target = cdf[i] + prob
        j = np.searchsorted(cdf, target)
        if j < len(xs):
            w = xs[j] - xs[i]
            if best is None or w < best[2]:
                best = (xs[i], xs[j], w)
    return best[0], best[1]
lam_hdi_lo, lam_hdi_hi = hdi_gamma(post_alpha, 1 / post_beta)
q_lo_g, q_hi_g = gd.ppf(0.025), gd.ppf(0.975)
# Prediction du nombre d'incidents le mois prochain
pred_next = stats.nbinom(post_alpha, post_beta / (post_beta + 1))
K["v4a_prior_alpha"] = alpha0
K["v4a_prior_beta"] = beta0
K["v4a_prior_mean"] = alpha0 / beta0
K["v4a_y_mois"] = [int(v) for v in y_mois]
K["v4a_somme_y"] = int(sum(y_mois))
K["v4a_mle_lambda"] = float(np.mean(y_mois))
K["v4a_post_alpha"] = post_alpha
K["v4a_post_beta"] = post_beta
K["v4a_post_mean_lambda"] = lam_moy
K["v4a_post_mediane_lambda"] = gd.median()
K["v4a_IC95_gamma"] = [q_lo_g, q_hi_g]
K["v4a_HDI95_gamma"] = [lam_hdi_lo, lam_hdi_hi]
K["v4a_P_lambda_sup_10"] = 1 - gd.cdf(10)
K["v4a_pred_mean_mois_prochain"] = post_alpha / post_beta   # meme que la moyenne posterior
K["v4a_P_pred_sup_12"] = 1 - pred_next.cdf(12)

# =====================================================================
#  VOLET 4b - Cout moyen d'un incident (Normal-Normal)
# =====================================================================
# 15 incidents relement factures (milliers de FCFA)
mu_vrai, sig_vrai = 42.0, 12.0
couts = RNG.normal(mu_vrai, sig_vrai, size=15).round(1).tolist()
ybar = float(np.mean(couts))
s_emp = float(np.std(couts, ddof=1))
se_emp = s_emp / np.sqrt(15)
# Prior mu ~ N(50, 15^2) : le logistique estime le cout a 50 kFCFA (+/-15)
mu0, tau0 = 50.0, 15.0
w_data = se_emp**2 / (se_emp**2 + tau0**2)
mu_post = w_data * mu0 + (1 - w_data) * ybar
sd_post = np.sqrt((se_emp**2 * tau0**2) / (se_emp**2 + tau0**2))
md = stats.norm(mu_post, sd_post)
K["v4b_prior_mu"] = mu0
K["v4b_prior_sd"] = tau0
K["v4b_n"] = 15
K["v4b_couts"] = couts
K["v4b_ybar"] = ybar
K["v4b_s_emp"] = s_emp
K["v4b_se_emp"] = se_emp
K["v4b_poids_donnees"] = 1 - w_data
K["v4b_mu_post"] = mu_post
K["v4b_sd_post"] = sd_post
K["v4b_IC95_mu"] = [mu_post - 1.96 * sd_post, mu_post + 1.96 * sd_post]

# =====================================================================
#  VOLET 4c - Regression bayesienne : incidents vs volume de trafic
# =====================================================================
# 12 mois de donnees : volume de trafic (milliers de tonnes) et incidents
x_trafic = np.array([110, 118, 125, 131, 140, 148, 155, 162, 170, 178, 185, 192])
y_inc = np.array([7, 8, 9, 8, 10, 11, 12, 11, 13, 13, 14, 15])
n_reg = len(x_trafic)
# OLS
b_ols, a_ols = np.polyfit(x_trafic, y_inc, 1)
resid = y_inc - (a_ols + b_ols * x_trafic)
sig_hat = float(np.sqrt(np.sum(resid**2) / (n_reg - 2)))
se_b = sig_hat / np.sqrt(np.sum((x_trafic - x_trafic.mean())**2))
# Prior beta ~ N(0.05, 0.02^2)
b0_mu, b0_sd = 0.05, 0.02
w_b = se_b**2 / (se_b**2 + b0_sd**2)
b_post = w_b * b0_mu + (1 - w_b) * b_ols
sd_b_post = np.sqrt(se_b**2 * b0_sd**2 / (se_b**2 + b0_sd**2))
K["v4c_n"] = n_reg
K["v4c_b_ols"] = b_ols
K["v4c_a_ols"] = a_ols
K["v4c_se_b"] = se_b
K["v4c_prior_b_mu"] = b0_mu
K["v4c_prior_b_sd"] = b0_sd
K["v4c_poids_donnees"] = 1 - w_b
K["v4c_b_post"] = b_post
K["v4c_sd_b_post"] = sd_b_post
K["v4c_IC95_b"] = [b_post - 1.96 * sd_b_post, b_post + 1.96 * sd_b_post]
K["v4c_P_b_sup_0"] = 1 - stats.norm.cdf(0, b_post, sd_b_post)
# Prediction pour un volume de 200 kt
x_new = 200.0
mu_pred = a_ols + b_post * x_new
K["v4c_pred_200kt"] = mu_pred

# =====================================================================
#  VOLET 5 - Comparaison MSE frequentiste vs bayesien (simulation)
# =====================================================================
# Meme exercice que h08 du cours : n=20, theta_vrai=0.10, prior Beta(3,17)
theta_vrai = 0.10
n_sim_mse = 500
for n_obs in [5, 20]:
    mle_err, bay_err = [], []
    for _ in range(n_sim_mse):
        yv = RNG.binomial(n_obs, theta_vrai)
        mle = yv / n_obs
        # prior Beta(3,17) -> posterior Beta(3+y, 17+n-y)
        bay = (3 + yv) / (20 + n_obs)
        mle_err.append((mle - theta_vrai)**2)
        bay_err.append((bay - theta_vrai)**2)
    K[f"v5_mse_n{n_obs}_mle"] = float(np.mean(mle_err))
    K[f"v5_mse_n{n_obs}_bayes"] = float(np.mean(bay_err))
    K[f"v5_ratio_n{n_obs}"] = float(np.mean(bay_err) / np.mean(mle_err))

# =====================================================================
#  FIGURES
# =====================================================================
# Fig 1 : prior / vraisemblance / posterior (Volet 1)
fig, ax = plt.subplots(figsize=(5.4, 3.0))
xs = np.linspace(0, 0.35, 500)
ax.plot(xs, stats.beta.pdf(xs, a0, b0), color=C_PRIOR, lw=2, label=f"Prior Beta({a0:g},{b0:g})")
xs_l = np.linspace(1e-4, 0.35, 500)
vrais = stats.binom.pmf(y_post, n_post, xs_l)
vrais = vrais / vrais.max() * stats.beta.pdf(p_mle, post_a, post_b)
ax.plot(xs_l, vrais, color=C_DONNEES, lw=2, ls="--", label="Vraisemblance (200, 9)")
ax.plot(xs, bd.pdf(xs), color=C_POST, lw=2.5, label=f"Posterior {K['v1_post_params_effectifs']}")
ax.axvline(p_moy, color=C_POST, ls=":", lw=1.2, label=f"Moyenne {100*p_moy:.1f}\\%")
ax.set_xlabel("Proportion d'incidents $p$"); ax.set_ylabel("Densite")
ax.set_yticks([])
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig("figures/fiche_ch3_prior_post.png"); plt.close(fig)

# Fig 2 : intervalles queue egale vs HDI (Volet 2)
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.plot(xs, bd.pdf(xs), color=C_POST, lw=2)
ax.fill_between(xs[(xs >= q_lo) & (xs <= q_hi)], bd.pdf(xs[(xs >= q_lo) & (xs <= q_hi)]),
                color=C_PRIOR, alpha=0.30, label=f"Queue egale [{q_lo:.3f}, {q_hi:.3f}]")
ax.fill_between(xs[(xs >= hdi_lo) & (xs <= hdi_hi)], bd.pdf(xs[(xs >= hdi_lo) & (xs <= hdi_hi)]),
                color=C_DONNEES, alpha=0.25, label=f"HDI 95\\% [{hdi_lo:.3f}, {hdi_hi:.3f}]")
ax.axvline(seuil_reglementaire, color="black", ls="--", lw=1.4,
           label="Seuil reglementaire 10\\%")
ax.set_xlabel("Proportion d'incidents $p$"); ax.set_ylabel("Densite"); ax.set_yticks([])
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig("figures/fiche_ch3_intervalles.png"); plt.close(fig)

# Fig 3 : prediction a posteriori (Volet 3)
fig, ax = plt.subplots(figsize=(5.4, 2.8))
ks = np.arange(0, n_next + 1)
pmf = bb.pmf(ks)
ax.bar(ks, pmf, color=C_POST, alpha=0.75, label="Loi predictive beta-binomiale")
ax.set_xticks(ks); ax.set_xlabel("Nombre d'incidents dans les 10 prochaines livraisons")
ax.set_ylabel("Probabilite")
for k, v in zip(ks, pmf):
    ax.text(k, v + 0.004, f"{v:.2f}", ha="center", fontsize=7)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/fiche_ch3_prediction.png"); plt.close(fig)

# Fig 4 : posterior Gamma-Poisson (Volet 4a)
fig, ax = plt.subplots(figsize=(5.4, 2.8))
xs_g = np.linspace(0, 18, 500)
ax.plot(xs_g, stats.gamma.pdf(xs_g, alpha0, scale=1/beta0), color=C_PRIOR, lw=2,
        label=f"Prior Gamma({alpha0:g},{beta0:g})")
ax.plot(xs_g, gd.pdf(xs_g), color=C_POST, lw=2.5,
        label=f"Posterior Gamma({post_alpha:g},{post_beta:g})")
ax.axvline(np.mean(y_mois), color=C_DONNEES, ls="--", lw=1.5,
           label=f"MLE $\\hat\\lambda$ = {np.mean(y_mois):.2f}")
ax.fill_between(xs_g[(xs_g >= q_lo_g) & (xs_g <= q_hi_g)],
                gd.pdf(xs_g[(xs_g >= q_lo_g) & (xs_g <= q_hi_g)]),
                color=C_POST, alpha=0.18, label="IC 95\\%")
ax.set_xlabel("Nombre moyen d'incidents par mois $\\lambda$"); ax.set_yticks([])
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig("figures/fiche_ch3_gamma.png"); plt.close(fig)

# Fig 5 : regression bayesienne (Volet 4c)
fig, ax = plt.subplots(figsize=(5.4, 2.8))
ax.scatter(x_trafic, y_inc, color=C_DONNEES, s=35, zorder=3, label="Donnees (12 mois)")
xg = np.linspace(105, 200, 100)
# courbe moyenne posterior + bande IC
y_hat = a_ols + b_post * xg
sd_pred = np.sqrt(sd_b_post**2 * (xg - x_trafic.mean())**2 + sig_hat**2)
ax.plot(xg, y_hat, color=C_POST, lw=2, label="Regression bayesienne")
ax.fill_between(xg, y_hat - 1.96 * sd_pred, y_hat + 1.96 * sd_pred,
                color=C_POST, alpha=0.15, label="Bande 95\\%")
ax.axvline(200, color="black", ls=":", lw=1)
ax.text(198, 17.5, "200 kt", ha="right", fontsize=8)
ax.set_xlabel("Volume de trafic (milliers de tonnes / mois)")
ax.set_ylabel("Incidents / mois")
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig("figures/fiche_ch3_regression.png"); plt.close(fig)

# Fig 6 : MSE n=5 vs n=20 (Volet 5)
fig, ax = plt.subplots(figsize=(5.4, 2.6))
ns = [5, 20]
mle = [K["v5_mse_n5_mle"], K["v5_mse_n20_mle"]]
bay = [K["v5_mse_n5_bayes"], K["v5_mse_n20_bayes"]]
xpos = np.arange(2); w = 0.35
ax.bar(xpos - w/2, mle, w, color=C_DONNEES, label="MLE (frequentiste)")
ax.bar(xpos + w/2, bay, w, color=C_POST, label="Bayesien (prior Beta(3,17))")
for i, (m, b) in enumerate(zip(mle, bay)):
    ax.text(i - w/2, m, f"{m:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(i + w/2, b, f"{b:.4f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(xpos); ax.set_xticklabels(["n = 5", "n = 20"])
ax.set_ylabel("MSE (500 simulations)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/fiche_ch3_mse.png"); plt.close(fig)

# =====================================================================
#  SAUVEGARDE
# =====================================================================
with open("figures/key_numbers_fiche_ch3.json", "w", encoding="utf-8") as f:
    json.dump(K, f, indent=2, ensure_ascii=False,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Figures + JSON ecrits. Nb de cles :", len(K))
for kk, vv in K.items():
    print(f"  {kk:38s} = {vv}")
