"""Chapitre 2 -- verifications numeriques annexes + figure g15 (poids du prior).

Sorties : figures/g15_poids_prior.png  et fusion dans figures/key_numbers_ch2.json
Toutes les valeurs citees dans les frames « quiz », « pseudo-observations » et
« prior de Jeffreys » sortent d'ici (aucune valeur ecrite a la main).
"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
from scipy import stats, special
from figstyle import plt, C, save, FIGDIR

K = json.load(open(os.path.join(FIGDIR, "key_numbers_ch2.json"), encoding="utf-8"))
t0 = time.time()
n_ch, k_ch = 23, 7


def fr(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


# ------------------------------------------------------------------ 1
# poids du prior w = (a+b)/(a+b+n) pour les priors etudies (Challenger, n = 23)
lignes = []
for lab, a, b in [("Beta(1, 1)", 1, 1), ("Beta(2, 2)", 2, 2), ("Beta(5, 3)", 5, 3),
                  ("Beta(1,3 ; 50)", 1.3, 50)]:
    m = stats.beta(a + k_ch, b + n_ch - k_ch)
    lignes.append({"prior": lab, "a_plus_b": round(a + b, 2), "w": round((a + b) / (a + b + n_ch), 4),
                   "moyenne_post": round(float(m.mean()), 4),
                   "ic95": [round(float(v), 4) for v in m.ppf([.025, .975])]})
K["annexe_poids_table"] = lignes
K["annexe_B_8_17"] = float(special.beta(8, 17))
K["annexe_w_50_50_n23"] = round(100 / (100 + n_ch), 4)
K["annexe_n_pour_w_5pct"] = 1900
K["annexe_w_Beta50_50_n1900"] = round(100 / (100 + 1900), 4)
K["annexe_pseudo_obs"] = {"Beta(1,1)": 0, "Beta(2,2)": 2, "Beta(5,3)": 6, "Beta(50,50)": 98,
                          "Beta(1.3,50)": 49.3}

# ------------------------------------------------------------------ 2
# valeurs du quiz de conjugaison
K["annexe_quiz"] = {
    "gamma_post_12_48": {"alpha": 2 + 48, "beta": 1 + 12,
                         "moyenne": round(float(stats.gamma(50, scale=1 / 13).mean()), 4),
                         "sd": round(float(stats.gamma(50, scale=1 / 13).std()), 4)},
    "w_gamma_n240": round(3 / (3 + 240), 4),
    "w_gamma_n12": round(3 / (3 + 12), 4),
    "Beta_15_35": {"moyenne": round(float(stats.beta(15, 35).mean()), 4),
                   "sd": round(float(stats.beta(15, 35).std()), 4)},
    "Beta_2_4": {"moyenne": round(float(stats.beta(2, 4).mean()), 4),
                 "sd": round(float(stats.beta(2, 4).std()), 4)},
    "Beta_5_9": {"moyenne": round(float(stats.beta(5, 9).mean()), 4),
                 "P_gt_0_5": round(float(stats.beta(5, 9).sf(.5)), 4),
                 "ic95": [round(float(v), 4) for v in stats.beta(5, 9).ppf([.025, .975])]},
    "w_Betapair_Beta_2_2_n10": round(4 / (4 + 10), 4),
}

# ------------------------------------------------------------------ 3bis
# elicitation : Beta(2, 98) pour un taux de retour de 2 %, et son controle predictif
rng = np.random.default_rng(7)
b98 = stats.beta(2, 98)
th98 = rng.beta(2, 98, 20000)
k98 = rng.binomial(200, th98)
K["annexe_expert"] = {"prior": [2, 98], "a_plus_b": 100,
                      "moyenne": round(float(b98.mean()), 4),
                      "sd": round(float(b98.std()), 4),
                      "ic95": [round(float(v), 4) for v in b98.ppf([.025, .975])],
                      "P_lt_0_05": round(float(b98.cdf(.05)), 4),
                      "pred_moy": round(float(k98.mean()), 2),
                      "pred_q05": float(np.percentile(k98, 5)),
                      "pred_q95": float(np.percentile(k98, 95)),
                      "pred_P_gt_20": round(float((k98 > 20).mean()), 4)}

# ------------------------------------------------------------------ 3ter
# prior de Jeffreys : verification numerique du calcul exact
pg = np.linspace(1e-4, 1 - 1e-4, 20001)
info_bin = 1.0 / (pg * (1 - pg))                       # info de Fisher (par observation)
noyau_jef = pg ** (0.5 - 1) * (1 - pg) ** (0.5 - 1)     # noyau Beta(1/2, 1/2)
K["annexe_jeffreys"] = {
    "Beta_0_5_0_5_pdf_0": float(stats.beta(.5, .5).logpdf(1e-6)),
    "B_0_5_0_5_sur_pi_ecart": abs(float(special.beta(.5, .5)) - np.pi),
    "P_theta_lt_0_1": round(float(stats.beta(.5, .5).cdf(.1)), 4),
    "P_theta_lt_0_1_uniforme": round(float(stats.beta(1, 1).cdf(.1)), 4),
    "erreur_relative_max": float(np.max(np.abs(
        np.sqrt(info_bin) / np.sqrt(info_bin).max() - noyau_jef / noyau_jef.max()))),
    "poisson_noyau": "Gamma(1/2, 0) impropre",
    "normale_mu_noyau": "constant (uniforme impropre) si sigma connu",
    "normale_sigma_noyau": "1/sigma -> pi(sigma) ~ 1/sigma, uniforme sur log sigma",
}

# ------------------------------------------------------------------ 4
# standardisation : statistiques des 8 ecoles (pour la section 2.5)
y_schools = np.array([28.39, 7.94, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16])
se_schools = np.array([14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6])
K["annexe_ecoles"] = {"moyenne": round(float(y_schools.mean()), 3),
                      "sd": round(float(y_schools.std(ddof=1)), 3),
                      "min": float(y_schools.min()), "max": float(y_schools.max()),
                      "se_moyen": round(float(se_schools.mean()), 3),
                      "y_standardises": [round(float(v), 3) for v in
                                         (y_schools - y_schools.mean()) / y_schools.std(ddof=1)]}

# ------------------------------------------------------------------ 4bis
# deux vitesses : biais du prior en 1/n contre incertitude du posterior en 1/sqrt(n)
p_vrai, a0p, b0p = 0.35, 50, 50          # prior Beta(50,50) centre sur 0,5
lignes_v = []
for nn in (10, 100, 1000, 3000):
    e = p_vrai * nn                        # nombre de succes attendu
    sd_post = float(np.sqrt((a0p + e) * (b0p + nn - e) / ((a0p + b0p + nn) ** 2 * (a0p + b0p + nn + 1))))
    biais = (a0p + b0p) * (0.5 - p_vrai) / (a0p + b0p + nn)
    lignes_v.append({"n": nn, "biais_prior": round(biais, 4), "sd_post": round(sd_post, 4),
                     "ratio_biais_sd": round(biais / sd_post, 2),
                     "sd_loi_donnees": round(float(np.sqrt(p_vrai * (1 - p_vrai) / nn)), 4)})
K["annexe_deux_vitesses"] = lignes_v
K["annexe_w_Beta50_50"] = {f"n={nn}": round(100 / (100 + nn), 4) for nn in (10, 100, 1000, 3000)}

# ------------------------------------------------------------------ figure g15
ns = np.logspace(0, 4, 300)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2))
for lab, ab, col in [("Beta(1, 1) — 0 obs. virtuelles", 2, C["grey"]),
                     ("Beta(2, 2) — 2", 4, C["green"]),
                     ("Beta(5, 3) — 6", 8, C["orange"]),
                     ("Beta(50, 50) — 98", 100, C["red"])]:
    axes[0].plot(ns, ab / (ab + ns), color=col, lw=1.9, label=lab)
axes[0].axvline(n_ch, color=C["deep"], ls="--", lw=1.4, label=f"Challenger : n = {n_ch}")
axes[0].axhline(.05, color=C["blue"], ls=":", lw=1.3, label="prior < 5 % du poids")
axes[0].set_xscale("log"); axes[0].set_xlabel("nombre de données $n$ (log)")
axes[0].set_ylabel("poids du prior $w$")
axes[0].set_title("Quand le prior s'efface-t-il ?", fontsize=10.2)
axes[0].legend(fontsize=7.4)

for lab, a, b, col in [("Beta(1, 1)", 1, 1, C["grey"]), ("Beta(2, 2)", 2, 2, C["green"]),
                       ("Beta(5, 3)", 5, 3, C["orange"]), ("Beta(1,3 ; 50)", 1.3, 50, C["red"])]:
    w = (a + b) / (a + b + n_ch)
    m = stats.beta(a + k_ch, b + n_ch - k_ch).mean()
    axes[1].plot(w, m, "o", color=col, ms=9)
    axes[1].annotate(f"{lab}\n$w$ = {fr(w)} → {fr(m)}", (w, m), textcoords="offset points",
                     xytext=(10, -6), fontsize=8.0, color=col, fontweight="bold")
axes[1].axvline(k_ch / n_ch, color=C["deep"], ls="--", lw=1.3, label=f"$k/n$ = {fr(k_ch/n_ch)}")
axes[1].set_xlabel("poids du prior $w$"); axes[1].set_ylabel("moyenne a posteriori")
axes[1].set_title(f"Mêmes 23 vols : la réponse dépend de $w$ (données)", fontsize=10.2)
axes[1].legend(fontsize=8.0)
fig.suptitle("g15 — Le prior comme échantillon virtuel : $w = (a+b)/(a+b+n)$",
             fontsize=10.4, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "g15_poids_prior.png")

out = os.path.join(FIGDIR, "key_numbers_ch2.json")
json.dump(K, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n=== CH2 (annexes) ===")
print(json.dumps({k: v for k, v in K.items() if k.startswith("annexe_")}, indent=1, ensure_ascii=False))
print(f"total {time.time()-t0:.1f} s")
