"""Remplace les @@PLACEHOLDER@@ de slides/parts/ch5_c.tex par les valeurs REELLES
calculees par src/fig_ch5b_pymc.py (figures/key_numbers_ch5.json).

Le fichier source de ch5_c.tex contenait des marqueurs non substitues (la
redaction a ete interrompue) : ce script termine le travail et verifie qu'il ne
reste aucun marqueur.

Usage : python src/fill_ch5_placeholders.py [--dry-run]
"""
import json, os, re, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KF = os.path.join(ROOT, "figures", "key_numbers_ch5.json")
CIBLE = os.path.join(ROOT, "slides", "parts", "ch5_c.tex")
K = json.load(open(KF, encoding="utf-8"))
d, pri, post = K["radon_donnees"], K["radon_priors"], K["radon_posterior"]
pr, com, val = K["radon_ppc"], K["radon_comtes"], K["validation_exacte"]


def fr(x, nd=3):
    """Nombre au format francais (virgule decimale)."""
    return f"{x:.{nd}f}".replace(".", ",")


def milliers(n):
    return f"{int(n):,}".replace(",", "\u202f")     # espace fine insecable


# --- pourcentage de logements au-dessus de 10 pCi/L dans l'echantillon retenu
df = pd.read_csv(os.path.join(ROOT, "src", "data", "srrs2.dat"), skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
df["state"] = df["state"].astype(str).str.strip()
df["county"] = df["county"].astype(str).str.strip()
df["activity"] = pd.to_numeric(df["activity"], errors="coerce")
mn = df[(df.state == "MN") & df.activity.notna()]
top = list(mn.county.value_counts().head(d["n_comtes_retenus"]).index)
sub = mn[mn.county.isin(top)]
pct10 = 100 * float((sub.activity > 10).mean())

b_lo, b_hi = post["b_ic"]
expbic = [float(np.exp(-b_hi)), float(np.exp(-b_lo))]
acc = float(np.mean(post["acc_par_chaine"]))

V = {
    "NOBS": milliers(d["n_obs_retenus"]),
    "NCOM": str(d["n_comtes_retenus"]),
    "NFLOOR": str(d["n_premier_etage"]),
    "NBASEMENT": str(d["n_sous_sol"]),
    "MEANY": fr(d["log_radon_moyen"], 4),
    "MEAN1": fr(d["moyenne_brute_premier_etage"], 3),
    "MEAN0": fr(d["moyenne_brute_sous_sol"], 3),
    "DIFF": fr(d["moyenne_brute_premier_etage"] - d["moyenne_brute_sous_sol"], 3),
    "SDY": fr(pr["sd_observee"], 3),
    "PCT10": fr(pct10, 0),
    "PRIORLO": fr(pri["log_radon_prior_min"], 2),
    "PRIORHI": fr(pri["log_radon_prior_max"], 2),
    "PRIOR10": fr(pri["pct_prior_au_dessus_de_10_pCi"], 1),
    "NOPOOLMIN": fr(com["intervalle_brut_min"], 3),
    "NOPOOLMAX": fr(com["intervalle_brut_max"], 3),
    "FORESTMIN": fr(com["intervalle_partiel_min"], 3),
    "FORESTMAX": fr(com["intervalle_partiel_max"], 3),
    "MUAICP": fr(com["mu_a_moyenne"], 3),
    "RHATMAX": fr(post["rhat_max"], 5),
    "RHATA": fr(post["rhat_max_a"], 5),
    "RHATBASE": fr(post["rhat_max"], 5),
    "ESSMIN": milliers(round(post["ess_min"])),
    "ESSA": milliers(round(post["ess_min_a"])),
    "DIV": str(post["divergences"]),
    "TIME": str(round(post["secondes_echantillonnage"])),
    "DRAWSTOT": milliers(post["n_draws_total"]),
    "ACC": fr(acc, 3),
    "NSTEPS": milliers(post["n_evaluations_fonction"]),
    "MUAMEAN": fr(post["mu_a_moy"], 3),
    "MUAIC": f"{fr(post['mu_a_ic'][0], 3)} ; {fr(post['mu_a_ic'][1], 3)}",
    "SIGAMEAN": fr(post["sigma_a_moy"], 3),
    "SIGAIC": f"{fr(post['sigma_a_ic'][0], 3)} ; {fr(post['sigma_a_ic'][1], 3)}",
    "BMEAN": fr(post["b_moy"], 3),
    "BIC": f"{fr(b_lo, 3)} ; {fr(b_hi, 3)}",
    "BICEFFET": f"facteur {fr(post['rapport_etage'], 2)} : le premier étage divise le radon par 2",
    "AMIN": fr(com["intervalle_partiel_min"], 3),
    "AMAX": fr(com["intervalle_partiel_max"], 3),
    "SIGYMEAN": fr(post["sigma_y_moy"], 3),
    "EXPB": fr(post["rapport_etage"], 3),
    "EXPBINV": fr(1 / post["rapport_etage"], 1),
    "EXPBIC": f"{fr(expbic[0], 2)} ; {fr(expbic[1], 2)}",
    "PPCCOV": fr(100 * pr["couverture_94pct"], 1) + " \\%",
    "PPCMEANOBS": fr(pr["moyenne_observee"], 3),
    "PPCMEANPRED": fr(pr["moyenne_predite"], 3),
    "PPCSDOBS": fr(pr["sd_observee"], 3),
    "PPCSDPRED": fr(pr["sd_predite"], 3),
    "MCMMEAN": fr(val["moyenne_mcmc"], 5),
    "MCMQ3": fr(val["quantiles_mcmc"][0], 4),
    "MCMMED": fr(val["quantiles_mcmc"][1], 4),
    "MCMQ97": fr(val["quantiles_mcmc"][2], 4),
    "MCMMR": fr(val["rhat_p"], 5),
    "MCMESS": fr(val["ess_p"], 1),
    "MCMDIFF": fr(val["ecart_absolu_moyenne"], 5),
}

src = open(CIBLE, encoding="utf-8").read()
utilises = set(re.findall(r"@@([A-Z_0-9]+)@@", src))
manquants = sorted(utilises - set(V))
print(f"marqueurs distincts : {len(utilises)} ; valeurs fournies : {len(V)}")
if manquants:
    print("!!! marqueurs SANS valeur :", manquants)
    sys.exit(1)

out = src
for k, v in sorted(V.items(), key=lambda kv: -len(kv[0])):
    out = out.replace(f"@@{k}@@", v)
# la légende « (2 × 1000) » correspondait à l'ancien réglage d'échantillonnage
out = out.replace(r"($2 \times 1000$)", rf"($2 \times {post['n_draws_par_chaine']}$)")

reste = re.findall(r"@@[A-Z_0-9]+@@", out)
print(f"marqueurs restants apres substitution : {len(reste)}")
if reste:
    print(sorted(set(reste)))
    sys.exit(1)

if "--dry-run" in sys.argv:
    print("(mode verification : fichier non ecrit)")
else:
    open(CIBLE, "w", encoding="utf-8").write(out)
    print(f"[fill] {os.path.relpath(CIBLE, ROOT)} mis a jour")

print("\nExemples de valeurs injectees :")
for k in ["NOBS", "NCOM", "MEANY", "BMEAN", "BIC", "RHATMAX", "ESSMIN", "DRAWSTOT",
          "PPCCOV", "EXPBINV", "PCT10"]:
    print(f"  @@{k}@@ -> {V[k]}")
