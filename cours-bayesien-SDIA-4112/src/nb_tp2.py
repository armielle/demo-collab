"""TP2 : choix des priors, conjugaison, prior predictive checks, shrinkage."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mknb import build, execute, PREAMBLE

C = []
C += [("md", r"""## Objectifs

1. Reconnaître et exploiter les **quatre couples conjugués** classiques.
2. Mesurer **l'information** qu'un prior apporte, et la comparer à celle des données.
3. Faire un **contrôle prédictif a priori** (avant toute donnée) pour valider l'échelle des priors.
4. Voir le **shrinkage** d'un modèle hiérarchique sur un exemple regroupé."""),
      ("code", PREAMBLE)]

C += [("md", r"""---
# 1. Les quatre conjugaisons, vérifiées par le calcul

| Vraisemblance | A priori | A posteriori | Mise à jour |
|---|---|---|---|
| $\mathrm{Bin}(n,\theta)$ | $\mathrm{Beta}(\alpha,\beta)$ | $\mathrm{Beta}(\alpha+k,\beta+n-k)$ | on additionne |
| $\mathrm{Poisson}(\lambda)$ | $\mathrm{Gamma}(a,b)$ | $\mathrm{Gamma}(a+\sum y_i,\,b+n)$ | on additionne |
| $\N(\mu,\sigma^2)$ ($\sigma$ connu) | $\N(\mu_0,\tau_0^2)$ | $\N(\mu_n,\tau_n^2)$ | précisions additives |
| $\mathrm{Multinomial}(n,\mathbf p)$ | $\mathrm{Dirichlet}(\boldsymbol\alpha)$ | $\mathrm{Dirichlet}(\boldsymbol\alpha+\mathbf k)$ | on additionne |

Vérifions le cas Gamma-Poisson en comparant la formule close à une intégration numérique."""),
      ("code", """# --- Gamma-Poisson : comptages d'accidents dans 8 carrefours
y_counts = np.array([2, 0, 3, 1, 5, 2, 1, 4])
a0, b0 = 2.5, 1.0          # prior Gamma(a0, b0) : moyenne 2.5 accidents, assez vague
a1, b1 = a0 + y_counts.sum(), b0 + len(y_counts)
post = stats.gamma(a1, scale=1 / b1)

print(f"donnees  : n = {len(y_counts)}, total = {y_counts.sum()}")
print(f"posterior: Gamma({a1:.1f}, {b1:.1f})  moyenne = {post.mean():.4f}  "
      f"ecart-type = {post.std():.4f}")
print(f"IC 95 %%  : [{post.ppf(.025):.4f} ; {post.ppf(.975):.4f}]")
print(f"P(lambda < 2 | y) = {post.cdf(2):.4f}")

# verification numerique : le posterior non normalise Gamma ameliore par la vraisemblance
lam = np.linspace(1e-4, 8, 120000)
nonnorm = stats.gamma.pdf(lam, a0, scale=1 / b0) * np.prod(
    stats.poisson.pmf(y_counts[:, None], lam[None, :]), axis=0)
nonnorm /= np.trapezoid(nonnorm, lam)
print(f"controle numerique : moyenne = {np.trapezoid(lam*nonnorm, lam):.4f}  "
      f"(formule close : {post.mean():.4f})")""")]

C += [("md", r"""---
# 2. Combien d'information un prior apporte-t-il ?

Un prior Beta$(\alpha_0,\beta_0)$ équivaut à $\alpha_0+\beta_0$ **pseudo-observations**.
Comparons trois priors sur les mêmes données (37 conversions sur 400 visites) : le prior
de droite est un dogme déguisé."""),
      ("code", """k, n = 37, 400
priors = [("uniforme Beta(1,1)", 1, 1),
          ("faiblement informatif Beta(2,2)", 2, 2),
          ("informatif Beta(20,80) (20 % attendu)", 20, 80),
          ("dogme Beta(380,20) (95 % attendu)", 380, 20)]

th = np.linspace(1e-4, .999, 600)
plt.figure(figsize=(7.6, 3.4))
rows = []
for nom, a0, b0 in priors:
    d = stats.beta(a0 + k, b0 + n - k)
    poids = (a0 + b0) / (a0 + b0 + n)
    rows.append((nom, d.mean(), d.ppf(.025), d.ppf(.975), 1 - d.cdf(.10), poids))
    plt.plot(th, d.pdf(th), lw=2, label=f"{nom} -> moy. {d.mean():.4f}")
plt.axvline(k / n, color="k", ls="--", lw=1.2, label="MLE = 0,0925")
plt.xlabel("p (taux de conversion)"); plt.ylabel("densite a posteriori"); plt.legend(fontsize=8)
plt.title("Le meme posterior ? Non : quatre reponses differentes"); plt.xlim(0, .35); plt.show()

print(f"{'prior':<44}{'moyenne':>9}{'IC95':>20}{'P(p>0,10)':>11}{'poids prior':>13}")
for nom, m, lo, hi, pg, po in rows:
    print(f"{nom:<44}{m:>9.4f}   [{lo:.4f};{hi:.4f}]{pg:>11.4f}{po:>12.1%}")
print()
print("Lecture : avec 400 visites, le prior Beta(20,80) apporte 100 pseudo-observations")
print("(20 % du total) et deplace encore la moyenne ; le dogme Beta(380,20) refuse")
print("de voir les donnees : sa moyenne bouge de 0,950 a", f"{rows[3][1]:.4f}", "seulement.")""")]

C += [("md", r"""---
# 3. Contrôle prédictif a priori : simuler AVANT d'avoir des données

Le test décisif d'un prior : les données qu'il *prédit* sont-elles physiquement possibles ?
Exemple d'un prior mal calibré sur une échelle : on modélise des concentrations dont on sait
qu'elles sont inférieures à 100 mg/L, mais on met un prior vague $\mathrm{Demi\!-\!Normale}(1000)$."""),
      ("code", """rng = np.random.default_rng(4112)

# --- a priori TROP vague sur l'echelle : on simule ce que le modele predit
pred_vague = np.abs(rng.normal(0, 1000, 4000))
# --- a priori calibre : on sait qu'on est entre 0 et 100
pred_calibre = np.abs(rng.normal(0, 30, 4000))

fig, ax = plt.subplots(1, 2, figsize=(10, 2.9))
ax[0].hist(pred_vague, bins=60, color="#C0392B", alpha=.75)
ax[0].axvline(100, color="k", ls="--", lw=1.4)
ax[0].set_title("Prior vague $\\mathcal{N}^+(0, 1000)$ : prédit l'impossible")
ax[0].set_xlabel("concentration prédite (mg/L)")
ax[1].hist(pred_calibre, bins=60, color="#1E8449", alpha=.75)
ax[1].axvline(100, color="k", ls="--", lw=1.4)
ax[1].set_title("Prior calibré $\\mathcal{N}^+(0, 30)$ : plausible")
ax[1].set_xlabel("concentration prédite (mg/L)")
plt.tight_layout(); plt.show()

frac = (pred_vague > 100).mean()
print(f"Prior vague  : {frac:.1%} des donnees predites depassent la limite physique de 100 mg/L")
print(f"Prior calibre: {(pred_calibre > 100).mean():.1%}")
print()
print("Un prior vague n'est PAS neutre : ici il affirme que des concentrations de 2 000 mg/L")
print("sont normales. Le controle predictif a priori l'aurait revele AVANT tout echantillonnage.")""")]

C += [("md", r"""---
# 4. Shrinkage hiérarchique : l'information circule entre les groupes

Huit magasins, très différents en volume. Estimés séparément, les petits magasins donnent des
taux extrêmes. Un modèle hiérarchique les **ramène vers la moyenne commune**, d'autant plus
que le groupe est petit."""),
      ("code", """import pymc as pm, arviz as az, time

# --- donnees simulees : 8 magasins, taux de conversion ~ autour de 8 %
vrais = np.array([.05, .06, .04, .09, .10, .12, .03, .08])
visites = np.array([60, 90, 40, 400, 500, 600, 30, 250])
conversions = np.random.default_rng(3).binomial(visites, vrais)

# --- estimateur separe (MLE par magasin)
mle = conversions / visites

t0 = time.time()
with pm.Model() as m:
    mu = pm.Beta("mu", 2, 20)                 # taux moyen du reseau
    kappa = pm.HalfNormal("kappa", 20)        # concentration
    alpha = pm.Deterministic("alpha", mu * kappa)
    beta = pm.Deterministic("beta", (1 - mu) * kappa)
    p = pm.Beta("p", alpha, beta, shape=len(vrais))
    pm.Binomial("obs", n=visites, p=p, observed=conversions)
    idata = pm.sample(draws=800, tune=800, chains=2, cores=1, target_accept=.9,
                      progressbar=False, random_seed=8)
t = time.time() - t0

p_post = idata.posterior["p"].values.reshape(-1, len(vrais))
moy = p_post.mean(axis=0)
lo, hi = np.percentile(p_post, [2.5, 97.5], axis=0)

print(f"echantillonnage : {t:.0f} s, {2*800} tirages retenus")
print(f"{'magasin':>8}{'visites':>9}{'MLE':>9}{'posterieur':>12}{'IC95':>22}")
for i in range(len(vrais)):
    print(f"{i:>8}{visites[i]:>9}{mle[i]:>9.3f}{moy[i]:>12.4f}   [{lo[i]:.3f};{hi[i]:.3f}]")

plt.figure(figsize=(7.8, 3.2))
plt.errorbar(visites, mle, yerr=None, fmt="o", color="#5B6470", label="MLE par magasin")
plt.errorbar(visites, moy, yerr=[moy - lo, hi - moy], fmt="o", color="#1F6FEB",
             capsize=4, label="moyenne postérieure + IC 95 %")
plt.axhline(float(idata.posterior["mu"].mean()), color="#C0392B", ls="--",
            label=f"taux moyen du réseau = {float(idata.posterior['mu'].mean()):.3f}")
plt.xscale("log"); plt.xlabel("visites (échelle log)"); plt.ylabel("taux de conversion")
plt.legend(fontsize=9); plt.title("Shrinkage : les petits groupes sont ramenés vers la moyenne")
plt.show()""")]

C += [("md", r"""**À vous** — Comparez l'écart entre MLE et posterior pour le magasin 7 (30 visites)
et pour le magasin 6 (600 visites). Expliquez en une phrase, en utilisant la notion de
**poids relatif** de la vraisemblance.

---
# Synthèse

* La conjugaison transforme une intégrale en une **addition de paramètres**.
* Un prior se mesure : $\alpha_0+\beta_0$ pseudo-observations face à $n$ observations réelles.
* Un prior se **valide** par un contrôle prédictif a priori — c'est la seule façon de
  déceler un prior qui affirme des choses physiquement impossibles.
* Un modèle hiérarchique **emprunte de l'information** aux groupes voisins : c'est la
  réponse bayésienne au problème des petits effectifs, et la principale raison pour laquelle
  cette approche domine aujourd'hui en biostatistique et en sciences sociales.

**À rendre** : notebook exécuté + une phrase de bilan par section.""")]

p = build("TP2_priors_conjugaison_shrinkage.ipynb", C,
          "TP2 — Choix des a priori, conjugaison et shrinkage")
execute(p, "TP2_priors_conjugaison_shrinkage.ipynb", timeout=1800)
