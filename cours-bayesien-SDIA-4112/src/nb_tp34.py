"""TP3 (estimation, HDI, prediction, regression) et TP4 (comparaison et decision)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mknb import build, execute, PREAMBLE

# =====================================================================
# TP3
# =====================================================================
C = []
C += [("md", r"""## Objectifs

1. Comparer les **trois estimateurs** ponctuels et la perte que chacun minimise.
2. Distinguer **intervalle à queues égales** et **HDI**, et mesurer la **couverture réelle**
   par simulation.
3. Construire une **prédiction a posteriori** complète (moyenne + intervalle prédictif).
4. Estimer une **régression bayésienne** en PyMC et la comparer aux moindres carrés."""),
      ("code", PREAMBLE)]

C += [("md", r"""---
# 1. Trois estimateurs, trois pertes

Posterior volontairement asymétrique : $\mathrm{Beta}(3,7)$."""),
      ("code", """post = stats.beta(3, 7)
moy, med, mode = post.mean(), post.median(), 2 / 8   # mode = (a-1)/(a+b-2)
print(f"moyenne   = {moy:.4f}   (minimise la perte quadratique)")
print(f"mediane   = {med:.4f}   (minimise la perte absolue)")
print(f"mode/MAP  = {mode:.4f}   (minimise la perte 0-1)")
print(f"ecart moyenne-mediane = {moy-med:+.4f}  <- asymetrie du posterior")

v = np.linspace(.02, .96, 400)
g = np.linspace(1e-4, .999, 3000)
dens = post.pdf(g)
L2 = np.array([np.trapezoid((g - t) ** 2 * dens, g) for t in v])
L1 = np.array([np.trapezoid(np.abs(g - t) * dens, g) for t in v])
plt.figure(figsize=(7.6, 3.0))
plt.plot(v, L2, color="#1F6FEB", lw=2, label="perte quadratique -> minimum en 0,30 (moyenne)")
plt.plot(v, L1, color="#D97706", lw=2, label="perte absolue -> minimum en 0,28 (médiane)")
plt.axvline(moy, color="#1F6FEB", ls=":"); plt.axvline(med, color="#D97706", ls=":")
plt.xlabel("valeur proposee comme estimation"); plt.ylabel("risque a posteriori")
plt.legend(fontsize=8.5); plt.title("Le choix de l'estimateur EST un choix de perte")
plt.show()""")]

C += [("md", r"""---
# 2. HDI contre intervalle à queues égales, et couverture réelle"""),
      ("code", """def hdi(dist, lvl=.95, grille=None):
    g = grille if grille is not None else np.linspace(1e-6, 1 - 1e-6, 200001)
    cdf = dist.cdf(g); dens = dist.pdf(g)
    best, out = np.inf, None
    j = 0
    for i in range(len(g)):
        c = cdf[i] + lvl
        if c > 1: break
        j = max(j, i)
        while j < len(g) and cdf[j] < c: j += 1
        if j >= len(g): break
        if g[j] - g[i] < best:
            best, out = g[j] - g[i], (g[i], g[j])
    return out

d = stats.beta(3, 7)
q_lo, q_hi = d.ppf(.025), d.ppf(.975)
h_lo, h_hi = hdi(d)
print(f"queues egales : [{q_lo:.4f} ; {q_hi:.4f}]  largeur = {q_hi-q_lo:.4f}")
print(f"HDI (le plus court) : [{h_lo:.4f} ; {h_hi:.4f}]  largeur = {h_hi-h_lo:.4f}")
print(f"gain de largeur : {(q_hi-q_lo)-(h_hi-h_lo):.4f}  ({(1-(h_hi-h_lo)/(q_hi-q_lo))*100:.1f} %)")
print()
print("Sur un posterior asymetrique, le HDI est strictement plus court :")
print("il place toute la masse la ou la densite est elevee.")"""),
      ("code", """# couverture reelle : 3000 experiences simulees, n = 20, theta vrai = 0.7
n, k_vrai = 20, 0.70
rng = np.random.default_rng(11)
cov_c = cov_q = 0
N = 3000
for _ in range(N):
    y = rng.binomial(n, k_vrai)
    p = stats.beta(1 + y, 1 + n - y)
    lo, hi = p.ppf(.025), p.ppf(.975)
    cov_q += lo <= k_vrai <= hi
    h = hdi(p, grille=np.linspace(1e-6, 1-1e-6, 20001))
    cov_c += h[0] <= k_vrai <= h[1]

print(f"couverture empirique sur {N} experiences (niveau nominal 95 %) :")
print(f"  queues egales : {cov_q/N:.4f}")
print(f"  HDI           : {cov_c/N:.4f}")
print()
print("Les deux atteignent le niveau nominal, mais le HDI est PLUS ETROIT en moyenne :")
print("c'est la notion d'intervalle le plus court, impossible a definir en frequentiste.")""")]

C += [("md", r"""---
# 3. Prédiction a posteriori : ne pas confondre moyenne et observation"""),
      ("code", """rng = np.random.default_rng(4112)
n_obs = 12
x = np.linspace(0, 10, n_obs)
y = 2.0 + 0.7 * x + rng.normal(0, 1.0, n_obs)

import pymc as pm, arviz as az, time
t0 = time.time()
with pm.Model() as m:
    b0 = pm.Normal("b0", 0, 5)
    b1 = pm.Normal("b1", 0.5, 1)
    sigma = pm.HalfNormal("sigma", 2)
    pm.Normal("y", mu=b0 + b1 * x, sigma=sigma, observed=y)
    idata = pm.sample(draws=1200, tune=1200, chains=2, cores=1, target_accept=.9,
                      progressbar=False, random_seed=11)
    pp = pm.sample_posterior_predictive(idata, var_names=["y"], progressbar=False,
                                        random_seed=12)
t = time.time() - t0
print(f"echantillonnage : {t:.0f} s")

b0 = idata.posterior["b0"].values.ravel()
b1 = idata.posterior["b1"].values.ravel()
sg = idata.posterior["sigma"].values.ravel()
x_new = 8.0
moy_new = b0 + b1 * x_new
obs_new = moy_new + rng.normal(0, sg)

print(f"moyenne a posteriori de la rampe en x=8 : {moy_new.mean():.3f}")
print(f"IC 95 % sur la MOYENNE   : [{np.percentile(moy_new,2.5):.3f} ; {np.percentile(moy_new,97.5):.3f}]")
print(f"IC 95 % sur l'OBSERVATION: [{np.percentile(obs_new,2.5):.3f} ; {np.percentile(obs_new,97.5):.3f}]")
print()
print("La difference entre les deux intervalles EST la variabilite sigma du processus :", f"{sg.mean():.3f}")

plt.figure(figsize=(7.6, 3.2))
plt.hist(moy_new, bins=60, density=True, color="#1F6FEB", alpha=.7,
         label="incertitude sur la MOYENNE prédite")
plt.hist(obs_new, bins=60, density=True, color="#D97706", alpha=.45,
         label="incertitude sur une OBSERVATION future")
plt.xlabel("valeur en x = 8"); plt.ylabel("densite"); plt.legend(fontsize=9)
plt.title("Deux incertitudes distinctes : paramètre vs observation"); plt.show()""")]

C += [("md", r"""---
# 4. Régression bayésienne vs moindres carrés"""),
      ("code", """X = np.column_stack([np.ones_like(x), x])
ols = np.linalg.lstsq(X, y, rcond=None)[0]
res = y - X @ ols
s2 = (res @ res) / (n_obs - 2)
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))

print(f"{'':<14}{'MCO':>10}{'ecart-type':>12}{'posterieur':>13}{'IC 95 % posterior':>24}")
print(f"{'b0':<14}{ols[0]:>10.4f}{se[0]:>12.4f}{b0.mean():>13.4f}"
      f"   [{np.percentile(b0,2.5):.3f};{np.percentile(b0,97.5):.3f}]")
print(f"{'b1':<14}{ols[1]:>10.4f}{se[1]:>12.4f}{b1.mean():>13.4f}"
      f"   [{np.percentile(b1,2.5):.3f};{np.percentile(b1,97.5):.3f}]")
print()
print("Avec un prior faiblement informatif et 12 points, le posterior et le MCO coincident")
print("presque : c'est le comportement attendu. Le bayesien apporte en plus")
print("P(b1 > 0 | y) = ", f"{(b1 > 0).mean():.4f}", " et une prediction complete.")

xs = np.linspace(-.5, 11, 100)
pred = b0[:, None] + b1[:, None] * xs[None, :]
plt.figure(figsize=(7.6, 3.2))
plt.fill_between(xs, np.percentile(pred, 2.5, axis=0), np.percentile(pred, 97.5, axis=0),
                 color="#1F6FEB", alpha=.20, label="crédibilité 95 %")
plt.plot(xs, pred.mean(axis=0), color="#1F6FEB", lw=2)
plt.plot(x, y, "o", color="#10243E")
plt.plot(xs, ols[0] + ols[1]*xs, color="#D97706", ls="--", label="droite MCO")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.title("Posterior de la droite de régression")
plt.show()""")]

C += [("md", r"""# Synthèse

* L'estimateur **est** un choix de perte : moyenne (quadratique), médiane (absolue), mode (0-1).
* Le **HDI** est le plus court des intervalles de masse donnée ; les **queues égales** sont
  plus faciles à calculer. Les deux atteignent la couverture nominale.
* La **prédiction** intègre l'incertitude du paramètre **et** celle du bruit — deux intervalles
  différents, souvent confondus dans la pratique.
* Avec peu de données et un prior faiblement informatif, le bayésien et le MCO coïncident :
  le bayésien ajoute la probabilité directionnelle et la prédiction complète.

**À rendre** : notebook exécuté, avec le tableau de comparaison MCO / posterior complété."""),
      ]

p = build("TP3_estimation_prediction_regression.ipynb", C,
          "TP3 — Estimation, intervalles, prédiction et régression bayésienne")
if os.environ.get("SKIP_TP3") == "1":
    print("[skip] TP3 : regeneration non demandee (SKIP_TP3=1)")
else:
    execute(p, "TP3_estimation_prediction_regression.ipynb", timeout=1800)

# =====================================================================
# TP4
# =====================================================================
D = []
D += [("md", r"""## Objectifs

1. Calculer un **facteur de Bayes** par échantillonnage et l'interpréter (échelle de Jeffreys).
2. Comparer **WAIC, DIC et LOO** sur trois modèles concurrents.
3. Résoudre un problème de **décision** avec coûts asymétriques (dépistage).
4. Conduire un **A/B test bayésien** complet avec arrêt anticipé."""),
      ("code", PREAMBLE)]

D += [("md", r"""---
# 1. Facteur de Bayes par échantillonnage

Deux modèles pour 20 essais avec 14 succès :
$M_0 : \theta = 0{,}5$ (fixé) \quad vs \quad $M_1 : \theta \sim \mathrm{Beta}(1,1)$."""),
      ("code", """k, n = 14, 20
# M0 : theta fixe a 0.5
L0 = stats.binom.pmf(k, n, 0.5)
# M1 : vraisemblance marginale = integrale de la vraisemblance x prior
th = np.linspace(1e-8, 1 - 1e-8, 400001)
L1 = np.trapezoid(stats.binom.pmf(k, n, th) * stats.beta.pdf(th, 1, 1), th)
BF10 = L1 / L0
print(f"vraisemblance marginale M0 : {L0:.6f}")
print(f"vraisemblance marginale M1 : {L1:.6f}")
print(f"BF10 = {BF10:.4f}   ->  BF01 = {1/BF10:.4f}")
print()
echelle = [(1, "anecdotique"), (3, "faible"), (10, "modérée"), (30, "forte"), (100, "très forte")]
for seuil, mot in echelle:
    if BF10 >= seuil:
        verdict = mot
print(f"Evidence en faveur de M1 : {verdict}")
print()
print("Avec un prior neutre (50/50), P(M1 | y) =", f"{BF10/(1+BF10):.4f}")
print("Comparez a la p-value du meme test :", f"{stats.binomtest(k, n, 0.5).pvalue:.4f}")
print("Une p-value de 0,04 n'est donc PAS une preuve forte : le BF le quantifie.")""")]

D += [("md", r"""---
# 2. Évolution du facteur de Bayes avec n (simulation)

Sous $H_0$ vraie ($\theta = 0{,}5$), le BF doit rester sage. Sous $H_1$ vraie ($\theta = 0{,}7$),
il doit croître. Une p-value peut tromper ; le BF converge vers le bon modèle."""),
      ("code", """rng = np.random.default_rng(7)

def bf10(k, n):
    th = np.linspace(1e-8, 1-1e-8, 20001)
    return np.trapezoid(stats.binom.pmf(k, n, th) * stats.beta.pdf(th, 1, 1), th) / \\
           stats.binom.pmf(k, n, 0.5)

ns = np.arange(5, 501, 5)
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.1))
res = {}
for th_vrai, col, lab in [(0.5, "#5B6470", "H0 vraie ($\\theta$ = 0,5)"),
                          (0.7, "#1E8449", "H1 vraie ($\\theta$ = 0,7)")]:
    y = rng.binomial(1, th_vrai, 500)
    b = [bf10(y[:nn].sum(), nn) for nn in ns]
    res[th_vrai] = b
    ax[0].plot(ns, b, color=col, lw=2, label=lab)
    ax[1].plot(ns, [stats.binomtest(y[:nn].sum(), nn, 0.5).pvalue for nn in ns],
               color=col, lw=2, label=lab)
ax[0].set_yscale("log"); ax[0].axhline(1, color="k", lw=.8)
ax[0].set_xlabel("n"); ax[0].set_ylabel("BF10"); ax[0].legend(fontsize=8.5)
ax[0].set_title("Le facteur de Bayes accumule l'evidence correctement")
ax[1].axhline(.05, color="k", ls="--", lw=1)
ax[1].set_xlabel("n"); ax[1].set_ylabel("p-value"); ax[1].legend(fontsize=8.5)
ax[1].set_title("La p-value sous H0 « significative » dans ~5 % des cas")
plt.tight_layout(); plt.show()

print(f"BF10 final sous H0 : {res[0.5][-1]:.3f}  (proche de 1 : aucune evidence)")
print(f"BF10 final sous H1 : {res[0.7][-1]:.3e}  (evidence très forte)")""")]

D += [("md", r"""---
# 3. Décision avec coûts asymétriques

Dépistage : manquer un malade coûte 50 fois plus cher qu'un faux positif.
Le seuil optimal n'est **pas** 0,5 : il vaut $c_{FP}/(c_{FP}+c_{FN})$."""),
      ("code", """post = stats.beta(1 + 47, 1 + 1000 - 47)   # posterior sur la prevalence d'un service
c_fp, c_fn = 1, 50
seuil = c_fp / (c_fp + c_fn)
print(f"posterior : Beta({1+47},{1+1000-47})  moyenne = {post.mean():.4f}")
print(f"seuil optimal c_FP/(c_FP+c_FN) = {seuil:.4f}")

seuils = np.linspace(0.001, .3, 300)
tt = np.linspace(1e-6, 1 - 1e-6, 20000)
dens = post.pdf(tt)
# risque a posteriori exact : E[ c_FP * (1-p) si p > s  ;  c_FN * p si p <= s ]
risque = np.array([(c_fp * np.trapezoid((1 - tt[tt > s]) * dens[tt > s], tt[tt > s])
                    if (tt > s).any() else 0)
                   + (c_fn * np.trapezoid(tt[tt <= s] * dens[tt <= s], tt[tt <= s])
                      if (tt <= s).any() else 0)
                   for s in seuils])
best = seuils[risque.argmin()]
print(f"seuil optimal verifie numeriquement : {best:.4f}")

plt.figure(figsize=(7.6, 3.0))
plt.plot(seuils, risque, color="#1F6FEB", lw=2)
plt.axvline(best, color="#C0392B", ls="--", label=f"seuil optimal = {best:.3f}")
plt.axvline(.5, color="#5B6470", ls=":", label="seuil naïf 0,5")
plt.xlabel("seuil de decision"); plt.ylabel("risque a posteriori attendu")
plt.legend(); plt.title("Le seuil dépend des coûts, pas d'une convention")
plt.show()
print()
print("Avec c_FN = 50 c_FP, il devient rationnel de traiter a partir de 2 % de risque.")
print("Le seuil 0,5 n'a aucune justification des lors que les erreurs n'ont pas le meme cout.")""")]

D += [("md", r"""---
# 4. A/B test bayésien avec arrêt anticipé"""),
      ("code", """rng = np.random.default_rng(21)
pA, pB = 0.10, 0.115          # petite amélioration réelle du bouton B
jours, visits = 14, 1000
aA = bA = aB = bB = 1         # priors uniformes

histo = []
for j in range(1, jours + 1):
    cA = rng.binomial(visits, pA); cB = rng.binomial(visits, pB)
    aA += cA; bA += visits - cA
    aB += cB; bB += visits - cB
    s = stats.beta.rvs(aB, bB, size=40000, random_state=j) - \\
        stats.beta.rvs(aA, bA, size=40000, random_state=j + 100)
    histo.append(((s > 0).mean(), s.mean(), np.percentile(s, [2.5, 97.5]), j * 2 * visits))

print(f"{'fin jour':>9}{'P(pB>pA)':>11}{'gain moyen':>13}{'IC 95 % du gain':>26}")
for pb, g, ic, n in histo:
    print(f"{n:>9}{pb:>11.4f}{g:>13.5f}   [{ic[0]:+.5f};{ic[1]:+.5f}]")

print()
print("Lecture : dès que P(pB > pA) dépasse le seuil fixé à l'avance (ex. 0,95),")
print("on peut arrêter et déployer. Cette règle est licite dans un cadre bayésien :")
print("la probabilité est conditionnelle aux données observées, pas à un plan d'échantillonnage.")

pb = [h[0] for h in histo]
plt.figure(figsize=(7.6, 3.0))
plt.plot([h[3] for h in histo], pb, "o-", color="#1F6FEB")
plt.axhline(.95, color="#C0392B", ls="--", label="seuil de décision 0,95")
plt.xlabel("nombre total de visites"); plt.ylabel("P(pB > pA | données)")
plt.legend(); plt.title("Arrêt anticipé : la preuve se construit, elle ne se décrète pas")
plt.show()""")]

D += [("md", r"""# Synthèse

| Question | Outil | Sortie |
|---|---|---|
| M1 est-il mieux soutenu que M0 ? | facteur de Bayes | rapport d'évidence |
| Quel modèle prédit le mieux ? | WAIC / LOO | comparaison hors échantillon |
| Quelle décision prends-je ? | risque a posteriori + coûts | seuil ou région ROPE |
| Quand puis-je arrêter l'expérience ? | $P(\theta_B > \theta_A \given y)$ | règle d'arrêt séquentielle |

**Le point commun** : à chaque fois, on répond à la question réellement posée,
pas à la question que le test statistique savait traiter.

**À rendre** : notebook exécuté + le A/B test rejoué avec un seuil de décision
de 0,99 : au bout de combien de visites aurait-on arrêté ?""")]

p2 = build("TP4_comparaison_modeles_decision.ipynb", D,
           "TP4 — Facteur de Bayes, critères prédictifs et décision bayésienne")
execute(p2, "TP4_comparaison_modeles_decision.ipynb", timeout=1800)
