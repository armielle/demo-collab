"""Mini-projet guide : modele hierarchique bayesien sur les donnees radon du Minnesota.

Jeu de donnees : src/data/srrs2.dat (Radon SRRS, Minnesota), telecharge depuis
l'hebergement du cours « Bayesian Data Analysis » de Gelman et al.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mknb import build, execute, PREAMBLE

C = []
C += [("md", r"""# Mini-projet — modélisation bayésienne complète

**Contexte.** Le radon est un gaz radioactif cancérigène qui s'accumule dans les sous-sols.
L'État du Minnesota a mesuré le radon dans des milliers de logements. Question appliquée :

> **Le niveau de radon dépend-il de l'étage du logement (sous-sol ou rez-de-chaussée), et cet
> effet est-il le même dans tous les comtés ?**

**Données réelles** : `src/data/srrs2.dat` (relevé SRRS, Minnesota), une ligne par logement.

## Les six étapes du pipeline (à reproduire dans votre propre rendu)

| # | Étape | Livrable attendu |
|---|---|---|
| 1 | Formulation du modèle | écriture de la vraisemblance et des priors |
| 2 | Justification des priors | échelle, ordre de grandeur, contrôle prédictif a priori |
| 3 | Vérification prédictive a priori | graphique + conclusion écrite |
| 4 | Échantillonnage | modèle PyMC, sortie `idata` |
| 5 | Diagnostics | $\hat R$, ESS, trace, divergences |
| 6 | Prédiction et décision | $P(\text{effet étage} < 0 \given y)$, recommandation |"""),
      ("code", PREAMBLE)]

C += [("md", r"""---
# Étape 0 — lecture et exploration des données"""),
      ("code", """import pandas as pd
DATA = os.path.join("..", "data", "srrs2.dat")
if not os.path.exists(DATA):
    DATA = os.path.join("..", "src", "data", "srrs2.dat")

df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
df = df[(df.state == "MN") & df.activity.notna() & (df.activity > 0)].copy()
df["log_radon"] = np.log(df.activity)
df["floor"] = df.floor.astype(int)              # 0 = sous-sol, 1 = rez-de-chaussee

# on restreint aux comtes suffisamment representes : le modele doit pouvoir estimer
# un effet par comte, sinon le pooling hierarchique n'a rien a apprendre
cnt = df.cntyfips.value_counts()
gardes = cnt[cnt >= 25].index[:8]
d = df[df.cntyfips.isin(gardes)].copy()
d["comte"] = d.cntyfips.astype("category").cat.codes
n_comtes = d.comte.nunique()

print(f"logements retenus : {len(d)}  |  comtes : {n_comtes}")
print("(reglage reduit pour tenir sur un CPU sans GPU : 8 comtes, 400 tirages par chaine)")
print(f"radon brut (Bq/m3) : mediane {df.activity.median():.0f}, "
      f"max {df.activity.max():.0f}")
print()
print(d.groupby("floor")["log_radon"].agg(["count", "mean", "std"]).round(3).to_string())
print()
print("Difference brute des moyennes (rez-de-chaussee - sous-sol) :",
      round(d[d.floor == 1].log_radon.mean() - d[d.floor == 0].log_radon.mean(), 3))
d.head()""")]

C += [("md", r"""---
# Étape 1 — le modèle

$$y_i \sim \N(a_{c[i]} + b\,\mathrm{floor}_i,\ \sigma_y^2)$$

$$a_c \sim \N(\mu_a, \sigma_a^2),\qquad b \sim \N(0, 1),\qquad \sigma_y \sim \N^+(0, 1)$$

* $a_c$ : log-radon moyen du comté $c$, **effet aléatoire** (le pooling partiel vient de là) ;
* $b$ : effet de l'étage, **effet fixe** supposé identique partout ;
* $\mu_a, \sigma_a$ : moyenne et dispersion des comtés (priors faiblement informatifs).

**Justification des échelles** : les données sont des log de concentrations en Bq/m³, donc des
valeurs typiquement entre 0 et 3. Un prior $\N(0,1)$ sur $b$ autorise des effets de 0 à ±2
(soit un facteur $e^2 \approx 7$ sur le radon) : large, mais pas absurde."""),
      ("code", """import pymc as pm, arviz as az, time

x = d.floor.values.astype(float)
y = d.log_radon.values
idx = d.comte.values
n = len(y)

# --- etape 3 : controle predictif A PRIORI (avant de conditionner sur les donnees)
with pm.Model() as mm:
    mu_a = pm.Normal("mu_a", 0.0, 1.0)
    sigma_a = pm.HalfNormal("sigma_a", 1.0)
    a_off = pm.Normal("a_off", 0.0, 1.0, shape=n_comtes)
    a = mu_a + a_off * sigma_a
    b = pm.Normal("b", 0.0, 1.0)
    sigma_y = pm.HalfNormal("sigma_y", 1.0)
    pm.Normal("log_radon", mu=a[idx] + b * x, sigma=sigma_y, observed=y)
    prior_pred = pm.sample_prior_predictive(samples=200, random_seed=1)

pp = prior_pred.prior_predictive["log_radon"].values.ravel()
print("Controle predictif a priori :")
print(f"  intervalle des donnees simulees : [{np.percentile(pp, .5):.2f} ; {np.percentile(pp, 99.5):.2f}]")
print(f"  intervalle des donnees reelles  : [{y.min():.2f} ; {y.max():.2f}]")
print(f"  part des simulations hors des bornes reelles : "
      f"{((pp < y.min()) | (pp > y.max())).mean():.1%}")"""),
      ("code", """plt.figure(figsize=(7.6, 3.0))
plt.hist(pp, bins=80, density=True, color="#D97706", alpha=.6,
         label="prédictive a priori")
plt.hist(y, bins=40, density=True, color="#1F6FEB", alpha=.75, label="données observées")
plt.xlabel("log(radon)"); plt.ylabel("densité")
plt.legend(); plt.title("Étape 3 — les données prédites par les priors sont-elles plausibles ?")
plt.show()""")]

C += [("md", r"""**Lecture à écrire dans votre rendu** : les deux distributions doivent se recouvrir largement.
Un prior qui prédit des log-radon de $-8$ (soit 0,0003 Bq/m³) ou de $+15$ (3 millions de Bq/m³)
doit être recalibré immédiatement.

---
# Étape 4 — échantillonnage"""),
      ("code", """t0 = time.time()
with mm:
    idata = pm.sample(draws=400, tune=400, chains=2, cores=1, target_accept=.9,
                      progressbar=False, random_seed=42)
    idata.extend(pm.sample_posterior_predictive(idata, progressbar=False, random_seed=43))
print(f"echantillonnage : {time.time()-t0:.0f} s")

# --- etape 5 : diagnostics
r = az.rhat(idata)
e = az.ess(idata)
print()
print(f"{'parametre':<12}{'r_hat':>9}{'ESS':>9}")
for v in ["mu_a", "sigma_a", "b", "sigma_y"]:
    print(f"{v:<12}{float(r[v].values):>9.4f}{float(e[v].values):>9.0f}")
print(f"divergences : {int(idata.sample_stats['diverging'].sum())}")
print(f"effet de l'etage b : moyenne {float(idata.posterior['b'].mean()):.4f}  "
      f"IC 95 % [{np.percentile(idata.posterior['b'].values, 2.5):.4f} ; "
      f"{np.percentile(idata.posterior['b'].values, 97.5):.4f}]")
print(f"P(b < 0 | y) = {(idata.posterior['b'].values < 0).mean():.4f}")""")]

C += [("md", r"""---
# Étape 6 — visualisation et décision

Le graphique en forêt (*forest plot*) ci-dessous est le livrable attendu : un effet par comté,
avec son intervalle de crédibilité, et la moyenne globale en référence."""),
      ("code", """a_post = idata.posterior["a_off"].values.reshape(-1, n_comtes) * \\
         idata.posterior["sigma_a"].values.reshape(-1, 1) + \\
         idata.posterior["mu_a"].values.reshape(-1, 1)
moy = a_post.mean(axis=0)
lo, hi = np.percentile(a_post, [2.5, 97.5], axis=0)
taille = d.groupby("comte").size().values
ordre = np.argsort(moy)

plt.figure(figsize=(7.8, 4.2))
plt.errorbar(moy[ordre], np.arange(n_comtes),
             xerr=[moy[ordre] - lo[ordre], hi[ordre] - moy[ordre]],
             fmt="o", color="#1F6FEB", capsize=4,
             label="log-radon moyen du comté + IC 95 %")
plt.axvline(float(idata.posterior["mu_a"].mean()), color="#C0392B", ls="--",
            label="moyenne de l'État")
for k, i in enumerate(ordre):
    plt.text(hi[i] + .02, k, f"n={taille[i]}", va="center", fontsize=8, color="#5B6470")
plt.yticks(np.arange(n_comtes), [f"comté {i}" for i in ordre], fontsize=8)
plt.xlabel("log(radon), sous-sol")
plt.title("Étape 6 — effets par comté (pooling partiel hiérarchique)")
plt.legend(fontsize=9); plt.tight_layout(); plt.show()

print("Interprétation : les comtés à petit effectif sont ramenés vers la moyenne de l'État ;")
print("ceux à fort effectif gardent leur valeur propre. C'est le pooling partiel.")

# decision
b_post = idata.posterior["b"].values.ravel()
print()
print("DECISION APPLIQUEE")
print(f"  effet de l'etage : {b_post.mean():+.4f} en log-radon, "
      f"soit un facteur {np.exp(b_post.mean()):.3f} sur la concentration")
print(f"  P(effet < 0 | y) = {(b_post < 0).mean():.4f}  ->  "
      f"{'le rez-de-chaussee est moins expose' if (b_post < 0).mean() > .95 else 'preuve insuffisante'}")
print(f"  probabilite que l'effet soit superieur a un facteur 2 : "
      f"{(b_post < -np.log(2)).mean():.4f}")""")]

C += [("md", r"""---
# Grille d'évaluation du rendu (20 points)

| Critère | Points | Attendu |
|---|---|---|
| Modèle écrit et justifié | 4 | vraisemblance, priors, échelles discutées |
| Contrôle prédictif a priori | 3 | graphique **et** conclusion écrite |
| Diagnostics rapportés | 3 | $\hat R$, ESS, divergences, trace |
| Résultats chiffrés | 3 | effet de l'étage avec intervalle de crédibilité |
| Visualisation | 3 | graphique lisible, légende, interprétation |
| Décision argumentée | 2 | réponse à la question posée, avec le risque assumé |
| Reproductibilité | 2 | script/notebook exécutable de bout en bout, graine fixée |

## Ce que l'on attend en une phrase

> « L'effet de l'étage vaut … en log-radon, avec un intervalle de crédibilité de … ;
> la probabilité qu'il soit négatif est de … ; les comtés se répartissent autour de …
> avec un pooling partiel visible sur les petits effectifs. »

**Pour aller plus loin** : remplacer l'effet fixe $b$ par un effet aléatoire par comté
(modèle à pentes variables), puis comparer les deux modèles par WAIC (chapitre 4)."""),
      ]

p = build("PROJET_modele_hierarchique_radon.ipynb", C,
          "Mini-projet — modèle hiérarchique bayésien sur les données radon")
execute(p, "PROJET_modele_hierarchique_radon.ipynb", timeout=3600)
