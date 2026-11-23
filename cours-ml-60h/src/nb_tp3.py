"""TP3 -- regression et regularisation : fabrique et execute le notebook."""
import os, time
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NBDIR = os.path.join(ROOT, "notebooks")
KERNEL = "ml-60h"
ENTETE = ("%matplotlib inline\n"
          "import numpy as np, pandas as pd\n"
          "import matplotlib.pyplot as plt\n"
          "np.random.seed(4112)\n"
          "print('environnement pret')")

c = [
 ("md", """## 1. Les données

Jeu **diabetes** (fourni avec scikit-learn) : 442 patients, 10 variables
cliniques, cible = progression de la maladie un an plus tard."""),
 ("code", r"""from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)
print("formes :", X.shape)
print("cible  : moyenne", round(y.mean(), 1), "| ecart-type", round(y.std(), 1))
X.head(3)"""),
 ("md", "## 2. Découper avant toute transformation"),
 ("code", r"""from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=4112)
print("entrainement :", Xtr.shape, "| test :", Xte.shape)"""),
 ("md", "## 3. La référence : OLS dans un pipeline"),
 ("code", r"""from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ols = Pipeline([("sc", StandardScaler()), ("reg", LinearRegression())]).fit(Xtr, ytr)
pred = ols.predict(Xte)
print(f"R2   : {r2_score(yte, pred):.3f}")
print(f"RMSE : {np.sqrt(mean_squared_error(yte, pred)):.2f}")
print(f"MAE  : {mean_absolute_error(yte, pred):.2f}")"""),
 ("md", "## 4. Lire les coefficients"),
 ("code", r"""coefs = pd.Series(ols.named_steps["reg"].coef_, index=X.columns).sort_values()
print(coefs.round(2).to_string())"""),
 ("md", "## 5. Diagnostiquer les résidus"),
 ("code", r"""res = yte - pred
plt.figure(figsize=(7, 3.4))
plt.scatter(pred, res, s=16, color="#D97706", alpha=.75)
plt.axhline(0, color="grey", lw=1)
plt.xlabel("valeur prédite"); plt.ylabel("résidu")
plt.title("Résidus : un nuage sans structure est le signe attendu")
plt.show()
print("correlation |residu| / prediction :", round(float(np.corrcoef(pred, np.abs(res))[0,1]), 3))"""),
 ("md", """## 6. Ridge et Lasso : les chemins

On fait varier $\\lambda$ et on observe les coefficients."""),
 ("code", r"""from sklearn.linear_model import Ridge, Lasso

alphas = np.logspace(-2, 3, 40)
Xtr_s = StandardScaler().fit_transform(Xtr)
chemins_r, chemins_l = [], []
for a in alphas:
    chemins_r.append(Ridge(alpha=a).fit(Xtr_s, ytr).coef_)
    chemins_l.append(Lasso(alpha=a, max_iter=20000).fit(Xtr_s, ytr).coef_)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for j in range(X.shape[1]):
    axes[0].semilogx(alphas, np.array(chemins_r)[:, j], lw=1.1)
    axes[1].semilogx(alphas, np.array(chemins_l)[:, j], lw=1.1)
axes[0].set_title("Ridge : contraction"); axes[1].set_title("Lasso : annulation")
for ax in axes: ax.set_xlabel("lambda")
plt.show()
print("coefficients Lasso non nuls au plus grand lambda :",
      int((np.abs(chemins_l[-1]) > 1e-8).sum()))"""),
 ("md", "## 7. Choisir $\\lambda$ par validation croisée"),
 ("code", r"""from sklearn.linear_model import RidgeCV
modele = Pipeline([("sc", StandardScaler()),
                   ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 40), cv=5))]).fit(Xtr, ytr)
print("lambda retenu :", round(float(modele.named_steps["ridge"].alpha_), 3))
print("RMSE test     :", round(float(np.sqrt(mean_squared_error(yte, modele.predict(Xte)))), 2))"""),
 ("md", """## 8. Grande dimension : quand la régularisation change tout

150 variables pour 120 observations, dont 8 seulement sont informatives."""),
 ("code", r"""rng = np.random.default_rng(4112)
n, p, k = 120, 150, 8
Xg = rng.normal(size=(n, p))
beta = np.zeros(p); beta[:k] = rng.uniform(1.5, 3.0, k) * rng.choice([-1, 1], k)
yg = Xg @ beta + rng.normal(0, 1.0, n)
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg, test_size=.3, random_state=4112)

def rmse(m):
    mod = Pipeline([("sc", StandardScaler()), ("m", m)]).fit(Xg_tr, yg_tr)
    return round(float(np.sqrt(mean_squared_error(yg_te, mod.predict(Xg_te)))), 2), mod

r_ols, _ = rmse(LinearRegression())
r_ridge, _ = rmse(RidgeCV(alphas=np.logspace(-2, 4, 40)))
r_lasso, m_lasso = rmse(Lasso(alpha=0.3, max_iter=50000))
print(f"OLS   : {r_ols}")
print(f"Ridge : {r_ridge}")
print(f"Lasso : {r_lasso}")
print("coefficients non nuls :", int((np.abs(m_lasso.named_steps['m'].coef_) > 1e-6).sum()))"""),
 ("md", """## 9. À vous (noté)

**1.** Ajoutez une variable $x^2$ (carré d'une des colonnes) : le $R^2$ de test
augmente-t-il ? Que se passe-t-il si vous ajoutez tous les carrés ?

**2.** Sur le jeu de grande dimension, faites varier l'alpha du Lasso
(0,01 ; 0,1 ; 1 ; 10). Combien de coefficients restent non nuls à chaque fois ?

**3.** Expliquez en deux phrases pourquoi Ridge n'apporte rien sur le jeu du
diabète mais pourrait aider si deux variables étaient corrélées à 0,99."""),
 ("code", r"""# ---- A VOUS ----
# 1)
# 2)
# 3)"""),
 ("md", """## 10. Ce qu'il faut retenir

| Geste | Pourquoi |
|---|---|
| OLS d'abord | c'est la référence à battre |
| standardiser dans le pipeline | sinon la pénalité dépend des unités |
| régulariser si p grand ou colinéaire | sinon l'OLS suffit |
| choisir lambda par CV | le test ne sert qu'une fois |
| lire les résidus | ils disent ce que le modèle n'a pas capté |"""),
]


def nb(titre, duree, bareme, obj, trav, cellules):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": KERNEL, "display_name": "ML 60h", "language": "python"}
    n.cells.append(nbf.v4.new_markdown_cell(
        f"""# {titre}

**Cours :** Introduction au Machine Learning — 60 h — Licence 3 SDIA
**Enseignante :** Dr. Noulapeu Ngaffo Armielle — ENSPD, Université de Douala
**Durée conseillée :** {duree} · **Barème :** {bareme}

---

## Objectifs

{obj}

## Travail demandé

{trav}

## Mode d'emploi

- Exécuter dans l'ordre ; tout tourne sur CPU en moins d'une minute.
- Les cellules **« À vous »** sont évaluées. Graine fixée."""))
    n.cells.append(nbf.v4.new_code_cell(ENTETE))
    for typ, src in cellules:
        n.cells.append(nbf.v4.new_markdown_cell(src) if typ == "md"
                       else nbf.v4.new_code_cell(src))
    return n


n = nb("TP3 — Régression et régularisation",
       "3 h", "15 points",
       "- Ajuster un OLS et le diagnostiquer.\n"
       "- Tracer les chemins Ridge et Lasso.\n"
       "- Choisir lambda par validation croisée.\n"
       "- Montrer expérimentalement quand la régularisation gagne.",
       "1. Répondre aux 3 questions de la section « À vous ».\n"
       "2. Rendre le notebook exécuté.", c)

chemin = os.path.join(NBDIR, "TP3_regression_regularisation.ipynb")
t0 = time.time()
NotebookClient(n, timeout=1200, kernel_name=KERNEL,
               resources={"metadata": {"path": NBDIR}}).execute()
with open(chemin, "w", encoding="utf-8") as f:
    nbf.write(n, f)
nimg = sum(1 for cc in n.cells if cc.cell_type == "code"
           for o in cc.get("outputs", []) if "image/png" in o.get("data", {}))
print(f"[nb] TP3_regression_regularisation.ipynb  {nimg} figures  "
      f"{os.path.getsize(chemin)/1024:.1f} Ko  ({time.time()-t0:.0f} s)")
for cc in n.cells:
    if cc.cell_type == "code":
        for o in cc.get("outputs", []):
            t = o.get("text") or ""
            t = "".join(t) if isinstance(t, list) else t
            for l in t.splitlines():
                if any(k in l for k in ("R2", "RMSE", "MAE", "lambda", "OLS", "Ridge", "Lasso", "non nuls")):
                    print("   ", l.strip())
