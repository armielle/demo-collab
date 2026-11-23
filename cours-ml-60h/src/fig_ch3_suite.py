"""Complement chapitre 3 : cas de grande dimension ou la regularisation gagne."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

from figstyle import C, save, FIGDIR

rng = np.random.default_rng(4112)
JSON = os.path.join(FIGDIR, "key_numbers_ch3.json")
cles = json.load(open(JSON, encoding="utf-8")) if os.path.exists(JSON) else {}

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# n petit, p grand : 8 variables informatives sur 150
n, p, k = 120, 150, 8
X = rng.normal(size=(n, p))
beta = np.zeros(p); beta[:k] = rng.uniform(1.5, 3.0, k) * rng.choice([-1, 1], k)
y = X @ beta + rng.normal(0, 1.0, n)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=4112)

def rmse(modele):
    m = Pipeline([("sc", StandardScaler()), ("m", modele)]).fit(Xtr, ytr)
    return float(np.sqrt(mean_squared_error(yte, m.predict(Xte)))), m

r_ols, m_ols = rmse(LinearRegression())
r_ridge, m_ridge = rmse(RidgeCV(alphas=np.logspace(-2, 4, 40)))
r_lasso, m_lasso = rmse(Lasso(alpha=0.3, max_iter=50000))

# nombre de coefficients non nuls
lasso_coef = m_lasso.named_steps["m"].coef_
nz = int((np.abs(lasso_coef) > 1e-6).sum())
vrais = int((np.abs(beta) > 1e-9).sum())
trouves = int((np.abs(lasso_coef[:k]) > 1e-6).sum())

cles["grande_dimension"] = {"n": n, "p": p, "informatives": vrais,
                            "rmse_ols": round(r_ols, 2),
                            "rmse_ridge": round(r_ridge, 2),
                            "rmse_lasso": round(r_lasso, 2),
                            "lasso_coefficients_non_nuls": nz,
                            "lasso_vraies_variables_retrouvees": trouves}

fig, ax = plt.subplots(figsize=(7.2, 3.4))
noms = ["OLS", "Ridge", "Lasso"]
vals = [r_ols, r_ridge, r_lasso]
b = ax.bar(noms, vals, color=[C["red"], C["blue"], C["green"]], alpha=.85, width=.55)
for bb, v in zip(b, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + max(vals) * .02, f"{v:.1f}",
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("RMSE sur le test")
ax.set_title(f"{p} variables pour {n} observations : la régularisation devient décisive")
ax.set_ylim(0, max(vals) * 1.2)
save(fig, "k09_grande_dimension.png")

with open(JSON, "w", encoding="utf-8") as f:
    json.dump(cles, f, ensure_ascii=False, indent=2)

print(json.dumps(cles["grande_dimension"], ensure_ascii=False))
