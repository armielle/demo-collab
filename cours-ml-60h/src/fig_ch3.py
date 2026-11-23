"""Figures du chapitre 3 -- Regression et regularisation (Licence 3 SDIA).

Toutes les valeurs sont calculees ici (graine 4112) et ecrites dans
figures/key_numbers_ch3.json.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

from figstyle import C, save, FIGDIR

rng = np.random.default_rng(4112)
cles = {}

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X, y = load_diabetes(return_X_y=True)
noms = load_diabetes().feature_names
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=4112)

# ==================================================== k01 : ajustement et residus
ols = Pipeline([("sc", StandardScaler()), ("reg", LinearRegression())]).fit(Xtr, ytr)
pred_te = ols.predict(Xte)
res = yte - pred_te
rmse_ols = float(np.sqrt(mean_squared_error(yte, pred_te)))
mae_ols = float(mean_absolute_error(yte, pred_te))
r2_ols = float(r2_score(yte, pred_te))
cles["ols"] = {"rmse_test": round(rmse_ols, 2), "mae_test": round(mae_ols, 2),
               "r2_test": round(r2_ols, 3), "n_train": int(len(ytr)), "n_test": int(len(yte))}

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ax.scatter(yte, pred_te, s=16, color=C["blue"], alpha=.7)
lim = [min(yte.min(), pred_te.min()), max(yte.max(), pred_te.max())]
ax.plot(lim, lim, "--", color=C["grey"], lw=1.2, label="prédiction parfaite")
ax.set_xlabel("valeur réelle"); ax.set_ylabel("valeur prédite")
ax.set_title(f"Test : R² = {r2_ols:.3f}, RMSE = {rmse_ols:.1f}")
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(pred_te, res, s=16, color=C["orange"], alpha=.7)
ax.axhline(0, color=C["grey"], lw=1.2)
ax.set_xlabel("valeur prédite"); ax.set_ylabel("résidu")
ax.set_title("Résidus : aucune structure attendue")
save(fig, "k01_ajustement.png")

# ==================================================== k02 : polynomes
x = np.sort(rng.uniform(0, 1, 60))
yv = np.sin(2 * np.pi * x) + rng.normal(0, .22, 60)
xg = np.linspace(0, 1, 300)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
for ax, d, titre in zip(axes, [1, 3, 15],
                        ["degré 1 : biais", "degré 3 : bon compromis", "degré 15 : variance"]):
    coef = np.polyfit(x, yv, d)
    ax.scatter(x, yv, s=14, color=C["grey"], alpha=.7)
    ax.plot(xg, np.polyval(coef, xg), color=[C["red"], C["green"], C["orange"]][[1, 3, 15].index(d)], lw=2)
    mse = float(np.mean((np.polyval(coef, x) - yv) ** 2))
    ax.set_title(f"{titre}\nMSE = {mse:.3f}")
    ax.set_ylim(-1.8, 1.8)
save(fig, "k02_polynomes.png")
cles["polynomes"] = {"deg1_mse": round(float(np.mean((np.polyval(np.polyfit(x, yv, 1), x) - yv) ** 2)), 3),
                     "deg3_mse": round(float(np.mean((np.polyval(np.polyfit(x, yv, 3), x) - yv) ** 2)), 3),
                     "deg15_mse": round(float(np.mean((np.polyval(np.polyfit(x, yv, 15), x) - yv) ** 2)), 4)}

# ==================================================== k03 : chemins de regularisation
alphas = np.logspace(-3, 3, 60)
Xtr_s = StandardScaler().fit_transform(Xtr)
chemins_ridge, chemins_lasso = [], []
for a in alphas:
    chemins_ridge.append(Ridge(alpha=a).fit(Xtr_s, ytr).coef_)
    chemins_lasso.append(Lasso(alpha=a, max_iter=20000).fit(Xtr_s, ytr).coef_)
chemins_ridge = np.array(chemins_ridge); chemins_lasso = np.array(chemins_lasso)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for j in range(X.shape[1]):
    axes[0].semilogx(alphas, chemins_ridge[:, j], lw=1.2)
    axes[1].semilogx(alphas, chemins_lasso[:, j], lw=1.2)
axes[0].set_title("Ridge : les coefficients se contractent")
axes[1].set_title("Lasso : les coefficients s'annulent")
for ax in axes:
    ax.set_xlabel("λ (échelle log)"); ax.axhline(0, color=C["grey"], lw=.8)
axes[0].set_ylabel("coefficient standardisé")
save(fig, "k03_chemins.png")

nz = int((np.abs(chemins_lasso[-1]) > 1e-8).sum())
cles["chemins"] = {"variables": int(X.shape[1]),
                   "lasso_non_nulles_alpha_max": nz}

# ==================================================== k04 : choix de lambda
lambdas = np.logspace(-2, 3, 40)
rmse_tr, rmse_te = [], []
for a in lambdas:
    m = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=a))]).fit(Xtr, ytr)
    rmse_tr.append(np.sqrt(mean_squared_error(ytr, m.predict(Xtr))))
    rmse_te.append(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
rmse_tr, rmse_te = np.array(rmse_tr), np.array(rmse_te)
best = int(np.argmin(rmse_te))
cles["lambda"] = {"optimal": float(lambdas[best]),
                  "rmse_test_optimal": round(float(rmse_te[best]), 2),
                  "rmse_test_ols": round(rmse_ols, 2),
                  "rmse_test_lambda_max": round(float(rmse_te[-1]), 2)}

fig, ax = plt.subplots(figsize=(8.4, 3.8))
ax.semilogx(lambdas, rmse_tr, "o-", ms=3, color=C["blue"], label="entraînement")
ax.semilogx(lambdas, rmse_te, "s-", ms=3, color=C["red"], label="test")
ax.axvline(lambdas[best], color=C["green"], ls="--", lw=1.3)
ax.annotate(f"λ = {lambdas[best]:.2f}\nRMSE test = {rmse_te[best]:.2f}",
            (lambdas[best], rmse_te[best]), xytext=(lambdas[best] * 6, rmse_te[best] + 8),
            fontsize=9, color=C["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["green"]))
ax.set_xlabel("λ (échelle log)"); ax.set_ylabel("RMSE")
ax.set_title("Régularisation : le compromis se lit sur le test")
ax.legend()
save(fig, "k04_lambda.png")

# ==================================================== k05 : descente de gradient
def descente(lr, n_iter=40):
    w, hist = 8.0, [8.0]
    for _ in range(n_iter):
        w = w - lr * 2 * (w - 2)
        hist.append(w)
    return np.array(hist)

fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
ws = np.linspace(-2, 9, 300)
axes[0].plot(ws, (ws - 2) ** 2 + 1, color=C["deep"], lw=2)
for lr, col in [(0.03, C["blue"]), (0.2, C["green"]), (1.02, C["red"])]:
    h = descente(lr, 12 if lr < 1 else 8)
    axes[0].plot(h, (h - 2) ** 2 + 1, "o-", ms=3, lw=1.2, color=col, label=f"η = {lr}")
axes[0].set_ylim(0, 40); axes[0].legend(fontsize=8)
axes[0].set_xlabel("w"); axes[0].set_ylabel("J(w)")
axes[0].set_title("Trois pas d'apprentissage")

for lr, col in [(0.03, C["blue"]), (0.2, C["green"])]:
    h = descente(lr, 40)
    axes[1].plot(np.abs(h - 2), color=col, lw=1.6, label=f"η = {lr}")
axes[1].set_yscale("log")
axes[1].set_xlabel("itération"); axes[1].set_ylabel("|w - w*| (log)")
axes[1].set_title("Vitesse de convergence")
axes[1].legend(fontsize=8)
save(fig, "k05_descente.png")
cler = {str(lr): round(float(np.abs(descente(lr, 40)[-1] - 2)), 4) for lr in (0.03, 0.2)}
cles["descente"] = {"ecart_final_" + k: v for k, v in cler.items()}

# ==================================================== k06 : heteroscedasticite
n = 200
xx = rng.uniform(0, 10, n)
bruit = rng.normal(0, .4 + .35 * xx, n)
yy = 2 + 1.5 * xx + bruit
m = LinearRegression().fit(xx.reshape(-1, 1), yy)
pr = m.predict(xx.reshape(-1, 1))
fig, ax = plt.subplots(figsize=(7.6, 3.5))
ax.scatter(pr, yy - pr, s=14, color=C["orange"], alpha=.7)
ax.axhline(0, color=C["grey"], lw=1)
ax.set_xlabel("valeur prédite"); ax.set_ylabel("résidu")
ax.set_title("Hétéroscédasticité : l'erreur croît avec le niveau")
save(fig, "k06_heteroscedasticite.png")
corr_abs = float(np.corrcoef(pr, np.abs(yy - pr))[0, 1])
cles["heteroscedasticite"] = {"corr_pred_abs_residu": round(corr_abs, 3)}

# ==================================================== k07 : multicolinearite
n2 = 80
x1 = rng.normal(0, 1, n2)
x2 = x1 + rng.normal(0, .05, n2)     # quasi-colinéaire
Xc = np.column_stack([x1, x2])
yc = 2 * x1 + rng.normal(0, .5, n2)
coefs = []
for _ in range(200):
    idx = rng.integers(0, n2, n2)
    coefs.append(LinearRegression().fit(Xc[idx], yc[idx]).coef_)
coefs = np.array(coefs)
cles["multicolinearite"] = {"corr_x1_x2": round(float(np.corrcoef(x1, x2)[0, 1]), 3),
                            "coef1_moyen": round(float(coefs[:, 0].mean()), 2),
                            "coef1_ecart_type": round(float(coefs[:, 0].std()), 2),
                            "coef2_moyen": round(float(coefs[:, 1].mean()), 2),
                            "coef2_ecart_type": round(float(coefs[:, 1].std()), 2)}

fig, ax = plt.subplots(figsize=(7.4, 3.5))
ax.hist(coefs[:, 0], bins=30, alpha=.65, color=C["blue"], label="coefficient de $x_1$")
ax.hist(coefs[:, 1], bins=30, alpha=.65, color=C["red"], label="coefficient de $x_2$")
ax.axvline(0, color=C["grey"], lw=1)
ax.set_xlabel("valeur du coefficient (200 rééchantillonnages)")
ax.set_ylabel("effectif")
ax.set_title("Deux variables quasi identiques : coefficients instables, signes opposés")
ax.legend()
save(fig, "k07_multicolinearite.png")

# ==================================================== k08 : comparaison
resultats = {}
for nom, mod in [("OLS", LinearRegression()),
                 ("Ridge", Ridge(alpha=float(lambdas[best]))),
                 ("Lasso", Lasso(alpha=0.1, max_iter=20000)),
                 ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=.5, max_iter=20000))]:
    p = Pipeline([("sc", StandardScaler()), ("m", mod)]).fit(Xtr, ytr)
    resultats[nom] = round(float(np.sqrt(mean_squared_error(yte, p.predict(Xte)))), 2)
cles["comparaison"] = resultats

fig, ax = plt.subplots(figsize=(7.2, 3.4))
b = ax.bar(list(resultats), list(resultats.values()),
           color=[C["grey"], C["blue"], C["orange"], C["green"]], alpha=.85, width=.6)
for bb, v in zip(b, resultats.values()):
    ax.text(bb.get_x() + bb.get_width() / 2, v + .4, f"{v:.1f}", ha="center",
            fontsize=10, fontweight="bold")
ax.set_ylabel("RMSE sur le test")
ax.set_ylim(0, max(resultats.values()) * 1.25)
ax.set_title("Sur ce jeu, la régularisation ne fait pas toujours mieux que l'OLS")
save(fig, "k08_comparaison.png")

with open(os.path.join(FIGDIR, "key_numbers_ch3.json"), "w", encoding="utf-8") as f:
    json.dump(cles, f, ensure_ascii=False, indent=2)

print("\n--- chiffres cles ---")
for k, v in cles.items():
    print(k, ":", v)
