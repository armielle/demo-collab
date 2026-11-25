"""Table de verification des chiffres cites dans les slides + frames de l'animation."""
import os, json
import numpy as np
from scipy import stats
from figstyle import plt, C, FIGDIR

os.environ.setdefault("OMP_NUM_THREADS", "2")
rng = np.random.default_rng(4112)
obs = (rng.random(200) < 0.65).astype(int)

T = {}
rows = []
for n in [0, 1, 2, 5, 20, 200]:
    k = int(obs[:n].sum())
    a, b = 1 + k, 1 + n - k
    rows.append((n, k, f"Beta({a},{b})", round(a / (a + b), 3)))
T["table_update"] = rows
print("=== mise a jour sequentielle (seed 4112, p=0.65) ===")
for r in rows:
    print("   n=%3d  k=%3d  %-14s moyenne=%.3f" % r)

def q(a, b, lvl=.95):
    lo, hi = stats.beta(a, b).ppf([(1 - lvl) / 2, 1 - (1 - lvl) / 2])
    return round(float(lo), 3), round(float(hi), 3)

T["beta95_975"] = q(9, 5)
T["beta95_85"] = q(8, 4)
T["beta65_mean"] = round(float(stats.beta(9, 5).mean()), 3)
T["beta85_mean"] = round(float(stats.beta(8, 4).mean()), 3)
T["beta85_map"] = round(float((8 - 1) / (8 + 4 - 2)), 3)
print("\nBeta(9,5) moy=%.3f  IC95=%s | Beta(8,4) moy=%.3f IC95=%s"
      % (stats.beta(9, 5).mean(), T["beta95_975"], stats.beta(8, 4).mean(), T["beta95_85"]))
print("Beta(9,5) HDI :", end=" ")
from scipy.optimize import brentq
def hdi(a, b, lvl=.95):
    """HDI par balayage : plus petit intervalle de densite de masse lvl."""
    d = stats.beta(a, b)
    g = np.linspace(1e-6, 1 - 1e-6, 200001)
    pdf = d.pdf(g)
    cdf = d.cdf(g)
    best, out = np.inf, None
    j = 0
    for i in range(len(g)):
        target = cdf[i] + lvl
        if target > 1:
            break
        j = max(j, i)
        while j < len(g) and cdf[j] < target:
            j += 1
        if j >= len(g):
            break
        w = g[j] - g[i]
        if w < best:
            best, out = w, (float(g[i]), float(g[j]))
    return (round(out[0], 3), round(out[1], 3))
T["beta95_hdi"] = hdi(9, 5)
T["beta85_hdi"] = hdi(8, 4)
print(T["beta95_hdi"], T["beta85_hdi"])

p = stats.beta(325, 277)
T["sondage_p_gt_052"] = round(float(1 - p.cdf(0.52)), 3)
T["sondage_p_gt_050"] = round(float(1 - p.cdf(0.50)), 4)
T["sondage_p_gt_055"] = round(float(1 - p.cdf(0.55)), 3)
T["sondage_ci"] = q(325, 277)
print("\nsondage Beta(325,277) : IC95=%s  P(p>.5)=%.4f  P(p>.52)=%.3f  P(p>.55)=%.3f"
      % (T["sondage_ci"], T["sondage_p_gt_050"], T["sondage_p_gt_052"], T["sondage_p_gt_055"]))

# exercice guide : 10 patients, 7 guerisons, prior Beta(2,2)
T["exo_beta"] = "Beta(9,5)"
T["exo_mean"] = round(9 / 14, 3)
T["exo_ci"] = q(9, 5)
print("exercice 10 patients / prior Beta(2,2) : Beta(9,5) moy=%.3f IC95=%s mode=%.3f"
      % (9 / 14, T["exo_ci"], 8 / 12))

# test medical du slide 3 : sens 99 %, spec 95 %, prevalence 1 %
T["med_ppv"] = round(0.99 * 0.01 / (0.99 * 0.01 + 0.05 * 0.99), 4)
T["med_quiz_ppv"] = round(0.99 * .2 / (0.99 * .2 + .05 * .8), 3)
print("medecine : PPV(1%%)=%.4f   quiz PPV(20%%)=%.3f" % (T["med_ppv"], T["med_quiz_ppv"]))

# spam : 1, 2, 3 occurrences + interpretation de l'hypothese d'independance
T["spam"] = [round(float(0.25 * 12 ** t / (1 + 0.25 * 12 ** t)), 4) for t in range(4)]
print("spam 0..3 occurrences :", T["spam"])

# regression de la diapo 45 : estimation avec prior vague, 12 points (donnees synthetiques reelles)
n_obs = 12
x = np.linspace(0, 10, n_obs)
y = 2.0 + 0.7 * x + rng.normal(0, 1.0, n_obs)
X = np.column_stack([np.ones_like(x), x])
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
sigma2 = float(((y - X @ beta_ols) ** 2).sum() / (n_obs - 2))
Vb = sigma2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(Vb))
T["reg_ols"] = [round(float(v), 3) for v in beta_ols]
T["reg_se"] = [round(float(v), 3) for v in se]
T["reg_data"] = {"x": [round(float(v), 2) for v in x], "y": [round(float(v), 2) for v in y]}
print("regression (12 pts) : beta0=%.3f (se %.3f)  beta1=%.3f (se %.3f)  sigma=%.3f"
      % (beta_ols[0], se[0], beta_ols[1], se[1], np.sqrt(sigma2)))

json.dump(T, open(os.path.join(FIGDIR, "key_numbers_ch1c.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ frames animation
FD = os.path.join(FIGDIR, "f04_frames")
os.makedirs(FD, exist_ok=True)
for f in os.listdir(FD):
    os.remove(os.path.join(FD, f))
th = np.linspace(0, 1, 500)
seq = list(range(0, 40, 2)) + list(range(40, 201, 10))
for i, n in enumerate(seq):
    k = int(obs[:n].sum())
    a, b = 1 + k, 1 + n - k
    fig = plt.figure(figsize=(5.6, 3.1))
    ax = fig.add_axes([.10, .16, .86, .74])
    ax.fill_between(th, 0, stats.beta.pdf(th, a, b), color=C["blue"], alpha=.30)
    ax.plot(th, stats.beta.pdf(th, a, b), color=C["blue"], lw=2.0)
    ax.axvline(0.65, color=C["red"], ls="--", lw=1.2)
    ax.plot([a / (a + b)], [0], marker="^", color=C["red"], ms=7, clip_on=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 9.5); ax.set_yticks([])
    ax.set_xlabel("$\\theta$")
    ax.set_title("n = %d   posterior Beta(%d, %d)   moyenne = %.3f" % (n, a, b, a / (a + b)),
                 fontsize=10.5, fontweight="bold", color=C["deep"])
    ax.grid(alpha=.22)
    fig.savefig(os.path.join(FD, "frame-%d.png" % i), dpi=150)
    plt.close(fig)
print("\n[anim] %d frames -> figures/f04_frames/" % len(seq))
