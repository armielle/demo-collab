"""TP2 -- preparation des donnees : fabrique et execute le notebook."""
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
 ("md", """## 1. Un jeu de données volontairement imparfait

Jeu **synthétique** construit pour ce TP (graine 4112) : 400 dossiers, comme
un export réel — des trous, du texte, des échelles incompatibles et des
valeurs aberrantes."""),
 ("code", r"""rng = np.random.default_rng(4112)
n = 400
revenu = rng.lognormal(10.5, .8, n)
df = pd.DataFrame({
    "moyenne": rng.normal(11.5, 3.2, n).clip(0, 20),
    "absences": rng.poisson(9, n),
    "revenu_parent": revenu,
    "filiere": rng.choice(["informatique", "telecom", "reseaux"], n),
    "heure_inscription": rng.integers(0, 24, n),
})
df.loc[rng.random(n) < .08, "moyenne"] = np.nan
df.loc[rng.random(n) < .26, "revenu_parent"] = np.nan
df.loc[rng.random(n) < .29, "filiere"] = np.nan
# une valeur aberrante de capteur
df.loc[3, "revenu_parent"] = 9_000_000
# etiquette : reussite, liee a la moyenne et aux absences
proba = 1 / (1 + np.exp(-(0.45 * df["moyenne"].fillna(df["moyenne"].median())
                          - 0.12 * df["absences"] - 2.2)))
y = (rng.random(n) < proba).astype(int)
print(df.shape, "| reussite :", y.mean().round(3))
df.head(4)"""),
 ("md", "## 2. Diagnostiquer avant de toucher"),
 ("code", r"""print("types :")
print(df.dtypes)
print()
print("valeurs manquantes (%) :")
print((df.isna().mean() * 100).round(1))
print()
print("etendues :")
print((df.max(numeric_only=True) - df.min(numeric_only=True)).round(2))"""),
 ("md", "## 3. Découper AVANT toute transformation"),
 ("code", r"""from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=.25, random_state=4112, stratify=y)
print("entrainement :", Xtr.shape, "| test :", Xte.shape)"""),
 ("md", "## 4. Premier pipeline, volontairement naïf"),
 ("code", r"""from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

num = ["moyenne", "absences", "revenu_parent"]
cat = ["filiere"]

prepa = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
modele = Pipeline([("prepa", prepa), ("clf", LogisticRegression(max_iter=1000))])
modele.fit(Xtr, ytr)
pred = modele.predict(Xte)
print(f"exactitude : {accuracy_score(yte, pred):.3f}")
print(f"rappel     : {recall_score(yte, pred):.3f}")"""),
 ("md", """## 5. Ajouter l'encodage cyclique de l'heure

23 h et 0 h sont voisines. On ajoute deux colonnes $\\sin$ et $\\cos$."""),
 ("code", r"""def ajoute_cyclique(X):
    X = X.copy()
    X["h_sin"] = np.sin(2 * np.pi * X["heure_inscription"] / 24)
    X["h_cos"] = np.cos(2 * np.pi * X["heure_inscription"] / 24)
    return X

Xtr2, Xte2 = ajoute_cyclique(Xtr), ajoute_cyclique(Xte)
num2 = num + ["h_sin", "h_cos"]
prepa2 = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num2),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
modele2 = Pipeline([("prepa", prepa2), ("clf", LogisticRegression(max_iter=1000))])
modele2.fit(Xtr2, ytr)
print(f"avec cyclique : {accuracy_score(yte, modele2.predict(Xte2)):.3f}")"""),
 ("md", """## 6. Fabriquer une fuite par encodage cible

Variable « agence » à 100 modalités, 2 lignes chacune, étiquette aléatoire.
L'encodage calculé **sur tout le jeu** contient la réponse."""),
 ("code", r"""from sklearn.model_selection import KFold, cross_val_score

n2 = 200
agence = np.repeat(np.arange(100), 2)
y2 = rng.integers(0, 2, n2)

# (a) fuite : moyenne de y par agence, calculee sur tout le jeu
moy_totale = np.array([y2[agence == c].mean() for c in agence])
s_fuite = cross_val_score(LogisticRegression(max_iter=1000),
                          moy_totale.reshape(-1, 1), y2,
                          cv=KFold(5, shuffle=True, random_state=4112)).mean()

# (b) propre : encodage recalcule dans chaque pli
kf = KFold(5, shuffle=True, random_state=4112)
enc = np.zeros(n2)
for tr, te in kf.split(agence):
    table = {c: y2[tr][agence[tr] == c].mean() for c in np.unique(agence[tr])}
    defaut = y2[tr].mean()
    enc[te] = [table.get(c, defaut) for c in agence[te]]
s_propre = cross_val_score(LogisticRegression(max_iter=1000),
                           enc.reshape(-1, 1), y2, cv=kf).mean()
print(f"avec fuite : {s_fuite:.3f}")
print(f"propre     : {s_propre:.3f}")
print(f"ecart      : {100*(s_fuite-s_propre):.1f} points")"""),
 ("code", r"""plt.figure(figsize=(6.4, 3.2))
b = plt.bar(["encodage global", "encodage par pli"], [s_fuite, s_propre],
            color=["#C0392B", "#1E8449"], width=.55)
for bb, v in zip(b, [s_fuite, s_propre]):
    plt.text(bb.get_x()+bb.get_width()/2, v+.02, f"{v:.1%}", ha="center", fontweight="bold")
plt.axhline(.5, color="grey", ls="--", lw=1)
plt.ylim(0, 1); plt.ylabel("exactitude (validation croisée)")
plt.title("Étiquettes aléatoires : la fuite fabrique un score")
plt.show()"""),
 ("md", """## 7. À vous (noté)

**1.** Remplacez l'imputation médiane du revenu par une imputation par la
moyenne. Le score change-t-il ? De combien ?

**2.** Retirez la valeur aberrante (`9 000 000`) avant l'entraînement :
le rappel s'améliore-t-il ?

**3.** Dans la section 6, expliquez en deux phrases pourquoi l'encodage global
produit un score élevé alors que l'étiquette est tirée au hasard."""),
 ("code", r"""# ---- A VOUS ----
# 1)

# 2)

# 3) reponse :"""),
 ("md", """## 8. Ce qu'il faut retenir

| Geste | Pourquoi |
|---|---|
| diagnostiquer avant de transformer | on ne corrige que ce qu'on a vu |
| découper avant tout | les paramètres de prétraitement ne doivent pas voir le test |
| `ColumnTransformer` | chaque type de colonne a son traitement |
| encodage par pli | sinon la cible fuit dans la variable |"""),
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


n = nb("TP2 — Préparer les données sans fuite",
       "2 h", "10 points",
       "- Diagnostiquer manquants, types et échelles.\n"
       "- Construire un ColumnTransformer complet.\n"
       "- Ajouter un encodage cyclique.\n"
       "- Fabriquer et corriger une fuite d'encodage cible.",
       "1. Répondre aux 3 questions de la section « À vous ».\n"
       "2. Rendre le notebook exécuté.", c)

chemin = os.path.join(NBDIR, "TP2_preparation_donnees.ipynb")
t0 = time.time()
NotebookClient(n, timeout=1200, kernel_name=KERNEL,
               resources={"metadata": {"path": NBDIR}}).execute()
with open(chemin, "w", encoding="utf-8") as f:
    nbf.write(n, f)
nimg = sum(1 for cc in n.cells if cc.cell_type == "code"
           for o in cc.get("outputs", []) if "image/png" in o.get("data", {}))
print(f"[nb] TP2_preparation_donnees.ipynb  {nimg} figures  "
      f"{os.path.getsize(chemin)/1024:.1f} Ko  ({time.time()-t0:.0f} s)")
for cc in n.cells:
    if cc.cell_type == "code":
        for o in cc.get("outputs", []):
            t = o.get("text") or ""
            t = "".join(t) if isinstance(t, list) else t
            for l in t.splitlines():
                if any(k in l for k in ("exactitude", "rappel", "avec fuite", "propre", "ecart")):
                    print("   ", l.strip())
