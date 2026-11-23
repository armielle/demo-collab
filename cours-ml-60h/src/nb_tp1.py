"""TP1 -- premier pipeline : fabrique et execute le notebook."""
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

Jeu **breast_cancer** (fourni avec scikit-learn, aucun téléchargement) :
569 patients, 30 mesures, étiquette bénin / malin."""),
 ("code", r"""from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print("formes :", X.shape, "| etiquettes :", y.shape)
print("repartition :", y.value_counts().to_dict())
X.head(3)"""),
 ("md", "## 2. Découpage : entraînement et test"),
 ("code", r"""from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=4112, stratify=y)
print("entrainement :", Xtr.shape, "| test :", Xte.shape)"""),
 ("md", "## 3. La baseline : sans elle, aucun score n'est interprétable"),
 ("code", r"""from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
acc_base = accuracy_score(yte, base.predict(Xte))
print(f"baseline (classe majoritaire) : {acc_base:.3f}")"""),
 ("md", "## 4. Un premier vrai modèle, dans un pipeline"),
 ("code", r"""from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

modele = Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000))])
modele.fit(Xtr, ytr)
acc = accuracy_score(yte, modele.predict(Xte))
print(f"regression logistique : {acc:.3f}")
print(f"gain sur la baseline  : {100*(acc-acc_base):.1f} points")"""),
 ("md", "## 5. Au-delà de l'exactitude : la matrice de confusion"),
 ("code", r"""from sklearn.metrics import confusion_matrix, classification_report
pred = modele.predict(Xte)
cm = confusion_matrix(yte, pred)
print("matrice de confusion :")
print(pd.DataFrame(cm, index=["reel benin", "reel malin"],
                   columns=["predit benin", "predit malin"]))
print()
print(classification_report(yte, pred, target_names=["benin", "malin"]))"""),
 ("md", """## 6. Validation croisée : un score, mais avec son incertitude

Un score unique cache l'instabilité. On rapporte **moyenne ± écart-type**."""),
 ("code", r"""from sklearn.model_selection import cross_val_score
scores = cross_val_score(modele, Xtr, ytr, cv=5, scoring="accuracy")
print("scores par pli :", np.round(scores, 3))
print(f"moyenne : {scores.mean():.3f}  ecart-type : {scores.std():.3f}")"""),
 ("md", """## 7. Fabriquer une fuite, et la mesurer

On ajoute **300 variables de pur bruit** et on tire les étiquettes au hasard :
il n'y a rien à apprendre. Pourtant, une sélection de variables faite *avant*
la validation croisée produit un score flatteur."""),
 ("code", r"""from sklearn.feature_selection import SelectKBest, f_classif
rng = np.random.default_rng(4112)
n, p, k = 90, 600, 10
Xb = rng.normal(size=(n, p))
yb = rng.integers(0, 2, n)          # etiquettes aleatoires

# (a) FUITE : selection sur tout le jeu, puis validation croisee
sel = SelectKBest(f_classif, k=k).fit(Xb, yb)
score_fuite = cross_val_score(LogisticRegression(max_iter=1000),
                              sel.transform(Xb), yb, cv=5).mean()

# (b) PROPRE : la selection vit dans le pipeline
propre = Pipeline([("sel", SelectKBest(f_classif, k=k)),
                   ("clf", LogisticRegression(max_iter=1000))])
score_propre = cross_val_score(propre, Xb, yb, cv=5).mean()

print(f"avec fuite : {score_fuite:.3f}")
print(f"propre     : {score_propre:.3f}")
print(f"ecart      : {100*(score_fuite-score_propre):.1f} points")"""),
 ("code", r"""plt.figure(figsize=(6.5, 3.4))
b = plt.bar(["selection\navant le split", "selection\ndans le pipeline"],
            [score_fuite, score_propre], color=["#C0392B", "#1E8449"], width=.55)
plt.axhline(.5, color="grey", ls="--", lw=1)
for bb, v in zip(b, [score_fuite, score_propre]):
    plt.text(bb.get_x()+bb.get_width()/2, v+.02, f"{v:.1%}", ha="center", fontweight="bold")
plt.ylim(0, 1); plt.ylabel("exactitude en validation croisée")
plt.title("Des étiquettes aléatoires, et pourtant 70 % ?")
plt.show()"""),
 ("md", """## 8. À vous (noté)

**1.** Remplacez la régression logistique par un `RandomForestClassifier`
       (`n_estimators=200, random_state=4112`). La baseline change-t-elle ?
       Le gain est-il plus grand ?

**2.** Le jeu est-il déséquilibré ? Regardez `y.value_counts()` : quel serait
       le danger de juger avec l'exactitude seule ?

**3.** Dans la section 7, expliquez en deux phrases **pourquoi** la sélection
       faite avant le split fabrique un score élevé alors que les étiquettes
       sont aléatoires."""),
 ("code", r"""# ---- A VOUS ----
# 1)

# 2)

# 3) reponse :"""),
 ("md", """## 9. Ce qu'il faut retenir

| Geste | Pourquoi |
|---|---|
| baseline d'abord | sans référence, un score ne veut rien dire |
| pipeline | le prétraitement ne doit jamais voir le test |
| validation croisée | un score seul cache l'instabilité |
| plusieurs métriques | l'exactitude masque les classes rares |"""),
]


def nb(titre, duree, bareme, obj, trav, cellules):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": KERNEL, "display_name": "ML 60h", "language": "python"}
    n.cells.append(nbf.v4.new_markdown_cell(
        f"""# {titre}

**Cours :** Introduction au Machine Learning — 60 h — Master 1 SDIA
**Enseignante :** Dr. Noulapeu Ngaffo Armielle — ENSPD, Université de Douala
**Durée conseillée :** {duree} · **Barème :** {bareme}

---

## Objectifs

{obj}

## Travail demandé

{trav}

## Mode d'emploi

- Exécuter les cellules dans l'ordre ; tout tourne sur CPU en moins d'une minute.
- Les cellules **« À vous »** sont évaluées. Graine fixée : deux exécutions
  donnent le même résultat."""))
    n.cells.append(nbf.v4.new_code_cell(ENTETE))
    for typ, src in cellules:
        n.cells.append(nbf.v4.new_markdown_cell(src) if typ == "md"
                       else nbf.v4.new_code_cell(src))
    return n


n = nb("TP1 — Premier pipeline et première fuite",
       "2 h", "10 points",
       "- Charger des données réelles et les découper.\n"
       "- Comparer un modèle à une baseline.\n"
       "- Lire une matrice de confusion et une validation croisée.\n"
       "- Fabriquer et mesurer une fuite de données.",
       "1. Répondre aux 3 questions de la section « À vous ».\n"
       "2. Rendre le notebook exécuté.", c)

chemin = os.path.join(NBDIR, "TP1_premier_pipeline.ipynb")
t0 = time.time()
NotebookClient(n, timeout=1200, kernel_name=KERNEL,
               resources={"metadata": {"path": NBDIR}}).execute()
with open(chemin, "w", encoding="utf-8") as f:
    nbf.write(n, f)
nimg = sum(1 for cc in n.cells if cc.cell_type == "code"
           for o in cc.get("outputs", []) if "image/png" in o.get("data", {}))
print(f"[nb] TP1_premier_pipeline.ipynb  {nimg} figures  "
      f"{os.path.getsize(chemin)/1024:.1f} Ko  ({time.time()-t0:.0f} s)")
for cc in n.cells:
    if cc.cell_type == "code":
        for o in cc.get("outputs", []):
            t = o.get("text") or ""
            t = "".join(t) if isinstance(t, list) else t
            for l in t.splitlines():
                if any(k in l for k in ("baseline", "gain", "moyenne", "avec fuite", "propre ")):
                    print("   ", l.strip())
