"""TP4 -- classification et evaluation : fabrique et execute le notebook."""
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
 ("md", """## 1. Un jeu déséquilibré

5\,000 dossiers, **5 % de positifs** (fraude, décrochage, panne). C'est le régime
réel de la plupart des problèmes de détection."""),
 ("code", r"""from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5000, n_features=5, n_informative=3,
                           n_redundant=0, weights=[0.95, 0.05],
                           flip_y=0.01, random_state=4112)
print("positifs :", int(y.sum()), f"({100*y.mean():.1f} %)")
print("forme    :", X.shape)"""),
 ("md", "## 2. Découper, puis la référence naïve"),
 ("code", r"""from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=4112, stratify=y)

# le modele qui ne detecte jamais rien
acc_naif = float((yte == 0).mean())
print(f"exactitude d'un modele tout-negatif : {acc_naif:.4f}")"""),
 ("md", "## 3. Régression logistique : les cinq métriques"),
 ("code", r"""from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

modele = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
proba = modele.predict_proba(Xte)[:, 1]
pred = (proba >= .5).astype(int)

print(f"exactitude : {accuracy_score(yte, pred):.4f}")
print(f"precision  : {precision_score(yte, pred):.4f}")
print(f"rappel     : {recall_score(yte, pred):.4f}")
print(f"F1         : {f1_score(yte, pred):.4f}")
print(f"AUC        : {roc_auc_score(yte, proba):.4f}")
print("\nmatrice :\n", confusion_matrix(yte, pred))"""),
 ("md", "## 4. Le seuil déplace tout"),
 ("code", r"""seuils = np.linspace(.05, .95, 40)
prec = [precision_score(yte, (proba >= s).astype(int), zero_division=0) for s in seuils]
rec  = [recall_score(yte, (proba >= s).astype(int)) for s in seuils]
f1s  = [f1_score(yte, (proba >= s).astype(int), zero_division=0) for s in seuils]

plt.figure(figsize=(8, 3.4))
plt.plot(seuils, prec, label="precision")
plt.plot(seuils, rec, label="rappel")
plt.plot(seuils, f1s, "--", label="F1")
plt.axvline(.5, color="grey", ls=":")
plt.xlabel("seuil"); plt.legend(); plt.title("Précision et rappel selon le seuil")
plt.show()
print("seuil du F1 maximal :", round(float(seuils[int(np.argmax(f1s))]), 2))"""),
 ("md", """## 5. Choisir le seuil par le coût

Une fraude manquée coûte **10 fois** plus qu'une fausse alerte."""),
 ("code", r"""C_FP, C_FN = 1, 10
couts = []
for s in seuils:
    tn, fp, fn, tp = confusion_matrix(yte, (proba >= s).astype(int)).ravel()
    couts.append(C_FP * fp + C_FN * fn)

k = int(np.argmin(couts))
print(f"seuil optimal : {seuils[k]:.2f} | cout : {couts[k]}")
print(f"cout au seuil 0,50 : {couts[int(np.argmin(np.abs(seuils - .5)))]}")"""),
 ("md", "## 6. ROC et précision-rappel"),
 ("code", r"""from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

fpr, tpr, _ = roc_curve(yte, proba)
pr, rc, _ = precision_recall_curve(yte, proba)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="grey")
axes[0].set_xlabel("faux positifs"); axes[0].set_ylabel("vrais positifs")
axes[0].set_title("ROC"); axes[0].legend()
axes[1].plot(rc, pr, lw=2, color="#DC2626", label=f"AP = {average_precision_score(yte, proba):.3f}")
axes[1].axhline(yte.mean(), color="grey", ls="--", label=f"hasard = {yte.mean():.2f}")
axes[1].set_xlabel("rappel"); axes[1].set_ylabel("précision")
axes[1].set_title("Précision-rappel"); axes[1].legend()
plt.show()"""),
 ("md", "## 7. La pondération des classes"),
 ("code", r"""m_bal = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(Xtr, ytr)
pred_b = m_bal.predict(Xte)

comparaison = pd.DataFrame({
    "sans poids": [accuracy_score(yte, pred), f1_score(yte, pred), recall_score(yte, pred)],
    "avec poids": [accuracy_score(yte, pred_b), f1_score(yte, pred_b), recall_score(yte, pred_b)],
}, index=["exactitude", "F1", "rappel"])
print(comparaison.round(3).to_string())"""),
 ("md", "## 8. Les probabilités sont-elles calibrées ?"),
 ("code", r"""from sklearn.calibration import calibration_curve
frac, moy = calibration_curve(yte, proba, n_bins=8, strategy="quantile")
plt.figure(figsize=(5.4, 3.8))
plt.plot([0, 1], [0, 1], "--", color="grey")
plt.plot(moy, frac, "o-", lw=2)
plt.xlabel("probabilité annoncée"); plt.ylabel("fréquence observée")
plt.title("Diagramme de fiabilité")
plt.show()
print("ecart maximal :", round(float(np.max(np.abs(frac - moy))), 3))"""),
 ("md", """## 9. À vous (noté)

**1.** Calculez le coût total au seuil qui maximise le $F_1$, puis comparez-le
au coût du seuil optimal trouvé en section 5. Lequel est le plus faible ?

**2.** Réentraînez le modèle avec `class_weight={0:1, 1:25}`. Que deviennent
précision, rappel et coût au seuil 0,5 ?

**3.** Expliquez en deux phrases pourquoi l'exactitude d'un modèle qui ne
détecte jamais rien est ici de 0,95."""),
 ("code", r"""# ---- A VOUS ----
# 1)
# 2)
# 3)"""),
 ("md", """## 10. Ce qu'il faut retenir

| Geste | Pourquoi |
|---|---|
| comparer à un modèle naïf | sinon le score ne veut rien dire |
| regarder la matrice | elle localise les erreurs |
| choisir le seuil par le coût | une métrique abstraite ne décide pas |
| préférer PR à ROC si très déséquilibré | les vrais négatifs gonflent la ROC |
| vérifier la calibration | un seuil calculé suppose des probabilités justes |"""),
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


n = nb("TP4 — Classification et évaluation",
       "3 h", "15 points",
       "- Évaluer un classifieur sur des données déséquilibrées.\n"
       "- Tracer ROC et précision-rappel.\n"
       "- Choisir un seuil par le coût des erreurs.\n"
       "- Mesurer l'effet de la pondération des classes.",
       "1. Répondre aux 3 questions de la section « À vous ».\n"
       "2. Rendre le notebook exécuté.", c)

chemin = os.path.join(NBDIR, "TP4_classification_evaluation.ipynb")
t0 = time.time()
NotebookClient(n, timeout=1200, kernel_name=KERNEL,
               resources={"metadata": {"path": NBDIR}}).execute()
with open(chemin, "w", encoding="utf-8") as f:
    nbf.write(n, f)
nimg = sum(1 for cc in n.cells if cc.cell_type == "code"
           for o in cc.get("outputs", []) if "image/png" in o.get("data", {}))
print(f"[nb] TP4_classification_evaluation.ipynb  {nimg} figures  "
      f"{os.path.getsize(chemin)/1024:.1f} Ko  ({time.time()-t0:.0f} s)")
for cc in n.cells:
    if cc.cell_type == "code":
        for o in cc.get("outputs", []):
            t = o.get("text") or ""
            t = "".join(t) if isinstance(t, list) else t
            for l in t.splitlines():
                if any(k in l for k in ("exactitude", "rappel", "F1", "AUC", "cout",
                                        "seuil", "precision", "non nuls", "positifs")):
                    print("   ", l.strip())
