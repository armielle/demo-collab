"""Fabrique des notebooks TP3 (LSTM/GRU) et TP4 (projet trajectoire)."""
import os, time
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NBDIR = os.path.join(ROOT, "notebooks")
KERNEL = "rnn-20h"
ENTETE = ("%matplotlib inline\n"
          "import numpy as np, torch, torch.nn as nn\n"
          "import matplotlib.pyplot as plt\n"
          "torch.manual_seed(4112); np.random.seed(4112)\n"
          "torch.set_num_threads(2)")


def nb(titre, duree, bareme, obj, trav, cellules):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": KERNEL, "display_name": "RNN 20h", "language": "python"}
    n.cells.append(nbf.v4.new_markdown_cell(
        f"""# {titre}

**Cours :** Réseaux de neurones récurrents (RNN) — 20 h — Master 1 SDIA
**Enseignante :** Dr. Noulapeu Ngaffo Armielle — ENSPD, Université de Douala
**Durée conseillée :** {duree} · **Barème :** {bareme}

---

## Objectifs

{obj}

## Travail demandé

{trav}

## Mode d'emploi

- Exécuter dans l'ordre ; tout tourne sur CPU.
- Les cellules **« À vous »** sont évaluées ; graines fixées."""))
    n.cells.append(nbf.v4.new_code_cell(ENTETE))
    for typ, src in cellules:
        n.cells.append(nbf.v4.new_markdown_cell(src) if typ == "md"
                       else nbf.v4.new_code_cell(src))
    return n


def execute(n, nom):
    chemin = os.path.join(NBDIR, nom)
    t0 = time.time()
    NotebookClient(n, timeout=2400, kernel_name=KERNEL,
                   resources={"metadata": {"path": NBDIR}}).execute()
    with open(chemin, "w", encoding="utf-8") as f:
        nbf.write(n, f)
    nimg = sum(1 for c in n.cells if c.cell_type == "code"
               for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    nc = sum(1 for c in n.cells if c.cell_type == "code")
    print(f"[nb] {nom:<44} {nc} cellules, {nimg} figures  "
          f"{os.path.getsize(chemin)/1024:.1f} Ko  ({time.time()-t0:.0f} s)")


# ===================================================================== TP3
c3 = [
 ("md", "## 1. Un corpus jouet, étiqueté\n\n"
        "Corpus **écrit pour ce TP** (18 phrases) : ce n'est pas un jeu de référence, "
        "c'est un support pour comparer des architectures."),
 ("code", r"""positifs = [
    "le film est excellent", "histoire captivante et superbe",
    "je recommande vivement ce film", "acteurs formidables et scenario solide",
    "un chef d oeuvre absolu", "mise en scene magnifique",
    "emouvant et intelligent", "dialogues brillants",
    "j ai adore du debut a la fin",
]
negatifs = [
    "le film est mauvais", "histoire ennuyeuse et confuse",
    "je deconseille ce film", "acteurs faibles et scenario vide",
    "un desastre absolu", "mise en scene pauvre",
    "long et sans interet", "dialogues plats",
    "je me suis ennuye du debut a la fin",
]
phrases = positifs + negatifs
labels = [1] * len(positifs) + [0] * len(negatifs)
print("positifs :", len(positifs), "| negatifs :", len(negatifs))"""),
 ("code", r"""from sklearn.model_selection import train_test_split
mots = sorted({m for p in phrases for m in p.split()})
vocab = {"<pad>": 0}
for m in mots:
    vocab[m] = len(vocab)
L = max(len(p.split()) for p in phrases)

def encode(p):
    ids = [vocab[m] for m in p.split()]
    return ids + [0] * (L - len(ids))

X = torch.tensor([encode(p) for p in phrases])
y = torch.tensor(labels, dtype=torch.long)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=4112, stratify=y)
print("vocabulaire :", len(vocab), "| longueur :", L)
print("train :", tuple(Xtr.shape), "| test :", tuple(Xte.shape))"""),
 ("md", "## 2. Une fonction d'entraînement commune"),
 ("code", r"""class Classifieur(nn.Module):
    def __init__(self, cellule="rnn", H=32, bidir=False):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), 24, padding_idx=0)
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cellule]
        self.rnn = cls(24, H, batch_first=True, bidirectional=bidir)
        self.out = nn.Linear(H * (2 if bidir else 1), 2)
    def forward(self, x):
        v = self.emb(x)
        s, _ = self.rnn(v)
        return self.out(s[:, -1, :])

def entraine(modele, epoques=80, lr=1e-2):
    opt = torch.optim.Adam(modele.parameters(), lr=lr)
    pf = nn.CrossEntropyLoss()
    for ep in range(epoques):
        modele.train(); opt.zero_grad()
        perte = pf(modele(Xtr), ytr)
        perte.backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 5.0)
        opt.step()
    modele.eval()
    with torch.no_grad():
        acc = (modele(Xte).argmax(1) == yte).float().mean().item()
    return round(float(perte), 3), round(acc, 3)"""),
 ("code", r"""resultats = {}
for nom, kw in [("RNN", dict(cellule="rnn")),
                ("GRU", dict(cellule="gru")),
                ("LSTM", dict(cellule="lstm")),
                ("LSTM bidirectionnel", dict(cellule="lstm", bidir=True))]:
    perte, acc = entraine(Classifieur(**kw))
    resultats[nom] = acc
    print(f"{nom:<22} perte={perte:.3f}   exactitude test={acc:.1%}")"""),
 ("code", r"""noms = list(resultats)
vals = [resultats[n] * 100 for n in noms]
plt.figure(figsize=(9, 3.2))
b = plt.barh(noms[::-1], vals[::-1], color=["#5B6470", "#1E8449", "#1F6FEB", "#12A594"][::-1], alpha=.88)
for bb, v in zip(b, vals[::-1]):
    plt.text(v + 1, bb.get_y() + bb.get_height()/2, f"{v:.1f} %", va="center", fontsize=9)
plt.xlim(0, 115); plt.xlabel("exactitude sur le test (%)")
plt.title("Attention : 18 phrases seulement — les ecarts sont fragiles")
plt.show()"""),
 ("md", "## 3. À vous\n\n"
        "**1.** Pourquoi ce corpus est-il trop petit pour conclure qu'un modèle est "
        "« meilleur » ?\n\n"
        "**2.** Ajoutez 3 phrases négatives et relancez : les classements changent-ils ?\n\n"
        "**3.** Remplacez `s[:, -1, :]` (dernier état) par une **moyenne masquée** des "
        "états : est-ce plus stable ?"),
 ("code", r"""# ---- A VOUS ----
# 1)
# 2)
# 3)"""),
 ("md", "## 4. Synthèse\n\n"
        "| Ce qu'il faut retenir | |\n|---|---|\n"
        "| même nombre de couches, coût différent | GRU < LSTM en paramètres |\n"
        "| bidirectionnel | meilleur, mais pas en temps réel |\n"
        "| petit corpus | écarts non significatifs |"),
]
execute(nb("TP3 — LSTM, GRU et sentiment", "2 h", "10 points",
           "- Comparer RNN, GRU et LSTM sur une même tâche.\n"
           "- Mesurer l'apport du bidirectionnel.\n"
           "- Prendre conscience des limites d'un petit corpus.",
           "1. Répondre aux 3 questions de la section « À vous ».\n"
           "2. Rendre le notebook exécuté.", c3), "TP3_LSTM_GRU_sentiment.ipynb")

# ===================================================================== TP4
c4 = [
 ("md", "## Projet — prédire la trajectoire d'une coccinelle\n\n"
        "On simule 200 trajectoires, on apprend à prédire les 10 positions suivantes "
        "à partir des 30 précédentes, et on compare à une **référence naïve**."),
 ("code", r"""rng = np.random.default_rng(4112)

def trajectoire(n=40, noms=200):
    trajs = []
    for _ in range(noms):
        angle = rng.uniform(0, 2 * np.pi)
        pos = np.zeros(2)
        pts = []
        for _ in range(n):
            angle += rng.normal(0, .25)
            pos = pos + .3 * np.array([np.cos(angle), np.sin(angle)])
            pts.append(pos.copy())
        trajs.append(np.array(pts, dtype=np.float32))
    return np.array(trajs)

T = trajectoire()
print("formes :", T.shape, "= (trajectoires, pas, coordonnees)")
obs, cible = T[:, :30, :], T[:, 30:, :]
print("observe :", obs.shape, "| a predire :", cible.shape)"""),
 ("code", r"""import torch.nn as nn
class Predicteur(nn.Module):
    def __init__(self, H=64, horizon=10):
        super().__init__()
        self.lstm = nn.LSTM(2, H, batch_first=True)
        self.out = nn.Linear(H, 2 * horizon)
        self.horizon = horizon
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.out(h[-1]).view(-1, self.horizon, 2)

Xo = torch.tensor(obs); Yo = torch.tensor(cible)
modele = Predicteur()
opt = torch.optim.Adam(modele.parameters(), lr=5e-3)
pf = nn.MSELoss()
for ep in range(120):
    opt.zero_grad()
    p = modele(Xo)
    perte = pf(p, Yo)
    perte.backward()
    torch.nn.utils.clip_grad_norm_(modele.parameters(), 5.0)
    opt.step()
    if ep % 30 == 0:
        print(f"epoque {ep:3d}  perte = {perte.item():.5f}")
print("perte finale :", round(float(perte), 5))"""),
 ("code", r"""with torch.no_grad():
    pred = modele(Xo).numpy()
naif = np.repeat(obs[:, -1:, :], 10, axis=1)

def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))

print("RMSE du LSTM      :", round(rmse(pred, cible), 4))
print("RMSE de la ref.   :", round(rmse(naif, cible), 4))
print("gain              :", round(100 * (1 - rmse(pred, cible) / rmse(naif, cible)), 1), "%")"""),
 ("code", r"""fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax in axes:
    i = np.random.randint(len(obs))
    ax.plot(obs[i, :, 0], obs[i, :, 1], color="#5B6470", lw=1.2, label="observe")
    ax.plot(cible[i, :, 0], cible[i, :, 1], color="#1E8449", lw=2, label="vrai futur")
    ax.plot(pred[i, :, 0], pred[i, :, 1], color="#C0392B", lw=2, ls="--", label="LSTM")
    ax.plot(naif[i, :, 0], naif[i, :, 1], color="#D97706", lw=1.4, ls=":", label="naif")
    ax.legend(fontsize=7.5); ax.set_title("trajectoire", fontsize=9)
    ax.axis("off")
plt.suptitle("Trois trajectoires : le modele suit la courbure, le naif reste immobile")
plt.show()"""),
 ("md", "## Grille d'évaluation (20 points)\n\n"
        "| Critère | Points |\n|---|---|\n"
        "| Le notebook s'exécute de bout en bout | 4 |\n"
        "| Découpage correct (pas de fuite) et justification | 4 |\n"
        "| RMSE mesuré **et** comparé à la référence naïve | 4 |\n"
        "| Une expérience contrôlée (horizon, H, GRU vs LSTM) | 4 |\n"
        "| Analyse des échecs (trajectoires à fort virage) | 2 |\n"
        "| Conclusion : trois limites honnêtes | 2 |\n\n"
        "**Règle :** tout chiffre du rapport doit être reproductible."""),
 ("md", "## À vous\n\n"
        "**1.** Faites varier l'horizon (5, 10, 20) : comment évolue le RMSE ?\n\n"
        "**2.** Remplacez le LSTM par une GRU : gain ou perte ?\n\n"
        "**3.** Le modèle se trompe-t-il davantage sur les trajectoires à fort virage ? "
        "Mesurez l'erreur en fonction de la courbure."),
 ("code", r"""# ---- A VOUS ----
# 1)
# 2)
# 3)"""),
]
execute(nb("Projet — Prédiction de trajectoire", "3 h", "20 points",
           "- Mener un projet complet : données, modèle, entraînement, évaluation.\n"
           "- Comparer à une référence naïve.\n"
           "- Analyser les cas d'échec.",
           "1. Mener le projet jusqu'au bout.\n"
           "2. Ajouter une expérience contrôlée mesurée.\n"
           "3. Rendre le notebook exécuté avec une conclusion écrite.", c4),
        "PROJET_prediction_trajectoire.ipynb")
print("\nTP3 et projet generes.")
