"""Fabrique des notebooks de TP du cours RNN (execution reelle, CPU)."""
import os, time
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NBDIR = os.path.join(ROOT, "notebooks")
os.makedirs(NBDIR, exist_ok=True)
KERNEL = "rnn-20h"

ENTETE = ("%matplotlib inline\n"
          "import numpy as np, torch, torch.nn as nn\n"
          "import matplotlib.pyplot as plt\n"
          "torch.manual_seed(4112); np.random.seed(4112)\n"
          "torch.set_num_threads(2)\n"
          "print('torch', torch.__version__)")


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

- Exécuter les cellules dans l'ordre. Tout tourne sur **CPU** (quelques minutes).
- Les cellules **« À vous »** sont évaluées.
- Graines fixées : deux exécutions donnent le même résultat."""))
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


# ===================================================================== TP1
c1 = [
 ("md", "## 1. Un vocabulaire, des identifiants\n\n"
        "On construit le vocabulaire d'un petit corpus de critiques, puis on "
        "convertit chaque phrase en suite d'entiers."),
 ("code", r"""corpus = [
    "le film est excellent et les acteurs sont formidables",
    "un chef d oeuvre absolu je recommande vivement",
    "histoire captivante et mise en scene superbe",
    "je me suis ennuye du debut a la fin",
    "le scenario est confus et les dialogues sont plats",
    "film mediocre sans interet et trop long",
]
mots = sorted({m for p in corpus for m in p.split()})
vocab = {"<pad>": 0, "<unk>": 1}
for m in mots:
    vocab[m] = len(vocab)
inv = {i: m for m, i in vocab.items()}
print("taille du vocabulaire :", len(vocab))
print("premiers identifiants :", list(vocab.items())[:8])"""),
 ("md", "## 2. Encoder une phrase"),
 ("code", r"""def encode(phrase, max_len=None):
    ids = [vocab.get(m, vocab["<unk>"]) for m in phrase.split()]
    if max_len:
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return ids

for p in corpus[:3]:
    print(len(encode(p)), encode(p))
print("\nlongueurs :", [len(encode(p)) for p in corpus])"""),
 ("md", "## 3. Padding et masque\n\n"
        "Un lot doit être rectangulaire. On complète avec 0, et on construit un "
        "**masque** pour que ces positions ne comptent pas."),
 ("code", r"""longueur = max(len(encode(p)) for p in corpus)
lot = torch.tensor([encode(p, longueur) for p in corpus])
masque = (lot != 0).float()
print("forme du lot :", tuple(lot.shape))
print("premiere phrase :", lot[0].tolist())
print("son masque      :", masque[0].tolist())
print("mots reels par phrase :", masque.sum(1).tolist())"""),
 ("md", "## 4. L'embedding : un mot devient un vecteur"),
 ("code", r"""dim = 16
emb = nn.Embedding(len(vocab), dim, padding_idx=0)
vecteurs = emb(lot)
print("forme :", tuple(vecteurs.shape), "= (lot, longueur, dimension)")

# le vecteur du mot 'film'
i = vocab["film"]
print("vecteur de 'film' (8 premieres valeurs) :")
print(vecteurs[0, lot[0].tolist().index(i), :8].detach().numpy().round(3))"""),
 ("md", "## 5. Moyenne masquée : un premier résumé de phrase\n\n"
        "Sans masque, les zéros de padding diluent la moyenne. Comparez."),
 ("code", r"""def moyenne_masquee(v, m):
    m = m.unsqueeze(-1)
    return (v * m).sum(1) / m.sum(1).clamp(min=1)

sans = vecteurs.mean(1)
avec = moyenne_masquee(vecteurs, masque)
print("sans masque, phrase 0 :", sans[0, :4].detach().numpy().round(3))
print("avec masque, phrase 0 :", avec[0, :4].detach().numpy().round(3))
print("norme de l'ecart      :", float((sans - avec).norm()))"""),
 ("md", "## 6. À vous\n\n"
        "**1.** Ajoutez deux phrases au corpus et vérifiez que le vocabulaire grandit.\n\n"
        "**2.** Que se passe-t-il si l'on met `padding_idx=0` puis qu'on oublie le "
        "masque ? Mesurez l'écart sur une phrase courte.\n\n"
        "**3.** Pourquoi réserve-t-on l'indice 0 au padding plutôt qu'au mot le plus "
        "fréquent ?"),
 ("code", r"""# ---- A VOUS ----
# 1)

# 2)

# 3) reponse :"""),
 ("md", "## 7. Synthèse\n\n"
        "| Étape | Résultat |\n|---|---|\n"
        "| tokenisation | liste de mots |\n"
        "| indexation | liste d'entiers |\n"
        "| padding | matrice rectangulaire |\n"
        "| masque | les positions de remplissage ne comptent pas |\n"
        "| embedding | tenseur (lot, longueur, dimension) |"),
]
execute(nb("TP1 — Embeddings et séquences", "1 h 30", "10 points",
           "- Construire un vocabulaire et indexer un corpus.\n"
           "- Padder un lot et construire le masque.\n"
           "- Manipuler une couche d'embedding et calculer une moyenne masquée.",
           "1. Répondre aux 3 questions de la section « À vous ».\n"
           "2. Rendre le notebook exécuté.", c1), "TP1_embeddings_sequences.ipynb")

# ===================================================================== TP2
c2 = [
 ("md", "## 1. La cellule récurrente, écrite à la main\n\n"
        "Avant PyTorch, on écrit l'équation : $h_t = \\tanh(W_x x_t + W_h h_{t-1} + b)$."),
 ("code", r"""E, H = 1, 16
Wx = nn.Linear(E, H)
Wh = nn.Linear(H, H, bias=False)
b = nn.Parameter(torch.zeros(H))

def cellule(x_t, h_pre):
    return torch.tanh(Wx(x_t) + Wh(h_pre) + b)

h = torch.zeros(H)
for k in range(5):
    h = cellule(torch.tensor([0.3]), h)
print("etat apres 5 pas :", h[:4].detach().numpy().round(3))
print("norme de h       :", round(float(h.norm()), 3), "(borne par tanh)")"""),
 ("md", "## 2. Le gradient s'évanouit : mesure\n\n"
        "On propage un gradient à travers $T$ pas et on mesure sa norme."),
 ("code", r"""def norme_gradient(T, rayon):
    torch.manual_seed(4112)
    w = nn.Parameter(torch.tensor(rayon))
    h = torch.tensor(1.0)
    for _ in range(T):
        h = torch.tanh(w * h)
    h.backward()
    return float(w.grad)

for T in (5, 10, 20, 40):
    g = norme_gradient(T, 0.55)
    print(f"T={T:3d}   |d h_T / d w| = {g:.3e}")"""),
 ("code", r"""Ts = np.arange(1, 41)
for rayon, coul, lbl in [(0.55, "#C0392B", "r = 0,55"),
                         (0.85, "#D97706", "r = 0,85"),
                         (1.15, "#12A594", "r = 1,15")]:
    plt.semilogy(Ts, [abs(norme_gradient(int(T), rayon)) for T in Ts],
                 color=coul, lw=2, label=lbl)
plt.axhline(1, color="k", ls="--", lw=1)
plt.xlabel("nombre de pas T"); plt.ylabel("norme du gradient (log)")
plt.title("Le gradient le long du temps")
plt.legend(); plt.show()"""),
 ("md", "## 3. Entraîner un RNN à prédire une série"),
 ("code", r"""t = np.arange(400)
serie = np.sin(2 * np.pi * t / 40) + 0.15 * np.random.default_rng(4112).normal(size=400)
serie = (serie - serie.mean()) / serie.std()
X = torch.tensor(serie[:-1], dtype=torch.float32).view(1, -1, 1)
Y = torch.tensor(serie[1:], dtype=torch.float32).view(1, -1, 1)

class RNN(nn.Module):
    def __init__(self, H=32):
        super().__init__()
        self.rnn = nn.RNN(1, H, batch_first=True)
        self.out = nn.Linear(H, 1)
    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h)

modele = RNN()
opt = torch.optim.Adam(modele.parameters(), lr=1e-2)
perte_f = nn.MSELoss()
for ep in range(60):
    opt.zero_grad()
    pred = modele(X)[:, :-1, :]
    perte = perte_f(pred, Y[:, :-1, :])
    perte.backward()
    torch.nn.utils.clip_grad_norm_(modele.parameters(), 5.0)
    opt.step()
    if ep % 15 == 0:
        print(f"epoque {ep:2d}  perte = {perte.item():.5f}")
print("perte finale :", round(float(perte), 5))"""),
 ("code", r"""with torch.no_grad():
    pred = modele(X)[0, :, 0].numpy()
plt.figure(figsize=(11, 3))
plt.plot(serie[1:], label="observé", color="grey", lw=1)
plt.plot(pred, label="prédit", color="#1F6FEB", lw=1.6)
plt.legend(); plt.title("Prédiction pas à pas (une entrée, une sortie)")
plt.xlabel("temps"); plt.show()
print("RMSE :", round(float(np.sqrt(np.mean((pred - serie[1:])**2))), 4))"""),
 ("md", "## 4. À vous\n\n"
        "**1.** Remplacez `nn.RNN` par une cellule écrite à la main et réentraînez. "
        "La perte finale est-elle comparable ?\n\n"
        "**2.** Supprimez le `clip_grad_norm_` : que devient la perte ?\n\n"
        "**3.** Augmentez la période de la sinusoïde (80 au lieu de 40) : le RNN "
        "simple y arrive-t-il encore ?"),
 ("code", r"""# ---- A VOUS ----
# 1)
# 2)
# 3)"""),
 ("md", "## 5. Synthèse\n\n"
        "| Observation | Interprétation |\n|---|---|\n"
        "| norme du gradient ↓ très vite | évanouissement |\n"
        "| norme ↑ | explosion, à clipper |\n"
        "| RNN correct sur période courte | mémoire effective limitée |"),
]
execute(nb("TP2 — RNN sur série temporelle", "2 h", "10 points",
           "- Écrire la cellule récurrente à la main.\n"
           "- Mesurer l'évanouissement du gradient.\n"
           "- Entraîner un RNN à prédire une série.",
           "1. Répondre aux 3 questions de la section « À vous ».\n"
           "2. Rendre le notebook exécuté.", c2), "TP2_RNN_serie_temporelle.ipynb")

print("\nTP1 et TP2 generes.")
