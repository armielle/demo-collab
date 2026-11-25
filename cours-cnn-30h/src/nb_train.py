"""Fabrique de notebooks -- TP d'entrainement : TP3, TP4, PROJET.

Machine CPU (8 coeurs, pas de GPU) : sous-ensembles de donnees, peu d'epoques,
threads limites a 2.
"""
import os, time
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NBDIR = os.path.join(ROOT, "notebooks")
os.makedirs(NBDIR, exist_ok=True)
KERNEL = "cnn-30h"

ENTETE = "%matplotlib inline\nimport numpy as np, torch, torch.nn as nn, torch.nn.functional as F\nimport matplotlib.pyplot as plt\ntorch.manual_seed(4112); np.random.seed(4112)\ntorch.set_num_threads(2)\ndevice = 'cpu'\nprint('torch', torch.__version__, '| appareil :', device)"


def nb(titre, duree, bareme, objectifs, travail, cellules):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": KERNEL, "display_name": "CNN 30h", "language": "python"}
    n.metadata["language_info"] = {"name": "python"}
    n.cells.append(nbf.v4.new_markdown_cell(
        f"""# {titre}

**Cours :** Réseaux de neurones convolutifs (CNN) — 30 h — Master 1 SDIA
**Enseignante :** Dr. Noulapeu Ngaffo Armielle — ENSPD, Université de Douala
**Durée conseillée :** {duree} · **Barème :** {bareme}

---

## Objectifs

{objectifs}

## Travail demandé

{travail}

## Mode d'emploi

- Exécuter les cellules **dans l'ordre**. L'entraînement tourne sur **CPU** : quelques minutes.
- Les cellules **« À vous »** sont évaluées.
- Toutes les graines sont fixées : deux exécutions donnent le même résultat."""))
    n.cells.append(nbf.v4.new_code_cell(ENTETE))
    for typ, src in cellules:
        n.cells.append(nbf.v4.new_markdown_cell(src) if typ == "md"
                       else nbf.v4.new_code_cell(src))
    return n


def execute(n, nom):
    chemin = os.path.join(NBDIR, nom)
    t0 = time.time()
    NotebookClient(n, timeout=3000, kernel_name=KERNEL,
                   resources={"metadata": {"path": NBDIR}}).execute()
    with open(chemin, "w", encoding="utf-8") as f:
        nbf.write(n, f)
    nimg = sum(1 for c in n.cells if c.cell_type == "code"
               for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    nc = sum(1 for c in n.cells if c.cell_type == "code")
    print(f"[nb] {nom:<46} {nc} cellules code, {nimg} figures  "
          f"{os.path.getsize(chemin)/1024:.1f} Ko  ({time.time()-t0:.0f} s)")


# ==================================================================== TP3
c3 = [
 ("md", r"""## 1. Le jeu de données MNIST

70 000 chiffres manuscrits, 28×28 pixels, en niveaux de gris. On en prend un
sous-ensemble pour que l'entraînement reste rapide sur CPU (pas de GPU ici).""")
, ("code", r"""from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

tf = transforms.Compose([transforms.Pad(2),
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
train_full = datasets.MNIST("../data", train=True, download=True, transform=tf)
test_full = datasets.MNIST("../data", train=False, download=True, transform=tf)

N_TRAIN, N_TEST = 6000, 1500
tr = Subset(train_full, range(N_TRAIN))
te = Subset(test_full, range(N_TEST))
train_loader = DataLoader(tr, batch_size=64, shuffle=True)
test_loader = DataLoader(te, batch_size=256)

print("entrainement :", len(tr), "images")
print("test         :", len(te), "images")

xb, yb = next(iter(train_loader))
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for ax, im, yy in zip(axes, xb[:8], yb[:8]):
    ax.imshow(im[0], cmap="gray"); ax.set_title(int(yy)); ax.axis("off")
plt.suptitle("Quelques images du sous-ensemble"); plt.show()""")
, ("md", r"""## 2. LeNet-5, en PyTorch

L'architecture du cours : conv 5×5 (6 filtres) → pool → conv 5×5 (16) → pool →
conv 5×5 (120) → dense 84 → sortie 10.""")
, ("code", r"""class LeNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c3 = nn.Conv2d(6, 16, 5)
        self.c5 = nn.Conv2d(16, 120, 5)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        x = F.relu(self.c5(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.f6(x))
        return self.out(x)

modele = LeNet().to(device)
n_param = sum(p.numel() for p in modele.parameters() if p.requires_grad)
print(f"parametres entraînables : {n_param:,}".replace(",", " "))""")
, ("md", r"""## 3. Entraînement

La boucle est toujours la même : sortie du modèle → perte → rétropropagation →
mise à jour des poids.""")
, ("code", r"""from tqdm.auto import tqdm
optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)
perte_f = nn.CrossEntropyLoss()
historique = {"perte": [], "acc_train": [], "acc_test": []}

def evalue(modele, loader):
    modele.eval(); bon = tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            bon += (modele(x).argmax(1) == y).sum().item(); tot += y.size(0)
    return bon / tot

EPOQUES = 3
for ep in range(EPOQUES):
    modele.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimiseur.zero_grad()
        perte = perte_f(modele(x), y)
        perte.backward(); optimiseur.step()
        historique["perte"].append(perte.item())
    a_tr = evalue(modele, train_loader); a_te = evalue(modele, test_loader)
    historique["acc_train"].append(a_tr); historique["acc_test"].append(a_te)
    print(f"epoque {ep+1}/{EPOQUES}  perte={np.mean(historique['perte'][-94:]):.4f}"
          f"  train={a_tr:.1%}  test={a_te:.1%}")

print("\nexactitude finale sur le test :", f"{historique['acc_test'][-1]:.1%}")""")
, ("code", r"""fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
axes[0].plot(historique["perte"], lw=.8, color="grey")
axes[0].set_xlabel("iteration"); axes[0].set_ylabel("perte"); axes[0].set_title("Perte d'entrainement")
axes[1].plot(range(1, EPOQUES+1), historique["acc_train"], "o-", label="train")
axes[1].plot(range(1, EPOQUES+1), historique["acc_test"], "s-", label="test")
axes[1].set_xlabel("epoque"); axes[1].set_ylabel("exactitude"); axes[1].legend()
axes[1].set_title("Exactitude train / test")
plt.tight_layout(); plt.show()""")
, ("md", r"""## 4. Où le modèle se trompe-t-il ?

On regarde les erreurs : c'est souvent plus instructif que le score global.""")
, ("code", r"""modele.eval()
x, y = next(iter(test_loader))
with torch.no_grad():
    pred = modele(x).argmax(1)
faux = (pred != y).nonzero().flatten()[:8]
print("erreurs affichees :", len(faux), "sur", len(y))

fig, axes = plt.subplots(1, 8, figsize=(12, 2.2))
for ax, i in zip(axes, faux):
    ax.imshow(x[i][0], cmap="gray")
    ax.set_title(f"vrai {int(y[i])}\npred {int(pred[i])}", fontsize=8)
    ax.axis("off")
plt.suptitle("Exemples mal classes"); plt.show()""")
, ("md", r"""## 5. À vous

**Question 1.** Quelle est la forme du tenseur d'entrée de la première couche
convolutive ? Combien de paramètres contient `c1` ?

**Question 2.** Ajoutez une couche de dropout (p = 0,5) après `f6` et relancez
l'entraînement. L'exactitude de test progresse-t-elle ?

**Question 3.** Remplacez `max_pool2d` par `avg_pool2d` : quel est l'effet ?""")
, ("code", r"""# ---- À VOUS ----
# 1) forme : (N, C, H, W) = ...
#    parametres de c1 = K*K*Cin*Cout + Cout = ...
print("parametres de c1 :", sum(p.numel() for p in modele.c1.parameters()))

# 2) a completer : ajouter nn.Dropout(0.5) et relancer

# 3) a completer : remplacer max_pool2d par avg_pool2d""")
, ("md", r"""## 6. Synthèse

| Étape | Rôle |
|---|---|
| `Conv2d` | extrait des caractéristiques locales |
| `ReLU` | introduit la non-linéarité |
| `MaxPool2d` | réduit la résolution, sans paramètre |
| `Flatten` | transforme le tenseur en vecteur |
| `Linear` | décide à partir des caractéristiques |

**Repère :** LeNet-5 complet ≈ 61 000 paramètres. Un MLP sur les mêmes 784 entrées en
demanderait 784 000 pour une seule couche.""")
]
execute(nb("TP3 — Premier CNN sur MNIST", "2 h", "10 points",
           r"""- Construire un CNN en PyTorch et l'entraîner sur CPU.
- Lire les courbes d'apprentissage et l'écart train / test.
- Repérer les erreurs du modèle et les interpréter.""",
           r"""1. Répondre aux 3 questions de la section « À vous » (les cellules doivent tourner).
2. Rendre le notebook exécuté.""", c3), "TP3_premier_CNN_MNIST.ipynb")

# ==================================================================== TP4
c4 = [
 ("md", r"""## 1. Surapprentissage, régularisation, augmentation

On entraîne **trois fois le même réseau** sur le même jeu de données, en changeant
un seul réglage à chaque fois.""")
, ("code", r"""from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

tf = transforms.Compose([transforms.Pad(2),
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
train_full = datasets.MNIST("../data", train=True, download=True, transform=tf)
test_full = datasets.MNIST("../data", train=False, download=True, transform=tf)

N_TRAIN, N_TEST = 4000, 1000
# jeu volontairement plus petit que celui du TP3 : le surapprentissage apparait mieux
tr = Subset(train_full, range(N_TRAIN))
te = Subset(test_full, range(N_TEST))
train_loader = DataLoader(tr, batch_size=64, shuffle=True)
test_loader = DataLoader(te, batch_size=256)
print(len(tr), "images d'entrainement,", len(te), "de test")""")
, ("code", r"""class LeNet(nn.Module):
    def __init__(self, dropout=0.0, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5); self.c3 = nn.Conv2d(6, 16, 5)
        self.c5 = nn.Conv2d(16, 120, 5); self.f6 = nn.Linear(120, 84)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        x = F.relu(self.c5(x))
        x = F.relu(self.f6(self.drop(torch.flatten(x, 1))))
        return self.out(x)


def entraine(dropout=0.0, augmenter=False, epoques=3, lr=1e-3):
    tf_ = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomAffine(degrees=12, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))] if augmenter else [
        transforms.Pad(2), transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    ds = Subset(datasets.MNIST("../data", train=True, download=True, transform=tf_), range(N_TRAIN))
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    m = LeNet(dropout=dropout).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr); pf = nn.CrossEntropyLoss()

    def acc(loader, m):
        m.eval(); b = t = 0
        with torch.no_grad():
            for x, y in loader:
                b += (m(x).argmax(1) == y).sum().item(); t += y.size(0)
        return b / t

    hist = {"train": [], "test": []}
    for ep in range(epoques):
        m.train()
        for x, y in dl:
            opt.zero_grad(); pf(m(x), y).backward(); opt.step()
        hist["train"].append(acc(train_loader, m)); hist["test"].append(acc(test_loader, m))
        print(f"  epoque {ep+1}: train={hist['train'][-1]:.1%}  test={hist['test'][-1]:.1%}")
    return hist

print("=== A. sans regularisation ===")
h_A = entraine(dropout=0.0, augmenter=False)
print("\n=== B. avec dropout (p = 0,5) ===")
h_B = entraine(dropout=0.5, augmenter=False)
print("\n=== C. avec dropout + augmentation ===")
h_C = entraine(dropout=0.5, augmenter=True)""")
, ("code", r"""fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.4))
for h, nom, coul in [(h_A, "A. brute", "#C0392B"), (h_B, "B. dropout", "#D97706"),
                     (h_C, "C. dropout + augmentation", "#1E8449")]:
    axes[0].plot(range(1, len(h["train"])+1), h["train"], "o-", color=coul, label=nom)
    axes[1].plot(range(1, len(h["test"])+1), h["test"], "s-", color=coul, label=nom)
axes[0].set_title("Exactitude — entrainement"); axes[1].set_title("Exactitude — test")
for a in axes:
    a.set_xlabel("epoque"); a.set_ylabel("exactitude"); a.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

ecarts = {n: h["train"][-1] - h["test"][-1] for n, h in [("A", h_A), ("B", h_B), ("C", h_C)]}
print("ecart train - test en fin d'entrainement :")
for n, e in ecarts.items():
    print(f"   {n} : {e:+.1%}")""")
, ("md", r"""## 2. Voir l'augmentation

L'augmentation fabrique des variantes plausibles de chaque image. Le réseau voit ainsi
davantage de situations sans qu'on collecte de nouvelles données.""")
, ("code", r"""base = datasets.MNIST("../data", train=True, download=True,
                        transform=transforms.ToTensor())
img = base[7][0]
aug = transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.12, 0.12))])

fig, axes = plt.subplots(2, 6, figsize=(12.5, 3.6))
for j in range(6):
    axes[0, j].imshow(img[0], cmap="gray"); axes[0, j].set_title("originale" if j == 0 else "", fontsize=8)
    axes[1, j].imshow(aug(img)[0], cmap="gray")
    axes[0, j].axis("off"); axes[1, j].axis("off")
axes[1, 0].set_ylabel("augmentee")
plt.suptitle("Une image source, six variantes d'entrainement", fontweight="bold")
plt.tight_layout(); plt.show()""")
, ("md", r"""## 3. À vous

**Question 1.** Pourquoi l'augmentation est-elle appliquée **uniquement** au jeu
d'entraînement et jamais à celui de test ?

**Question 2.** Le dropout ralentit l'apprentissage par époque mais améliore le test :
expliquez ce compromis.

**Question 3.** Avec un jeu d'entraînement beaucoup plus grand, le bénéfice du dropout
diminuerait-il ou augmenterait-il ? Justifiez.""")
, ("code", r"""# ---- À VOUS ----
# 1)
# 2)
# 3)""")
, ("md", r"""## 4. Synthèse

| Régularisation | Principe | Coût |
|---|---|---|
| **Dropout** | éteindre aléatoirement des neurones | apprentissage plus lent |
| **Augmentation** | multiplier les variantes des données | calcul par époque |
| **Weight decay** | pénaliser les grands poids | 1 hyperparamètre de plus |
| **Early stopping** | arrêter au minimum de validation | nécessite un jeu de validation |
| **Batch norm** | recaler les activations | quasi gratuit |

**Le meilleur levier reste… davantage de données.**""")
]
execute(nb("TP4 — Régularisation et augmentation", "2 h", "10 points",
           r"""- Mesurer le surapprentissage et l'effet de trois stratégies de régularisation.
- Comparer les courbes train / test entre configurations.
- Comprendre quand l'augmentation aide et quand elle ne suffit plus.""",
           r"""1. Répondre aux 3 questions de la section « À vous ».
2. Rendre le notebook exécuté.""", c4), "TP4_regularisation_augmentation.ipynb")

# ================================================================ PROJET
cP = [
 ("md", r"""## Projet — Classification d'images réelles de bout en bout

**Jeu de données.** Le projet du cours porte sur **GTSRB** (panneaux routiers, 43 classes,
39 209 images d'entraînement). **GTSRB n'est pas accessible depuis cette machine** (serveur
injoignable) : le projet utilise donc **CIFAR-10**, autre jeu réel de 60 000 images couleur
32×32 en 10 classes. **Le code est identique** : pour passer à GTSRB, il suffit de remplacer
le `DataLoader` (une cellule est prévue pour cela en fin de notebook).

**Contrainte :** machine **CPU**, pas de GPU. Sous-ensemble + peu d'époques.""")
, ("code", r"""from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

moy, ect = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(moy, ect)])
tf_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(moy, ect)])

tr_full = datasets.CIFAR10("../data", train=True, download=True, transform=tf_train)
te_full = datasets.CIFAR10("../data", train=False, download=True, transform=tf_test)
CLASSES = ['avion', 'automobile', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion']
N_TR, N_TE = 8000, 2000
tr = Subset(tr_full, range(N_TR)); te = Subset(te_full, range(N_TE))
train_loader = DataLoader(tr, batch_size=128, shuffle=True)
test_loader = DataLoader(te, batch_size=256)
print(N_TR, "images d'entrainement |", N_TE, "de test |", len(CLASSES), "classes")""")
, ("code", r"""xb, yb = next(iter(train_loader))
fig, axes = plt.subplots(1, 10, figsize=(13, 1.8))
for ax, im, yy in zip(axes, xb[:10], yb[:10]):
    ax.imshow(np.transpose(im, (1, 2, 0)) * np.array(ect) + np.array(moy))
    ax.set_title(CLASSES[int(yy)], fontsize=7.5); ax.axis("off")
plt.suptitle("CIFAR-10 : 10 classes d'images couleur reelles", fontweight="bold")
plt.show()""")
, ("code", r"""class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        def bloc(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.corps = nn.Sequential(bloc(3, 32), bloc(32, 64), bloc(64, 128))
        self.tete = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128*4*4, 256),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.tete(self.corps(x))

modele = CNN().to(device)
print("parametres :", f"{sum(p.numel() for p in modele.parameters()):,}".replace(",", " "))""")
, ("code", r"""opt = torch.optim.Adam(modele.parameters(), lr=1.5e-3, weight_decay=5e-4)
EPOQUES = 6
plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOQUES)
pf = nn.CrossEntropyLoss()
hist = {"perte": [], "train": [], "test": []}

def evalue(m, loader):
    m.eval(); b = t = 0
    with torch.no_grad():
        for x, y in loader:
            b += (m(x).argmax(1) == y).sum().item(); t += y.size(0)
    return b / t

EPOQUES = 8
for ep in range(EPOQUES):
    modele.train(); acc_p = 0
    for x, y in train_loader:
        opt.zero_grad(); perte = pf(modele(x), y)
        perte.backward(); opt.step(); hist["perte"].append(perte.item())
    plan.step()
    a_tr = evalue(modele, train_loader); a_te = evalue(modele, test_loader)
    hist["train"].append(a_tr); hist["test"].append(a_te)
    print(f"epoque {ep+1}/{EPOQUES}  perte={np.mean(hist['perte'][-94:]):.4f}"
          f"  train={a_tr:.1%}  test={a_te:.1%}")

print("\nexactitude test :", f"{hist['test'][-1]:.2%}")""")
, ("code", r"""fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.4))
axes[0].plot(hist["perte"], lw=.8, color="grey"); axes[0].set_title("Perte")
axes[0].set_xlabel("iteration"); axes[0].set_ylabel("perte")
axes[1].plot(range(1, EPOQUES+1), hist["train"], "o-", label="train")
axes[1].plot(range(1, EPOQUES+1), hist["test"], "s-", label="test")
axes[1].set_title("Exactitude"); axes[1].set_xlabel("epoque"); axes[1].legend()
plt.tight_layout(); plt.show()""")
, ("md", r"""## Évaluation détaillée

L'exactitude globale ne suffit pas : on regarde classe par classe.""")
, ("code", r"""from sklearn.metrics import confusion_matrix, classification_report

modele.eval(); P, Y = [], []
with torch.no_grad():
    for x, y in test_loader:
        P.append(modele(x).argmax(1)); Y.append(y)
P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()

M = confusion_matrix(Y, P)
print(classification_report(Y, P, target_names=CLASSES, digits=3, zero_division=0))""")
, ("code", r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
im = axes[0].imshow(M, cmap="Blues")
axes[0].set_xticks(range(10)); axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
axes[0].set_yticks(range(10)); axes[0].set_yticklabels(CLASSES, fontsize=8)
axes[0].set_xlabel("classe predite"); axes[0].set_ylabel("classe reelle")
axes[0].set_title(f"Matrice de confusion (exactitude {M.trace()/M.sum():.1%})")
for i in range(10):
    for j in range(10):
        if M[i, j]:
            axes[0].text(j, i, M[i, j], ha="center", va="center", fontsize=6.2,
                         color="white" if M[i, j] > M.max()*.55 else "#10243E")
axes[0].grid(False)
rappel = M.diagonal() / M.sum(1)
axes[1].barh(CLASSES, rappel, color="#1F6FEB", alpha=.85)
axes[1].set_xlim(0, 1); axes[1].set_xlabel("rappel")
axes[1].set_title("Rappel par classe")
plt.tight_layout(); plt.show()

pire = int(np.argmin(rappel))
print(f"classe la plus difficile : {CLASSES[pire]} (rappel {rappel[pire]:.1%})")""")
, ("md", r"""## Top-k et robustesse

Un modèle peut se tromper tout en classant la bonne réponse au second rang. On mesure
aussi sa résistance à une transformation géométrique simple.""")
, ("code", r"""corrects = []
with torch.no_grad():
    for x, y in test_loader:
        s = modele(x)
        top5 = s.topk(5, dim=1).indices
        corrects.append((top5 == y[:, None]).any(1))
top5_acc = torch.cat(corrects).float().mean().item()
print(f"top-1 : {hist['test'][-1]:.2%}")
print(f"top-5 : {top5_acc:.2%}")

# robustesse : rotation de 20 degres
tf_rot = transforms.Compose([transforms.RandomRotation((-20, -20)), transforms.ToTensor(),
                             transforms.Normalize(moy, ect)])
te_rot = Subset(datasets.CIFAR10("../data", train=False, download=True, transform=tf_rot), range(N_TE))
print(f"top-1 sous rotation -20 deg : {evalue(modele, DataLoader(te_rot, batch_size=256)):.2%}")""")
, ("md", r"""## Gradient-weighted Class Activation Mapping (Grad-CAM)

Quelles régions de l'image ont pesé dans la décision ? On remonte le gradient de la
classe prédite jusqu'à la dernière carte convolutive.""")
, ("code", r"""def gradcam(modele, img, classe=None):
    modele.eval()
    x = img.unsqueeze(0).to(device)
    if classe is None:
        with torch.no_grad():
            classe = int(modele(x).argmax(1))
    cartes, grads = [], []
    cible = modele.corps[-1][3]          # derniere couche convolutive du corps
    h = cible.register_forward_hook(lambda m, i, o: cartes.append(o))
    hs = cible.register_full_backward_hook(lambda m, gi, go: grads.append(gi[0]))
    sortie = modele(x)[0, classe]; modele.zero_grad(); sortie.backward()
    h.remove(); hs.remove()
    A, G = cartes[0][0], grads[0][0]
    poids = G.mean(dim=(1, 2))
    cam = torch.relu((poids[:, None, None] * A).sum(0))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy(), classe

xb, yb = next(iter(test_loader))
fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
for j in range(4):
    img = xb[j]
    cam, cl = gradcam(modele, img)
    aff = np.transpose(img, (1, 2, 0)) * np.array(ect) + np.array(moy)
    axes[0, j].imshow(np.clip(aff, 0, 1)); axes[0, j].set_title(CLASSES[cl], fontsize=9)
    axes[1, j].imshow(np.clip(aff, 0, 1))
    axes[1, j].imshow(F.interpolate(torch.tensor(cam)[None, None], size=(32, 32),
                                    mode="bilinear")[0, 0], cmap="jet", alpha=.5)
    axes[0, j].axis("off"); axes[1, j].axis("off")
axes[0, 0].set_ylabel("image", fontsize=9); axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
plt.suptitle("Ou le reseau regarde-t-il pour decider ?", fontweight="bold")
plt.tight_layout(); plt.show()""")
, ("md", r"""## Passer à GTSRB (panneaux routiers)

Le code du projet est identique : il suffit de changer le jeu de données. Téléchargez
GTSRB manuellement, puis :

```python
# racine contenant les dossiers gtsrb-german-traffic-sign/ ou le .zip officiel
gtsrb_train = datasets.GTSRB(root="../data", split="train",
                             download=False, transform=tf_train)
gtsrb_test  = datasets.GTSRB(root="../data", split="test",
                             download=False, transform=tf_test)
# 43 classes au lieu de 10 ; adaptez n_classes dans CNN()
```

## Grille d'évaluation du projet (20 points)

| Critère | Points |
|---|---|
| Le notebook s'exécute de bout en bout sans erreur | 4 |
| Le modèle atteint une exactitude de test raisonnable (> 60 % sur CIFAR-10) | 4 |
| La matrice de confusion est commentée (classes confondues, pourquoi) | 4 |
| Une expérience de régularisation est menée **et mesurée** | 4 |
| Le Grad-CAM est interprété (le réseau regarde-t-il au bon endroit ?) | 2 |
| Conclusion : trois limites honnêtes du modèle | 2 |

**Rappel méthodologique :** un chiffre annoncé sans la commande qui l'a produit ne vaut rien.
Toute valeur de votre rapport doit être reproductible.""")
]
execute(nb("Projet — Classification d'images de bout en bout", "4 h", "20 points",
           r"""- Mener un projet complet : données, modèle, entraînement, évaluation.
- Interpréter les résultats (matrice de confusion, rappel par classe, top-k).
- Visualiser les zones de décision par Grad-CAM.
- Rédiger une conclusion critique sur les limites du modèle.""",
           r"""1. Mener le projet jusqu'au bout (le notebook doit s'exécuter entièrement).
2. Ajouter au moins **une expérience de régularisation** supplémentaire, mesurée.
3. Rendre le notebook exécuté accompagné d'une conclusion écrite.""", cP),
        "PROJET_classification_images.ipynb")

print("\nTP d'entrainement generes.")