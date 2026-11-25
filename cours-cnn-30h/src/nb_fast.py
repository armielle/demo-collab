"""Fabrique de notebooks -- TP rapides (sans telechargement) : TP1, TP2, TP5."""
import os, sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NBDIR = os.path.join(ROOT, "notebooks")
os.makedirs(NBDIR, exist_ok=True)

KERNEL = "cnn-30h"

CONSIGNES = r"""# %(titre)s

**Cours :** Réseaux de neurones convolutifs (CNN) — 30 h — Master 1 SDIA
**Enseignante :** Dr. Noulapeu Ngaffo Armielle — ENSPD, Université de Douala
**Durée conseillée :** %(duree)s  ·  **Barème :** %(bareme)s

---

## Objectifs

%(objectifs)s

## Travail demandé

%(travail)s

## Mode d'emploi

- Exécuter les cellules **dans l'ordre** (`Shift + Entrée`).
- Les cellules marquées **« À vous »** sont à compléter : ce sont elles qui sont évaluées.
- Toutes les figures s'affichent dans le notebook ; si ce n'est pas le cas, vérifier que la
  première cellule a bien été exécutée.
- Reproductibilité : les graines aléatoires sont fixées. Deux exécutions donnent le même résultat.
"""


def nb(titre, duree, bareme, objectifs, travail, cellules):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": KERNEL, "display_name": "CNN 30h", "language": "python"}
    n.metadata["language_info"] = {"name": "python"}
    entete = CONSIGNES % dict(titre=titre, duree=duree, bareme=bareme,
                              objectifs=objectifs, travail=travail)
    n.cells.append(nbf.v4.new_markdown_cell(entete))
    n.cells.append(nbf.v4.new_code_cell("%matplotlib inline\nimport numpy as np\n"
                                        "import matplotlib.pyplot as plt\n"
                                        "np.random.seed(4112)"))
    for typ, src in cellules:
        n.cells.append(nbf.v4.new_markdown_cell(src) if typ == "md"
                       else nbf.v4.new_code_cell(src))
    return n


def execute(n, nom):
    chemin = os.path.join(NBDIR, nom)
    NotebookClient(n, timeout=2400, kernel_name=KERNEL,
                   resources={"metadata": {"path": NBDIR}}).execute()
    with open(chemin, "w", encoding="utf-8") as f:
        nbf.write(n, f)
    nimg = 0
    ncode = 0
    for c in n.cells:
        if c.cell_type == "code":
            ncode += 1
            for o in c.get("outputs", []):
                if "image/png" in o.get("data", {}):
                    nimg += 1
    print(f"[nb] {nom:<46} {ncode} cellules code, {nimg} figures  "
          f"{os.path.getsize(chemin)/1024:.1f} Ko")


# ===================================================================== TP1
c1 = [
 ("md", r"""## 1. Une image, c'est une matrice

On charge une image réelle en niveaux de gris et on l'affiche.""")
, ("code", r"""img = np.load("../data/face_gray.npy").astype(float)
print("dimensions :", img.shape)
print("valeurs   : min = %.0f, max = %.0f" % (img.min(), img.max()))

plt.figure(figsize=(6, 4))
plt.imshow(img, cmap="gray")
plt.title("Image de travail")
plt.axis("off")
plt.show()""")
, ("md", r"""## 2. Les noyaux de convolution

Un **noyau** (ou *filtre*) est une petite matrice. Appliqué par balayage sur l'image, il
produit une nouvelle image. Voici les noyaux classiques.""")
, ("code", r"""noyaux = {
    "flou (moyenneur)": np.ones((3, 3)) / 9,
    "flou gaussien":    np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    "nettete":          np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "contours (Sobel X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "contours (Sobel Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "laplacien":        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    "relief (emboss)":  np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
}
for nom, k in noyaux.items():
    print(f"{nom:>22} :\n{k}\n")""")
, ("md", r"""## 3. Appliquer un noyau : `scipy.signal.convolve2d`

> **Attention à la convention.** `convolve2d` effectue une *vraie* convolution : le noyau est
> retourné avant d'être appliqué. Les réseaux de neurones utilisent la **corrélation croisée**
> (noyau non retourné). Pour un noyau symétrique, les deux coïncident.""")
, ("code", r"""from scipy import signal

# correlation=True : convention des CNN (noyau NON retourne)
# correlation=False : vraie convolution (noyau retourne)
def applique(im, noyau, correlation=True):
    k = noyau if correlation else noyau[::-1, ::-1]
    return signal.convolve2d(im, k, mode="same", boundary="symm")

fig, axes = plt.subplots(2, 4, figsize=(15, 7))
axes = axes.ravel()
axes[0].imshow(img, cmap="gray"); axes[0].set_title("originale")
for ax, (nom, k) in zip(axes[1:], noyaux.items()):
    r = applique(img, k)
    ax.imshow(np.abs(r) if "contours" in nom or "laplacien" in nom else r, cmap="gray")
    ax.set_title(nom, fontsize=9)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()""")
, ("md", r"""## 4. Vérification du calcul, à la main

Avant de faire confiance à une bibliothèque, on refait le calcul soi-même sur une petite
matrice. Le **premier élément** de la sortie vaut $\sum (\text{zone} \times \text{noyau})$.""")
, ("code", r"""I = np.array([[1, 2, 0, 1, 3],
              [4, 5, 1, 0, 2],
              [2, 3, 6, 1, 1],
              [0, 1, 4, 8, 2],
              [1, 2, 1, 3, 5]], dtype=float)
K = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]], dtype=float)

zone = I[:3, :3]
print("zone examinee :\n", zone)
print("produits terme a terme :", (zone * K).ravel())
print("premier element de sortie =", (zone * K).sum())

# comparaison avec scipy
S = signal.convolve2d(I, K[::-1, ::-1], mode="valid")  # correlation croisee
print("\nsortie complete (correlation croisee) :\n", S)""")
, ("md", r"""## 5. À vous

**Question 1.** Le noyau ci-dessous détecte-t-il plutôt les contours horizontaux ou
verticaux ? Répondez en une phrase, puis vérifiez en l'appliquant.

**Question 2.** Construisez un noyau 5×5 qui floute l'image plus fortement que le
moyenneur 3×3, puis appliquez-le.

**Question 3.** Que se passe-t-il si la somme des coefficients d'un noyau vaut 0 ?
Testez sur une zone uniforme de l'image.""")
, ("code", r"""# ---- À VOUS (question 1) ----
K1 = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]])
# votre reponse en commentaire :
# ...

# ---- À VOUS (question 2) ----
K2 = None   # a completer : np.ones((5,5))/25 par exemple

# ---- À VOUS (question 3) ----
# appliquez un noyau de somme nulle a une zone uniforme et expliquez :
# ...""")
, ("md", r"""## 6. Synthèse

| Ce qu'il faut retenir | |
|---|---|
| Un noyau de moyenne = **flou** | la somme des coefficients vaut 1 |
| Un noyau de différence = **contours** | la somme des coefficients vaut 0 |
| Le résultat s'appelle une **carte d'activation** | c'est ce que le réseau apprendra à produire |
| La convolution est **locale** | chaque pixel de sortie ne dépend que du voisinage |

**Suite :** le TP2 aborde le padding, le stride et le pooling.""")
]
execute(nb("TP1 — Noyaux de convolution", "1 h 30", "10 points",
           r"""- Manipuler une image comme une matrice de nombres.
- Construire et reconnaître les noyaux classiques (flou, netteté, contours, relief).
- Appliquer un noyau et interpréter la carte d'activation obtenue.
- Vérifier « à la main » le résultat d'une convolution sur une petite matrice.""",
           r"""1. Répondre aux 3 questions de la section « À vous ».
2. Rendre le notebook exécuté (toutes les cellules doivent avoir tourné sans erreur).""",
           c1), "TP1_noyaux_convolution.ipynb")

# ===================================================================== TP2
c2 = [
 ("md", r"""## 1. Padding, stride, pooling : trois réglages qui changent tout

On travaille sur de petits tenseurs pour voir exactement ce qui se passe.""")
, ("code", r"""I = np.arange(1, 26).reshape(5, 5).astype(float)
print(I)""")
, ("md", r"""## 2. La formule de dimension

$$N_{out} = \left\lfloor \frac{N_{in} - K + 2P}{S} \right\rfloor + 1$$

$N_{in}$ : côté d'entrée · $K$ : côté du noyau · $P$ : padding · $S$ : stride.""")
, ("code", r"""def dim(N_in, K, P=0, S=1):
    return (N_in - K + 2 * P) // S + 1

for (N, K, P, S) in [(28, 3, 0, 1), (28, 3, 1, 1), (28, 5, 2, 1),
                     (28, 3, 0, 2), (32, 5, 0, 1), (224, 7, 3, 2)]:
    print(f"N_in={N:3d}  K={K}  P={P}  S={S}   ->   N_out={dim(N, K, P, S)}")""")
, ("md", r"""## 3. Padding et stride, en image

Le padding ajoute des zéros autour de l'entrée ; le stride décide du pas de balayage.""")
, ("code", r"""def avec_padding(M, P):
    return np.pad(M, P, constant_values=0)

def correlation(M, K, S=1):
    kh, kw = K.shape
    h = (M.shape[0] - kh) // S + 1
    w = (M.shape[1] - kw) // S + 1
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = (M[i*S:i*S+kh, j*S:j*S+kw] * K).sum()
    return out

K = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], float)
print("valid  (P=0) :", correlation(I, K).shape)
print("same   (P=1) :", correlation(avec_padding(I, 1), K).shape)
print("stride 2     :", correlation(I, K, S=2).shape)""")
, ("md", r"""## 4. Pooling

Le pooling réduit la résolution sans rien apprendre. C'est l'opération la plus simple
du réseau — et l'une des plus efficaces.""")
, ("code", r"""def pool(M, mode="max", k=2, s=2):
    n = (M.shape[0] - k) // s + 1
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            blk = M[i*s:i*s+k, j*s:j*s+k]
            out[i, j] = blk.max() if mode == "max" else blk.mean()
    return out

M = np.array([[1, 2, 5, 3],
              [4, 9, 2, 1],
              [7, 3, 8, 6],
              [2, 1, 4, 5]], float)
print("entree  :\n", M)
print("\nmax pooling 2x2 :\n", pool(M, "max"))
print("\nmoyenne 2x2 :\n", pool(M, "avg"))""")
, ("md", r"""## 5. À vous

**Question 1.** Quelle combinaison $(K, P, S)$ donne une sortie de même taille que l'entrée
$32 \times 32$ avec un noyau $5 \times 5$ ?

**Question 2.** Pourquoi le max pooling rend-il la détection moins sensible à un léger
décalage du motif ? Vérifiez en décalant d'un pixel la matrice $M$ ci-dessus.

**Question 3.** Combien de paramètres apprend une couche de pooling 3×3 ?""")
, ("code", r"""# ---- À VOUS ----
# 1) cherchez (K, P, S) tel que dim(32, K, P, S) == 32
# 2)

# 3) reponse :""")
, ("md", r"""## 6. Synthèse

| Réglage | Effet |
|---|---|
| **Padding** $P$ | contrôle la taille de sortie ; `same` la conserve |
| **Stride** $S$ | réduit la résolution d'un facteur $S$ |
| **Pooling** | réduit la résolution, **sans paramètre appris** |
| **Formule** | $N_{out} = \lfloor (N_{in} - K + 2P)/S \rfloor + 1$ |""")
]
execute(nb("TP2 — Padding, stride et pooling", "1 h 30", "10 points",
           r"""- Maîtriser la formule de dimension d'une couche convolutive.
- Visualiser l'effet du padding et du stride.
- Implémenter le max pooling et la moyenne, et comprendre leur rôle.""",
           r"""1. Répondre aux 3 questions de la section « À vous ».
2. Rendre le notebook exécuté.""", c2), "TP2_padding_stride_pooling.ipynb")

# ===================================================================== TP5
c5 = [
 ("md", r"""## 1. Un CNN ne se limite pas aux images

Un signal 1D — une mesure qui évolue dans le temps — se traite avec exactement les mêmes
principes : un noyau qui balaye, une activation, un pooling. On travaille ici sur un signal
de type **électrocardiogramme** synthétique.""")
, ("code", r"""def ecg(n=1200, bpm=72, seed=4112):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, n)
    sig = np.zeros(n)
    for k in range(int(10 * bpm / 60)):
        c = k * 60 / bpm
        idx = np.argmin(np.abs(t - c))
        for dt, amp in [(-.05, .08), (0, -0.12), (0.012, 1.0), (0.03, -0.22), (0.16, 0.22)]:
            j = idx + int(dt * n / 10)
            if 0 <= j < n:
                w = max(2, int(.008 * n))
                lo, hi = max(0, j - w), j + w
                sig[lo:hi] += amp * np.exp(-((np.arange(lo, hi) - j) ** 2) / (2 * (.004 * n) ** 2))
    return t, sig + rng.normal(0, .012, n)

t, s = ecg()
print("duree :", t[-1], "s   |   points :", len(s), "  |   frequence simulee : 72 bpm")

plt.figure(figsize=(11, 3))
plt.plot(t, s, lw=.9)
plt.xlabel("temps (s)"); plt.ylabel("amplitude (mV)")
plt.title("Signal ECG synthetique")
plt.show()""")
, ("md", r"""## 2. Un noyau 1D

Un noyau 1D est un simple vecteur. Appliqué par balayage, il produit un nouveau signal.
Voici quatre noyaux aux rôles bien distincts.""")
, ("code", r"""noyaux1d = {
    "lissage (Hann 11)":   np.hanning(11) / np.hanning(11).sum(),
    "derivee (detecteur de fronts)": np.array([-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1], float),
    "moyenne (5 points)":  np.ones(5) / 5,
    "detecteur de pics":   np.array([-1, -2, 6, -2, -1], float),
}
fig, axes = plt.subplots(len(noyaux1d) + 1, 1, figsize=(11, 9), sharex=True)
axes[0].plot(t, s, color="grey", lw=.9); axes[0].set_ylabel("brut")
for ax, (nom, k) in zip(axes[1:], noyaux1d.items()):
    ax.plot(t, np.convolve(s, k, mode="same"), lw=1.3)
    ax.set_ylabel(nom, fontsize=8)
plt.suptitle("Effet des noyaux 1D sur le signal ECG", fontweight="bold")
plt.tight_layout()
plt.show()""")
, ("md", r"""## 3. Détecter les battements

Un détecteur de pics bien calibré, suivi d'un seuil, localise les complexes QRS.
C'est exactement ce que fait une couche convolutive apprise : elle découvre *seule*
le noyau qui fait ressortir le motif.""")
, ("code", r"""noyau = np.exp(-np.linspace(-1, 1, 41) ** 2 * 6)
noyau /= noyau.sum()
activation = np.convolve(np.maximum(s, 0) ** 2, noyau, mode="same")
seuil = activation.mean() + 2.2 * activation.std()
pics = (activation > seuil) & (np.roll(activation, 1) <= seuil)

print("seuil      :", round(float(seuil), 4))
print("battements :", int(pics.sum()), "detectes  (attendu ≈ 12 en 10 s a 72 bpm)")

plt.figure(figsize=(11, 3))
plt.plot(t, activation, lw=1.2, label="activation du filtre 1D")
plt.axhline(seuil, color="red", ls="--", lw=1.2, label="seuil")
plt.plot(t[pics], activation[pics], "v", ms=8, label="battements")
plt.legend(); plt.xlabel("temps (s)")
plt.title("Detection des complexes QRS")
plt.show()""")
, ("md", r"""## 4. À vous

**Question 1.** Un noyau 1D de longueur 21 appliqué à un signal de 600 points : combien de
points compte la sortie en mode `valid` ?

**Question 2.** Remplacez le noyau de détection par une simple moyenne mobile de 41 points.
Les pics sont-ils encore détectables ?

**Question 3.** Quelle est la différence essentielle entre un noyau 1D et un noyau 2D,
du point de vue du nombre de paramètres pour une même longueur $K$ ?""")
, ("code", r"""# ---- À VOUS ----
# 1) calcul : ...
# 2)

# 3) reponse :""")
, ("md", r"""## 5. Synthèse

| Notion | Images (2D) | Séries temporelles (1D) |
|---|---|---|
| Noyau | petite matrice | petit vecteur |
| Balayage | 2 directions | 1 direction |
| Paramètres d'un noyau | $K^2$ | $K$ |
| Champ récepteur | surface | intervalle de temps |
| Applications | classification, détection | ECG, trafic, météo, capteurs |

**Suite :** le TP3 construit un vrai CNN entraîné sur des images.""")
]
execute(nb("TP5 — CNN 1D sur séries temporelles", "1 h 30", "10 points",
           r"""- Transposer les principes de convolution à un signal 1D.
- Construire un détecteur de battements sur un ECG.
- Mesurer l'effet de différents noyaux 1D.""",
           r"""1. Répondre aux 3 questions de la section « À vous ».
2. Rendre le notebook exécuté.""", c5), "TP5_CNN_1D_series_temporelles.ipynb")

print("\nTP rapides generes.")
