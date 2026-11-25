"""Donnees reelles pour la video CNN : cartes de convolutions, pooling, sortie dense.

Tout est calcule ici (numpy/scipy, graine 4112) et sauvegarde en PNG + JSON.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

ICI = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ICI, "assets")
os.makedirs(A, exist_ok=True)
GR = np.random.default_rng(4112)
K = {}

# ---------------------------------------------------------------- image source
img = np.load(os.path.join(ICI, "..", "data", "face_gray.npy")).astype(float)
H, W = img.shape
# decoupe carree centree
cote = min(H, W) // 2
y0, x0 = (H - cote) // 2, (W - cote) // 2
crop = img[y0:y0 + cote, x0:x0 + cote]
crop = crop[::2, ::2]                     # 384 x 384
K["image_source"] = list(crop.shape)
plt.imsave(os.path.join(A, "entree.png"), crop, cmap="gray")

# ---------------------------------------------------------------- couche 1 : contours
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)
c1 = np.hypot(signal.convolve2d(crop, sobelx, mode="same", boundary="symm"),
              signal.convolve2d(crop, sobely, mode="same", boundary="symm"))
c1 /= c1.max()
plt.imsave(os.path.join(A, "carte_contours.png"), c1, cmap="magma")
K["c1_extrema"] = [round(float(c1.min()), 3), round(float(c1.max()), 3)]

# relu explicite (les valeurs negatives n'existent pas ici, on le dit dans la video)
c1r = np.clip(c1, 0, None)

# ---------------------------------------------------------------- pooling 2x2
def pool(m, k=2):
    h, w = m.shape
    h2, w2 = h // k, w // k
    return m[:h2 * k, :w2 * k].reshape(h2, k, w2, k).max(axis=(1, 3))

c1p = pool(c1r)
plt.imsave(os.path.join(A, "carte_pool.png"), c1p, cmap="magma")
K["pooling_dim"] = [list(c1.shape), list(c1p.shape)]

# ---------------------------------------------------------------- couche 2 : textures
blur = np.ones((7, 7)) / 49
lisse = signal.convolve2d(crop, blur, mode="same", boundary="symm")
c2 = np.abs(signal.convolve2d(lisse, sobelx, mode="same", boundary="symm"))
c2 = pool(c2)
c2 /= (c2.max() + 1e-9)
plt.imsave(os.path.join(A, "carte_textures.png"), c2, cmap="magma")

# ---------------------------------------------------------------- couche 3 : parties
grand = np.ones((25, 25)) / 625
obj = signal.convolve2d(c1r, grand, mode="same", boundary="symm")
obj = pool(pool(obj))
obj /= (obj.max() + 1e-9)
plt.imsave(os.path.join(A, "carte_parties.png"), obj, cmap="magma")
K["hierarchie_dims"] = [list(c1.shape), list(c2.shape), list(obj.shape)]

# ---------------------------------------------------------------- petite grille 12x12
pet = crop[::16, ::16]                     # 12 x 12
pet = (pet - pet.min()) / (pet.max() - pet.min())
np.save(os.path.join(A, "grille12.npy"), pet)
K["grille12_shape"] = list(pet.shape)
K["grille12_coin"] = pet[:3, :3].round(2).tolist()
K["noyau_sobel"] = sobelx.tolist()

# convolution pas a pas sur la petite grille
def corr2d(m, k):
    kh, kw = k.shape
    o = np.zeros((m.shape[0] - kh + 1, m.shape[1] - kw + 1))
    for i in range(o.shape[0]):
        for j in range(o.shape[1]):
            o[i, j] = (m[i:i + kh, j:j + kw] * k).sum()
    return o

noyau = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], float)
sort = corr2d(pet, noyau)
np.save(os.path.join(A, "sortie10.npy"), sort)
K["sortie10_shape"] = list(sort.shape)
K["premier_calcul"] = {
    "zone": pet[:3, :3].round(2).tolist(),
    "produits": (pet[:3, :3] * noyau).round(2).tolist(),
    "somme": round(float((pet[:3, :3] * noyau).sum()), 2),
}
K["relu_negatifs"] = int((sort < 0).sum())
K["relu_total"] = int(sort.size)

# pooling numerique 10 -> 5
p5 = sort[:10, :10].reshape(5, 2, 5, 2).max(axis=(1, 3))
np.save(os.path.join(A, "pool5.npy"), p5)
K["pool5_shape"] = list(p5.shape)

# ---------------------------------------------------------------- flatten + dense
final = pool(p5)                            # 2 x 2 -> trop petit ; on prend 4x4
final = sort[1:9, 1:9].reshape(4, 2, 4, 2).max(axis=(1, 3))    # 4x4
vec = final.ravel()                          # 16 valeurs
K["flatten_taille"] = int(vec.size)
K["flatten_valeurs"] = vec.round(2).tolist()

# petit reseau dense 16 -> 8 -> 3 (poids de demonstration, graine 4112)
W1 = GR.normal(0, .7, (16, 8)); b1 = GR.normal(0, .1, 8)
W2 = GR.normal(0, .7, (8, 3));  b2 = GR.normal(0, .1, 3)
h1 = np.tanh(vec @ W1 + b1)
logits = h1 @ W2 + b2
e = np.exp(logits - logits.max())
soft = e / e.sum()
K["dense_hidden"] = h1.round(3).tolist()
K["logits"] = logits.round(3).tolist()
K["softmax"] = soft.round(4).tolist()
K["argmax"] = int(np.argmax(soft))

# petit reseau dense visualisable : 16 -> 8 -> 3
np.save(os.path.join(A, "vec16.npy"), vec)
np.save(os.path.join(A, "h8.npy"), h1)

# ---------------------------------------------------------------- ratios
K["param_mlp_48M"] = 48_000_000 * 1000
K["param_conv_1792"] = 3 * 3 * 3 * 64 + 64

with open(os.path.join(A, "donnees.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(f"[key] donnees.json -- {len(K)} cles")
for f in sorted(os.listdir(A)):
    print(f"   {f:<26} {os.path.getsize(os.path.join(A, f))/1024:7.1f} Ko")
