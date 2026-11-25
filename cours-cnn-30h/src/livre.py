"""Dossier de livraison + archive ZIP du cours CNN."""
import os, shutil, glob, zipfile, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
HOME = os.path.expanduser("~")
D = os.path.join(HOME, "Downloads", "Cours-CNN-30h")
ZIP = D + ".zip"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


shutil.rmtree(D, ignore_errors=True)
os.makedirs(os.path.join(D, "TP"), exist_ok=True)
copies = []

# ---- supports
for nom in ["Cours-CNN-complet.pdf", "Cours-CNN-complet-leger.pdf"]:
    src = os.path.join(OUT, nom)
    if os.path.exists(src):
        dst = os.path.join(D, nom)
        shutil.copy2(src, dst)
        copies.append((src, dst))

# ---- chapitres, tries
i = 1
for f in sorted(glob.glob(os.path.join(OUT, "Chapitre-*.pdf"))):
    dst = os.path.join(D, f"{i:02d}-" + os.path.basename(f))
    shutil.copy2(f, dst)
    copies.append((f, dst))
    i += 1

# ---- notebooks -> PDF de TP et notebooks
for f in sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb"))):
    dst = os.path.join(D, "TP", os.path.basename(f))
    shutil.copy2(f, dst)
    copies.append((f, dst))
for f in sorted(glob.glob(os.path.join(OUT, "TP*.pdf"))) + \
        sorted(glob.glob(os.path.join(OUT, "Projet*.pdf"))):
    if os.path.exists(f):
        dst = os.path.join(D, "TP", os.path.basename(f))
        shutil.copy2(f, dst)
        copies.append((f, dst))

# ---- documentation
for f in ["README.md", "PLAN.md", "RESSOURCES.md"]:
    src = os.path.join(ROOT, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(D, f))
        copies.append((src, os.path.join(D, f)))

# ---- copies directes du maitre
for cible in [os.path.join(HOME, "Downloads", "Cours-CNN-complet.pdf"),
              os.path.join(HOME, "Downloads", "Cours-CNN-complet-leger.pdf"),
              os.path.join(HOME, "Desktop", "Cours-CNN-complet.pdf"),
              os.path.join(HOME, "Desktop", "Cours-CNN-complet-leger.pdf")]:
    src = os.path.join(OUT, os.path.basename(cible))
    if os.path.exists(src):
        try:
            shutil.copy2(src, cible)
            copies.append((src, cible))
        except OSError as e:
            print("[verrouille]", os.path.basename(cible), e)

# ---- archive
n = 0
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in sorted(os.listdir(D)):
        p = os.path.join(D, f)
        if os.path.isfile(p):
            z.write(p, os.path.join("Cours-CNN-30h", f)); n += 1
        elif os.path.isdir(p):
            for r, _, fs in os.walk(p):
                for g in fs:
                    z.write(os.path.join(r, g), os.path.relpath(os.path.join(r, g), HOME))
                    n += 1

print(f"dossier : {D}")
h_ref = sha(os.path.join(OUT, "Cours-CNN-complet.pdf"))
h_leg = sha(os.path.join(OUT, "Cours-CNN-complet-leger.pdf"))
print(f"master {h_ref} | leger {h_leg}")
for src, dst in copies:
    if "Cours-CNN-complet" in os.path.basename(dst):
        h = sha(dst)
        etat = "master OK" if h == h_ref else ("leger OK " if h == h_leg else "PERIME   ")
        print(f"  {etat}  {dst.replace(HOME, '~')}")
print(f"archive : {ZIP}  {os.path.getsize(ZIP)/1e6:.1f} Mo")
