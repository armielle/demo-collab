"""Fabrique et execute reellement les notebooks de TP (.ipynb).

Chaque notebook est decrit par une liste de cellules, execute avec le noyau
virtuel du projet, puis sauvegarde avec ses sorties reelles.
"""
import os, sys, json
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = os.path.join(ROOT, "notebooks")
os.makedirs(NB, exist_ok=True)
KERNEL = "sdia-bayes"

PREAMBLE = """%matplotlib inline
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
RNG = np.random.default_rng(4112)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
print("numpy", np.__version__, "| scipy", stats.__name__)
"""


def build(name, cells, title):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n"
                                         "_Cours EPD-SDIA-4112 — Statistique Bayésienne — "
                                         "Master 1 SDIA. Exécutez les cellules dans l'ordre._")]
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                        else nbf.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": "SDIA Bayes",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def execute(nb, name, timeout=900):
    client = NotebookClient(nb, timeout=timeout, kernel_name=KERNEL,
                            resources={"metadata": {"path": NB}},
                            allow_errors=False)
    client.execute()
    path = os.path.join(NB, name)
    nbf.write(nb, path)
    imgs = sum(1 for c in nb.cells for o in c.get("outputs", [])
               if "data" in o and "image/png" in o.get("data", {}))
    ncode = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"[nb] {name:<40} {len(nb.cells):>3} cellules ({ncode} code, {imgs} figures) "
          f"{os.path.getsize(path)/1024:7.1f} Ko")
    return path
