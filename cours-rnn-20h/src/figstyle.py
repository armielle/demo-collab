"""Style commun aux figures du cours CNN -- 30 h -- Master 1 SDIA."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)

C = dict(
    deep="#10243E", blue="#1F6FEB", cyan="#12A594", orange="#D97706",
    red="#C0392B", green="#1E8449", purple="#6C5CE7", grey="#5B6470",
    light="#F4F6F8", pink="#D6336C", teal="#0B7285",
)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.edgecolor": C["grey"],
    "axes.labelcolor": C["deep"], "text.color": C["deep"],
    "xtick.color": C["grey"], "ytick.color": C["grey"],
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "grid.color": C["grey"], "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.framealpha": .92, "legend.edgecolor": "#DDDDDD",
    "legend.fontsize": 9.5,
})


def save(fig, name, transparent=False):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, transparent=transparent)
    plt.close(fig)
    print(f"[fig] {name:<38} {os.path.getsize(path)/1024:7.1f} Ko")
    return path


def annot(ax, x, y, txt, color=None, **kw):
    kw.setdefault("fontsize", 9)
    kw.setdefault("fontweight", "bold")
    ax.annotate(txt, (x, y), color=color or C["deep"], **kw)
