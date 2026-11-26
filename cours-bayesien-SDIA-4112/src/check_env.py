"""Verificateur de coherence des fichiers de chapitre (avant compilation).

Controle : (1) environnements LaTeX equilibres, (2) toute frame contenant
lstlisting est declaree [fragile], (3) pas d'emoji, (4) figures referencees
presentes sur le disque, (5) compteur de frames.
"""
import os, re, sys, glob
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")
FIG = os.path.join(ROOT, "figures")

ENVS = ["frame", "columns", "column", "itemize", "enumerate", "center", "tabular",
        "tcolorbox", "codebox", "qbox", "abox", "ebox", "kbox", "wbox", "solbox",
        "figbox", "tikzpicture", "axis", "equation", "align", "aligned", "lstlisting",
        "tcolorbox", "multic", "grid"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def check(path):
    src = open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    problems = []
    for e in set(ENVS):
        b = len(re.findall(r"\\begin\{" + e + r"\}", src))
        n = len(re.findall(r"\\end\{" + e + r"\}", src))
        if b != n:
            problems.append(f"environnement INEQUILIBRE {e}: begin={b} end={n}")
    nframes = len(re.findall(r"\\begin\{frame\}", src))
    # frames avec lstlisting
    for m in re.finditer(r"\\begin\{frame\}(?!\[fragile)([^\n]*?)\n(.*?)\\end\{frame\}",
                         src, re.S):
        if "lstlisting" in m.group(2):
            problems.append("frame avec lstlisting SANS [fragile]")
    frag = len(re.findall(r"\\begin\{frame\}\[fragile\]", src))
    bad_frag = 0
    for m in re.finditer(r"\\begin\{frame\}\[fragile\](.*?)\\end\{frame\}", src, re.S):
        if "lstlisting" not in m.group(1):
            bad_frag += 1
    if bad_frag:
        problems.append(f"frame [fragile] inutile (sans lstlisting): {bad_frag}")
    emo = EMOJI.findall(src)
    if emo:
        problems.append(f"emoji detectes: {''.join(sorted(set(emo)))[:40]}")
    figs = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", src))
    missing = [f for f in figs if not os.path.exists(os.path.join(FIG, f))]
    if missing:
        problems.append("figures MANQUANTES: " + ", ".join(sorted(missing)))
    ok = not problems
    print(f"[{'OK ' if ok else 'KO '}] {name:<22} frames={nframes:>3} fragile={frag:>2} "
          f"figures={len(figs):>2} caracteres={len(src):>6}")
    for p in problems:
        print("       -", p)
    return ok


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob(os.path.join(PARTS, "ch*.tex")))
    files = args if args else sorted(glob.glob(os.path.join(PARTS, "ch*.tex")))
    tot_ok = True
    per_ch = Counter()
    for f in files:
        if not os.path.exists(f):
            f = os.path.join(PARTS, f)
        tot_ok &= check(f)
        ch = re.match(r"ch(\d)", os.path.basename(f))
        if ch:
            per_ch[ch.group(1)] += len(re.findall(r"\\begin\{frame\}",
                                                  open(f, encoding="utf-8").read()))
    print("\nFrames par chapitre (hors page de garde \\chapterslide) :",
          dict(sorted(per_ch.items())))
    print("RESULTAT :", "TOUT EST COHERENT" if tot_ok else "PROBLEMES A CORRIGER")
    sys.exit(0 if tot_ok else 1)
