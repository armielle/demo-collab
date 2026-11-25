"""Ferme les colonnes restees ouvertes avant un \\end{columns} (insertion chirurgicale).

Pour chaque frame, on suit la pile : si un \\end{columns} arrive alors que la pile est
['columns','column'], on insere \\end{column} juste avant.
"""
import glob, os, re, sys

PARTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "slides", "parts")
JEUX = ["columns", "column", "itemize", "enumerate", "tabular", "kbox", "ebox", "wbox", "abox", "qbox",
        "solbox", "codebox", "figbox", "tcolorbox", "center", "minipage", "tikzpicture", "description"]
motif = sys.argv[1] if len(sys.argv) > 1 else "ch*.tex"
total = 0
for f in sorted(glob.glob(os.path.join(PARTS, motif))):
    src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    out, i = [], 0
    while True:
        a = src.find("\\begin{frame}", i)
        if a < 0:
            out.append(src[i:]); break
        b = src.find("\\end{frame}", a) + len("\\end{frame}")
        out.append(src[i:a]); out.append(src[a:b]); i = b
    modifie = False
    for k, bloc in enumerate(out):
        if not bloc.startswith("\\begin{frame}"):
            continue
        corps = bloc[:bloc.rfind("\\end{frame}")]
        net = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", corps, flags=re.S)
        pile, insertions = [], []
        for m in re.finditer(r"\\(begin|end)\{(\w+)\}", net):
            g, nom = m.group(1), m.group(2)
            if nom not in JEUX:
                continue
            if g == "begin":
                pile.append(nom)
            elif nom == "columns" and pile and pile[-1] == "column":
                insertions.append(m.start())      # position dans `net` = position dans `corps`
                pile.pop(); pile.pop() if pile and pile[-1] == "columns" else None
            elif pile and pile[-1] == nom:
                pile.pop()
        if insertions:
            neuf = corps
            for p in reversed(insertions):
                neuf = neuf[:p] + "\\end{column}\n" + neuf[p:]
            out[k] = neuf + bloc[bloc.rfind("\\end{frame}"):]
            modifie = True
            total += len(insertions)
            print(f"  {os.path.basename(f)} frame l.{src[:src.find(bloc)].count(chr(10))+1 if bloc in src else '?'} : {len(insertions)} \\end{{column}} insere(s)")
    if modifie:
        open(f, "w", encoding="utf-8", newline="").write("".join(out).replace("\n", "\r\n"))
print(f"\n{total} insertion(s)")
