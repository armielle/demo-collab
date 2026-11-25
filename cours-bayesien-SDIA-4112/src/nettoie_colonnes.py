"""Supprime les fermetures de colonne prématurées ou orphelines.

Deux cas observes apres les scissions automatiques :
  - un \\end{columns} rencontre alors que la pile est ['columns'] (donc aucune colonne
    ouverte) ALORS qu'une autre \\begin{column} suit dans la meme frame : fermeture prematuree ;
  - un \\end{column} sans \\begin{column} ouvert : orphelin.
Les deux sont supprimes.
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
        pile, a_supprimer = [], []
        for m in re.finditer(r"\\(begin|end)\{(\w+)\}", net):
            g, nom = m.group(1), m.group(2)
            if nom not in ("column", "columns"):
                continue
            suite = net[m.end():]
            if g == "begin":
                pile.append(nom)
            elif nom == "columns":
                if pile and pile[-1] == "column":
                    pile.pop(); pile.pop() if pile and pile[-1] == "columns" else None
                elif pile and pile[-1] == "columns" and "\\begin{column}" in suite:
                    a_supprimer.append((m.start(), m.end()))          # fermeture prematuree
                    pile.pop()
                elif pile and pile[-1] == "columns":
                    pile.pop()
            else:  # end column
                if pile and pile[-1] == "column":
                    pile.pop()
                else:
                    a_supprimer.append((m.start(), m.end()))          # orphelin
        if a_supprimer:
            neuf = corps
            for d, fin in sorted(a_supprimer, reverse=True):
                neuf = neuf[:d] + neuf[fin:]
            neuf = re.sub(r"\n{3,}", "\n\n", neuf)
            out[k] = neuf + bloc[bloc.rfind("\\end{frame}"):]
            modifie = True
            total += len(a_supprimer)
            print(f"  {os.path.basename(f)} : {len(a_supprimer)} fermeture(s) supprimee(s)")
    if modifie:
        open(f, "w", encoding="utf-8", newline="").write("".join(out).replace("\n", "\r\n"))
print(f"\n{total} fermeture(s) supprimee(s)")
