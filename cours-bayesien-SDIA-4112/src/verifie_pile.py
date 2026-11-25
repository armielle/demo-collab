"""Detecte les fermetures orphelines et les environnements laisses ouverts, frame par frame."""
import glob, os, re, sys

PARTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "slides", "parts")
JEUX = ["columns", "column", "itemize", "enumerate", "tabular", "kbox", "ebox", "wbox", "abox", "qbox",
        "solbox", "codebox", "figbox", "tcolorbox", "center", "minipage", "tikzpicture", "description"]
motif = sys.argv[1] if len(sys.argv) > 1 else "ch*.tex"
anomalies = 0
for f in sorted(glob.glob(os.path.join(PARTS, motif))):
    src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    i = 0
    while True:
        a = src.find("\\begin{frame}", i)
        if a < 0:
            break
        b = src.find("\\end{frame}", a)
        if b < 0:
            break
        corps = src[a:b]
        ligne = src[:a].count("\n") + 1
        corps_n = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", corps, flags=re.S)
        pile = []
        for m in re.finditer(r"\\(begin|end)\{(\w+)\}", corps_n):
            geste, nom = m.group(1), m.group(2)
            if nom not in JEUX:
                continue
            if geste == "begin":
                pile.append(nom)
            else:
                if not pile or pile[-1] != nom:
                    pos = corps_n[:m.start()].count("\n") + 1
                    print(f"  {os.path.basename(f)} frame l.{ligne} : \\end{{{nom}}} orphelin (l.{pos} de la frame, pile={pile})")
                    anomalies += 1
                else:
                    pile.pop()
        if pile:
            print(f"  {os.path.basename(f)} frame l.{ligne} : non fermes = {pile}")
            anomalies += 1
        i = b + 1
print(f"\n{anomalies} anomalie(s)")
