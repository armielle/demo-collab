"""Equilibre des ACCOLADES dans chaque frame (en ignorant verbatim/lstlisting et commentaires)."""
import glob, os, re, sys

PARTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "slides", "parts")
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
        # retirer les corps verbatim
        corps = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", corps, flags=re.S)
        corps = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", corps, flags=re.S)
        net = []
        for l in corps.split("\n"):
            l = re.sub(r"(?<!\\)%.*$", "", l)          # commentaires
            l = l.replace(r"\{", "").replace(r"\}", "")
            net.append(l)
        net = "\n".join(net)
        o, c = net.count("{"), net.count("}")
        if o != c:
            anomalies += 1
            t = corps.split("\n")[0][:80]
            print(f"  {os.path.basename(f)}:{ligne:<5} accolades {o}/{c}   {t}")
        i = b + 1
print(f"\n{anomalies} frame(s) a accolades desequilibrees")
