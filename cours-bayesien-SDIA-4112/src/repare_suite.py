"""Repare les scissions ou le bloc deplace a emporte la fermeture de sa colonne.

Etat fautif produit par split_frames4.py :
    frame A :  ... \\end{column}\\end{columns}          (colonne 2 supprimee, vide)
    suite   :  <bloc deplace> \\end{column} \\begin{column}{..}<reste>   (sans \\end{columns})

Reparation :
    frame A :  ... \\end{column} \\begin{column}{..}<reste> \\end{columns}
    suite   :  <bloc deplace>
"""
import glob, os, re

PARTS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "slides", "parts")
BLOCS = ("columns", "tcolorbox", "kbox", "abox", "ebox", "wbox", "qbox", "solbox", "codebox",
         "itemize", "enumerate", "tabular", "center", "figbox", "minipage", "tikzpicture")


def premier_bloc(txt):
    """Fin du premier bloc de premier niveau (celui qui ouvre le texte) ou None."""
    m = re.match(r"\s*\\begin\{(\w+)\}", txt)
    if not m:
        return None
    nom = m.group(1)
    k = m.end()
    prof = 1
    while prof:
        o = txt.find(r"\begin{" + nom + "}", k)
        c = txt.find(r"\end{" + nom + "}", k)
        if c < 0:
            return None
        if o >= 0 and o < c:
            prof += 1
            k = o + len(nom) + 8
        else:
            prof -= 1
            k = c + len(nom) + 6
    return k


total = 0
for f in sorted(glob.glob(os.path.join(PARTS, "ch*.tex"))):
    src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    out, i = [], 0
    while True:
        a = src.find("\\begin{frame}", i)
        if a < 0:
            out.append(src[i:]); break
        b = src.find("\\end{frame}", a) + len("\\end{frame}")
        out.append(src[i:a]); out.append(src[a:b]); i = b
    modifie = False
    frames = [k for k, x in enumerate(out) if x.startswith("\\begin{frame}")]
    for j in range(len(frames) - 1):
        k = frames[j + 1]
        paire_ok = "(suite)}" in out[k].split("\n")[0]
        if not paire_ok:
            continue
        prec = out[frames[j]]
        if " (suite)" in prec.split("\n")[0]:
            continue
        entete_a = prec.split("\n")[0]
        corps_a = prec[prec.find("\n") + 1:prec.rfind("\\end{frame}")]
        entete_s = out[k].split("\n")[0]
        corps_s = out[k][out[k].find("\n") + 1:out[k].rfind("\\end{frame}")]
        if "\\end{column}" not in corps_s or "\\end{columns}" in corps_s:
            continue
        fin_bloc = premier_bloc(corps_s.strip())
        if not fin_bloc:
            continue
        deplace = corps_s.strip()[:fin_bloc].strip()
        reste = corps_s.strip()[fin_bloc:].strip()
        if not reste.startswith("\\end{column}"):
            continue
        reste = reste[len("\\end{column}"):].lstrip()
        if "\\end{columns}" in corps_a:
            corps_a = corps_a[:corps_a.rfind("\\end{columns}")].rstrip()
        out[frames[j]] = entete_a + "\n" + corps_a + "\n" + reste + "\n\\end{columns}\n\\end{frame}"
        out[k] = entete_s + "\n" + deplace + "\n\\end{frame}"
        modifie = True
        total += 1
        print(f"  {os.path.basename(f)}  paire reparee : bloc deplace = {deplace.splitlines()[0][:60]}")
    if modifie:
        open(f, "w", encoding="utf-8", newline="").write("".join(out).replace("\n", "\r\n"))
print(f"\n{total} paire(s) reparee(s)")
