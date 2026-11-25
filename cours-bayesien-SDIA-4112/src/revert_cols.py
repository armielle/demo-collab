"""Annule les scissions de colonnes (marqueur `% [cols]`) : elles n'ont rien corrige.

Deux frames issues de split_frames3.py sont recollées en une seule, en reconstruisant
l'environnement `columns` d'origine (les colonnes des deux groupes, dans l'ordre).
"""
import os, re, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")
MARK = "% [cols]"

PAT = re.compile(
    re.escape(MARK) + r"\n(\\begin\{frame\}(?:\[[^\]]*\])?\{([^\n]*)\})\n(.*?)\\end\{frame\}\n\n"
    + re.escape(MARK) + r"\n\\begin\{frame\}\{([^\n]*)\}\n(.*?)\\end\{frame\}",
    re.S)

total = 0
for f in sorted(glob.glob(os.path.join(PARTS, "ch*.tex"))):
    src = open(f, encoding="utf-8", newline="").read()
    m = PAT.search(src)
    n = 0
    while m:
        entete, titre, corps1, titre2, corps2 = m.groups()

        def cols_de(txt):
            out, i = [], 0
            while True:
                a = txt.find(r"\begin{column}", i)
                if a < 0:
                    return out
                prof, k = 1, a + len(r"\begin{column}")
                while prof:
                    o = txt.find(r"\begin{column}", k)
                    e = txt.find(r"\end{column}", k)
                    if e < 0:
                        break
                    if 0 <= o < e:
                        prof += 1
                        k = o + len(r"\begin{column}")
                    else:
                        prof -= 1
                        k = e + len(r"\end{column}")
                out.append(txt[a:k])
                i = k

        c1, c2 = cols_de(corps1), cols_de(corps2)
        avant = corps1[:corps1.index(r"\begin{columns}")].strip()
        apres = corps2[corps2.rindex(r"\end{columns}") + len(r"\end{columns}"):].strip() \
            if r"\end{columns}" in corps2 else ""
        fusion = ("\n".join(x for x in [avant,
                                        "\\begin{columns}[t]\n" + "\n".join(c1 + c2) + "\n\\end{columns}",
                                        apres] if x))
        src = src[:m.start()] + entete + "\n" + fusion + "\n\\end{frame}" + src[m.end():]
        n += 1
        total += 1
        m = PAT.search(src)
    if n:
        open(f, "w", encoding="utf-8", newline="").write(src)
        print(f"  {os.path.basename(f):<14} {n} scission(s) de colonnes annulee(s)")
print(f"\n{total} frame(s) recollée(s)")
