"""Localise la frame ouverte dans ch1_f : compile ch1_a..ch1_e + les N premieres lignes de ch1_f."""
import os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SL = os.path.join(ROOT, "slides")
TEC = os.path.expanduser("~/texbin/tectonic.exe")
fich = sys.argv[1] if len(sys.argv) > 1 else "ch1_f"
L = open(os.path.join(SL, "parts", fich + ".tex"), encoding="utf-8", newline="").read().replace("\r\n", "\n").split("\n")
print(f"{fich}.tex : {len(L)} lignes")

for n in list(range(30, len(L) + 1, 30)) + [len(L)]:
    extrait = "\n".join(L[:n])
    # on ne garde que des frames completes : on s'arrete a la derniere \end{frame}
    k = extrait.rfind("\\end{frame}")
    if k < 0:
        continue
    extrait = extrait[:k + len("\\end{frame}")]
    tex = (f"% bisect\n\\input{{preamble.tex}}\n\\title[Ch 1]{{Introduction}}\n\\subtitle{{EPD-SDIA-4112}}\n"
           f"\\author{{Test}}\n\\institute{{DGI}}\n\\date{{2026}}\n\\begin{{document}}\n"
           f"\\input{{parts/ch1_a}}\\input{{parts/ch1_b}}\\input{{parts/ch1_c}}\\input{{parts/ch1_d}}\\input{{parts/ch1_e}}\n"
           f"{extrait}\n\\end{{document}}\n")
    open(os.path.join(SL, "_bis.tex"), "w", encoding="utf-8").write(tex)
    r = subprocess.run([TEC, "-X", "compile", "_bis.tex", "--outdir", os.path.join(ROOT, "out")],
                       cwd=SL, capture_output=True, text=True, encoding="utf-8", errors="replace")
    brut = (r.stdout + r.stderr).splitlines()
    err = [l.strip() for l in brut if ("error" in l.lower() and "fontconfig" not in l.lower()
                                       and "Cannot load default" not in l) or l.strip().startswith("!")]
    ouvert = L[n - 1].strip()[:70] if n - 1 < len(L) else ""
    print(f"  {n:>4} lignes  {'OK   ' if not err else 'ECHEC'}  {ouvert}")
