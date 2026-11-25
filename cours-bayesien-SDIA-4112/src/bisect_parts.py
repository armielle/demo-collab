"""Compile le cours chapitre 1 en ajoutant les fichiers un a un, pour localiser la frame ouverte."""
import os, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SL = os.path.join(ROOT, "slides")
TEC = os.path.expanduser("~/texbin/tectonic.exe")
lettres = list("abcdef")

for k in range(1, len(lettres) + 1):
    corps = "\n".join(f"\\input{{parts/ch1_{l}}}" for l in lettres[:k])
    tex = (f"% bisect\n\\input{{preamble.tex}}\n\\title[Ch 1]{{Introduction}}\n"
           f"\\subtitle{{EPD-SDIA-4112}}\n\\author{{Test}}\n\\institute{{DGI}}\n\\date{{2026}}\n"
           f"\\begin{{document}}\n{corps}\n\\end{{document}}\n")
    f = os.path.join(SL, "_bis.tex")
    open(f, "w", encoding="utf-8").write(tex)
    r = subprocess.run([TEC, "-X", "compile", "_bis.tex", "--outdir", os.path.join(ROOT, "out")],
                       cwd=SL, capture_output=True, text=True, encoding="utf-8", errors="replace")
    brut = (r.stdout + r.stderr).splitlines()
    erreurs = [l.strip() for l in brut
               if ("error" in l.lower() and "fontconfig" not in l.lower() and "Cannot load default" not in l)
               or l.strip().startswith("!")]
    etat = "OK   " if not erreurs else "ECHEC"
    print(f"  ch1_a..ch1_{lettres[k-1]}   {etat}  {erreurs[0][:110] if erreurs else ''}")
    if erreurs:
        break
