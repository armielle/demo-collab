# -*- coding: utf-8 -*-
"""Bissection des frames de git1_d.tex : trouve la frame fautive."""
import re, subprocess, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SL = os.path.join(ROOT, "slides")
src = open(os.path.join(SL, "parts", "git1_d.tex"), encoding="utf-8").read()
frames = re.split(r"(?=\\begin\{frame\})", src)
frames = [f for f in frames if f.strip()]
print("nb morceaux:", len(frames), flush=True)

TECT = os.path.expanduser("~/texbin/tectonic.exe")
OUT = os.path.join(ROOT, "out")
OK = None
for n in range(1, len(frames) + 1):
    body = "".join(frames[:n])
    with open(os.path.join(SL, "_bis_d.tex"), "w", encoding="utf-8") as f:
        f.write("% bisect\n\\input{preamble.tex}\n\\title[t]{t}\\author{a}\\institute{i}\\date{d}\n"
                "\\begin{document}\n" + body + "\n\\end{document}\n")
    pdf = os.path.join(OUT, "_bis_d.pdf")
    if os.path.exists(pdf):
        os.remove(pdf)
    r = subprocess.run([TECT, "-X", "compile", "_bis_d.tex", "--outdir", OUT],
                       capture_output=True, text=True, cwd=SL)
    ok = os.path.exists(pdf)
    if not ok:
        err = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("error")]
        print(f"frame {n}: ECHEC -> {err[0][:90] if err else '?'}", flush=True)
        OK = n
        break
    else:
        print(f"frame {n}: OK", flush=True)
        os.remove(pdf)
print("premiere frame fautive:", OK)
