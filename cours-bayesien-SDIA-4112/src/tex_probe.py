import subprocess, os, textwrap
D = r"C:\Users\HP\Documents\cours-bayesien-SDIA-4112\slides"
TEX = r"C:\Users\HP\texbin\tectonic.exe"

variants = {
"a_plain": r"""
\begin{frame}{A}\titlepage\end{frame}
\begin{frame}{Simple} Bonjour é à ç « test » \end{frame}
""",
"b_boxes": r"""
\begin{frame}{B}
\begin{ebox}[Exemple] Beta-Binomial $\theta \given y$ \end{ebox}
\begin{qbox}[Quiz] Combien ? \end{qbox}
\end{frame}
""",
"c_lstlisting": r"""
\begin{frame}[fragile]{C}
\begin{codebox}[PyMC]
\begin{lstlisting}
import pymc as pm   # a priori
theta = pm.Beta("theta", 2, 2)
\end{lstlisting}
\end{codebox}
\end{frame}
""",
"d_tikz": r"""
\begin{frame}{D}
\begin{tikzpicture}
\begin{axis}[width=5cm,height=3cm,axis lines=left]
\addplot[byBlue,thick,domain=0:1,samples=50]{x^3*(1-x)*140};
\end{axis}
\end{tikzpicture}
\end{frame}
""",
"e_cols": r"""
\begin{frame}{E}
\begin{columns}
\begin{column}{0.48\textwidth} gauche \end{column}
\begin{column}{0.48\textwidth} droite \end{column}
\end{columns}
\end{frame}
""",
}

for name, body in variants.items():
    f = os.path.join(D, f"_v_{name}.tex")
    open(f, "w", encoding="utf-8").write("\\input{preamble.tex}\n\\begin{document}\n" + body + "\n\\end{document}\n")
    r = subprocess.run([TEX, "-X", "compile", f"_v_{name}.tex", "--outdir", "../out", "--keep-logs"],
                       cwd=D, capture_output=True, text=True, timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    bad = [l for l in out.splitlines() if "error" in l.lower() or "Illegal" in l or "!" == l[:1]]
    print(f"--- {name}: rc={r.returncode} pdf={'OK' if os.path.exists(os.path.join(D,'..','out',f'_v_{name}.pdf')) else 'NO'}")
    for l in bad[:6]:
        print("    ", l[:130])
