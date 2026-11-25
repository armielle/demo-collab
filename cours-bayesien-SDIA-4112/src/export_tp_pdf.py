"""Exporte les six notebooks de TP en PDF imprimables.

Chaîne : nbconvert (notebook -> HTML autonome, sorties incluses) puis Edge en
mode sans interface (HTML -> PDF). Aucun LaTeX n'est requis, ce qui evite la
dependance a pandoc.

Usage :  ./.venv/Scripts/python.exe src/export_tp_pdf.py
"""
import os, sys, glob, shutil, subprocess, pathlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = os.path.join(ROOT, "notebooks")
HTML = os.path.join(ROOT, "out", "tp_html")
OUT = os.path.join(ROOT, "out")

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# ordre et noms de sortie
ORDRE = [
    ("TP1_mise_a_jour_bayesienne.ipynb", "TP1-mise-a-jour-bayesienne.pdf"),
    ("TP2_priors_conjugaison_shrinkage.ipynb", "TP2-priors-conjugaison-shrinkage.pdf"),
    ("TP3_estimation_prediction_regression.ipynb", "TP3-estimation-prediction-regression.pdf"),
    ("TP4_comparaison_modeles_decision.ipynb", "TP4-comparaison-modeles-decision.pdf"),
    ("TP5_mcmc_diagnostics.ipynb", "TP5-mcmc-diagnostics.pdf"),
    ("PROJET_modele_hierarchique_radon.ipynb", "Mini-projet-modele-hierarchique-radon.pdf"),
]


def main():
    os.makedirs(HTML, exist_ok=True)
    py = sys.executable

    print("1/2  notebook -> HTML (nbconvert)")
    for nb, _ in ORDRE:
        r = subprocess.run([py, "-m", "nbconvert", "--to", "html",
                            "--output-dir", HTML, os.path.join(NB, nb)],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0
        print(f"   {'OK ' if ok else 'ECHEC'} {nb}")
        if not ok:
            print("      ", (r.stderr or r.stdout)[-300:])

    if not os.path.exists(EDGE):
        print(f"Edge introuvable : {EDGE}")
        return

    print("2/2  HTML -> PDF (Edge headless)")
    for nb, pdf in ORDRE:
        src = os.path.join(HTML, nb[:-6] + ".html")
        if not os.path.exists(src):
            print(f"   ECHEC {nb} : HTML absent")
            continue
        dst = os.path.join(OUT, pdf)
        url = pathlib.Path(src).as_uri()
        r = subprocess.run([EDGE, "--headless=new", "--disable-gpu",
                            "--no-pdf-header-footer",
                            "--virtual-time-budget=20000",
                            f"--print-to-pdf={dst}", url],
                           capture_output=True, text=True, timeout=300)
        ok = os.path.exists(dst)
        # pages et poids
        info = ""
        if ok:
            import pymupdf
            info = f"{os.path.getsize(dst)/1024:7.0f} Ko  {pymupdf.open(dst).page_count:>3} pages"
        print(f"   {'OK ' if ok else 'ECHEC'} {pdf:<46} {info}")
    print("\nPDF de TP dans out/")


if __name__ == "__main__":
    main()
