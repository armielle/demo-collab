"""Exporte les 3 notebooks de TP corrigés en PDF (nbconvert -> HTML -> Edge headless).

Usage :  <venv>/python.exe src/export_tp_pdf.py
"""
import os, sys, subprocess, pathlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = os.path.join(ROOT, "notebooks")
HTML = os.path.join(ROOT, "out", "tp_html")
OUT = os.path.join(ROOT, "out")

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

ORDRE = [
    ("TP1_cartographie.ipynb", "TP1-cartographie.pdf"),
    ("TP2_dataset_supervise.ipynb", "TP2-dataset-supervise.pdf"),
    ("TP3_anomalies_classification.ipynb", "TP3-anomalies-classification.pdf"),
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
        return 1

    print("2/2  HTML -> PDF (Edge headless)")
    n_ok = 0
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
        info = ""
        if ok:
            n_ok += 1
            import pymupdf
            info = f"{os.path.getsize(dst)/1024:7.0f} Ko  {pymupdf.open(dst).page_count:>3} pages"
        print(f"   {'OK ' if ok else 'ECHEC'} {pdf:<40} {info}")
    print(f"\n{n_ok}/{len(ORDRE)} PDF dans out/")
    return 0 if n_ok == len(ORDRE) else 2


if __name__ == "__main__":
    sys.exit(main())
