"""Diagnostic page -> frame pour un deck EPD-GIT-5534.

Usage : python diag_pages.py <n> [pages...]
Affiche, pour chaque page demandee, le titre de la frame correspondante
(en supposant 1 frame = 1 page, verifie par comparaison de titres).
"""
import re
import sys
import pathlib
import pymupdf

ROOT = pathlib.Path(r"C:\Users\HP\Documents\cours-aiops-GIT-5534")
SLIDES = ROOT / "slides"


def frames_of(n: int):
    build = (SLIDES / f"_build_git{n}.tex").read_text(encoding="utf-8")
    parts = re.findall(r"\\input\{parts/([^}]+)\}", build)
    out = []
    for part in parts:
        p = SLIDES / "parts" / (part + ".tex")
        txt = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\\begin\{frame\}(\[[^\]]*\])?\{([^}]*)\}", txt):
            out.append({
                "file": p.name,
                "line": txt[:m.start()].count("\n") + 1,
                "title": m.group(2),
            })
    return out


def main():
    n = int(sys.argv[1])
    pages = [int(x) for x in sys.argv[2:]] or []
    d = pymupdf.open(ROOT / "out" / f"_build_git{n}.pdf")
    fr = frames_of(n)
    print(f"frames={len(fr)} pages={d.page_count}")
    for i in range(d.page_count):
        first = d[i].get_text().splitlines()
        head = first[0][:55] if first else ""
        ftitle = fr[i]["title"][:55] if i < len(fr) else "-"
        mark = " <<<" if (i + 1) in pages else ""
        print(f"{i+1:3d} | {head:<55} || {ftitle}{mark}")


if __name__ == "__main__":
    main()
