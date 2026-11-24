"""Insere un facteur de taille dans les frames signalees -- parseur a accolades
equilibrees (corrige le bug de l'insertion dans un titre contenant \\emph{...}).

Usage : python fix_overflows2.py <n> [--apply]
"""
import re
import sys
import subprocess
import pathlib
import pymupdf

ROOT = pathlib.Path(r"C:\Users\HP\Documents\cours-aiops-GIT-5534")
SLIDES = ROOT / "slides"
PY = r"C:/Users/HP/Documents/cours-bayesien-SDIA-4112/.venv/Scripts/python.exe"
CHECK = r"C:/Users/HP/Documents/cours-bayesien-SDIA-4112/src/check_overflow.py"

FINALS = {
    1: "Seance1-Introduction-ML-et-infrastructures.pdf",
    2: "Seance2-Donnees-infrastructures-et-preparation.pdf",
    3: "Seance3-Modeles-ML-infrastructures.pdf",
    4: "Seance4-Supervision-optimisation-prediction.pdf",
    5: "Seance5-Securite-et-fiabilite.pdf",
    6: "Seance6-Industrialisation-et-cloture.pdf",
}


def read_frame_headers(txt):
    """Retourne [(start, body_start, title)] avec accolades equilibrees."""
    out = []
    for m in re.finditer(r"\\begin\{frame\}", txt):
        i = m.end()
        # options [..] equilibrees
        while i < len(txt) and txt[i] in " \t\n":
            i += 1
        if i < len(txt) and txt[i] == "[":
            depth = 0
            while i < len(txt):
                if txt[i] == "[":
                    depth += 1
                elif txt[i] == "]":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            while i < len(txt) and txt[i] in " \t\n":
                i += 1
        if i < len(txt) and txt[i] == "{":
            depth = 0
            start_title = i
            while i < len(txt):
                if txt[i] == "{" and (i == 0 or txt[i - 1] != "\\"):
                    depth += 1
                elif txt[i] == "}" and txt[i - 1] != "\\":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            title = txt[start_title + 1:i - 1]
        else:
            title = ""
        out.append({"start": m.start(), "body": i, "title": title})
    return out


def frames_of(n):
    build = (SLIDES / f"_build_git{n}.tex").read_text(encoding="utf-8")
    parts = re.findall(r"\\input\{parts/([^}]+)\}", build)
    out = []
    for part in parts:
        p = SLIDES / "parts" / (part + ".tex")
        txt = p.read_text(encoding="utf-8")
        for h in read_frame_headers(txt):
            h["file"] = p
            h["line"] = txt[:h["start"]].count("\n") + 1
            out.append(h)
    return out


def overflow_pages(pdf):
    r = subprocess.run([PY, CHECK, str(pdf), "--seuil=4"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"pages concernees\s*:\s*\[([^\]]*)\]", r.stdout)
    return [] if not m else [int(x) for x in m.group(1).split(",") if x.strip()]


def norm(s):
    for a, b in (("é", "e"), ("è", "e"), ("ê", "e"), ("à", "a"), ("ç", "c"), ("ù", "u")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def detect_offset(pdf, frames):
    d = pymupdf.open(pdf)
    best, bestscore = 1, -1
    for off in (0, 1, 2):
        score = sum(1 for i in range(d.page_count)
                    if 0 <= i - off < len(frames)
                    and norm(frames[i - off]["title"])[:20] in norm(d[i].get_text())[:400])
        if score > bestscore:
            best, bestscore = off, score
    return best, bestscore, d.page_count


def main():
    n = int(sys.argv[1])
    apply = "--apply" in sys.argv
    pdf = ROOT / "out" / FINALS[n]
    if not pdf.exists():
        raise SystemExit(f"PDF absent : {pdf}")
    pages = overflow_pages(pdf)
    frames = frames_of(n)
    off, score, npages = detect_offset(pdf, frames)
    print(f"seance {n}: {len(pages)} page(s) {pages} | offset={off} ({score}/{npages})")
    todo = []
    for p in pages:
        fi = p - 1 - off
        if 0 <= fi < len(frames):
            f = frames[fi]
            todo.append(f)
            print(f"  p{p:3d} -> {f['file'].name}:{f['line']}  {f['title'][:60]}")
    if not apply:
        return
    for f in sorted(todo, key=lambda x: (x["file"].name, -x["body"])):
        txt = f["file"].read_text(encoding="utf-8")
        if txt[f["body"]:f["body"] + 40].lstrip().startswith("\\footnotesize"):
            continue
        txt = txt[:f["body"]] + "\n\\footnotesize%" + txt[f["body"]:]
        f["file"].write_text(txt, encoding="utf-8")
    print("insertions faites :", len(todo))


if __name__ == "__main__":
    main()
