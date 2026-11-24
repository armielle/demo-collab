"""Corrige les debordements verticaux des decks EPD-GIT-5534.

Strategie : pour chaque page signalee par check_overflow (seuil 4 pt), insérer
\\footnotesize (puis \\scriptsize si besoin) au debut du corps de la frame
correspondante. Ne touche jamais au padding du pied ni au interlignage.

Usage : python fix_overflows.py <n> [--apply]
Sans --apply : affiche seulement le diagnostic (pages -> frames).
"""
import re
import sys
import subprocess
import pathlib

ROOT = pathlib.Path(r"C:\Users\HP\Documents\cours-aiops-GIT-5534")
SLIDES = ROOT / "slides"
PY = r"C:/Users/HP/Documents/cours-bayesien-SDIA-4112/.venv/Scripts/python.exe"
CHECK = r"C:/Users/HP/Documents/cours-bayesien-SDIA-4112/src/check_overflow.py"


def frames_of(n):
    build = (SLIDES / f"_build_git{n}.tex").read_text(encoding="utf-8")
    parts = re.findall(r"\\input\{parts/([^}]+)\}", build)
    out = []
    for part in parts:
        p = SLIDES / "parts" / (part + ".tex")
        txt = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\\begin\{frame\}(\[[^\]]*\])?\{([^}]*)\}", txt):
            out.append({
                "file": p,
                "start": m.start(),
                "end": m.end(),
                "line": txt[:m.start()].count("\n") + 1,
                "opts": m.group(1) or "",
                "title": m.group(2),
                "text": txt,
            })
    return out


def overflow_pages(pdf):
    r = subprocess.run([PY, CHECK, str(pdf), "--seuil=4"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout
    m = re.search(r"pages concernees\s*:\s*\[([^\]]*)\]", out)
    if not m:
        return []
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower().replace("é", "e").replace("è", "e")
                  .replace("ê", "e").replace("à", "a").replace("ç", "c"))


def detect_offset(pdf, frames):
    import pymupdf
    d = pymupdf.open(pdf)
    best, bestscore = 1, -1
    for off in (0, 1, 2):
        score = 0
        for i in range(d.page_count):
            fi = i - off
            if 0 <= fi < len(frames):
                ptxt = norm(d[i].get_text())[:400]
                if norm(frames[fi]["title"])[:20] in ptxt:
                    score += 1
        if score > bestscore:
            best, bestscore = off, score
    return best, bestscore, d.page_count


def resolve_pdf(n):
    cand = [
        ROOT / "out" / f"_build_git{n}.pdf",
        ROOT / "out" / {
            1: "Seance1-Introduction-ML-et-infrastructures.pdf",
            2: "Seance2-Donnees-infrastructures-et-preparation.pdf",
            3: "Seance3-Modeles-ML-infrastructures.pdf",
            4: "Seance4-Supervision-optimisation-prediction.pdf",
            5: "Seance5-Securite-et-fiabilite.pdf",
            6: "Seance6-Industrialisation-et-cloture.pdf",
        }[n],
    ]
    for c in cand:
        if c.exists():
            return c
    raise SystemExit(f"aucun PDF pour la seance {n}")


def main():
    n = int(sys.argv[1])
    apply = "--apply" in sys.argv
    pdf = resolve_pdf(n)
    pages = overflow_pages(pdf)
    frames = frames_of(n)
    off, score, npages = detect_offset(pdf, frames)
    print(f"seance {n}: {len(pages)} page(s) signalee(s) {pages} | offset={off} (score {score}/{npages})")
    plan = []
    for p in pages:
        fi = p - 1 - off
        if 0 <= fi < len(frames):
            f = frames[fi]
            plan.append((p, f))
            print(f"  p{p:3d} -> {f['file'].name}:{f['line']}  {f['title'][:60]}")
    if not apply:
        return
    # groupe par fichier, insertion du plus grand decalage au plus petit
    touched = []
    for p, f in sorted(plan, key=lambda t: (t[1]["file"].name, -t[1]["end"])):
        txt = f["file"].read_text(encoding="utf-8")
        start, end = f["start"], f["end"]
        seg = txt[start:end]
        if "\\footnotesize" in txt[end:end + 120] or "\\scriptsize" in txt[end:end + 120]:
            continue
        new = txt[:end] + "\n\\footnotesize%" + txt[end:]
        f["file"].write_text(new, encoding="utf-8")
        touched.append(f"{f['file'].name}:{f['line']}")
    print("frames modifiees :", len(touched))
    for t in touched:
        print("   ", t)


if __name__ == "__main__":
    main()
