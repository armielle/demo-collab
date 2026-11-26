"""Detection automatique des debordements dans un PDF de slides Beamer.

Calibration automatique : la zone de texte est deduite des titres de frames
(la barre de titre occupe exactement \textwidth) et la zone de pied de page
des numeros de page. Aucune constante "a la main".

Usage : python src/check_overflow.py out/_build_ch3.pdf [autres.pdf]
        (option --seuil=6 pour tolerer 6 pt de depassement)
"""
import sys, os, re
import pymupdf

TOL = 8.0          # pt : au-dela, le debordement est visible a la projection


def analyse(path, tol):
    doc = pymupdf.open(path)
    W, H = doc[0].rect.width, doc[0].rect.height

    # --- calibration automatique -------------------------------------
    # 1) bord droit du texte : valeur de x1 la plus FREQUENTE parmi les
    #    lignes de corps (c'est la ou le texte se coupe naturellement) ;
    # 2) haut du pied de page : bloc contenant le numero "n / N".
    from collections import Counter
    x1_pool, foot_bands = [], []
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                txt = "".join(s["text"] for s in l["spans"]).strip()
                if not txt:
                    continue
                y0, x1 = l["bbox"][1], l["bbox"][2]
                if H * 0.16 < y0 < H * 0.9 and x1 > W * 0.72:
                    x1_pool.append(round(x1))
                if re.fullmatch(r"\d+\s*/\s*\d+", txt) and y0 > H * 0.85:
                    foot_bands.append(l["bbox"][1])
    right = Counter(x1_pool).most_common(1)[0][0] if x1_pool else W - 30
    foot_top = (min(foot_bands) - 1.5) if foot_bands else H - 17
    bottom = foot_top

    problems = []
    for i, page in enumerate(doc):
        pn = i + 1
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            # controle au niveau des LIGNES : un bloc peut melanger une
            # derniere ligne de corps et la ligne de pied de page
            for l in b["lines"]:
                x1, y1, y0 = l["bbox"][2], l["bbox"][3], l["bbox"][1]
                txt = "".join(s["text"] for s in l["spans"]).strip()
                if not txt or y0 >= foot_top:
                    continue
                if x1 > right + tol:
                    problems.append((pn, "DROITE", x1 - right, txt[:64]))
                if y1 > bottom + tol:
                    problems.append((pn, "BAS", y1 - bottom, txt[:64]))
        for im in page.get_images(full=True):
            try:
                rects = page.get_image_rects(im[0])
            except Exception:
                continue
            for r in rects:
                if r.y0 >= foot_top:
                    continue
                if r.x1 > right + tol or r.y1 > bottom + tol:
                    problems.append((pn, "IMAGE", max(r.x1 - right, r.y1 - bottom),
                                     f"image {im[0]}"))
    doc.close()
    print(f"\n=== {os.path.basename(path)} : zone de texte droite = {right:.1f} pt, "
          f"limite basse = {bottom:.1f} pt (page {W:.0f}x{H:.0f}) ===")
    if not problems:
        print("Aucun debordement detecte.")
        return True, []
    pages = sorted(set(p for p, *_ in problems))
    print(f"{len(problems)} debordement(s) sur {len(pages)} page(s), "
          f"tolerance {tol} pt :")
    for pn, kind, d, txt in sorted(problems, key=lambda t: -t[2]):
        print(f"  p{pn:>3} [{kind:<6}] +{d:5.1f} pt | {txt}")
    print("pages concernees :", pages)
    return False, pages


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tol = TOL
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            tol = float(a.split("=")[1])
    args = args or [os.path.join("out", "_build_ch1.pdf")]
    allok = True
    for a in args:
        if not os.path.exists(a):
            print(f"[ignore] {a} absent")
            continue
        ok, _ = analyse(a, tol)
        allok &= ok
    print("\nRESULTAT :", "PROPRE" if allok else "DEBORDEMENTS A CORRIGER")
    sys.exit(0 if allok else 1)
