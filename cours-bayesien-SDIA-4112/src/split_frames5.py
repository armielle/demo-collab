"""Dernier levier automatique : scinder une liste trop longue, ou un encadre d'une autre colonne.

Trois strategies, essayees dans cet ordre :
  1. un bloc de premier niveau autre que `columns` -> deplace sur une slide « (suite) » ;
  2. une colonne contenant >= 2 blocs -> son dernier bloc est deplace (la colonne est refermee) ;
  3. une liste `itemize`/`enumerate` de >= 6 entrees -> coupee en deux slides equilibrees.

    python src/split_frames5.py [--dry-run] [--seuil=8]
"""
import os, re, sys, glob
import check_boxes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")
PARTS = os.path.join(ROOT, "slides", "parts")
CH = {"1": "fondements-bayesiens", "2": "a-priori-a-posteriori", "3": "estimation-inference",
      "4": "comparaison-decision", "5": "mcmc-mini-projet"}
MARK = "% [suite5]"
BLOCS = ("columns", "tcolorbox", "kbox", "abox", "ebox", "wbox", "qbox", "solbox", "codebox",
         "itemize", "enumerate", "tabular", "center", "figbox", "minipage", "tikzpicture")


def frames_of(src):
    out, pos = [], 0
    while True:
        i = src.find("\\begin{frame}", pos)
        if i < 0:
            return out
        j = src.find("\\end{frame}", i)
        if j < 0:
            return out
        out.append((i, j + len("\\end{frame}")))
        pos = j


def spans(texte, nom, debut=0):
    res, i = [], debut
    while True:
        m = re.compile(r"\\begin\{" + nom + r"\}(\[[^\]]*\])?").search(texte, i)
        if not m:
            return res
        prof, k = 1, m.end()
        while prof:
            o = texte.find("\\begin{" + nom + "}", k)
            c = texte.find("\\end{" + nom + "}", k)
            if c < 0:
                return res
            if o >= 0 and o < c:
                prof += 1
                k = o + len(nom) + 8
            else:
                prof -= 1
                k = c + len(nom) + 6
        res.append((m.start(), k, m.end(), k - len(nom) - 6))
        i = k


def racines(corps):
    tous = []
    for nom in BLOCS:
        for d, f, _, _ in spans(corps, nom):
            tous.append((d, f, nom))
    return sorted([(d, f, n) for d, f, n in tous if not any(dd < d and f <= ff for dd, ff, _ in tous)])


def coupe(corps):
    """(debut, fin, libelle, fermetures, texte_deplace_force) ou None."""
    r = racines(corps)
    if len(r) >= 2:
        d, f, n = r[-1]
        return d, f, n, "", None
    if not r:
        return None
    d0, f0, n0 = r[0]
    if n0 != "columns":
        # bloc unique : tenter de couper une longue liste a l'interieur
        coupe_liste = None
        for nom in ("itemize", "enumerate"):
            for d, f, dc, fc in spans(corps, nom):
                entrees = re.findall(r"\\item\b", corps[dc:fc])
                if len(entrees) >= 6:
                    coupe_liste = (nom, d, f, dc, fc, len(entrees))
                    break
        if not coupe_liste:
            return None
        nom, d, f, dc, fc, n = coupe_liste
        corps_int = corps[dc:fc]
        items, i = [], 0
        for m in re.finditer(r"\\item\b", corps_int):
            items.append(m.start())
        moitie = len(items) // 2
        coupure = items[moitie]
        avant, apres = corps_int[:coupure].rstrip(), corps_int[coupure:].strip()
        force = ("\\begin{" + nom + "}\n" + avant + "\n\\end{" + nom + "}",
                 "\\begin{" + nom + "}\n" + apres + "\n\\end{" + nom + "}")
        return dc, fc, f"{nom} de {n} entrees coupee", "", force
    # colonnes : essayer chaque colonne, de la plus remplie a la moins remplie
    cols = spans(corps, "column", d0)
    for i in sorted(range(len(cols)), key=lambda k: cols[k][3] - cols[k][2], reverse=True):
        cd, cf, cdc, cfc = cols[i]
        blocs = racines(corps[cdc:cfc])
        if len(blocs) < 2:
            continue
        d2, f2, n2 = blocs[-1]
        debut = cdc + d2
        return debut, cdc + f2, n2 + " (colonne)", "\n\\end{column}\n\\end{columns}", None
    # aucune colonne multi-blocs : tenter une liste dans la colonne la plus remplie
    for i in sorted(range(len(cols)), key=lambda k: cols[k][3] - cols[k][2], reverse=True):
        cd, cf, cdc, cfc = cols[i]
        for nom in ("itemize", "enumerate"):
            for d, f, dc, fc in spans(corps[cdc:cfc], nom):
                if len(re.findall(r"\\item\b", corps[cdc:cfc][dc:fc])) >= 6:
                    corps_int = corps[cdc:cfc][dc:fc]
                    items = [m.start() for m in re.finditer(r"\\item\b", corps_int)]
                    coupure = items[len(items) // 2]
                    avant, apres = corps_int[:coupure].rstrip(), corps_int[coupure:].strip()
                    force = ("\\begin{" + nom + "}\n" + avant + "\n\\end{" + nom + "}",
                             "\\begin{" + nom + "}\n" + apres + "\n\\end{" + nom + "}")
                    return cdc + dc, cdc + fc, f"{nom} coupee (colonne)", "", force
    return None


def main():
    dry = "--dry-run" in sys.argv
    seuil = 8.0
    for a in sys.argv[1:]:
        if a.startswith("--seuil="):
            seuil = float(a.split("=")[1])
    total = 0
    for ch in CH:
        pdf = os.path.join(OUT, f"Chapitre-{ch}-{CH[ch]}.pdf")
        if not os.path.exists(pdf):
            continue
        r = check_boxes.analyse(pdf, 1.0)
        pages = sorted({p for p, dep in r["graves"] if dep > seuil}, reverse=True)
        if not pages:
            print(f"=== chapitre {ch} : rien a scinder ===")
            continue
        index = [(None, None, None)]
        for f in sorted(glob.glob(os.path.join(PARTS, f"ch{ch}_*.tex"))):
            src = open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            index += [(f, a, b) for a, b in frames_of(src)]
        par_fichier = {}
        print(f"\n=== chapitre {ch} : {len(pages)} page(s) ===")
        for pn in pages:
            if pn - 1 >= len(index):
                continue
            f, a, b = index[pn - 1]
            if f is None:
                continue
            src = par_fichier.get(f) or open(f, encoding="utf-8", newline="").read().replace("\r\n", "\n")
            bloc = src[a:b]
            if MARK in bloc:
                print(f"  p{pn:<3} deja traitee")
                continue
            entete = re.match(r"(\\begin\{frame\}(?:\[[^\]]*\])?\{[^\n]*\})\n", bloc)
            fin = bloc.rfind("\\end{frame}")
            if not entete:
                continue
            corps = bloc[entete.end():fin]
            c = coupe(corps)
            if not c:
                print(f"  p{pn:<3} impossible : contenu insécable")
                continue
            d, f_, nom, fermetures, force = c
            reste = corps[:d].rstrip()
            apres = corps[f_:].strip()
            if force:
                garder, deplace = force
            else:
                deplace = corps[d:f_].strip()
                garder = ""
            titre = entete.group(1)
            t = re.match(r"\\begin\{frame\}(\[[^\]]*\])?\{([^\n]*)\}", titre)
            opts, nom_titre = t.group(1) or "", t.group(2)
            # les fermetures de colonne ne doivent pas etre ajoutees deux fois
            if "\\end{columns}" in apres:
                corps_a = reste + ("\n" + garder if garder else "") + "\n" + apres
            else:
                corps_a = (reste + ("\n" + garder if garder else "") + fermetures
                           + (("\n" + apres) if apres else ""))
            nouvelle = (f"{MARK}\n{titre}\n{corps_a}\n\\end{{frame}}\n\n{MARK}\n"
                        f"\\begin{{frame}}{opts}{{{nom_titre} (suite)}}\n{deplace}\n\\end{{frame}}")
            par_fichier[f] = src[:a] + nouvelle + src[b:]
            total += 1
            print(f"  p{pn:<3} scindee ({nom})")
        if not dry:
            for f, contenu in par_fichier.items():
                open(f, "w", encoding="utf-8", newline="").write(contenu.replace("\n", "\r\n"))
    print(f"\n{total} frame(s) scindee(s)" + (" (simulation)" if dry else ""))


if __name__ == "__main__":
    main()
