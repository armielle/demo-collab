"""Correcteur de debordements du deck CNN -- leviers successifs, idempotent.

Levier 0 : image plus large que sa colonne -> largeur reduite a celle de la colonne.
Levier 1 : frame avec figure -> la figure sort sur une slide dediee.
Levier 2 : frame sans figure -> le dernier bloc de premier niveau part sur une
           slide « (suite) ».
Chaque frame corrigee recoit un marqueur % [fix] et n'est pas retouchee deux fois.
"""
import os, re, glob, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")

CH = {1: "ch1", 2: "ch2", 3: "ch3", 4: "ch4", 5: "ch5", 6: "ch6"}
CODE = {v: k for k, v in CH.items()}


def split_frames(txt):
    """Retourne [(debut, fin, texte)] des frames, avec les \section et chapterslide."""
    res = []
    for m in re.finditer(r"\\begin\{frame\}", txt):
        i = m.start()
        j = txt.find(r"\end{frame}", i)
        res.append((i, (j + len(r"\end{frame}")) if j >= 0 else len(txt)))
    return res


def largeur_colonne(avant_frame):
    m = re.findall(r"\\begin\{column\}\{([\d.]+)\\textwidth\}", avant_frame)
    return float(m[-1]) if m else 1.0


def corrige(path, pages_defect, debut_page=1, offset=0):
    txt = open(path, encoding="utf-8").read()
    frames = split_frames(txt)
    if not frames:
        return []
    n_fix = 0
    logs = []
    out = txt
    for (a, b) in reversed(frames):
        idx = len([1 for (aa, _) in frames if aa < a])
        page = debut_page + idx + offset
        if page not in pages_defect:
            continue
        corps = out[a:b]
        if corps.count("% [fix]") >= 2:      # deux leviers deja appliques
            continue
        nom_frame = re.search(r"\\begin\{frame\}(?:\[[^\]]*\])?\{([^}]*)\}", corps)
        titre = nom_frame.group(1) if nom_frame else "Frame"
        mimg = re.search(r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}", corps)
        if mimg and "% [fix] " not in corps:
            opts, fig = mimg.group(1), mimg.group(2)
            if "width=" not in opts:
                opts = opts + ",width=.9\\textwidth" if opts else "width=.9\\textwidth"
            wm = re.search(r"width=([\d.]+)\\textwidth", opts)
            if wm:
                w = float(wm.group(1))
                # largeur de la colonne dans laquelle se trouve l'image
                pos = corps.find(mimg.group(0))
                colw = largeur_colonne(corps[:pos])
                if w > colw + 0.02:      # levier 0 : image trop large
                    new = f"width={colw - 0.02:.2f}\\textwidth"
                    corps2 = corps[:mimg.start()] + mimg.group(0).replace(wm.group(0), new) + corps[mimg.end():]
                    out = out[:a] + corps2 + out[b:]
                    logs.append(f"  levier0 p{page}: image {w}->{colw-0.02:.2f} ({fig})")
                    n_fix += 1
                    continue
            # levier 1 : la figure sort sur sa propre slide
            bloc_img = (r"\begin{frame}{%s --- la figure}\begin{center}"
                        % titre
                        + r"\includegraphics[width=.92\textwidth]{" + fig + "}"
                        + r"\end{center}\end{frame}")
            corps_sans = re.sub(r"\\begin\{center\}\s*\\includegraphics\[[^\]]*\]\{"
                                + re.escape(fig) + r"\}\s*\\end\{center\}", "", corps)
            if corps_sans == corps:      # la figure n'etait pas dans un center : on la retire
                corps_sans = corps.replace(mimg.group(0) + "\n", "").replace(mimg.group(0), "")
            corps_sans = corps_sans.replace("\n\n\n", "\n\n").rstrip() + "\n"
            corps_sans = corps_sans.replace(r"\end{frame}", "% [fix] figure sortie\n\\end{frame}", 1)
            remplace = corps_sans + "\n" + bloc_img
            out = out[:a] + remplace + out[b:]
            logs.append(f"  levier1 p{page}: figure sorti ({fig})")
            n_fix += 1
            continue
        # levier 2 : dernier bloc de premier niveau
        blocs = [m.start() for m in re.finditer(
            r"\\begin\{(columns|itemize|enumerate|kbox|wbox|ebox|abox|qbox|solbox|codebox|center)\}",
            corps)]
        closes = [m.end() for m in re.finditer(
            r"\\end\{(columns|itemize|enumerate|kbox|wbox|ebox|abox|qbox|solbox|codebox|center)\}",
            corps)]
        if not blocs or not closes:
            logs.append(f"  p{page}: SANS BLOC decoupable (a traiter a la main)")
            continue
        # dernier couple begin/end de premier niveau
        dernier_debut = blocs[-1]
        env = re.match(r"\\begin\{(\w+)\}", corps[dernier_debut:]).group(1)
        # find matching end
        prof = 0
        pos = dernier_debut
        fin = None
        for m in re.finditer(r"\\(begin|end)\{" + env + r"\}", corps[dernier_debut:]):
            prof += 1 if m.group(1) == "begin" else -1
            if prof == 0:
                fin = dernier_debut + m.end()
                break
        if fin is None:
            logs.append(f"  p{page}: bloc non ferme (a traiter a la main)")
            continue
        bloc = corps[dernier_debut:fin]
        reste = corps[:dernier_debut] + corps[fin:]
        # la fermeture de frame doit rester dans la premiere moitie
        if "\\end{frame}" not in reste:
            logs.append(f"  p{page}: ne peut pas couper (fermeture dans le bloc)")
            continue
        titre2 = titre if len(titre) < 44 else titre[:40] + "…"
        bloc_img = (r"\begin{frame}{%s (suite)}" % titre2 + "\n" + bloc + "\n\\end{frame}")
        reste = reste.replace(r"\end{frame}", "% [fix] bloc deplace\n\\end{frame}", 1)
        remplace = reste + "\n" + bloc_img
        out = out[:a] + remplace + out[b:]
        logs.append(f"  levier2 p{page}: dernier bloc deplace ({env})")
        n_fix += 1
    if n_fix:
        open(path, "w", encoding="utf-8", newline="\n").write(out)
    return logs


def main():
    seuil = 6.0
    import json
    rapport = os.path.join(ROOT, "out", "journal", "qa_defects.json")
    if not os.path.exists(rapport):
        print("pas de rapport (out/journal/qa_defects.json) ; rien a faire")
        return
    data = json.load(open(rapport, encoding="utf-8"))
    for pdf, pages in data.items():
        base = os.path.basename(pdf)
        m = re.match(r"Chapitre-(\d)-", base)
        if not m:
            continue
        n = int(m.group(1))
        pages_defect = {p + 1 for p in pages}
        frames_glob = sorted(glob.glob(os.path.join(PARTS, f"{CH[n]}_*.tex")))
        total = 0
        for f in frames_glob:
            t = open(f, encoding="utf-8").read()
            k = len(re.findall(r"\\begin\{frame\}", t))
            has_ch = "\\chapterslide" in t          # la page de garde compte pour 1 page
            k_eff = k + (1 if has_ch else 0)
            debut = total + 1
            pages_i = {pg for pg in pages_defect if debut <= pg < debut + k_eff}
            if pages_i:
                for lg in corrige(f, pages_i, debut, 1 if has_ch else 0):
                    print(os.path.basename(f) + " " + lg)
            total += k_eff
    print("correcteur : passe terminee")


if __name__ == "__main__":
    main()
