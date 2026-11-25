"""Genere out/tableau-de-bord.html : page de synthese visuelle du kit de cours.

Autonome (figures encodees en base64) : peut etre ouverte, envoyee ou imprimee
sans dependance externe.
"""
import os, re, glob, json, base64, datetime, html

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = os.path.join(ROOT, "slides", "parts")
FIG = os.path.join(ROOT, "figures")
NB = os.path.join(ROOT, "notebooks")
OUT = os.path.join(ROOT, "out")

CH = {
    "1": ("Fondements bayésiens", "8 h", "Théorème de Bayes, prior/vraisemblance/posterior, contre le fréquentisme"),
    "2": ("A priori et a posteriori", "8 h", "Conjugaison, Jeffreys, contrôles prédictifs, shrinkage hiérarchique"),
    "3": ("Estimation et inférence", "8 h", "Estimateurs et pertes, HDI, couverture, prédiction, régression"),
    "4": ("Comparaison et décision", "8 h", "Facteur de Bayes, WAIC/LOO, théorie de la décision, A/B testing"),
    "5": ("MCMC et mini-projet", "8 h", "Monte-Carlo, Metropolis-Hastings, Gibbs, NUTS, diagnostics, projet"),
}
GALLERY = [
    ("f07_covid_ppv.png", "Ch.1 — le même test, six vérités selon la prévalence"),
    ("f08_pvalue_vs_posterior.png", "Ch.1 — une p-value de 0,05 laisse 71 % de chance que H0 soit vraie"),
    ("g10_sensibilite_n.png", "Ch.2 — le prior s'efface en 1/racine(n)"),
    ("g10_prior_predictive.png", "Ch.2 — contrôle prédictif a priori"),
    ("g13_shrinkage_8groupes.png", "Ch.2 — shrinkage hiérarchique sur 8 groupes"),
    ("h04_couverture.png", "Ch.3 — couverture empirique mesurée sur 2 000 simulations"),
    ("h07_regression_bayesienne.png", "Ch.3 — régression bayésienne, bande de crédibilité"),
    ("i02_bf_vs_n.png", "Ch.4 — évolution du facteur de Bayes avec n"),
    ("i04_pvalue_vs_bf.png", "Ch.4 — p-value contre facteur de Bayes, 2 000 simulations"),
    ("j03_mh_beta.png", "Ch.5 — Metropolis-Hastings écrit à la main"),
    ("j08_hmc_nuts.png", "Ch.5 — dynamique hamiltonienne et sauts NUTS"),
]


def frames(c):
    n = sum(len(re.findall(r"\\begin\{frame\}", open(p, encoding="utf-8").read()))
            for p in sorted(glob.glob(os.path.join(PARTS, f"ch{c}_*.tex"))))
    return n + 1 if n else 0


MIME = {".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def b64(path):
    """Encode une figure en data-URI ; le type MIME suit l'extension (GIF animes inclus)."""
    mime = MIME.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def pick(name):
    """Retrouve une figure meme si le prefixe numerique exact differe."""
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        return p
    motif = re.sub(r"^([a-z])\d+_", r"\1*_", name)
    cand = sorted(glob.glob(os.path.join(FIG, motif)))
    return cand[0] if cand else None


def main():
    nframes = sum(frames(c) for c in CH)
    figs = sorted(glob.glob(os.path.join(FIG, "*.png"))
              + glob.glob(os.path.join(FIG, "*.gif")))   # meme perimetre que make_readme.py
    nbs = sorted(glob.glob(os.path.join(NB, "*.ipynb")))
    nval = 0
    for k in glob.glob(os.path.join(FIG, "key_numbers_*.json")):
        try:
            nval += len(json.load(open(k, encoding="utf-8")))
        except Exception:
            pass
    nq = sum(len(re.findall(r"\\begin\{qbox\}", open(p, encoding="utf-8").read()))
             for p in glob.glob(os.path.join(PARTS, "ch*.tex")))
    na = sum(len(re.findall(r"\\begin\{abox\}", open(p, encoding="utf-8").read()))
             for p in glob.glob(os.path.join(PARTS, "ch*.tex")))
    ne = sum(len(re.findall(r"\\begin\{ebox\}", open(p, encoding="utf-8").read()))
             for p in glob.glob(os.path.join(PARTS, "ch*.tex")))

    H = []
    A = H.append
    A("""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Statistique Bayésienne — tableau de bord du kit</title>
<style>
:root{--deep:#10243E;--blue:#1F6FEB;--cyan:#12A594;--orange:#D97706;--red:#C0392B;
--green:#1E8449;--purple:#6C5CE7;--grey:#5B6470;--light:#F4F6F8;}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI",system-ui,-apple-system,sans-serif;color:var(--deep);
background:#fff;line-height:1.5}
header{background:linear-gradient(120deg,#10243E 0%,#123A63 55%,#0E6E63 100%);
color:#fff;padding:38px 44px 30px}
header .k{font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:#7FE3D5;font-weight:600}
h1{margin:.25em 0 .1em;font-size:33px;font-weight:700;letter-spacing:-.5px}
header p{margin:.2em 0 0;color:#C9D6E4;max-width:70ch}
main{padding:30px 44px 60px;max-width:1240px}
h2{font-size:20px;margin:38px 0 14px;padding-bottom:7px;border-bottom:2px solid var(--light)}
.grid{display:grid;gap:14px}
.kpi{grid-template-columns:repeat(auto-fit,minmax(158px,1fr))}
.kpi div{background:var(--light);border-radius:10px;padding:16px 18px;border-left:4px solid var(--blue)}
.kpi b{display:block;font-size:29px;font-weight:700;line-height:1.1}
.kpi span{font-size:12.5px;color:var(--grey);text-transform:uppercase;letter-spacing:.05em}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid #E3E8EE}
th{background:var(--light);font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--grey)}
tr:hover td{background:#FAFCFE}
.bar{height:7px;border-radius:4px;background:var(--blue);display:inline-block;vertical-align:middle}
.gal{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
figure{margin:0;border:1px solid #E3E8EE;border-radius:10px;overflow:hidden;background:#fff}
figure img{width:100%;display:block}
figcaption{padding:10px 13px;font-size:13px;color:var(--grey);border-top:1px solid #EEF2F6}
.note{background:var(--light);border-left:4px solid var(--cyan);padding:14px 18px;
border-radius:0 8px 8px 0;font-size:14px}
code{background:#EEF2F6;padding:1px 5px;border-radius:4px;font-size:13px}
footer{padding:22px 44px 44px;color:var(--grey);font-size:12.5px}
</style></head><body>""")
    A("<header><div class='k'>EPD-SDIA-4112 &middot; Master 1 Science des Données et IA</div>")
    A("<h1>Statistique Bayésienne — kit de cours exécutable</h1>")
    A("<p>5 chapitres, 40 heures, 3 crédits. Support LaTeX Beamer compilable, "
      "figures produites par du code exécuté, notebooks de TP pré-exécutés. "
      "Reconstruit à partir du document source et de son audit.</p></header><main>")

    A("<h2>Vue d'ensemble</h2><div class='grid kpi'>")
    for v, t in [(nframes, "slides"), (len(figs), "figures générées"),
                 (len(nbs), "notebooks exécutés"), (nq, "quiz"), (na, "activités"),
                 (ne, "exemples"), (nval, "valeurs traçables")]:
        A(f"<div><b>{v}</b><span>{t}</span></div>")
    A("</div>")

    A("<h2>Chapitres</h2><table><tr><th>#</th><th>Titre</th><th>Volume</th>"
      "<th>Slides</th><th>Contenu clé</th></tr>")
    for c, (t, h, d) in CH.items():
        n = frames(c)
        w = min(100, int(n / 50 * 100))
        A(f"<tr><td><b>{c}</b></td><td>{t}</td><td>{h}</td><td>{n} "
          f"<span class='bar' style='width:{w}px'></span></td><td>{d}</td></tr>")
    A(f"<tr><td></td><td><b>Total</b></td><td><b>40 h</b></td><td><b>{nframes}</b></td><td></td></tr></table>")

    A("<h2>Figures du support — échantillon</h2><div class='grid gal'>")
    for name, cap in GALLERY:
        p = pick(name)
        if p:
            A(f"<figure><img src='{b64(p)}' alt='{html.escape(cap)}'>"
              f"<figcaption>{cap} <br><code>{os.path.basename(p)}</code></figcaption></figure>")
    A("</div>")

    A("<h2>Notebooks de travaux pratiques</h2><table><tr><th>Notebook</th>"
      "<th>Cellules</th><th>Figures produites</th></tr>")
    for p in nbs:
        try:
            nbk = json.load(open(p, encoding="utf-8"))
            ic = sum(1 for c in nbk["cells"] if c["cell_type"] == "code")
            im = sum(1 for c in nbk["cells"] for o in c.get("outputs", [])
                     if "data" in o and "image/png" in o.get("data", {}))
            A(f"<tr><td><code>{os.path.basename(p)}</code></td><td>{ic}</td><td>{im}</td></tr>")
        except Exception:
            A(f"<tr><td><code>{os.path.basename(p)}</code></td><td>—</td><td>—</td></tr>")
    A("</table>")

    A("<h2>Traçabilité des chiffres</h2>")
    A("<div class='note'><b>Aucune valeur numérique n'est saisie à la main.</b> "
      "Les scripts <code>src/fig_chN*.py</code> calculent et écrivent "
      f"<code>figures/key_numbers_chN.json</code> ({nval} valeurs tracées) ; "
      "les slides et notebooks citent ces valeurs. Graines aléatoires fixées : "
      "les résultats sont reproductibles à l'identique. "
      "Les corrections apportées au matériel d'origine sont listées dans <code>AUDIT.md</code>.</div>")

    A("</main><footer>Tableau de bord généré par <code>src/make_dashboard.py</code> le "
      + datetime.date.today().isoformat() + ".</footer></body></html>")

    path = os.path.join(OUT, "tableau-de-bord.html")
    open(path, "w", encoding="utf-8").write("\n".join(H))
    print(f"[dashboard] {path}  ({os.path.getsize(path)/1024:.0f} Ko)")
    print(f"  slides={nframes} figures={len(figs)} notebooks={len(nbs)} valeurs={nval}")
    miss = [n for n, _ in GALLERY if not pick(n)]
    if miss:
        print("  figures de galerie introuvables :", miss)


if __name__ == "__main__":
    main()
