"""Figure pedagogique : moyenne mobile (SMA / EMA) sur une serie calculee."""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- donnees
serie = np.array([12.0, 14.5, 13.0, 16.0, 18.5, 17.0, 15.5, 19.0, 21.0, 20.0])
n = len(serie)


def sma(x, k):
    return np.array([x[i - k + 1:i + 1].mean() for i in range(k - 1, len(x))])


def ema(x, k):
    a = 2 / (k + 1)
    out = [x[0]]
    for v in x[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return np.array(out)


k = 3
s3, e3 = sma(serie, k), ema(serie, k)
alpha = 2 / (k + 1)

# effet de la fenetre : lissage vs retard
effet = []
for kk in (2, 3, 5):
    s = sma(serie, kk)
    effet.append({"k": kk,
                  "perdus": kk - 1,
                  "rugosite": round(float(np.std(np.diff(s))), 3),
                  "etendue": round(float(s.max() - s.min()), 2),
                  "retard": kk - 1})
effet.insert(0, {"k": 0, "perdus": 0,
                 "rugosite": round(float(np.std(np.diff(serie))), 3),
                 "etendue": round(float(serie.max() - serie.min()), 2), "retard": 0})

cles = {
    "serie": serie.tolist(),
    "k": k,
    "sma": [round(float(v), 3) for v in s3],
    "ema": [round(float(v), 3) for v in e3],
    "alpha": round(alpha, 3),
    "premiers_calculs": [
        {"t": i, "bloc": serie[i - k + 1:i + 1].tolist(),
         "somme": round(float(serie[i - k + 1:i + 1].sum()), 2),
         "moyenne": round(float(s3[i - k + 1]), 3)} for i in range(k - 1, k + 2)
    ],
    "effet_fenetre": effet,
    "dernier_ema": round(float(e3[-1]), 3),
    "dernier_sma": round(float(s3[-1]), 3),
}
with open(os.path.join(OUT, "key_numbers_moyenne_mobile.json"), "w", encoding="utf-8") as f:
    json.dump(cles, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- graphique
L, H = 560, 270
X0, X1, Y0, Y1 = 60, 540, 30, 222
vmin, vmax = 10.5, 22.0


def px(i):
    return X0 + (X1 - X0) * i / (n - 1)


def py(v):
    return Y1 - (Y1 - Y0) * (v - vmin) / (vmax - vmin)


grille = []
for v in (12, 14, 16, 18, 20, 22):
    y = py(v)
    grille.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" '
                  f'class="gl" stroke-width="1" opacity=".55"/>')
    grille.append(f'<text x="{X0-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                  f'class="tx">{v}</text>')
for i in (0, 3, 6, 9):
    grille.append(f'<text x="{px(i):.1f}" y="{Y1+18}" text-anchor="middle" font-size="10" '
                  f'class="tx">t={i}</text>')

pts_serie = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(serie))
pts_sma = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in zip(range(k - 1, n), s3))
pts_ema = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(e3))
cercles = "".join(
    f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.6" class="pt"/>'
    for i, v in enumerate(serie))

svg = f'''<svg viewBox="0 0 {L} {H}" width="100%" style="width:100%;max-width:540px;height:auto" role="img">
  <desc>Serie brute, moyenne mobile simple de fenetre 3 et moyenne mobile exponentielle</desc>
  {''.join(grille)}
  <polyline points="{pts_serie}" fill="none" class="lserie" stroke-width="1.6" opacity=".8"/>
  {cercles}
  <polyline points="{pts_ema}" fill="none" class="lema" stroke-width="2.6"
            opacity=".95" stroke-dasharray="7 4"/>
  <polyline points="{pts_sma}" fill="none" class="lsma" stroke-width="3"/>
  <text x="{X0}" y="14" font-size="11" class="tx">
    serie brute, SMA(3) pleine, EMA(3) pointillee</text>
</svg>'''

lignes = "".join(
    f'<tr><td>t={c["t"]}</td><td>({" + ".join(f"{v:g}" for v in c["bloc"])}) / 3</td>'
    f'<td>{c["somme"]:g}</td><td><b>{c["moyenne"]:.3f}</b></td></tr>'
    for c in cles["premiers_calculs"])

lignes_effet = "".join(
    f'<tr><td>{"série brute" if e["k"] == 0 else "k = " + str(e["k"])}</td>'
    f'<td>{e["perdus"] or "—"}</td><td>{e["rugosite"]:.3f}</td>'
    f'<td>{e["etendue"]:.2f}</td></tr>'
    for e in effet)

html = f'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<style>
  body {{ margin:0; padding:0; background:transparent; color:var(--foreground, #111827);
         font-family:inherit; font-size:14px; }}
  h2 {{ font-size:15px; margin:0 0 10px; font-weight:650; }}
  .grille {{ display:flex; gap:22px; flex-wrap:wrap; align-items:flex-start; }}
  .grille > div:first-child {{ flex:1 1 420px; min-width:320px; }}
  .carte {{ border:1px solid var(--border); border-radius:10px; padding:12px 14px;
            background:color-mix(in srgb, var(--card) 55%, transparent); }}
  table {{ border-collapse:collapse; font-size:12.5px; }}
  th, td {{ padding:5px 9px; text-align:left; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted-foreground); font-weight:600; }}
  td:last-child {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .note {{ color:var(--muted-foreground); font-size:12px; margin-top:8px; line-height:1.5; }}
  .formule {{ font-family:ui-monospace,Consolas,monospace; font-size:13px;
              background:color-mix(in srgb, var(--accent) 12%, transparent);
              border-left:3px solid var(--accent); padding:7px 10px; border-radius:4px;
              margin:0 0 12px; }}
  .mesure {{ display:flex; gap:18px; margin-top:10px; flex-wrap:wrap; }}
  .mesure div {{ font-size:12px; color:var(--muted-foreground); }}
  .mesure b {{ display:block; font-size:19px; color:var(--foreground, #111827); font-weight:650; }}
  .gl {{ stroke: var(--border, #d4d4d8); }}
  .tx {{ fill: var(--muted-foreground, #6b7280); }}
  .pt {{ fill: var(--muted-foreground, #6b7280); }}
  .lserie {{ stroke: var(--muted-foreground, #6b7280); }}
  .lsma {{ stroke: var(--foreground, #111827); }}
  .lema {{ stroke: var(--accent, #2563eb); }}
</style></head>
<body>
  <h2>Moyenne mobile : lisser une série sans perdre sa tendance</h2>
  <div class="formule">SMA(k) à l'instant t = (x<sub>t−k+1</sub> + … + x<sub>t</sub>) / k</div>
  <div class="grille">
    <div>{svg}
      <div class="mesure">
        <div><b>{cles["dernier_sma"]:.3f}</b> SMA(3) au dernier point</div>
        <div><b>{cles["dernier_ema"]:.3f}</b> EMA(3) au dernier point</div>
        <div><b>{alpha:.2f}</b> poids α de l'EMA</div>
      </div>
    </div>
    <div class="carte">
      <table>
        <tr><th>t</th><th>calcul</th><th>somme</th><th>moyenne</th></tr>
        {lignes}
      </table>
      <div class="note">Les {k-1} premiers points n'ont pas de moyenne :
      la série passe de {n} à {n-(k-1)} valeurs.</div>
    </div>
  </div>
  <div class="carte" style="margin-top:14px">
    <table>
      <tr><th>fenêtre</th><th>points perdus</th><th>rugosité</th><th>étendue</th></tr>
      {lignes_effet}
    </table>
    <div class="note"><b>Rugosité</b> = écart-type des écarts successifs : elle mesure
    les dents de scie. Elle chute de 2,065 (brut) à 0,408 (k = 5). Plus k grandit, plus la
    courbe est lisse — mais plus elle <b>retarde</b> sur le signal : retard de k−1 pas en mode
    « trailing » (celui d'ici), (k−1)/2 seulement si la fenêtre est centrée.</div>
  </div>
</body></html>'''

chemin = os.path.join(OUT, "moyenne_mobile.html")
with open(chemin, "w", encoding="utf-8") as f:
    f.write(html)
print("figure :", chemin)
print("chiffres :", json.dumps(cles, ensure_ascii=False)[:320])
