"""Construit et execute reellement le TP1 (corrigé officiel) avec nbclient."""
import os
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNEL = "sdia-bayes"
SEED = 5534

PREAMBLE = """%matplotlib inline
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
SEED = 5534                      # graine unique du projet (reproductibilité)
np.random.seed(SEED)
print("OK — graine", SEED)
"""

HEADER = """## EPD-GIT-5534 — TP 1 (corrigé officiel)

**Séance 1 — Cartographie de l'état des lieux : le dossier NOSYS**
**Enseignante : Dr. Noulapeu Ngaffo Armielle**

**Objectifs** : charger et contrôler le dossier `nosys_metriques.csv`
(8 serveurs × 30 jours × pas de 5 min), produire un inventaire de qualité,
visualiser srv-01 sur 30 jours avec les 2 incidents annotés, calculer les
statistiques par serveur (CPU, latence p99, consommation, ratio de capacité)
et rédiger 5 constats chiffrés + 3 risques SLA + 1 recommandation.

**Barème (20 pts)** : constats chiffrés et corrects — 10 pts · risques
reliés aux données — 5 pts · notebook exécutable, commenté, reproductible — 5 pts.

*Corrigé officiel exécuté de bout en bout (noyau `sdia-bayes`, graine 5534).*
"""

CELLS = [
    ("md", HEADER),
    ("code", PREAMBLE),

    # ---------------- B1 ----------------
    ("md", "## Étape B1 — Chargement et contrôles de base (30 min)\n\n"
           "Charger le CSV **avec `parse_dates`** (piège n°1 du TP), vérifier la "
           "forme attendue (69 120 × 10), la liste des 8 serveurs, la période, "
           "la constance du pas d'échantillonnage et l'absence de valeur manquante."),
    ("code", """m = pd.read_csv("data/nosys_metriques.csv", parse_dates=["horodatage"])
print("forme :", m.shape)                      # attendu (69120, 10)
print("serveurs :", sorted(m["serveur"].unique()))
print("période :", m["horodatage"].min(), "->", m["horodatage"].max())
print("colonnes :", list(m.columns))

# pas d'échantillonnage constant ? (premier point = NaT, attendu)
pas = m.sort_values(["serveur", "horodatage"]).groupby("serveur")["horodatage"].diff()
print("pas d'échantillonnage (s) :")
print(pas.value_counts().head())

# valeurs manquantes ?
print("NaN par colonne :", m.isna().sum().sum(), "au total")"""),
    ("md", "**Livrable B1** — le fichier contient bien **69 120 lignes × 10 colonnes**, "
           "8 serveurs (`srv-01` … `srv-08`), du **01/09/2026 00:00 au 30/09/2026 23:55**, "
           "un pas **constant de 300 s (5 min)** pour chaque serveur et **0 valeur manquante**. "
           "Le chargement est validé ; toute analyse ultérieure part de ces contrôles."),

    # ---------------- B2 ----------------
    ("md", "## Étape B2 — Inventaire de qualité (20 min)\n\n"
           "Tableau par colonne (type, min/max, manquants), complétude "
           "(8 640 fenêtres par serveur = 30 j × 288/j) et bornes physiques "
           "(CPU et RAM dans [0, 100], température plausible pour une salle)."),
    ("code", """inv = pd.DataFrame({
    "type": m.dtypes.astype(str),
    "min": m.min(numeric_only=True),
    "max": m.max(numeric_only=True),
    "manquants": m.isna().sum(),
})
display(inv)

# complétude : chaque serveur doit avoir 30 jours x 288 fenêtres = 8 640
n_par_srv = m.groupby("serveur").size()
print("fenêtres par serveur (attendu 8 640) :")
print(n_par_srv.to_string())

# cohérence physique
print("cpu_pct dans [0,100] :", m["cpu_pct"].between(0, 100).all(),
      "| max :", m["cpu_pct"].max())
print("ram_pct dans [0,100] :", m["ram_pct"].between(0, 100).all(),
      "| max :", m["ram_pct"].max())
print("température > 20 °C partout :", (m["temperature_c"] > 20).all(),
      "| max :", m["temperature_c"].max())
print("latence max :", m["latence_ms"].max(), "ms")"""),
    ("md", "**Livrable B2** — l'inventaire est **sain** : 0 valeur manquante sur toutes "
           "les colonnes, chaque serveur possède exactement **8 640 fenêtres**, "
           "CPU max **96,0 %**, RAM dans [0, 100], température max **42,7 °C** "
           "(plausible pour une salle serveur). Aucun outlier de saisie à neutraliser : "
           "on peut l'affirmer **avec la preuve**, pas par intuition."),

    # ---------------- B3 ----------------
    ("md", "## Étape B3 — Visualiser srv-01 sur 30 jours, incidents annotés (30 min)\n\n"
           "Vue journalière (CPU, latence, alarmes) avec les **2 créneaux d'incident "
           "annotés** (9/09 14 h et 22/09 3 h), puis zoom à la fenêtre près. "
           "Attention au piège : `drop(columns=[\"serveur\"])` avant le `resample`."),
    ("code", """s1 = m[m["serveur"] == "srv-01"].set_index("horodatage").drop(columns=["serveur"])
jour = s1.resample("D").mean()                     # 30 lignes, une par jour

incidents = s1[s1["alarme"] == 1].index
print("fenêtres d'alarme srv-01 :", len(incidents))
print("créneaux :", sorted({t.strftime('%d/%m %Hh') for t in incidents}))

fig, ax = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
ax[0].plot(jour.index, jour["cpu_pct"], color="C0")
ax[0].set_ylabel("CPU % (moyen/j)")
ax[1].plot(jour.index, jour["latence_ms"], color="C1")
ax[1].set_ylabel("latence ms (moy./j)")
ax[2].bar(jour.index, jour["alarme"] * 4, width=0.7, color="C3")  # x4 pour visibilité
ax[2].set_ylabel("alarmes (×4)")

# annotation des 2 incidents (identiques sur les 8 serveurs)
for d, lab in [("2026-09-09", "incident 1\\n9/09 14h — 35 min"),
               ("2026-09-22", "incident 2\\n22/09 3h — 50 min")]:
    for a in ax:
        a.axvspan(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(hours=2),
                  color="crimson", alpha=.12)
    ax[0].annotate(lab, xy=(pd.Timestamp(d) + pd.Timedelta(hours=1),
                            jour["cpu_pct"].max()),
                   fontsize=8, color="crimson", ha="center", va="bottom")
ax[0].set_title("srv-01 — 30 jours : CPU, latence et alarmes (incidents annotés)")
plt.tight_layout(); plt.show()"""),
    ("code", """# zoom sur les deux incidents : CPU et latence fenêtre par fenêtre (5 min)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
for a, d in zip(axes, ["2026-09-09", "2026-09-22"]):
    deb = pd.Timestamp(d) - pd.Timedelta(minutes=90)
    fin = pd.Timestamp(d) + pd.Timedelta(hours=2)
    z = s1.loc[deb:fin]
    a.plot(z.index, z["cpu_pct"], label="CPU %")
    a2 = a.twinx()
    a2.plot(z.index, z["latence_ms"], color="C1", label="latence ms")
    a2.set_ylabel("latence ms", color="C1")
    a.axvspan(pd.Timestamp(d) + pd.Timedelta(hours=0),
              pd.Timestamp(d) + pd.Timedelta(hours=2), color="crimson", alpha=.08)
    a.set_title(f"srv-01 — zoom {d}"); a.set_ylabel("CPU %")
plt.tight_layout(); plt.show()

# le seuil fixe « CPU > 90 % » aurait-il détecté les incidents ?
print("fenêtres CPU > 90 % :", (m["cpu_pct"] > 90).sum(),
      "— toutes avec alarme =", m.loc[m['cpu_pct'] > 90, 'alarme'].all())
print("corrélation latence–CPU hors incidents :",
      round(m.loc[m['alarme']==0, 'latence_ms'].corr(m.loc[m['alarme']==0, 'cpu_pct']), 3))"""),
    ("md", "**Livrable B3** — les 2 incidents de srv-01 tombent sur les créneaux "
           "**9/09 à 14 h (7 fenêtres, 35 min)** et **22/09 à 3 h (10 fenêtres, 50 min)**, "
           "identiques sur les 8 serveurs → **cause partagée probable** (batch/déploiement), "
           "pas 16 pannes indépendantes. Le seuil fixe « CPU > 90 % » les aurait détectés "
           "**a posteriori** (136 fenêtres, toutes avec `alarme = 1`), mais il ne dit rien "
           "de l'avenir : la latence monte **en même temps** que le CPU "
           "(corrélation **0,80** hors incidents)."),

    # ---------------- B4 ----------------
    ("md", "## Étape B4 — Statistiques par serveur, consommation, ratio de capacité (40 min)\n\n"
           "Le tableau de classement du parc, le ratio d'utilisation effective "
           "`cpu_moy / (capacité × 100)` et la consommation totale "
           "(piège du lambda : ne pas oublier le pas de 5 min)."),
    ("code", """par_srv = (m.groupby("serveur")
             .agg(cpu_moy=("cpu_pct", "mean"),
                  cpu_max=("cpu_pct", "max"),
                  lat_p99=("latence_ms", lambda s: s.quantile(0.99)),
                  temp_max=("temperature_c", "max"),
                  alarmes=("alarme", "sum"),
                  conso_kwh_tot=("conso_kwh", lambda s: s.sum() * 5 / 60),
                  capacite=("capacite_cpu", "first")))
par_srv["ratio_capacite"] = par_srv["cpu_moy"] / (par_srv["capacite"] * 100)
par_srv = par_srv.sort_values("cpu_moy", ascending=False)
display(par_srv.round(2))

print("CPU moyen du parc : %.1f %%" % m["cpu_pct"].mean())
print("p99 de latence du parc : %.1f ms" % m["latence_ms"].quantile(0.99))
print("consommation totale du parc : %.0f kWh" % par_srv["conso_kwh_tot"].sum())
print("total fenêtres d'alarme :", int(m["alarme"].sum()))"""),
    ("code", """fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ps = par_srv.sort_values("cpu_moy")
ax[0].barh(ps.index, ps["cpu_moy"],
           color=["crimson" if s == "srv-01" else "C0" for s in ps.index])
ax[0].axvline(m["cpu_pct"].mean(), color="k", ls="--", lw=1,
              label="parc %.1f %%" % m["cpu_pct"].mean())
ax[0].set_xlabel("CPU moyen (%)"); ax[0].set_title("CPU moyen par serveur")
ax[0].legend()
pr = par_srv.sort_values("ratio_capacite")
ax[1].barh(pr.index, pr["ratio_capacite"],
           color=["crimson" if s == "srv-01" else "C2" for s in pr.index])
ax[1].set_xlabel("CPU moyen / capacité × 100")
ax[1].set_title("Ratio d'utilisation effective")
plt.tight_layout(); plt.show()"""),
    ("md", "**Livrable B4** — classement : **srv-05 24,6 %** et **srv-01 24,3 %** en tête "
           "de charge, srv-02 le plus calme (**14,4 %**). Le ratio d'utilisation effective "
           "place **srv-01 en premier (capacité 0,83)** : c'est le serveur le plus sollicité "
           "*relativement* à ses moyens. Consommation totale du parc : **11 614 kWh/mois** — dont srv-05 (1 585 kWh) et srv-01 (1 577 kWh) en tête. "
           "Tous les p99 de latence restent ≤ 16,4 ms hors incidents, pour un SLO à 100 ms."),

    # ---------------- B5 ----------------
    ("md", "## Étape B5 — Livrable : la fiche « état des lieux » (20 min + finition)\n\n"
           "**5 constats chiffrés** (chiffre + référence + interprétation), **3 risques SLA** "
           "reliés aux données, **1 recommandation** d'automatisation.\n\n"
           "### Les 5 constats chiffrés\n\n"
           "1. **Le parc est globalement sain** : CPU moyen **20,5 %**, p99 de latence "
           "**16,3 ms** pour un SLO à **100 ms** — une marge de **× 6**.\n"
           "2. **Mais la disponibilité calculée est 99,80 % < SLA 99,9 %** : "
           "les 136 fenêtres d'alarme représentent **11,3 h d'indisponibilité cumulée** "
           "sur le mois.\n"
           "3. **Les 16 incidents suivent 2 créneaux communs à tous les serveurs** "
           "(9/09 à 14 h, 22/09 à 3 h) : cause **partagée** probable (batch de sauvegarde, "
           "déploiement, réseau), pas 16 pannes matérielles indépendantes.\n"
           "4. **Corrélation latence–CPU 0,80 hors incidents** : la charge *explique* la "
           "latence normale — un pic de charge est prévisible, donc la latence aussi "
           "(utile pour la prévision de la séance 4).\n"
           "5. **Consommation totale 11 614 kWh/mois** ; srv-05 (**1 585 kWh**) et "
           "srv-01 (**1 577 kWh**) en tête — les deux serveurs les plus chargés sont "
           "aussi les plus gourmands.\n\n"
           "### 3 risques pour le SLA\n\n"
           "- **Incidents synchrones entre serveurs** : une cause commune frappe les 8 "
           "serveurs à la fois → le SLA du *parc* tombe dès qu'un batch déraille.\n"
           "- **CPU × 3,2 pendant les incidents** (96 % contre 30 % en journée normale) : "
           "srv-01, déjà à 24,3 % de moyenne et au ratio de capacité le plus élevé, "
           "n'a aucune marge d'absorption.\n"
           "- **Corrélation latence–CPU 0,80** : tout pic de charge non anticipé se "
           "traduit immédiatement en latence, seule variable visible de l'utilisateur.\n\n"
           "### 1 recommandation d'automatisation\n\n"
           "Automatiser la **détection précoce d'incident** (prédire « incident ≤ 30 min » "
           "à partir des signaux passés — séances 2 et 3) pour traiter le risque n° 1 : "
           "aujourd'hui l'alarme n'est qu'a posteriori, alors que la montée du CPU est "
           "visible et corrélée à la latence *avant* l'incident."),
    ("code", """# reproductibilité : chiffres clés recalculés pour vérification
verif = {
    "n_lignes": len(m),
    "cpu_moyen_parc": round(m["cpu_pct"].mean(), 1),
    "lat_p99_parc": round(m["latence_ms"].quantile(0.99), 1),
    "n_alarmes": int(m["alarme"].sum()),
    "conso_totale_kwh": round(par_srv["conso_kwh_tot"].sum(), 1),
    "corr_lat_cpu_hors_alarme": round(m.loc[m['alarme']==0, 'latence_ms']
                                      .corr(m.loc[m['alarme']==0, 'cpu_pct']), 3),
}
print(verif)"""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for kind, src in CELLS:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                        else nbf.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": "SDIA Bayes",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    client = NotebookClient(nb, timeout=600, kernel_name=KERNEL,
                            resources={"metadata": {"path": ROOT}},
                            allow_errors=False)
    client.execute()
    name = "TP1_cartographie.ipynb"
    path = os.path.join(ROOT, "notebooks", name)
    nbf.write(nb, path)
    imgs = sum(1 for c in nb.cells for o in c.get("outputs", [])
               if "data" in o and "image/png" in o.get("data", {}))
    ncode = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"[nb] {name} — {len(nb.cells)} cellules ({ncode} code, {imgs} figures) "
          f"{os.path.getsize(path)/1024:.1f} Ko")


if __name__ == "__main__":
    main()
