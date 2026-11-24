# -*- coding: utf-8 -*-
"""Figures de la Seance 1 (EPD-GIT-5534) -- chiffres calcules depuis data/nosys_metriques.csv.
Ecrit figures/key_numbers_git1.json AVANT les figures lourdes (traçabilite)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data", "nosys_metriques.csv")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#5B6470", "axes.labelcolor": "#10243E",
    "text.color": "#10243E", "xtick.color": "#5B6470", "ytick.color": "#5B6470",
    "axes.titlecolor": "#10243E", "figure.facecolor": "white",
    "axes.grid": True, "grid.color": "#D8DEE6", "grid.linewidth": 0.6,
})
C = {"blue": "#1F6FEB", "deep": "#10243E", "cyan": "#12A594", "orange": "#D97706",
     "red": "#C0392B", "green": "#1E8449", "purple": "#6C5CE7", "grey": "#5B6470"}
DPI = 190

m = pd.read_csv(DATA, parse_dates=["horodatage"])
m = m.sort_values(["serveur", "horodatage"]).reset_index(drop=True)

# ================================================================ key numbers
K = {}
K["n_lignes"] = int(len(m))
K["n_serveurs"] = int(m["serveur"].nunique())
K["serveurs"] = sorted(m["serveur"].unique().tolist())
K["debut"] = str(m["horodatage"].min())
K["fin"] = str(m["horodatage"].max())
K["n_jours"] = float((m["horodatage"].max() - m["horodatage"].min()).total_seconds() / 86400)
pas = m["horodatage"].diff().dropna().value_counts()
K["pas_minutes"] = sorted({round(x.total_seconds() / 60) for x in pas.index})
K["n_alarme"] = int(m["alarme"].sum())
K["alarme_par_serveur"] = m.groupby("serveur")["alarme"].sum().astype(int).to_dict()

g = m.groupby("serveur")
K["cpu_moyen"] = g["cpu_pct"].mean().round(2).to_dict()
K["cpu_max"] = g["cpu_pct"].max().round(1).to_dict()
K["ram_moyen"] = g["ram_pct"].mean().round(1).to_dict()
K["lat_p99"] = g["latence_ms"].quantile(0.99).round(1).to_dict()
K["lat_p99_parc"] = float(m["latence_ms"].quantile(0.99))
K["lat_moy_parc"] = float(m["latence_ms"].mean())
K["lat_max"] = float(m["latence_ms"].max())
K["temp_max"] = float(m["temperature_c"].max())
K["temp_moy"] = float(m["temperature_c"].mean())
K["paquet_p99"] = float(m["paquets_perdus_pct"].quantile(0.99))
K["capacite"] = {s: float(c) for s, c in g["capacite_cpu"].first().items()}
K["manquants_par_col"] = m.isna().sum().to_dict()
# consommation totale du mois (kWh) : conso_kwh semble une puissance moyenne (kW) par fenetre
K["conso_totale_kwh"] = float(round(m["conso_kwh"].sum() * (5 / 60), 1))
K["conso_par_serveur"] = {s: round(v * 5 / 60, 1) for s, v in g["conso_kwh"].sum().items()}

# seuil fixe 90 % : fenetres au-dessus, dont combien ne sont PAS des alarmes declarees
au_dessus = m[m["cpu_pct"] > 90]
K["n_au_dessus_90"] = int(len(au_dessus))
K["n_au_dessus_90_hors_alarme"] = int((au_dessus["alarme"] == 0).sum())
K["n_alarmes_cpu_le_90"] = int(m[(m["alarme"] == 1) & (m["cpu_pct"] <= 90)].shape[0])
# incidents = blocs consecutifs d'alarme par serveur
inc = []
for s, sub in m.groupby("serveur"):
    a = sub["alarme"].values
    starts = np.where((a == 1) & (np.r_[0, a[:-1]] == 0))[0]
    for i in starts:
        j = i
        while j + 1 < len(a) and a[j + 1] == 1:
            j += 1
        inc.append({"serveur": s, "debut": str(sub["horodatage"].iloc[i]),
                    "n_fenetres": int(j - i + 1),
                    "cpu_max": float(sub["cpu_pct"].iloc[i:j + 1].max()),
                    "lat_max": float(sub["latence_ms"].iloc[i:j + 1].max())})
K["n_incidents"] = len(inc)
K["incidents"] = inc
K["duree_incidents_h"] = [round(i["n_fenetres"] * 5 / 60, 2) for i in inc]

# correlation latence vs cpu (hors alarmes puis globalement)
hors = m[m["alarme"] == 0]
K["corr_lat_cpu_global"] = float(round(m["latence_ms"].corr(m["cpu_pct"]), 3))
K["corr_lat_cpu_hors_alarme"] = float(round(hors["latence_ms"].corr(hors["cpu_pct"]), 3))
K["corr_paquet_cpu"] = float(round(m["paquets_perdus_pct"].corr(m["cpu_pct"]), 3))

# saisonnalite : CPU moyen par heure (jour vs nuit)
K["cpu_moyen_parc"] = float(round(m["cpu_pct"].mean(), 1))
K["cpu_moyen_jour"] = float(round(m.loc[m["horodatage"].dt.hour.between(9, 17), "cpu_pct"].mean(), 1))
K["cpu_moyen_nuit"] = float(round(m.loc[m["horodatage"].dt.hour.isin([0, 1, 2, 3, 4, 23]), "cpu_pct"].mean(), 1))

with open(os.path.join(FIG, "key_numbers_git1.json"), "w", encoding="utf-8") as f:
    json.dump(K, f, ensure_ascii=False, indent=1)
print("JSON ecrit :", len(K), "cles")

def sauve(fig, nom):
    fig.savefig(os.path.join(FIG, nom), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", nom)

# ================================================================ f01 profil srv-01
s1 = m[m["serveur"] == "srv-01"].set_index("horodatage").drop(columns=["serveur"])
jour = s1.resample("D").mean()
fig, ax = plt.subplots(3, 1, figsize=(11.2, 5.4), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1.4, 1.2]})
ax[0].plot(jour.index, jour["cpu_pct"], color=C["blue"], lw=1.8, marker="o", ms=3)
ax[0].fill_between(jour.index, jour["cpu_pct"], color=C["blue"], alpha=.10)
for d in (8, 21):
    idx = jour.index[d - 1]
    ax[0].axvline(idx, color=C["red"], lw=1.0, ls="--", alpha=.8)
    ax[0].annotate(f"incident (jour {d})\ncpu max {K['cpu_max']['srv-01']:.0f} %",
                   xy=(idx, jour["cpu_pct"].max()), xytext=(idx + pd.Timedelta(days=2.5),
                   jour["cpu_pct"].max() - 2), fontsize=8, color=C["red"],
                   arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))
ax[0].set_ylabel("CPU moyen du jour (%)")
ax[0].set_title("srv-01 : 30 jours de métriques — le profil « normal » et deux incidents", fontsize=10)
ax[1].plot(jour.index, jour["latence_ms"], color=C["cyan"], lw=1.6, marker="s", ms=2.5)
ax[1].set_ylabel("latence moyenne (ms)")
ax[2].bar(jour.index, jour["alarme"] * 136 / 30, width=0.7, color=C["orange"])
ax[2].set_ylabel("fenêtres\nd'alarme/j")
ax[2].set_ylim(0, 6)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
sauve(fig, "f01_profil_srv01.png")

# ================================================================ f02 saisonnalite horaire
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.2))
mh = m.groupby(m["horodatage"].dt.hour)["cpu_pct"].mean()
mh2 = m.groupby(m["horodatage"].dt.hour)["latence_ms"].mean()
ax[0].bar(mh.index, mh.values, color=C["blue"], alpha=.85)
ax[0].axhline(K["cpu_moyen_parc"] if "cpu_moyen_parc" in K else m["cpu_pct"].mean(),
              color=C["red"], ls="--", lw=1, label="moyenne du parc")
ax[0].set_title("CPU moyen par heure — saisonnalité jour/nuit", fontsize=9.5)
ax[0].set_xlabel("heure de la journée"); ax[0].set_ylabel("CPU (%)"); ax[0].legend(fontsize=8)
ax[1].bar(mh2.index, mh2.values, color=C["cyan"], alpha=.85)
ax[1].set_title("Latence moyenne par heure", fontsize=9.5)
ax[1].set_xlabel("heure de la journée"); ax[1].set_ylabel("latence (ms)")
sauve(fig, "f02_saisonnalite.png")

# ================================================================ f03 classement serveurs
ordre = g["cpu_pct"].mean().sort_values(ascending=False)
p99 = g["latence_ms"].quantile(0.99).reindex(ordre.index)
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.4))
ax[0].barh(list(ordre.index)[::-1], list(ordre.values)[::-1], color=[C["red"] if s == "srv-01" else C["blue"] for s in ordre.index][::-1], alpha=.85)
ax[0].axvline(ordre.mean(), color=C["grey"], ls="--", lw=1)
ax[0].set_title("CPU moyen sur 30 jours (srv-01 le plus chargé)", fontsize=9.5)
ax[0].set_xlabel("CPU (%)")
ax[1].barh(list(p99.index)[::-1], list(p99.values)[::-1], color=C["cyan"], alpha=.85)
ax[1].axvline(100, color=C["red"], ls="--", lw=1.2)
ax[1].text(101, 0.2, "SLO 100 ms", color=C["red"], fontsize=8)
ax[1].set_title("Latence p99 par serveur", fontsize=9.5)
ax[1].set_xlabel("ms")
sauve(fig, "f03_classement.png")

# ================================================================ f04 fenetres d'alarme
fig, ax = plt.subplots(figsize=(11.2, 3.0))
colors = {"srv-01": C["red"], "srv-02": C["blue"], "srv-03": C["cyan"], "srv-04": C["green"],
          "srv-05": C["orange"], "srv-06": C["purple"], "srv-07": C["grey"], "srv-08": "#8C564B"}
for i, s in enumerate(K["serveurs"]):
    sub = m[(m["serveur"] == s) & (m["alarme"] == 1)]
    ax.scatter(sub["horodatage"], [i] * len(sub), s=16, color=colors[s], marker="|", lw=2)
    if len(sub):
        first = sub["horodatage"].iloc[0]
        ax.text(first, i + 0.35, f"{len(sub)} fen.", fontsize=7.5, color=colors[s])
ax.set_yticks(range(8)); ax.set_yticklabels(K["serveurs"], fontsize=8)
ax.set_ylim(-0.6, 8.2)
ax.set_title("Les 136 fenêtres d'alarme du mois — 2 blocs de ~85 min par serveur", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
sauve(fig, "f04_alarmes.png")

# ================================================================ f05 seuil fixe vs distribution
fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.2))
ax[0].hist(m["cpu_pct"], bins=80, color=C["blue"], alpha=.75)
ax[0].axvline(90, color=C["red"], lw=1.6, ls="--")
ax[0].text(90.5, ax[0].get_ylim()[1] * 0.9, f"seuil fixe 90 %\n→ {K['n_au_dessus_90']} fen. dont "
           f"{K['n_au_dessus_90_hors_alarme']} hors incident", fontsize=8, color=C["red"])
ax[0].set_title("Distribution de CPU du parc (69 120 fenêtres)", fontsize=9.5)
ax[0].set_xlabel("CPU (%)"); ax[0].set_ylabel("nb de fenêtres")
x = m["cpu_pct"].values; y = m["latence_ms"].values
idx = m["alarme"].values.astype(bool)
ax[1].scatter(x[~idx], y[~idx], s=2, color=C["grey"], alpha=.30, label="normal")
ax[1].scatter(x[idx], y[idx], s=5, color=C["red"], alpha=.85, label="incident (alarme)")
ax[1].set_title("Latence vs CPU — corrélation globale r = %.2f" % K["corr_lat_cpu_global"], fontsize=9.5)
ax[1].set_xlabel("CPU (%)"); ax[1].set_ylabel("latence (ms)"); ax[1].legend(fontsize=8, loc="upper left")
sauve(fig, "f05_seuil_vs_scatter.png")

# ================================================================ f06 echelle de disponibilite
niv = [99, 99.9, 99.99, 99.999]
labels = ["99 %", "99,9 %", "99,99 %", "99,999 %"]
mins = [(100 - n) / 100 * 365 * 24 * 60 for n in niv]
fig, ax = plt.subplots(figsize=(9.6, 3.0))
b = ax.bar(labels, mins, color=[C["red"], C["orange"], C["cyan"], C["green"]], alpha=.88)
for r, v in zip(b, mins):
    ax.text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:,.0f} min\n≈ {v/60/24:.1f} j".replace(",", " "),
            ha="center", fontsize=8.5)
ax.set_yscale("log"); ax.set_ylim(1, 1e6)
ax.set_title("Coût d'un « neuf » de plus : indisponibilité annuelle autorisée (échelle log)", fontsize=10)
ax.set_ylabel("minutes/an")
sauve(fig, "f06_sla.png")

# ================================================================ f07 vue journaliere srv-01 zoom jour 8
d0 = s1.index[0]
j8 = s1.loc[d0 + pd.Timedelta(days=7): d0 + pd.Timedelta(days=7) + pd.Timedelta(hours=23, minutes=59)]
fig, ax = plt.subplots(figsize=(11.2, 3.2))
ax.plot(j8.index, j8["cpu_pct"], color=C["blue"], lw=1.6, label="CPU %")
ax.plot(j8.index, j8["latence_ms"] * 3, color=C["cyan"], lw=1.4, label="latence (ms) ×3 (visualisation)")
al = j8[j8["alarme"] == 1]
ax.fill_between(al.index, 0, 100, color=C["red"], alpha=.14, label="alarme déclarée")
ax.axhline(90, color=C["orange"], ls="--", lw=1)
ax.text(j8.index[0], 92, "seuil fixe 90 %", color=C["orange"], fontsize=8)
ax.set_ylim(0, 105)
ax.set_title("Zoom sur l'incident du jour 8 (srv-01) : CPU ×4,7, latence ×6,3 pendant ~85 min", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
sauve(fig, "f07_zoom_incident.png")

print("TERMINE -- figures seance 1")
print(json.dumps({k: K[k] for k in ["n_lignes", "n_alarme", "n_incidents", "n_au_dessus_90",
      "n_au_dessus_90_hors_alarme", "corr_lat_cpu_global", "cpu_moyen_jour", "cpu_moyen_nuit",
      "conso_totale_kwh", "lat_p99_parc", "pas_minutes", "n_jours"]}, ensure_ascii=False, indent=1))
