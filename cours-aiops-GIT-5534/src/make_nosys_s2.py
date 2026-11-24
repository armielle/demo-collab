# -*- coding: utf-8 -*-
"""Seance 2 -- generation du fichier 'brut' avec defauts semes (reproductible).

Le dossier NOSYS de la seance 1 est PROPRE (0 trou). Pour enseigner la
preparation de donnees, on part d'un fichier d'acquisition 'brute' contenant
les defauts typiques d'une collecte reelle :
  - trous d'echantillonnage (agent redemarre),
  - valeurs manquantes isolees (capteur flottant),
  - valeurs aberrantes (pic capteur : temperature=1800 C),
  - lignes dupliquees (retentile reseau),
  - derive de capteur sur un serveur (drift +0.02 C/5min),
  - granularite mixte (un serveur releve toutes les 15 min).

Sorties : data/nosys_metriques_brutes.csv + figures/defauts_s2.json
Graine fixee : 5534. Aucune valeur citee dans les slides ne doit etre saisie
a la main : tout vient de ce script ou de fig_git2.py.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(5534)

m = pd.read_csv(RACINE / "data" / "nosys_metriques.csv", parse_dates=["horodatage"])
n0 = len(m)
defauts = {}

# --- 1. Granularite mixte : srv-07 releve toutes les 15 min ---------------
s07 = m["serveur"] == "srv-07"
m15 = m[s07 & (m["horodatage"].dt.minute % 15 == 0)].copy()
m = pd.concat([m[~s07], m15], ignore_index=True)
defauts["srv07_points_15min"] = int(len(m15))

# --- 2. Trous d'echantillonnage : 6 redemarrages d'agent ------------------
trous = []
grille = pd.date_range(m["horodatage"].min(), m["horodatage"].max(), freq="5min")
for serveur in ["srv-01", "srv-02", "srv-04", "srv-05", "srv-08", "srv-06"]:
    t0 = grille[rng.integers(len(grille) // 3, 2 * len(grille) // 3)]
    duree = int(rng.integers(3, 8)) * 5          # 15 a 40 min
    t1 = t0 + pd.Timedelta(minutes=duree)
    masque = (m["serveur"] == serveur) & (m["horodatage"] >= t0) & (m["horodatage"] <= t1)
    trous.append({"serveur": serveur, "debut": str(t0), "fin": str(t1),
                  "points_supprimes": int(masque.sum())})
    m = m[~masque]
defauts["trous"] = trous
defauts["points_tous"] = int(n0)
defauts["points_apres_trous"] = int(len(m))

# --- 3. Valeurs manquantes isolees : 1.8 % des cellules numeriques --------
cols_num = ["cpu_pct", "ram_pct", "latence_ms", "paquets_perdus_pct", "temperature_c"]
idx = rng.choice(len(m), size=int(0.018 * len(m) * len(cols_num)), replace=False)
col_choisie = rng.choice(cols_num, size=len(idx))
lignes, colonnes = idx // len(cols_num), col_choisie
avant = m[cols_num].isna().sum().sum()
for l, c in zip(lignes, colonnes):
    m.iloc[l, m.columns.get_loc(c)] = np.nan
defauts["nan_ajoutes"] = int(m[cols_num].isna().sum().sum() - avant)
defauts["nan_par_colonne"] = {c: int(m[c].isna().sum()) for c in cols_num}

# --- 4. Valeurs aberrantes : 12 pics de capteur ---------------------------
pics = []
for _ in range(12):
    l = int(rng.integers(0, len(m)))
    c = rng.choice(["temperature_c", "cpu_pct", "latence_ms"])
    val = float(m.iloc[l][c])
    if c == "temperature_c":
        m.iat[l, m.columns.get_loc(c)] = val * 20 + 900      # capteur decale
    elif c == "cpu_pct":
        m.iat[l, m.columns.get_loc(c)] = 350.0 + float(rng.random()) * 100
    else:
        m.iat[l, m.columns.get_loc(c)] = val * 40 + 5000.0
    pics.append({"ligne_absolue": l, "colonne": c, "valeur": float(m.iloc[l][c])})
defauts["pics"] = pics

# --- 5. Lignes dupliquees : retentile reseau ------------------------------
dup_idx = rng.choice(len(m), size=200, replace=False)
dup = m.iloc[dup_idx].copy()
m = pd.concat([m, dup], ignore_index=True)
m = m.sample(frac=1.0, random_state=5534).reset_index(drop=True)  # ordre brule
defauts["doublons_ajoutes"] = 200

# --- 6. Derive de capteur : srv-06, +0.01 C/heure (capteur vieillissant) ---
s10 = m["serveur"] == "srv-06"
s10 = s10 & ~s10.index.isin([p["ligne_absolue"] for p in pics if False])  # noqa
heures = (m["horodatage"] - m["horodatage"].min()).dt.total_seconds() / 3600
m.loc[s10 & m["temperature_c"].notna(), "temperature_c"] += 0.01 * heures[s10]
defauts["derive_srv06"] = "temperature +0.01 C/heure (capteur vieillissant)"

m.to_csv(RACINE / "data" / "nosys_metriques_brutes.csv", index=False)

# Resume chiffre pour le JSON
resume = {
    "lignes_brutes": int(len(m)),
    "doublons_exactes": int(m.duplicated().sum()),
    "nan_total": int(m[cols_num].isna().sum().sum()),
    "nan_pct": round(100 * float(m[cols_num].isna().sum().sum())
                     / (len(m) * len(cols_num)), 3),
    "serveurs": sorted(m["serveur"].unique().tolist()),
    **defauts,
}
(RACINE / "figures").mkdir(exist_ok=True)
(RACINE / "figures" / "defauts_s2.json").write_text(
    json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in resume.items() if k != "pics"}, indent=2,
                 ensure_ascii=False))
