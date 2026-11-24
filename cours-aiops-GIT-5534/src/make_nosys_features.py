# -*- coding: utf-8 -*-
"""Construit data/nosys_features.csv : le dataset supervise du TP2 (corrige officiel).
Une ligne = (serveur, fenetre de 10 min). Signaux sur le PASSE de la fenetre,
etiquette 'incident <= 30 min' sur le FUTUR uniquement (colonne alarme).
Graine 5534, aucune valeur saisie a la main."""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = pd.read_csv(os.path.join(ROOT, "data", "nosys_metriques.csv"),
                parse_dates=["horodatage"]).sort_values(["serveur", "horodatage"])

# 1. signaux par fenetre de 10 min (passe de la fenetre uniquement)
f = (m.set_index("horodatage").groupby("serveur")
     .resample("10min")
     .agg(cpu_moy=("cpu_pct", "mean"), cpu_max=("cpu_pct", "max"),
          ram_moy=("ram_pct", "mean"),
          lat_moy=("latence_ms", "mean"),
          lat_p99=("latence_ms", lambda s: s.quantile(0.99)),
          perte_moy=("paquets_perdus_pct", "mean"),
          temp_moy=("temperature_c", "mean"),
          conso_moy=("conso_kwh", "mean"),
          capacite=("capacite_cpu", "first")))
f["cpu_delta"] = f.groupby(level=0)["cpu_moy"].diff()
f["cpu_amplitude"] = f["cpu_max"] - f["cpu_moy"]
h = f.index.get_level_values("horodatage").hour
f["h_sin"] = np.sin(2 * np.pi * h / 24)
f["h_cos"] = np.cos(2 * np.pi * h / 24)
jr = f.index.get_level_values("horodatage").dayofweek
f["js_sin"] = np.sin(2 * np.pi * jr / 7)
f["js_cos"] = np.cos(2 * np.pi * jr / 7)

# 2. etiquette : alarme presente dans la fenetre courante ou les 3 suivantes
al = (m.assign(fen=m["horodatage"].dt.floor("10min"))
      .groupby(["serveur", "fen"])["alarme"].max().unstack(0).fillna(0).astype(int))
fut = al.copy()
for k in (1, 2, 3):
    fut = fut | al.shift(-k, fill_value=0)
fut = fut.stack()
fut.index = fut.index.set_names(["horodatage", "serveur"])
fut = fut.reorder_levels(["serveur", "horodatage"]).rename("incident_30min")
data = f.join(fut).dropna(subset=["incident_30min"])
data["incident_30min"] = data["incident_30min"].astype(int)
data = data.reset_index().rename(columns={"horodatage": "fenetre"})
data.to_csv(os.path.join(ROOT, "data", "nosys_features.csv"), index=False)

# 3. chiffres de controle
n_pos = int(data["incident_30min"].sum())
K = {"lignes": int(len(data)), "positifs": n_pos,
     "taux_positifs_pct": round(100 * n_pos / len(data), 3),
     "colonnes": list(data.columns),
     "serveurs": int(data["serveur"].nunique()),
     "split_train_jours": "21 premiers jours (jusqu'au 21/09 inclus)",
     "split_test_jours": "9 derniers jours (22-30/09, contient l'incident du 22)",
     "train_lignes": int((data["fenetre"] < "2026-09-22").sum()),
     "test_lignes": int((data["fenetre"] >= "2026-09-22").sum()),
     "train_positifs": int(data.loc[data["fenetre"] < "2026-09-22", "incident_30min"].sum()),
     "test_positifs": int(data.loc[data["fenetre"] >= "2026-09-22", "incident_30min"].sum())}
with open(os.path.join(ROOT, "figures", "key_numbers_features.json"), "w") as fh:
    json.dump(K, fh, indent=1)
print(json.dumps(K, indent=1))
