"""Compare les trois fichiers du jeu NOSYS : metriques, metriques_brutes, features."""
import json
import pandas as pd
import numpy as np

D = r"C:\Users\HP\Documents\cours-aiops-GIT-5534\data"
res = {}

for nom in ["nosys_metriques_brutes.csv", "nosys_metriques.csv", "nosys_features.csv"]:
    d = pd.read_csv(f"{D}\\{nom}")
    info = {"lignes": int(len(d)), "colonnes": list(d.columns)}
    if "horodatage" in d.columns:
        h = pd.to_datetime(d["horodatage"])
        info["serveurs"] = sorted(d["serveur"].unique().tolist())
        info["debut"] = str(h.min())
        info["fin"] = str(h.max())
        info["pas_median_min"] = float(
            np.median(np.diff(np.sort(h.unique())).astype("timedelta64[m]").astype(float)))
        info["doublons_exacts"] = int(d.duplicated().sum())
        info["manquants"] = {c: int(d[c].isna().sum()) for c in d.columns if d[c].isna().any()}
        info["alarme_taux"] = round(float(d["alarme"].mean()), 4)
        info["temp_max"] = round(float(d["temperature_c"].max()), 1)
        if "capacite_cpu" in d.columns:
            info["capacite"] = sorted(d["capacite_cpu"].unique().tolist())[:6]
    else:
        info["serveurs"] = sorted(d["serveur"].unique().tolist())
        info["fenetres"] = int(d["fenetre"].nunique())
        info["taux_incident"] = round(float(d["incident_30min"].mean()), 4)
        info["nb_features"] = int(len([c for c in d.columns
                                       if c not in ("serveur", "fenetre", "incident_30min")]))
    res[nom] = info

for nom, i in res.items():
    print("==", nom)
    for k, v in i.items():
        print(f"   {k:18s}: {v}")
    print()

with open(r"C:\Users\HP\Documents\cours-aiops-GIT-5534\figures\compare_nosys.json", "w",
          encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# verification de la filiation : features = agregats de metriques par fenetre de 10 min
m = pd.read_csv(f"{D}\\nosys_metriques.csv", parse_dates=["horodatage"])
f = pd.read_csv(f"{D}\\nosys_features.csv", parse_dates=["fenetre"])
srv = m["serveur"].iloc[0]
fen = f[f["serveur"] == srv].iloc[100]
bloc = m[(m["serveur"] == srv) & (m["horodatage"] >= fen["fenetre"]) &
         (m["horodatage"] < fen["fenetre"] + pd.Timedelta("10min"))]
print("CONTROLE de filiation (serveur", srv, "fenetre", fen["fenetre"], ")")
print("  points de metriques dans la fenetre :", len(bloc))
print("  cpu_moy recalcule :", round(float(bloc["cpu_pct"].mean()), 4),
      "| valeur du fichier :", round(float(fen["cpu_moy"]), 4))
print("  lat_p99 recalcule :", round(float(bloc["latence_ms"].quantile(.99)), 3),
      "| valeur du fichier :", round(float(fen["lat_p99"]), 3))
print("  cpu_max recalcule :", round(float(bloc["cpu_pct"].max()), 4),
      "| valeur du fichier :", round(float(fen["cpu_max"]), 4))
