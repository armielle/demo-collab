# -*- coding: utf-8 -*-
"""Genere les deux jeux de donnees du dossier NOSYS (support EPD-GIT-5534).

Sorties :
  data/nosys_metriques.csv  - 8 serveurs, 30 jours a pas de 5 min (6 912 lignes/serveur)
  data/nosys_logs.csv       - 100 000 lignes syslog/auth sur 7 jours
  data/nosys_data.json      - provenance, graine, statistiques de controle
Toutes les valeurs proviennent de ce script (graine 5534) : aucune saisie a la main.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

GRAINE = 5534
rng = np.random.default_rng(GRAINE)
racine = Path(__file__).resolve().parents[1]
dossier = racine / "data"

# ---------------------------------------------------------------- metriques
N_SERVEURS = 8
PAS_MIN = 5                     # pas d'echantillonnage
N_POINTS = 30 * 24 * 60 // PAS_MIN   # 8 640 points par serveur
t0 = pd.Timestamp("2026-09-01 00:00")
horodatage = pd.date_range(t0, periods=N_POINTS, freq=f"{PAS_MIN}min")

serveurs = []
for i in range(N_SERVEURS):
    base_cpu = rng.uniform(25, 55)          # charge moyenne du serveur
    capac = rng.uniform(0.6, 1.0)           # CPU max (coeurs normalises)
    srv = f"srv-{i+1:02d}"

    minutes = np.arange(N_POINTS) * PAS_MIN
    heure = (minutes // 60) % 24
    jour_sem = ((minutes // 1440) + 0) % 7   # 0 = lundi (t0 = lundi)

    # saisonnalite jour/nuit + creux week-end + bruit AR(1)
    profil = 0.55 + 0.35 * np.sin((heure - 6) / 24 * 2 * np.pi)
    profil *= np.where(jour_sem < 5, 1.0, 0.45)
    bruit = np.zeros(N_POINTS)
    for k in range(1, N_POINTS):            # autocorrelation realiste
        bruit[k] = 0.92 * bruit[k - 1] + rng.normal(0, 0.9)
    cpu = base_cpu * profil + bruit
    cpu = np.clip(cpu, 2.0, 99.0)

    # 2 incidents planifies par serveur : pic CPU + latence, jour 8 et 21
    cpu_pic = cpu.copy()
    lat_ms = 8 + 6 * profil + np.abs(rng.normal(0, 1.5, N_POINTS))
    incidents = []
    for jour, h_debut, duree in [(8, 14, 35), (21, 3, 50)]:
        idx = (jour * 288) + int(h_debut * 60 / PAS_MIN)
        largeur = duree // PAS_MIN
        cpu_pic[idx:idx + largeur] = np.minimum(
            97.0, np.maximum(cpu_pic[idx:idx + largeur] + rng.uniform(28, 42),
                             rng.uniform(92, 96, largeur)))
        lat_ms[idx:idx + largeur] += rng.uniform(60, 140)
        incidents.append({"serveur": srv,
                          "debut": str(horodatage[idx]),
                          "duree_min": duree})

    ram = 35 + 30 * profil + rng.normal(0, 4, N_POINTS).cumsum() * 0.02
    ram = np.clip(ram, 20, 92)
    perte = np.clip(0.02 + 0.01 * profil + rng.exponential(0.03, N_POINTS), 0, 2.5)
    perte[cpu_pic > 90] += rng.uniform(0.2, 1.2, (cpu_pic > 90).sum())
    temp = 28 + 0.22 * cpu + rng.normal(0, 1.2, N_POINTS)
    conso_kwh = 1.1 + 0.045 * cpu + rng.normal(0, 0.08, N_POINTS)
    # alarmes : CPU > 90 % pendant une fenetre => alarme
    alarme = (cpu_pic > 90).astype(int)

    serveurs.append(pd.DataFrame({
        "horodatage": horodatage,
        "serveur": srv,
        "cpu_pct": np.round(cpu_pic, 2),
        "ram_pct": np.round(ram, 2),
        "latence_ms": np.round(lat_ms, 2),
        "paquets_perdus_pct": np.round(perte, 3),
        "temperature_c": np.round(temp, 2),
        "conso_kwh": np.round(conso_kwh, 3),
        "capacite_cpu": capac,
        "alarme": alarme,
    }))
    print(f"[metriques] {srv} : pic CPU {cpu_pic.max():.1f} %, "
          f"{alarme.sum()} fenetres en alarme, 2 incidents")

metriques = pd.concat(serveurs, ignore_index=True)
f_metrics = dossier / "nosys_metriques.csv"
metriques.to_csv(f_metrics, index=False)
print(f"[metriques] {len(metriques):,} lignes -> {f_metrics.name}")

# ---------------------------------------------------------------- logs
N_LOGS = 100_000
jours = 7
t_log = pd.date_range("2026-10-01", periods=N_LOGS, freq=f"{7*24*3600/N_LOGS:.1f}s")
machines = np.array([f"srv-{i+1:02d}" for i in range(N_SERVEURS)] + ["fw-edge", "bdd-principal"])
niveaux = np.array(["INFO", "WARNING", "ERROR", "CRITICAL"])
messages = {
    "INFO":    ["Connexion SSH acceptee", "Sauvegarde terminee", "Service sain",
                "Rotation des logs", "Requete traitée en {d} ms"],
    "WARNING": ["Latence elevee detectee", "Espace disque a 80 %", "Retry automatique",
                "Certificat expire dans 14 jours"],
    "ERROR":   ["Echec de connexion a la base", "Timeout de service",
                "Echec d'ecriture disque", "Processus redemarre"],
    "CRITICAL": ["Service indisponible", "Perte de quorum", "Alarme matériel déclenchée"],
}
utilisateurs = np.array(["svc-backup", "admin", "deploy", "audit", "dev1", "dev2",
                         "ops1", "inconnu-1", "inconnu-2", "root"])
proba_user = np.array([0.30, 0.12, 0.12, 0.08, 0.10, 0.10, 0.10, 0.04, 0.02, 0.02])

niv = rng.choice(niveaux, N_LOGS, p=[0.72, 0.13, 0.11, 0.04])
machine = rng.choice(machines, N_LOGS, p=np.full(len(machines), 1 / len(machines)))
user = rng.choice(utilisateurs, N_LOGS, p=proba_user)
msg = [rng.choice(messages[n]) for n in niv]
msg = [m.replace("{d}", str(int(rng.uniform(30, 900)))) for m in msg]

logs = pd.DataFrame({
    "horodatage": t_log,
    "machine": machine,
    "niveau": niv,
    "utilisateur": user,
    "message": msg,
    "auth_reussie": np.where(
        (user == "root") | (user == "admin"),
        rng.random(N_LOGS) > 0.15,          # comptes privilegies : souvent cibles
        rng.random(N_LOGS) > 0.03),
})
# intrusion simulee : brute force sur bdd-principal, nuit du jour 4 (2 h, tous comptes)
heure_arr = logs["horodatage"].dt.hour
attaque = (logs["machine"] == "bdd-principal") & (logs["horodatage"].dt.day == 5) \
    & (heure_arr >= 2) & (heure_arr < 4)
logs.loc[attaque, "auth_reussie"] = False
logs.loc[attaque, "niveau"] = "ERROR"
logs.loc[attaque, "message"] = "Echec d'authentification (mot de passe invalide)"
logs.loc[attaque, "utilisateur"] = rng.choice(["root", "admin", "inconnu-1", "inconnu-2"],
                                              attaque.sum())
n_att = int(attaque.sum())
print(f"[logs] {N_LOGS:,} lignes, intrusion simulee : {n_att} echecs concentres")

f_logs = dossier / "nosys_logs.csv"
logs.to_csv(f_logs, index=False)

# ---------------------------------------------------------------- controle
stats = {
    "provenance": "src/make_nosys_data.py (graine 5534)",
    "cours": "EPD-GIT-5534 - Apprentissage automatique applique aux infrastructures",
    "metriques": {
        "fichier": f_metrics.name,
        "lignes": len(metriques),
        "serveurs": N_SERVEURS,
        "pas_min": PAS_MIN,
        "periode": [str(horodatage[0]), str(horodatage[-1])],
        "fenetres_alarme": int(metriques["alarme"].sum()),
        "cpu_moyen": float(metriques["cpu_pct"].mean()),
        "latence_p99": float(metriques["latence_ms"].quantile(0.99)),
        "incidents_planifies": sum(len(s) for s in [incidents]),
    },
    "logs": {
        "fichier": f_logs.name,
        "lignes": N_LOGS,
        "periode": [str(t_log[0]), str(t_log[-1])],
        "echecs_auth": int((~logs["auth_reussie"]).sum()),
        "lignes_intrusion": n_att,
        "niveaux": logs["niveau"].value_counts().to_dict(),
    },
    "incidents_planifies": incidents,
}
with open(dossier / "nosys_data.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
print("[controle] data/nosys_data.json ecrit")
