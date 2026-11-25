"""Construit le jeu de donnees reel du TP du chapitre 3 : prix de logements, n = 200.

Source : DVF (Demandes de valeurs foncieres), DGFiP / Etalab, open data.
    https://files.data.gouv.fr/geo-dvf/latest/csv/2024/departements/75.csv.gz
    (fichier geo-dvf, millesime 2024, departement 75 -- Paris ; 1,8 Mo gz)

On ne garde que des ventes reelles de logements : nature_mutation = Vente,
type_local dans {Appartement, Maison}, surface et prix renseignes et plausibles.
Echantillon de 200 lignes a graine fixe (reproductible).

Sorties :
    data/logements_paris_200.csv     (surface_m2, prix_euros, prix_m2, ...)
    data/logements_paris_200.json    (provenance + statistiques de controle)
"""
import csv, gzip, json, os, sys, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
DVFDIR = os.path.join(DATA, "dvf")
URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/2024/departements/75.csv.gz"
GZ = os.path.join(DVFDIR, "dvf75_2024.csv.gz")
N = 200
GRAINE = 4112

# bornes de plausibilite (logements reels, hors erreurs de saisie et hors parkings)
SURF_MIN, SURF_MAX = 9.0, 400.0
PRIX_MIN, PRIX_MAX = 20_000.0, 8_000_000.0
SEUIL_M2 = 15_000.0      # au-dela : vente atypique (immeuble entier, part de societe)


def telecharge():
    os.makedirs(DVFDIR, exist_ok=True)
    if os.path.exists(GZ) and os.path.getsize(GZ) > 100_000:
        print(f"  fichier deja present : {os.path.getsize(GZ)} octets")
        return
    print(f"  telechargement : {URL}")
    urllib.request.urlretrieve(URL, GZ)
    print(f"  recu : {os.path.getsize(GZ)} octets")


def charge():
    """Une ligne par mutation : le logement principal (plus grande surface)."""
    par_mutation = {}
    with gzip.open(GZ, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            if d["nature_mutation"] != "Vente" or d["type_local"] not in ("Appartement", "Maison"):
                continue
            try:
                surf = float(d["surface_reelle_bati"] or 0)
                prix = float(d["valeur_fonciere"] or 0)
            except ValueError:
                continue
            if not (SURF_MIN <= surf <= SURF_MAX and PRIX_MIN <= prix <= PRIX_MAX):
                continue
            if prix / surf > SEUIL_M2 or prix / surf < 500:
                continue
            cle = d["id_mutation"]
            ancien = par_mutation.get(cle)
            if ancien is None or surf > ancien["surface_m2"]:
                par_mutation[cle] = {
                    "surface_m2": round(surf, 1),
                    "prix_euros": round(prix, 0),
                    "prix_m2": round(prix / surf, 0),
                    "pieces": int(d["nombre_pieces_principales"] or 0),
                    "type_local": d["type_local"],
                    "arrondissement": d["code_postal"],
                    "date_mutation": d["date_mutation"],
                    "latitude": round(float(d["latitude"]), 5) if d["latitude"] else None,
                    "longitude": round(float(d["longitude"]), 5) if d["longitude"] else None,
                }
    return list(par_mutation.values())


def main():
    import random
    print("=== jeu de donnees du TP (chapitre 3) ===")
    telecharge()
    lignes = charge()
    print(f"  ventes de logements exploitables : {len(lignes)}")
    if len(lignes) < N:
        sys.exit("  erreur : pas assez de ventes exploitables")
    random.Random(GRAINE).shuffle(lignes)
    ech = sorted(lignes[:N], key=lambda d: d["surface_m2"])

    csv_out = os.path.join(DATA, "logements_paris_200.csv")
    champs = ["surface_m2", "prix_euros", "prix_m2", "pieces", "type_local",
              "arrondissement", "date_mutation", "latitude", "longitude"]
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=champs)
        w.writeheader()
        w.writerows(ech)

    surf = [d["surface_m2"] for d in ech]
    prix = [d["prix_euros"] for d in ech]
    pm2 = [d["prix_m2"] for d in ech]
    import statistics as st
    pente = st.fmean(pm2)
    stats = {
        "source": URL,
        "producteur": "DGFiP / Etalab -- DVF (Demandes de valeurs foncieres), geo-dvf 2024, departement 75",
        "licence": "Licence Ouverte / Open Licence 2.0",
        "n": len(ech),
        "graine_echantillon": GRAINE,
        "surface_m2": {"min": min(surf), "max": max(surf), "moyenne": round(st.fmean(surf), 1)},
        "prix_euros": {"min": min(prix), "max": max(prix), "moyenne": round(st.fmean(prix), 0)},
        "prix_m2": {"min": min(pm2), "max": max(pm2), "moyenne": round(st.fmean(pm2), 0),
                    "median": round(st.median(pm2), 0)},
        "ventes_totales_exploitables": len(lignes),
        "note": ("Fichier construit par src/make_dataset_logements.py. "
                 "Les prix sont les valeurs foncieres reelles declarees aux mutations."),
    }
    json.dump(stats, open(os.path.join(DATA, "logements_paris_200.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    print(f"  ecrit : {csv_out}")
    print(f"  surface  : {min(surf):.0f} a {max(surf):.0f} m2 (moyenne {st.fmean(surf):.0f})")
    print(f"  prix     : {min(prix):,.0f} a {max(prix):,.0f} EUR".replace(",", " "))
    print(f"  prix/m2  : {min(pm2):,.0f} a {max(pm2):,.0f} EUR (median {st.median(pm2):,.0f})".replace(",", " "))


if __name__ == "__main__":
    main()
