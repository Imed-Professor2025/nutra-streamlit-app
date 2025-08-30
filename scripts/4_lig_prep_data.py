# script4.py
# -*- coding: utf-8 -*-
import os
import sys
from rdkit import Chem
import pandas as pd

def clean_sdf(input_path, output_path):
    """
    Nettoyage d'un fichier SDF : suppression des molécules invalides,
    recalcul des SMILES, et sauvegarde du SDF nettoyé.
    """
    suppl = Chem.SDMolSupplier(input_path, sanitize=False)
    writer = Chem.SDWriter(output_path)

    valid_count = 0
    for mol in suppl:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            writer.write(mol)
            valid_count += 1
        except Exception:
            continue
    writer.close()
    return valid_count

def main():
    # Forcer UTF-8 pour la console sur Windows
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    if len(sys.argv) != 4:
        print("[ERROR] Usage: python script4.py <actives.sdf> <inactives.sdf> <output_dir>")
        sys.exit(1)

    actives_file, inactives_file, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    # Assurer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Nettoyer actives
    actives_out = os.path.join(output_dir, "actives_clean.sdf")
    n_actives = clean_sdf(actives_file, actives_out)
    print(f"[OK] {n_actives} molécules valides sauvegardées dans {actives_out}")

    # Nettoyer inactives
    inactives_out = os.path.join(output_dir, "inactives_clean.sdf")
    n_inactives = clean_sdf(inactives_file, inactives_out)
    print(f"[OK] {n_inactives} molécules valides sauvegardées dans {inactives_out}")

    # Sauvegarder un petit rapport
    report = pd.DataFrame({
        "Fichier": ["Actives", "Inactives"],
        "Molécules valides": [n_actives, n_inactives],
        "Output": [actives_out, inactives_out]
    })
    report_path = os.path.join(output_dir, "report_script4.csv")
    report.to_csv(report_path, index=False)
    print(f"[INFO] Rapport sauvegardé : {report_path}")

if __name__ == "__main__":
    main()
