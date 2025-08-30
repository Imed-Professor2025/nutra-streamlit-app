# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from rdkit import Chem

def clean_sdf(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")
    if os.path.getsize(input_path) == 0:
        raise ValueError(f"Le fichier est vide : {input_path}")

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
    if len(sys.argv) != 3:
        print("Usage: python script5.py <input.sdf> <output_dir>")
        sys.exit(1)

    input_sdf = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Nom fixe pour le fichier final
    output_sdf = os.path.join(output_dir, "actives_clean.sdf")
    n_valid = clean_sdf(input_sdf, output_sdf)

    # Affichage Unicode-safe
    try:
        print(f"[OK] {n_valid} molécules valides sauvegardées dans {output_sdf}")
    except UnicodeEncodeError:
        print(f"[OK] {n_valid} molecules valides sauvegardees dans {output_sdf}")

    # Rapport CSV
    report = pd.DataFrame({
        "Fichier": [os.path.basename(input_sdf)],
        "Molécules valides": [n_valid],
        "Output": [output_sdf]
    })
    report_path = os.path.join(output_dir, "report_script5.csv")
    report.to_csv(report_path, index=False)

    try:
        print(f"[INFO] Rapport sauvegardé : {report_path}")
    except UnicodeEncodeError:
        print(f"[INFO] Rapport sauvegarde : {report_path}")

if __name__ == "__main__":
    main()
