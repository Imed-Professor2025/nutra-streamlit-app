# -*- coding: utf-8 -*-
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter
import streamlit as st
import traceback

def main(input_file, output_dir):
    progress_bar = st.progress(0)
    step_text = st.empty()

    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"❌ Fichier introuvable : {input_file}")

        # === Création du dossier principal pour le script 3 ===
        step3_dir = os.path.join(output_dir, "3_generate_sdf_actives_inactives")
        os.makedirs(step3_dir, exist_ok=True)
        step_text.text("📂 Dossier de sortie créé")
        progress_bar.progress(5)

        # === Charger le fichier Excel ===
        df = pd.read_excel(input_file)
        step_text.text(f"✅ Fichier chargé : {input_file} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
        progress_bar.progress(10)
        st.dataframe(df.head())

        # Vérification des colonnes nécessaires
        required_cols = ["Smiles", "Activity_Class", "Standard Value"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Colonnes manquantes : {missing_cols}")
        step_text.text("✅ Colonnes essentielles présentes")
        progress_bar.progress(20)

        # Conversion Smiles → ROMol
        df["ROMol"] = df["Smiles"].apply(Chem.MolFromSmiles)
        df = df[df["ROMol"].notnull()]
        step_text.text(f"✅ SMILES valides conservés ({len(df)})")
        progress_bar.progress(40)

        # Séparation Actives / Inactives
        df_active = df[df["Activity_Class"].str.strip() == "Active"].copy()
        df_inactive = df[df["Activity_Class"].str.strip() == "Inactive"].copy()

        # Filtrage des valeurs extrêmes
        df_active = df_active[df_active["Standard Value"] <= 20]
        df_inactive = df_inactive[df_inactive["Standard Value"] <= 5000]
        step_text.text(f"✅ Filtrage extrêmes : {len(df_active)} actives, {len(df_inactive)} inactives")
        progress_bar.progress(60)

        # === Fonction pour écrire SDF ===
        def write_sdf(df_sub, output_file):
            writer = SDWriter(output_file)
            for _, row in df_sub.iterrows():
                mol = Chem.AddHs(row["ROMol"])
                for col in df_sub.columns:
                    if col != "ROMol":
                        value = row[col]
                        if pd.notna(value):
                            mol.SetProp(str(col), str(value))
                writer.write(mol)
            writer.close()
            st.success(f"✅ Fichier généré : {output_file} ({len(df_sub)} molécules)")

        # === Sauvegarde SDF ===
        active_sdf = os.path.join(step3_dir, "actives.sdf")
        inactive_sdf = os.path.join(step3_dir, "inactives.sdf")
        write_sdf(df_active, active_sdf)
        write_sdf(df_inactive, inactive_sdf)
        progress_bar.progress(100)
        step_text.text("🎉 Script 3 terminé")

        # Affichage fichiers générés
        st.write("📂 Fichiers générés :")
        st.write([active_sdf, inactive_sdf])

        # Affichage 5 premières lignes de chaque fichier
        for f_path in [active_sdf, inactive_sdf]:
            try:
                with open(f_path, "r", errors="ignore") as f:
                    lines = [next(f) for _ in range(5)]
                st.text(f"--- Contenu brut {os.path.basename(f_path)} ---")
                st.text("".join(lines))
            except Exception as e:
                st.warning(f"⚠️ Impossible d’afficher les 5 premières lignes de {os.path.basename(f_path)} : {e}")

        # Affichage dataframe actives et inactives
        st.write("### Aperçu Actives")
        st.dataframe(df_active.head())
        st.write("### Aperçu Inactives")
        st.dataframe(df_inactive.head())

    except Exception as e:
        st.error(f"❌ ERREUR DÉTECTÉE DANS SCRIPT 3 : {e}")
        st.text(traceback.format_exc())
        progress_bar.progress(0)

# === Permet l'exécution standalone ===
if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_file, output_dir)
