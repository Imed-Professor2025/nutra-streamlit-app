# -*- coding: utf-8 -*-
import os
import pandas as pd
import traceback
import streamlit as st

def main(uploaded_file, output_dir):
    progress_bar = st.progress(0)
    step_text = st.empty()

    try:
        if uploaded_file is None:
            raise ValueError("❌ Aucun fichier uploadé trouvé. Assurez-vous de lancer le script depuis l'app.")

        # === Création du dossier principal pour le script ===
        script_output_dir = os.path.join(output_dir, "1_filtred_Chembl_Data")
        os.makedirs(script_output_dir, exist_ok=True)
        step_text.text("📂 Dossier de sortie créé")
        progress_bar.progress(5)

        # === Chargement du fichier original ChEMBL ===
        df = pd.read_excel(uploaded_file)
        step_text.text("📥 Fichier ChEMBL chargé")
        progress_bar.progress(10)

        # Vérifier colonnes essentielles
        required_cols = ['Target Name', 'Target Organism', 'Assay Description', 
                         'Molecule ChEMBL ID', 'Smiles', 'Standard Type', 
                         'Standard Relation', 'Standard Value', 'Standard Units', 
                         'pChEMBL Value']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"❌ Colonnes manquantes : {missing_cols}")
        step_text.text("✅ Colonnes vérifiées")
        progress_bar.progress(15)

        # Filtrage cible et organisme
        df = df[df['Target Name'].str.contains('C-X-C chemokine receptor type 3', case=False, na=False)]
        df = df[df['Target Organism'].str.contains('Homo sapiens|human', case=False, na=False)]
        step_text.text("🎯 Filtrage cible et organisme effectué")
        progress_bar.progress(30)

        # Filtrage sur essais pertinents
        keywords = ['CXCL10', 'CXCL9', 'CXCL11', 'displacement', 'binding', 'inhibition', 
                    'activation', 'calcium mobilization', 'FLIPR']
        pattern = '|'.join(keywords)
        df = df[df['Assay Description'].str.contains(pattern, case=False, na=False)]
        step_text.text("🧪 Filtrage sur essais pertinents")
        progress_bar.progress(40)

        # Sélection colonnes utiles
        cols = ['Molecule ChEMBL ID', 'Molecule Name', 'Smiles', 'Standard Type', 'Standard Relation',
                'Standard Value', 'Standard Units', 'pChEMBL Value', 'Assay Description', 'Assay Organism',
                'Assay Type', 'Target Name', 'Target Organism', 'Data Validity Comment',
                'Assay ChEMBL ID', 'Document ChEMBL ID']
        df_filtered = df[cols].copy()

        # Nettoyage des valeurs invalides
        df_filtered = df_filtered[df_filtered['Data Validity Comment'].isnull() | (df_filtered['Data Validity Comment'] == '')]
        step_text.text("🧹 Nettoyage des valeurs invalides")
        progress_bar.progress(50)

        # Sauvegarde fichier intermédiaire 1
        intermediate_file1 = os.path.join(script_output_dir, 'CXCR3_human_inhibition_filtered.xlsx')
        df_filtered.to_excel(intermediate_file1, index=False)
        step_text.text(f"✅ Fichier intermédiaire sauvegardé : {intermediate_file1}")
        progress_bar.progress(60)

        # Nettoyage supplémentaire
        df = df_filtered.drop_duplicates(subset=['Smiles', 'Molecule ChEMBL ID'])
        df = df[df['Standard Type'].str.contains('IC50', case=False, na=False)]
        df = df[df['Standard Value'].notna()]
        df['Standard Value'] = pd.to_numeric(df['Standard Value'], errors='coerce')
        df = df[df['Standard Value'].notna()]

        # Filtrage bornes extrêmes
        borne_inf, borne_sup = 0.1, 100000
        df = df[(df['Standard Value'] >= borne_inf) & (df['Standard Value'] <= borne_sup)]
        step_text.text("📊 Filtrage IC50 effectué")
        progress_bar.progress(75)

        # Sauvegarde fichier intermédiaire 2
        intermediate_file2 = os.path.join(script_output_dir, 'CXCR3_IC50_clean.xlsx')
        df.to_excel(intermediate_file2, index=False)
        step_text.text(f"✅ Nettoyage IC50 terminé et fichier sauvegardé : {intermediate_file2}")
        progress_bar.progress(90)

        # Sélection finale des colonnes
        colonnes_utiles = ['Molecule ChEMBL ID', 'Smiles', 'Standard Type', 'Standard Relation',
                           'Standard Value', 'Standard Units', 'pChEMBL Value']
        final_file = os.path.join(script_output_dir, "donnees_nettoyees_CXCR3.xlsx")
        df[colonnes_utiles].to_excel(final_file, index=False)
        step_text.text(f"✅ Fichier final nettoyé enregistré : {final_file}")
        progress_bar.progress(100)

        st.success(f"🎉 Script terminé. 3 fichiers Excel générés dans : {script_output_dir}")
        st.write("📂 Fichiers générés :")
        st.write([intermediate_file1, intermediate_file2, final_file])

    except Exception as e:
        st.error(f"❌ ERREUR DÉTECTÉE DANS SCRIPT 1 : {e}")
        st.text(traceback.format_exc())
        progress_bar.progress(0)

# === Permet l'exécution standalone ===
if __name__ == "__main__":
    import sys
    uploaded_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(uploaded_file, output_dir)
