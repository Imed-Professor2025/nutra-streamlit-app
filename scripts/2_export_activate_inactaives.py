# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import traceback

def main(input_file, output_dir):
    progress_bar = st.progress(0)
    step_text = st.empty()

    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"❌ Fichier introuvable : {input_file}")

        # === Création du dossier principal pour le script 2 ===
        step2_dir = os.path.join(output_dir, "2_export_activate_inactives")
        os.makedirs(step2_dir, exist_ok=True)
        step_text.text("📂 Dossier de sortie créé")
        progress_bar.progress(5)

        # === Étape 1 : Charger le fichier nettoyé ===
        df = pd.read_excel(input_file)
        step_text.text(f"✅ Fichier chargé : {input_file} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
        progress_bar.progress(10)
        st.dataframe(df.head())

        # Colonnes utiles
        cols = ['Molecule ChEMBL ID', 'Smiles', 'Standard Type',
                'Standard Relation', 'Standard Value',
                'Standard Units', 'pChEMBL Value']
        df = df[cols].dropna()
        step_text.text("✅ Colonnes essentielles conservées")
        progress_bar.progress(20)

        # === Étape 2 : Sauvegarde intermédiaire ===
        intermediate_file = os.path.join(step2_dir, "CXCR3_etape1_nettoye.xlsx")
        df.to_excel(intermediate_file, index=False)
        step_text.text(f"✅ Fichier intermédiaire sauvegardé : {intermediate_file}")
        progress_bar.progress(40)

        # === Étape 3 : Histogramme pChEMBL ===
        plt.figure(figsize=(8,5))
        df['pChEMBL Value'].hist(bins=40)
        plt.xlabel("pChEMBL Value")
        plt.ylabel("Nombre de molécules")
        plt.title("Distribution des pChEMBL Value")
        plt.grid(True)
        plt.tight_layout()
        hist_file = os.path.join(step2_dir, "pChEMBL_distribution.png")
        plt.savefig(hist_file)
        plt.close()
        step_text.text(f"📊 Histogramme pChEMBL généré et sauvegardé : {hist_file}")
        progress_bar.progress(60)

        # === Affichage PNG dans Streamlit ===
        if os.path.exists(hist_file):
            st.image(hist_file, caption="Distribution pChEMBL", use_column_width=True)
        else:
            st.warning("⚠️ L'histogramme n'a pas pu être généré.")

        # === Étape 4 : Classification de la bioactivité ===
        df['Standard Value'] = pd.to_numeric(df['Standard Value'], errors='coerce')
        df = df[df['Standard Value'].notna()]

        def classify_activity(val):
            if val < 100:
                return "Active"
            elif val > 1000:
                return "Inactive"
            else:
                return "Intermediate"

        df['Activity_Class'] = df['Standard Value'].apply(classify_activity)
        step_text.text("🎯 Bioactivité classifiée (Active / Inactive / Intermediate)")
        progress_bar.progress(75)

        # Ne garder que Active et Inactive
        df_filtered = df[df['Activity_Class'].isin(['Active', 'Inactive'])].copy()

        # === Étape 5 : Sauvegarde fichier final ===
        final_file = os.path.join(step2_dir, "bioactivity_classified.xlsx")
        df_filtered.to_excel(final_file, index=False)
        step_text.text(f"✅ Fichier final sauvegardé : {final_file}")
        progress_bar.progress(100)

        st.success(f"🎉 Script terminé. 2 fichiers Excel et 1 histogramme générés dans : {step2_dir}")
        st.write("📂 Fichiers générés :")
        st.write([intermediate_file, final_file, hist_file])
        st.dataframe(df_filtered.head())

    except Exception as e:
        st.error(f"❌ ERREUR DÉTECTÉE DANS SCRIPT 2 : {e}")
        st.text(traceback.format_exc())
        progress_bar.progress(0)

# === Permet l'exécution standalone ===
if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]  # chemin vers donnees_nettoyees_CXCR3.xlsx
    output_dir = sys.argv[2]  # dossier output
    main(input_file, output_dir)
