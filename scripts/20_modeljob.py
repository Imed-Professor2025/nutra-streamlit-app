# -*- coding: utf-8 -*-
import streamlit as st
import os
import pickle
import joblib

st.set_page_config(page_title="Convert PKL to JOBLIB", layout="wide")
st.title("🔄 Conversion modèle 3D-QSAR PKL → JOBLIB")

# -*- coding: utf-8 -*-
import pickle
import joblib
import os
import streamlit as st

def run(uploaded_file_paths):
    """
    Prend en entrée une liste avec un seul fichier .pkl à convertir en .joblib
    """
    if not uploaded_file_paths or len(uploaded_file_paths) != 1:
        st.error("⚠️ Veuillez uploader exactement 1 fichier .pkl")
        return

    pkl_path = uploaded_file_paths[0]
    st.write(f"[INFO] Chargement du modèle .pkl : {pkl_path}")

    # Chargement du modèle
    with open(pkl_path, "rb") as f:
        model_obj = pickle.load(f)

    # Conversion en joblib
    joblib_path = os.path.splitext(pkl_path)[0] + ".joblib"
    joblib.dump(model_obj, joblib_path)
    st.success(f"[OK] Modèle converti en .joblib : {joblib_path}")


# ------------------------------
# Upload du fichier model_final.pkl
# ------------------------------
uploaded_file = st.file_uploader("📂 Sélectionnez votre fichier model_final.pkl", type=["pkl"])
if uploaded_file is not None:
    st.write(f"Fichier chargé : {uploaded_file.name}")

    # ------------------------------
    # Choix du dossier de sauvegarde
    # ------------------------------
    output_dir = st.text_input("📁 Dossier de sauvegarde du fichier .joblib",
                              value=os.path.dirname(uploaded_file.name) or ".")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # Conversion PKL → JOBLIB
    # ------------------------------
    if st.button("✅ Convertir en JOBLIB"):
        try:
            uploaded_file.seek(0)
            model_obj = pickle.load(uploaded_file)  # Charge exactement le modèle comme généré
            joblib_path = os.path.join(output_dir, "model_final.joblib")
            joblib.dump(model_obj, joblib_path)
            st.success(f"Modèle converti avec succès et sauvegardé : {joblib_path}")
        except Exception as e:
            st.error(f"Erreur lors de la conversion : {e}")
