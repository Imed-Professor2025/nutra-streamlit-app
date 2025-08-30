
# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==============================
# Configuration
# ==============================
EXCLUDE_COLS = [
    "_rdkit_mol", "Smiles", "source_file",
    "Standard Value", "log_StandardValue", "StandardValue_raw",
    "Activity_Class", "Activity_Class_raw",
    "Standard Type", "Standard Units",
    "Molecule ChEMBL ID", "Conf_ID", "Valid_Conformer",
    "Energy", "Relative_Energy",
    "RMSD_to_conf0", "Min_Bond_Distance",
    "Centroid_XYZ", "pChEMBL Value",
]

threshold_ic50 = 10  # µM

# ==============================
# Fonctions utilitaires
# ==============================
def read_sdf_to_df(sdf_path):
    st.write(f"Lecture du fichier SDF : {sdf_path}")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    rows = []
    for mol in tqdm(suppl, desc="Chargement molécules"):
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        smiles = props.get("Smiles") or props.get("SMILES") or Chem.MolToSmiles(mol)
        mol_id = props.get("Molecule ChEMBL ID", smiles)
        clean_props = {str(k): (str(v) if isinstance(v, str) else v) for k, v in props.items()}
        rows.append({"_rdkit_mol": mol, "SMILES": smiles, "mol_id": mol_id, **clean_props})
    df = pd.DataFrame(rows)
    st.write(f"Total molécules chargées : {len(df)}")
    return df

def clean_and_prepare_df(df):
    df.columns = [str(c).strip() for c in df.columns]
    std_col = next((c for c in df.columns if c.lower().replace("_"," ") in ["standard value","standardvalue"]), None)
    df["StandardValue_raw"] = pd.to_numeric(df[std_col], errors='coerce')
    df = df[~df["_rdkit_mol"].isna()]
    df = df[df["StandardValue_raw"]>0].reset_index(drop=True)
    df["log_StandardValue"] = np.log10(df["StandardValue_raw"])
    df.reset_index(drop=True, inplace=True)
    return df

def select_features(df, exclude_cols=EXCLUDE_COLS):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    return numeric_cols

# ==============================
# Fonction principale run
# ==============================
def run(uploaded_file_paths=None):
    st.write("🚀 Démarrage de la validation 3D-QSAR")

    if not uploaded_file_paths or len(uploaded_file_paths) != 3:
        st.error("⚠️ Veuillez uploader exactement 3 fichiers : 1 .joblib et 2 .sdf")
        return

    # ----------------------
    # Charger modèle joblib directement
    # ----------------------
    model_path = uploaded_file_paths[0]  # le .joblib
    try:
        model_final = joblib.load(model_path)
        st.write(f"[INFO] Modèle final chargé avec succès : {model_path}")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return

    # ----------------------
    # Lecture fichiers SDF
    # ----------------------
    df_act = read_sdf_to_df(uploaded_file_paths[1])
    df_inact = read_sdf_to_df(uploaded_file_paths[2])
    df_val = pd.concat([df_act, df_inact], ignore_index=True)
    df_val = clean_and_prepare_df(df_val)

    # ----------------------
    # Préparation features
    # ----------------------
    features = select_features(df_val)
    X_val = df_val[features].apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf,-np.inf],0).values
    y_true_val = df_val["log_StandardValue"].values

    imputer = SimpleImputer(strategy="mean")
    X_val = imputer.fit_transform(X_val)
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)

    # ----------------------
    # Prédictions
    # ----------------------
    y_pred_val = model_final.predict(X_val_scaled)

    # ----------------------
    # Métriques
    # ----------------------
    r2 = r2_score(y_true_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
    mae = mean_absolute_error(y_true_val, y_pred_val)
    q2 = 1 - np.sum((y_true_val - y_pred_val)**2) / np.sum((y_true_val - np.mean(y_true_val))**2)

    st.write("\n### Métriques sur validation externe (régression)")
    st.write(f"R²      : {r2:.4f}")
    st.write(f"Q²      : {q2:.4f}")
    st.write(f"RMSE    : {rmse:.4f}")
    st.write(f"MAE     : {mae:.4f}")

    # ----------------------
    # Sauvegarde prédictions
    # ----------------------
    output_dir = os.path.join(os.path.dirname(model_path), "validation_3DQSAR_results")
    os.makedirs(output_dir, exist_ok=True)

    df_val["StandardValue_predite"] = 10**y_pred_val
    df_val["Activity_Class_predite"] = np.where(df_val["StandardValue_predite"] <= threshold_ic50, "Active", "Inactive")

    pred_file = os.path.join(output_dir, "predictions_validation.xlsx")
    df_val_to_save = df_val[[
        "Molecule ChEMBL ID",
        "SMILES",
        "Activity_Class",
        "Standard Value",
        "StandardValue_predite",
        "Activity_Class_predite"
    ]]
    df_val_to_save.to_excel(pred_file, index=False)
    st.write(f"[INFO] Prédictions sauvegardées dans {pred_file}")

    # ----------------------
    # Graphiques
    # ----------------------
    fig1, ax1 = plt.subplots(figsize=(6,6))
    ax1.scatter(y_true_val, y_pred_val, alpha=0.7, color='blue')
    ax1.plot([min(y_true_val), max(y_true_val)],
             [min(y_true_val), max(y_true_val)], 'r--')
    ax1.set_xlabel("log(StandardValue) Observé")
    ax1.set_ylabel("log(StandardValue) Prévu")
    ax1.set_title("Observed vs Predicted (Validation)")
    ax1.grid(True)
    plt.tight_layout()
    fig1_file = os.path.join(output_dir, "observed_vs_predicted.png")
    fig1.savefig(fig1_file, dpi=300)
    st.pyplot(fig1)

    residus_val = y_true_val - y_pred_val
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.histplot(residus_val, bins=20, kde=True, color='green', ax=ax2)
    ax2.set_title("Distribution des résidus (Validation)")
    ax2.set_xlabel("Résidu (Observé - Prévu)")
    ax2.set_ylabel("Nombre de molécules")
    ax2.grid(True)
    plt.tight_layout()
    fig2_file = os.path.join(output_dir, "residus_distribution.png")
    fig2.savefig(fig2_file, dpi=300)
    st.pyplot(fig2)

    st.write("✅ Validation 3D-QSAR terminée.")