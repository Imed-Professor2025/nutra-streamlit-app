# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
import streamlit as st

def run(uploaded_file_paths):
    EXCEL_FILE = uploaded_file_paths[0]
    RESULTS_DIR = os.path.join(os.path.dirname(EXCEL_FILE),"Post_analysis_3DQSAR_results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df = pd.read_excel(EXCEL_FILE)
    df = df.dropna(subset=["Standard Value","SMILES","StandardValue_predite"])
    
    y_true_reg = df["Standard Value"].values
    y_pred_reg = df["StandardValue_predite"].values
    smiles = df["SMILES"].values
    
    threshold_ic50 = 10
    df["Activity_Class_predite"] = ["Active" if x <= threshold_ic50 else "Inactive" for x in y_pred_reg]
    y_true_clf = df["Activity_Class"].values
    y_pred_clf = df["Activity_Class_predite"].values

    # --- Régression
    pearson_corr, _ = pearsonr(y_true_reg, y_pred_reg)
    spearman_corr, _ = spearmanr(y_true_reg, y_pred_reg)
    st.write(f"### Régression StandardValue")
    st.write(f"Pearson corr: {pearson_corr:.4f}, Spearman corr: {spearman_corr:.4f}")

    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(x=y_true_reg, y=y_pred_reg, ax=ax)
    ax.plot([min(y_true_reg), max(y_true_reg)], [min(y_true_reg), max(y_true_reg)], 'r--')
    ax.set_xlabel("StandardValue Observé")
    ax.set_ylabel("StandardValue Prévu")
    ax.set_title(f"Observed vs Predicted")
    st.pyplot(fig)

    residus = y_true_reg - y_pred_reg
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.histplot(residus, bins=20, kde=True, ax=ax2)
    ax2.set_title("Distribution des résidus")
    st.pyplot(fig2)

    # --- Classification
    cm = confusion_matrix(y_true_clf, y_pred_clf, labels=["Active","Inactive"])
    fig3, ax3 = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Active","Inactive"], yticklabels=["Active","Inactive"], ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix Activity_Class")
    st.pyplot(fig3)

    mcc = matthews_corrcoef(y_true_clf, y_pred_clf)
    acc = accuracy_score(y_true_clf, y_pred_clf)
    st.write(f"Accuracy: {acc:.4f}, MCC: {mcc:.4f}")

    # --- Similarité Tanimoto
    fps_true = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2, nBits=2048)
                for s, l in zip(smiles, y_true_clf) if l=="Active"]
    fps_pred = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2, nBits=2048)
                for s, l in zip(smiles, y_pred_clf) if l=="Active"]
    sims = []
    for f_true in fps_true:
        max_sim = max([DataStructs.TanimotoSimilarity(f_true, f_pred) for f_pred in fps_pred]) if fps_pred else 0
        sims.append(max_sim)
    st.write(f"Tanimoto similarity moyenne: {np.mean(sims):.4f}")

    fig4, ax4 = plt.subplots(figsize=(6,4))
    sns.histplot(sims, bins=20, kde=True, ax=ax4)
    ax4.set_title("Distribution Tanimoto similarity (Active)")
    st.pyplot(fig4)

    # --- Sauvegarde
    results_summary = pd.DataFrame({
        "Pearson": [pearson_corr],
        "Spearman": [spearman_corr],
        "Accuracy": [acc],
        "MCC": [mcc],
        "Tanimoto_mean": [np.mean(sims)]
    })
    summary_file = os.path.join(RESULTS_DIR,"analysis_summary.xlsx")
    results_summary.to_excel(summary_file, index=False)
    st.success(f"✅ Résumé des analyses sauvegardé : {summary_file}")
