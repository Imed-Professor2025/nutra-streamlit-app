# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    StackingRegressor, StackingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import streamlit as st

RANDOM_STATE = 42
EXCLUDE_COLS = [
    "_rdkit_mol", "Smiles", "source_file",
    "Standard Value", "log_StandardValue", "StandardValue_raw",
    "Activity_Class", "Activity_Class_raw",
    "Standard Type", "Standard Units",
    "Molecule ChEMBL ID", "Conf_ID", "Valid_Conformer",
    "Energy", "Relative_Energy",
    "RMSD_to_conf0", "Min_Bond_Distance",
    "Centroid_XYZ"
]

# ------------------------------
# Barre de progression Streamlit avec pourcentage
# ------------------------------
class StreamlitProgress:
    def __init__(self, iterable, desc="Progress"):
        self.iterable = iterable
        self.total = len(iterable)
        self.desc = desc
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def __iter__(self):
        for i, item in enumerate(self.iterable):
            yield item
            pct = (i + 1) / self.total
            self.progress_bar.progress(pct)
            self.status_text.text(f"{self.desc}: {i + 1}/{self.total} ({pct*100:.1f}%)")
        self.progress_bar.empty()
        self.status_text.empty()

# ------------------------------
# Fonctions principales
# ------------------------------
def run_script(actives_file, inactives_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    st.write(f"📂 Output folder: `{output_dir}`")

    def read_sdf_to_df(sdf_path):
        st.write(f"[INFO] Lecture du fichier SDF : {sdf_path}")
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        rows = []
        iterable = list(suppl)
        for mol in StreamlitProgress(iterable, desc="Chargement molécules"):
            if mol is None:
                continue
            props = mol.GetPropsAsDict()
            smiles = props.get("Smiles") or props.get("SMILES") or Chem.MolToSmiles(mol)
            mol_id = props.get("Molecule ChEMBL ID", smiles)
            clean_props = {str(k): (str(v) if isinstance(v, str) else v) for k, v in props.items()}
            rows.append({"_rdkit_mol": mol, "SMILES": smiles, "mol_id": mol_id, **clean_props})
        df = pd.DataFrame(rows)
        st.write(f"[INFO] Total molécules chargées : {len(df)}")
        return df

    def clean_and_prepare_df(df):
        df.columns = [str(c).strip() for c in df.columns]
        std_col = next((c for c in df.columns if c.lower().replace("_"," ") in ["standard value","standardvalue"]), None)
        act_col = next((c for c in df.columns if c.lower()=="activity_class"), None)
        df["StandardValue_raw"] = pd.to_numeric(df[std_col], errors='coerce')
        df["Activity_Class_raw"] = df[act_col]
        df = df[~df["_rdkit_mol"].isna()]
        df = df[df["StandardValue_raw"]>0].reset_index(drop=True)
        df["log_StandardValue"] = np.log10(df["StandardValue_raw"])
        df["Activity_Class"] = df["Activity_Class_raw"]
        lower, upper = df["log_StandardValue"].quantile([0.01,0.99])
        df = df[(df["log_StandardValue"] >= lower) & (df["log_StandardValue"] <= upper)]
        df.reset_index(drop=True, inplace=True)
        return df

    def select_features(df, threshold_corr=0.95):
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_COLS]
        corr_matrix = df[numeric_cols + ["log_StandardValue"]].corr().abs()
        to_remove = set()
        for col in numeric_cols:
            if corr_matrix.loc[col, "log_StandardValue"] > threshold_corr:
                to_remove.add(col)
        features = [c for c in numeric_cols if c not in to_remove]
        return features

    def stacking_regression(X_train, y_train):
        estimators = [
            ("RF", RandomForestRegressor(random_state=RANDOM_STATE)),
            ("GB", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            ("SVR", SVR()),
            ("Ridge", Ridge()),
            ("KNN", KNeighborsRegressor())
        ]
        stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), n_jobs=-1)
        stack.fit(X_train, y_train)
        return stack

    def stacking_classification(X_train, y_train):
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        estimators = [
            ("RF", RandomForestClassifier(random_state=RANDOM_STATE)),
            ("GB", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            ("SVC", SVC(probability=True)),
            ("LR", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ("KNN", KNeighborsClassifier())
        ]
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1)
        stack.fit(X_train, y_train_enc)
        return stack, le

    def run_strategy(df, strategy_name="Strategy"):
        st.write(f"\n[INFO] === {strategy_name} ===")
        features = select_features(df)
        df_mol = df.groupby("mol_id").agg({**{f:"mean" for f in features},
                                          "log_StandardValue":"first",
                                          "Activity_Class":"first"}).reset_index()
        df_mol[features] = df_mol[features].apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0).astype(float)

        unique_ids = df_mol["mol_id"].values
        mol_to_class = df_mol.set_index("mol_id")["Activity_Class"]

        train_ids, test_ids = train_test_split(
            unique_ids,
            train_size=0.8,
            random_state=RANDOM_STATE,
            stratify=mol_to_class[unique_ids] if len(mol_to_class.unique())==2 else None
        )

        df_train = df_mol[df_mol["mol_id"].isin(train_ids)].reset_index(drop=True)
        df_test  = df_mol[df_mol["mol_id"].isin(test_ids)].reset_index(drop=True)

        X_train = df_train[features].values
        y_reg_train = df_train["log_StandardValue"].values
        y_clf_train = df_train["Activity_Class"].values

        X_test = df_test[features].values
        y_reg_test = df_test["log_StandardValue"].values
        y_clf_test = df_test["Activity_Class"].values

        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        reg_model = stacking_regression(X_train_scaled, y_reg_train)
        y_reg_pred = reg_model.predict(X_test_scaled)

        R2 = r2_score(y_reg_test, y_reg_pred)
        RMSE = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        Q2 = r2_score(y_reg_train, cross_val_predict(reg_model, X_train_scaled, y_reg_train, cv=5, n_jobs=-1))

        clf_model, le = stacking_classification(X_train_scaled, y_clf_train)
        y_clf_true_enc = LabelEncoder().fit_transform(y_clf_test)
        y_clf_prob = clf_model.predict_proba(X_test_scaled)[:,1]
        y_clf_pred = (y_clf_prob >= 0.5).astype(int)

        metrics = {
            "R2": R2, "Q2": Q2, "RMSE": RMSE,
            "MAE": mean_absolute_error(y_reg_test, y_reg_pred),
            "MAPE": mean_absolute_percentage_error(y_reg_test, y_reg_pred),
            "Accuracy": accuracy_score(y_clf_true_enc, y_clf_pred),
            "F1": f1_score(y_clf_true_enc, y_clf_pred),
            "Precision": precision_score(y_clf_true_enc, y_clf_pred),
            "Recall": recall_score(y_clf_true_enc, y_clf_pred)
        }

        st.write(f"[INFO] [{strategy_name}] Regression R²={R2:.4f}, Q²={Q2:.4f}, RMSE={RMSE:.4f}")
        st.write(f"[INFO] [{strategy_name}] Classification Accuracy={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}")

        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x=y_reg_test, y=y_reg_pred, ax=ax)
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.set_title(f"Regression {strategy_name}")
        st.pyplot(fig)

        return reg_model, clf_model, metrics, y_reg_test, y_reg_pred, X_test_scaled, features

    # ------------------------------
    # Execution
    # ------------------------------
    df_act = read_sdf_to_df(actives_file)
    df_inact = read_sdf_to_df(inactives_file)
    df_act["source_file"] = os.path.basename(actives_file)
    df_inact["source_file"] = os.path.basename(inactives_file)
    df_all = pd.concat([df_act, df_inact], ignore_index=True)
    df_all = clean_and_prepare_df(df_all)

    reg_model, clf_model, metrics, y_true, y_pred, X_test_scaled, features = run_strategy(df_all, strategy_name="Strategy Streamlit")

    model_path = os.path.join(output_dir,"model_final.pkl")
    with open(model_path,"wb") as f:
        pickle.dump(reg_model,f)
    st.success(f"✅ Modèle final sauvegardé dans : {model_path}")
    st.write("📊 Metrics :", metrics)

# ------------------------------
# Wrapper pour Streamlit
# ------------------------------
def run(files):
    if len(files) != 2:
        raise ValueError("You must provide exactly 2 files: [actives_file, inactives_file]")
    actives_file = files[0]
    inactives_file = files[1]
    output_dir = os.path.join(os.path.dirname(actives_file), "script15_output")
    os.makedirs(output_dir, exist_ok=True)
    run_script(actives_file, inactives_file, output_dir)
