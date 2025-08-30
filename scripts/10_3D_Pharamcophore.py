# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import py3Dmol
import pickle
from tqdm import tqdm
import streamlit as st

# --------------------------
# Récupération des fichiers uploadés depuis main.py
# --------------------------
import sys
if len(sys.argv) < 4:
    st.error("❌ Veuillez passer 2 fichiers SDF et un dossier de sortie en argument !")
    st.stop()

actives_file = sys.argv[1]
inactives_file = sys.argv[2]
OUTPUT_DIR = sys.argv[3]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# CONFIGURATION
# --------------------------
random.seed(42)
np.random.seed(42)
MAX_HYPOTHESES = 5
DIST_THRESH = 5.0  # tolérance plus souple
N_FOLDS = 5
THRESHOLD = 0.3  # seuil permissif initial

st.title("🧬 Génération et évaluation du pharmacophore 3D")

# --------------------------
# 1. Chargement molécules uploadées
# --------------------------
st.info("📥 Chargement des molécules uploadées...")
def load_molecules_from_sdf(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    return [mol for mol in suppl if mol is not None]

actives = load_molecules_from_sdf(actives_file)
inactives = load_molecules_from_sdf(inactives_file)
st.success(f"🔹 Total Actives : {len(actives)}")
st.success(f"🔹 Total Inactives : {len(inactives)}")

# --------------------------
# 2. Split 70/30/30 train/test/validation
# --------------------------
st.info("🔄 Split train/test/validation...")
train_actives, temp_actives = train_test_split(actives, test_size=0.6, random_state=42)
test_actives, val_actives = train_test_split(temp_actives, test_size=0.5, random_state=42)
train_inactives, temp_inactives = train_test_split(inactives, test_size=0.6, random_state=42)
test_inactives, val_inactives = train_test_split(temp_inactives, test_size=0.5, random_state=42)
st.success(f"🔹 Train Actives : {len(train_actives)}, Test : {len(test_actives)}, Validation : {len(val_actives)}")
st.success(f"🔹 Train Inactives : {len(train_inactives)}, Test : {len(test_inactives)}, Validation : {len(val_inactives)}")

# --------------------------
# 3. Features standardisées
# --------------------------
st.info("⚙️ Extraction des features...")
def convert_feature_type(ftype):
    ftype_clean = ftype.strip().lower()
    if "donor" in ftype_clean:
        return "Donor"
    elif "acceptor" in ftype_clean:
        return "Acceptor"
    elif "arom" in ftype_clean:
        return "Aromatic"
    elif "hydrophobe" in ftype_clean or ftype_clean.startswith("rh"):
        return "Hydrophobe"
    elif "ring" in ftype_clean or ftype_clean in ["imidazole"]:
        return "Ring"
    elif "positive" in ftype_clean or "basic" in ftype_clean or ftype_clean == "posn":
        return "PositiveIonizable"
    elif "negative" in ftype_clean or "acidic" in ftype_clean or ftype_clean == "nitro2":
        return "NegativeIonizable"
    else:
        return None

def extract_features(mol):
    mol_h = Chem.AddHs(mol)
    fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    feats_raw = factory.GetFeaturesForMol(mol_h)
    feats = []
    for f in feats_raw:
        ftype_std = convert_feature_type(f.GetType())
        if ftype_std:
            feats.append({"type": ftype_std, "pos": f.GetPos()})
    return feats

# --------------------------
# 4. Génération hypothèses intelligentes
# --------------------------
st.info("⚗️ Génération des hypothèses pharmacophores...")
def generate_hypotheses(molecules, N=MAX_HYPOTHESES, min_feat=4, max_feat=7):
    hypotheses = []
    tries = 0
    max_tries = N*50
    while len(hypotheses) < N and tries < max_tries:
        mol = random.choice(molecules)
        feats = extract_features(mol)
        if len(feats) >= min_feat:
            n = random.randint(min_feat, min(max_feat, len(feats)))
            weights = [2 if f["type"] in ["Donor","Acceptor","Hydrophobe","Aromatic"] else 1 for f in feats]
            sampled = random.choices(feats, weights=weights, k=n)
            hypotheses.append(sampled)
        tries += 1
    return hypotheses

# --------------------------
# 5. Score et évaluation
# --------------------------
def compute_score(hypo, mol, dist_thresh=DIST_THRESH):
    mol_feats = extract_features(mol)
    hits = 0
    for h_feat in hypo:
        matched = False
        for m_feat in mol_feats:
            if h_feat["type"] == m_feat["type"]:
                dist = np.linalg.norm(np.array(h_feat["pos"]) - np.array(m_feat["pos"]))
                if dist <= dist_thresh:
                    matched = True
                    break
        if matched:
            hits += 1
    return hits / len(hypo) if len(hypo) > 0 else 0.0

def evaluate_hypothesis(hypo, actives, inactives, threshold=THRESHOLD):
    y_true = [1]*len(actives) + [0]*len(inactives)
    y_scores = [compute_score(hypo, mol) for mol in actives + inactives]
    if len(set(y_scores)) < 2:
        return {"AUC":0.5, "Precision":0.0, "Recall":0.0, "F1":0.0, "y_true": y_true, "y_scores": y_scores}
    auc_val = roc_auc_score(y_true, y_scores)
    y_pred = [1 if s>=threshold else 0 for s in y_scores]
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"AUC":auc_val, "Precision":precision, "Recall":recall, "F1":f1, "y_true": y_true, "y_scores": y_scores}

# --------------------------
# 6. K-Fold sur le train
# --------------------------
st.info(f"🔁 Génération de {MAX_HYPOTHESES} hypothèses avec k-fold CV...")
def kfold_generate(train_act, train_inact, k=N_FOLDS):
    act_labels = np.ones(len(train_act))
    inact_labels = np.zeros(len(train_inact))
    molecules = train_act + train_inact
    labels = np.concatenate([act_labels, inact_labels])
    indices = np.arange(len(molecules))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    best_hypotheses = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        st.info(f"📊 K-Fold {fold_idx+1}/{k}")
        train_act_fold = [molecules[i] for i in train_idx if labels[i]==1]
        hypotheses = generate_hypotheses(train_act_fold, N=MAX_HYPOTHESES//k)
        best_hypotheses.extend(hypotheses)
    return best_hypotheses

hypotheses = kfold_generate(train_actives, train_inactives)
st.success(f"✅ {len(hypotheses)} hypothèses générées.")

# --------------------------
# 7. Évaluation sur train/test/validation
# --------------------------
st.info("🧮 Évaluation des hypothèses...")
results = []
splits = {"Train":(train_actives, train_inactives),
          "Test":(test_actives, test_inactives),
          "Validation":(val_actives, val_inactives)}

for i, hypo in enumerate(tqdm(hypotheses)):
    st.info(f"Évaluation hypothèse {i+1}/{len(hypotheses)}")
    metrics = {}
    for split_name, (act, inact) in splits.items():
        m = evaluate_hypothesis(hypo, act, inact)
        metrics[f"AUC_{split_name}"] = m["AUC"]
        metrics[f"Precision_{split_name}"] = m["Precision"]
        metrics[f"Recall_{split_name}"] = m["Recall"]
        metrics[f"F1_{split_name}"] = m["F1"]
        metrics[f"y_true_{split_name}"] = m["y_true"]
        metrics[f"y_scores_{split_name}"] = m["y_scores"]
    results.append({"Hypothesis": hypo, **metrics})

df_results = pd.DataFrame(results)

# --------------------------
# 8. Nom des hypothèses
# --------------------------
def abbrev_feat(f):
    mapping = {"Donor":"D","Acceptor":"A","Aromatic":"R","Hydrophobe":"H",
               "PositiveIonizable":"P","NegativeIonizable":"N","Ring":"G"}
    return mapping.get(f["type"], "?")

def abbrev_hypo(hypo):
    return ''.join([abbrev_feat(f) for f in hypo])

df_results["Hypothesis_Name"] = df_results["Hypothesis"].apply(abbrev_hypo)

# --------------------------
# 9. Sélection meilleure hypothèse selon AUC Test
# --------------------------
best_idx = df_results["AUC_Test"].idxmax()
best_hypo = df_results.loc[best_idx, "Hypothesis"]
best_metrics = df_results.loc[best_idx, ["AUC_Train","Precision_Train","Recall_Train","F1_Train",
                                        "AUC_Test","Precision_Test","Recall_Test","F1_Test",
                                        "AUC_Validation","Precision_Validation","Recall_Validation","F1_Validation"]]
st.write(f"📈 Meilleure hypothèse : {df_results.loc[best_idx,'Hypothesis_Name']}")
st.write(best_metrics)

# --------------------------
# 10. Sauvegarde
# --------------------------
with open(os.path.join(OUTPUT_DIR, "best_hypothesis.pkl"),"wb") as f:
    pickle.dump({"hypothesis":best_hypo, "metrics":best_metrics.to_dict(),
                 "y_true_Train": df_results.loc[best_idx,"y_true_Train"],
                 "y_scores_Train": df_results.loc[best_idx,"y_scores_Train"],
                 "y_true_Test": df_results.loc[best_idx,"y_true_Test"],
                 "y_scores_Test": df_results.loc[best_idx,"y_scores_Test"],
                 "y_true_Validation": df_results.loc[best_idx,"y_true_Validation"],
                 "y_scores_Validation": df_results.loc[best_idx,"y_scores_Validation"]
                }, f)

df_best = pd.DataFrame([best_metrics])
df_best["Hypothesis_Name"] = df_results.loc[best_idx,'Hypothesis_Name']
df_best.to_excel(os.path.join(OUTPUT_DIR, "best_hypothesis.xlsx"), index=False)

# --------------------------
# 11. Visualisation Py3Dmol Streamlit (pharmacophore)
# --------------------------
st.info("🧪 Visualisation 3D du meilleur pharmacophore...")

viewer = py3Dmol.view(width=900, height=700)
def color_for_type(ftype):
    return {"Donor":"blue","Acceptor":"red","Aromatic":"orange","Hydrophobe":"yellow",
            "PositiveIonizable":"purple","NegativeIonizable":"green","Ring":"cyan"}.get(ftype,"gray")

for i, f in enumerate(best_hypo):
    pos = f["pos"]
    x, y, z = float(pos.x), float(pos.y), float(pos.z)
    c = color_for_type(f["type"])
    viewer.addSphere({'center': {'x': x, 'y': y, 'z': z}, 'radius': 1.2, 'color': c, 'alpha': 0.7})
    viewer.addLabel(abbrev_feat(f), {
        'position': {'x': x, 'y': y, 'z': z},
        'backgroundColor': c, 'fontColor': 'white', 'fontSize': 14
    })

viewer.zoomTo()

# Générer le HTML complet manuellement comme dans script 9
viewer_html = f"""
<div style="width:900px; height:700px; position: relative;">
{viewer.getViewer()}
</div>
<script type="text/javascript">
{viewer.js()}
</script>
"""

st.components.v1.html(viewer_html, height=700, scrolling=True)