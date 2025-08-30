# -*- coding: utf-8 -*-
import streamlit as st
import os
from pathlib import Path
import sys
import importlib.util
import re
import joblib
import pickle

# ==============================
# Configuration dossiers
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SCRIPTS_FOLDER = os.path.join(BASE_DIR, "scripts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCRIPTS_FOLDER, exist_ok=True)

# ==============================
# Barre de progression Streamlit
# ==============================
def st_progress(iterable, total=None, prefix="Progress"):
    total = total or len(iterable)
    progress_bar = st.progress(0)
    for i, item in enumerate(iterable):
        yield item
        progress_bar.progress((i + 1) / total)
    progress_bar.empty()

# ==============================
# Interface Streamlit
# ==============================
st.set_page_config(page_title="Drug Discovery App", layout="wide")
st.title("💊 Drug Discovery Platform")
st.write("Upload your dataset(s) to start processing each script.")

# ==============================
# Liste scripts
# ==============================
def list_scripts_ordered():
    scripts = [f for f in os.listdir(SCRIPTS_FOLDER) if f.endswith(".py") and f != "__init__.py"]
    scripts = ["0_script_standard.py"] + sorted(
        [s for s in scripts if s != "0_script_standard.py"],
        key=lambda x: int(re.match(r"(\d+)", x).group(1)) if re.match(r"(\d+)", x) else 999
    )
    return scripts

scripts = list_scripts_ordered()
selected_script = st.selectbox("Select a script to run", scripts)

# ==============================
# Upload fichiers
# ==============================
uploaded_file_paths = []
can_run = False

def save_uploaded_file(uploaded_file):
    path = os.path.join(OUTPUT_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ------------------------------
# Script16 : 1 .pkl ou .joblib + 2 .sdf
# ------------------------------
if selected_script.startswith("16"):
    uploaded_files = st.file_uploader(
        f"📂 Upload 3 input files for {selected_script}: 1 .pkl/.joblib + 2 .sdf",
        type=["pkl","joblib","sdf"], accept_multiple_files=True
    )
    if uploaded_files:
        if len(uploaded_files) != 3:
            st.warning("⚠️ Please upload exactly 3 files: 1 .pkl/.joblib and 2 .sdf")
            can_run = False
        else:
            uploaded_files = sorted(uploaded_files, key=lambda x: 0 if x.name.endswith((".pkl",".joblib")) else 1)
            first_file = uploaded_files[0]
            path_first = save_uploaded_file(first_file)
            if first_file.name.endswith(".pkl"):
                with open(path_first, "rb") as f:
                    model_obj = pickle.load(f)
                joblib_path = os.path.splitext(path_first)[0] + ".joblib"
                joblib.dump(model_obj, joblib_path)
                st.success(f"[INFO] .pkl converted to .joblib at {joblib_path}")
                uploaded_file_paths.append(joblib_path)
            else:
                uploaded_file_paths.append(path_first)
            for f in uploaded_files[1:]:
                uploaded_file_paths.append(save_uploaded_file(f))
            for f in uploaded_file_paths:
                st.success(f"[OK] File saved at: {f}")
            can_run = True

# ------------------------------
# Scripts 10 ou 15 : 2 fichiers
# ------------------------------
elif selected_script.startswith(("10","15")):
    uploaded_files = st.file_uploader(
        f"📂 Upload 2 input files for {selected_script}",
        type=["sdf","csv","xlsx","pkl"], accept_multiple_files=True
    )
    if uploaded_files:
        uploaded_file_paths = [save_uploaded_file(f) for f in uploaded_files]
        for f in uploaded_file_paths:
            st.success(f"[OK] File saved at: {f}")
        can_run = len(uploaded_file_paths) == 2

# ------------------------------
# Script17 : 1 fichier Excel
# ------------------------------
elif selected_script.startswith("17"):
    uploaded_file = st.file_uploader(
        f"📂 Upload 1 Excel file for {selected_script}",
        type=["xlsx","xls"]
    )
    if uploaded_file:
        path = save_uploaded_file(uploaded_file)
        uploaded_file_paths = [path]
        st.success(f"[OK] File saved at: {path}")
        can_run = True

# ------------------------------
# Autres scripts : 1 fichier
# ------------------------------
else:
    uploaded_file = st.file_uploader(
        f"📂 Upload input file for {selected_script}",
        type=["sdf","csv","xlsx","pkl"]
    )
    if uploaded_file:
        uploaded_file_paths = [save_uploaded_file(uploaded_file)]
        st.success(f"[OK] File saved at: {uploaded_file_paths[0]}")
        can_run = True

# ==============================
# Exécution des scripts
# ==============================
run_btn = st.button("Run selected script", disabled=not can_run)
if run_btn and can_run:
    script_path = os.path.join(SCRIPTS_FOLDER, selected_script)
    spec = importlib.util.spec_from_file_location("script_module", script_path)
    script_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_module)

    if hasattr(script_module, "run"):
        script_module.run(uploaded_file_paths)
    elif hasattr(script_module, "run_script"):
        st.warning(f"[INFO] Le script {selected_script} n'a pas de fonction 'run', on utilise 'run_script'")
        script_module.run_script(*uploaded_file_paths)
    else:
        st.error(f"Le script {selected_script} ne contient pas de fonction 'run' ou 'run_script'")
