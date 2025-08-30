# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle
from rdkit import Chem
import os

def run(files):
    """
    files: liste de chemins vers les fichiers uploadés
    Affiche dans Streamlit le contenu de chaque fichier :
    Excel, CSV, PKL, SDF
    """
    st.title("📂 Universal File Reader")
    st.write("Lecture de vos fichiers : **Excel, CSV, SDF, PKL**")

    for file_path in files:
        st.write(f"### 📑 Fichier détecté : `{os.path.basename(file_path)}`")
        file_type = file_path.split(".")[-1].lower()

        try:
            if file_type in ["xlsx", "xls"]:
                df = pd.read_excel(file_path)
                st.success("✅ Fichier Excel chargé avec succès")
                st.write(df)

            elif file_type == "csv":
                df = pd.read_csv(file_path)
                st.success("✅ Fichier CSV chargé avec succès")
                st.write(df)

            elif file_type == "pkl":
                with open(file_path, "rb") as f:
                    df = pickle.load(f)
                st.success("✅ Fichier Pickle chargé avec succès")
                st.write(df)

            elif file_type == "sdf":
                suppl = Chem.SDMolSupplier(file_path)
                mols = [mol for mol in suppl if mol is not None]

                if len(mols) == 0:
                    st.error("❌ Aucun molécule valide trouvée dans ce SDF")
                else:
                    st.success(f"✅ {len(mols)} molécules chargées depuis le SDF")
                    data = []
                    for i, mol in enumerate(mols):
                        props = mol.GetPropsAsDict()
                        props["SMILES"] = Chem.MolToSmiles(mol)
                        props["Index"] = i
                        data.append(props)
                    df = pd.DataFrame(data)
                    st.write(df)

            else:
                st.warning("⚠️ Type de fichier non supporté")
        except Exception as e:
            st.error(f"⚠️ Erreur lors de la lecture : {e}")
