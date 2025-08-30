# -*- coding: utf-8 -*-
import streamlit as st
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors3D, rdFreeSASA
from mordred import Calculator, descriptors

st.set_page_config(page_title="3D Descriptors Generator", layout="wide")

st.title("📊 Générateur de Descripteurs 3D")

# === Upload SDF ===
uploaded_file = st.file_uploader("📥 Téléversez votre fichier SDF", type=["sdf"])
output_dir = "input_3DQSAR"
os.makedirs(output_dir, exist_ok=True)

if uploaded_file:
    sdf_path = os.path.join(output_dir, uploaded_file.name)
    with open(sdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Fichier sauvegardé : {sdf_path}")

    @st.cache_data(show_spinner=True)
    def load_molecules(path):
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mols = [m for m in suppl if m is not None]
        return mols

    mols = load_molecules(sdf_path)
    st.write(f"✔ {len(mols)} molécules chargées depuis le SDF.")

    # Filtrer molécules avec conformers 3D
    valid_mols = [m for m in mols if m.GetNumConformers() > 0]
    st.write(f"✔ {len(valid_mols)} molécules avec conformer 3D valide")

    # Calcul des descripteurs 3D
    st.info("⚙ Calcul des descripteurs 3D... ceci peut prendre quelques minutes.")
    calc = Calculator(descriptors, ignore_3D=False)

    data = []
    output_sdf = os.path.join(output_dir, "INactives3D_descriptors.sdf")
    output_excel = os.path.join(output_dir, "INactives3D_descriptors.xlsx")
    writer = Chem.SDWriter(output_sdf)

    for mol in valid_mols:
        desc = mol.GetPropsAsDict()
        desc["Molecule_Name"] = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"

        # --- Descripteurs RDKit 3D ---
        try:
            radiis = rdFreeSASA.classifyAtoms(mol)
            desc["SASA"] = rdFreeSASA.CalcSASA(mol, radiis)
            desc["Asphericity"] = Descriptors3D.Asphericity(mol)
            desc["Eccentricity"] = Descriptors3D.Eccentricity(mol)
            desc["InertialShapeFactor"] = Descriptors3D.InertialShapeFactor(mol)
            desc["NPR1"] = Descriptors3D.NPR1(mol)
            desc["NPR2"] = Descriptors3D.NPR2(mol)
            desc["PMI1"] = Descriptors3D.PMI1(mol)
            desc["PMI2"] = Descriptors3D.PMI2(mol)
            desc["PMI3"] = Descriptors3D.PMI3(mol)
            desc["RadiusOfGyration"] = Descriptors3D.RadiusOfGyration(mol)
            desc["SpherocityIndex"] = Descriptors3D.SpherocityIndex(mol)
        except Exception as e:
            st.warning(f"Erreur RDKit 3D descriptors pour {desc['Molecule_Name']}: {e}")

        # --- Descripteurs Mordred 3D ---
        try:
            mordred_desc = calc(mol).asdict()
            for k, v in mordred_desc.items():
                k = str(k)
                if "_3D" in k:
                    desc[k] = v
        except Exception as e:
            st.warning(f"Erreur Mordred descriptors pour {desc['Molecule_Name']}: {e}")

        # Ajouter les propriétés dans le SDF
        for k, v in desc.items():
            try:
                mol.SetProp(str(k), str(v))
            except:
                pass

        writer.write(mol)
        data.append(desc)

    writer.close()
    st.success(f"✔ SDF enrichi sauvegardé → {output_sdf}")

    # Export Excel
    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False)
    st.success(f"✔ Tableau descripteurs sauvegardé → {output_excel}")

    # Affichage limité pour ne pas saturer Streamlit
    st.write("### Aperçu des 10 premières lignes des descripteurs")
    st.dataframe(df.head(10))
