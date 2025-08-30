# -*- coding: utf-8 -*-
import streamlit as st
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign
from rdkit.Chem.rdmolfiles import SDMolSupplier, SDWriter
import py3Dmol
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import sys

st.title("🧬 Alignement 3D des conformers actives")

# --- Récupération du fichier SDF
# Si le script est lancé via main.py (subprocess), récupérer le chemin du fichier
if len(sys.argv) > 1:
    uploaded_file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "Aligned_actives3D"
else:
    uploaded_file = st.file_uploader("📂 Charger SDF des conformers actives", type=["sdf"])
    if uploaded_file is None:
        st.warning("⚠️ Veuillez uploader un fichier SDF pour continuer")
        st.stop()
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp:
        tmp.write(uploaded_file.read())
        uploaded_file_path = tmp.name
    output_dir = "Aligned_actives3D"

os.makedirs(output_dir, exist_ok=True)
output_sdf_file = os.path.join(output_dir, "actives_aligned3d.sdf")
output_excel_file = os.path.join(output_dir, "actives_aligned3d.xlsx")

# Lecture SDF
st.info("📥 Chargement des molécules...")
suppl = SDMolSupplier(uploaded_file_path, removeHs=False)
conformers = [mol for mol in suppl if mol is not None]
st.write(f"Nombre total de conformers chargés : {len(conformers)}")

# Regrouper par Molecule ChEMBL ID
mol_dict = defaultdict(list)
for mol in conformers:
    try:
        mol_id = mol.GetProp("Molecule ChEMBL ID")
    except KeyError:
        continue
    mol_dict[mol_id].append(mol)
st.write(f"Molécules uniques détectées : {len(mol_dict)}")

# Calcul du MCS
st.info("🧠 Calcul du MCS en cours...")
unique_mols = [mol_list[0] for mol_list in mol_dict.values()]
mcs_result = rdFMCS.FindMCS(unique_mols, completeRingsOnly=True, matchValences=True, timeout=60)
mcs_smarts = mcs_result.smartsString
mcs_mol = Chem.MolFromSmarts(mcs_smarts)
st.success(f"MCS SMARTS : {mcs_smarts}")

# Sélection des molécules contenant le MCS
mol_with_mcs = [mol_id for mol_id, mols in mol_dict.items() if mols[0].HasSubstructMatch(mcs_mol)]
st.write(f"Molécules contenant le MCS : {len(mol_with_mcs)} / {len(mol_dict)}")

# Liste des conformers à aligner
confs_to_align = []
for mol_id in mol_with_mcs:
    confs_to_align.extend(mol_dict[mol_id])
st.write(f"Conformers totaux à aligner : {len(confs_to_align)}")

# Référence
if not mol_with_mcs:
    st.error("❌ Aucun MCS commun trouvé — arrêt")
    st.stop()
ref_mol = mol_dict[mol_with_mcs[0]][0]
ref_match = ref_mol.GetSubstructMatch(mcs_mol)
if not ref_match:
    st.error("❌ La molécule de référence ne matche pas le MCS — arrêt")
    st.stop()
st.write(f"Référence choisie : {mol_with_mcs[0]} (conformer index 0)")

# Alignement
st.info("🚀 Alignement des conformers sur la référence...")
writer = SDWriter(output_sdf_file)
aligned_mols = []
aligned_data = []
success_count = 0

for mol in tqdm(confs_to_align):
    try:
        mol_match = mol.GetSubstructMatch(mcs_mol)
        if not mol_match or len(mol_match) != len(ref_match):
            continue
        atomMap = list(zip(mol_match, ref_match))
        rmsd = rdMolAlign.AlignMol(prbMol=mol, refMol=ref_mol, atomMap=atomMap)
        mol.SetProp("RMSD_to_ref", f"{rmsd:.3f}")
        writer.write(mol)
        aligned_mols.append(mol)
        prop_dict = {prop: mol.GetProp(prop) for prop in mol.GetPropNames()}
        prop_dict["RMSD"] = f"{rmsd:.3f}"
        aligned_data.append(prop_dict)
        success_count += 1
    except Exception as e:
        continue

writer.close()
st.success(f"✅ Alignement terminé. Total réussites : {success_count} / {len(confs_to_align)}")
st.write(f"SDF écrit : {output_sdf_file}")

# Export Excel
df = pd.DataFrame(aligned_data)
df.to_excel(output_excel_file, index=False)
st.write(f"📤 Export Excel terminé : {output_excel_file}")

# Visualisation 3D
st.info("🧪 Visualisation 3D des molécules alignées...")
viewer = py3Dmol.view(width=900, height=700)

colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

for i, mol in enumerate(aligned_mols):
    mb = Chem.MolToMolBlock(mol)
    viewer.addModel(mb, 'mol')
    viewer.setStyle({'model': i}, {'stick': {'colorscheme': f'Jmol:{colors[i % len(colors)]}'}})

viewer.zoomTo()

# Générer le HTML complet manuellement
viewer_html = f"""
<div style="width:900px; height:700px; position: relative;">
{viewer.getViewer()}
</div>
<script type="text/javascript">
{viewer.js()}
</script>
"""

st.components.v1.html(viewer_html, height=700, scrolling=True)

