# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdmolfiles import SDWriter, SDMolSupplier
from rdkit.Chem import rdMolAlign, ChemicalFeatures
from rdkit import RDConfig
import math

# ---------------------------
# Fonction log temps réel compatible Windows UTF-8
# ---------------------------
def log(msg):
    try:
        sys.stdout.buffer.write((msg + "\n").encode('utf-8', errors='replace'))
        sys.stdout.flush()
    except Exception as e:
        # fallback si stdout pose problème
        print(msg)

# ---------------------------
# Validation conformer
# ---------------------------
def validate_conformer_geometry(mol, confId, rmsd_threshold=3.999, min_bond_length=0.9):
    try:
        rmsd_val = 0.0
        if confId != 0:
            rmsd_val = rdMolAlign.GetBestRMS(mol, mol, prbId=confId, refId=0)
            if rmsd_val >= rmsd_threshold:
                return False, rmsd_val, None
        conf = mol.GetConformer(confId)
        min_bond_dist = float('inf')
        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            pos1 = np.array(conf.GetAtomPosition(idx1))
            pos2 = np.array(conf.GetAtomPosition(idx2))
            dist = np.linalg.norm(pos1 - pos2)
            if dist < min_bond_dist:
                min_bond_dist = dist
            if dist < min_bond_length:
                return False, rmsd_val, dist
        return True, rmsd_val, min_bond_dist
    except Exception as e:
        log(f"⚠️ Validation failed for conformer {confId}: {e}")
        return False, None, None

# ---------------------------
# Arguments
# ---------------------------
if len(sys.argv) != 3:
    log("Usage: python script7.py <input.sdf> <output_dir>")
    sys.exit(1)

input_sdf = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

output_sdf_file = os.path.join(output_dir, "actives_clean_conformers_validated.sdf")
output_excel_file = os.path.join(output_dir, "actives_clean_conformers_validated_with_input_cols.xlsx")

log(f"✅ Fichiers de sortie : {output_dir}")

# Colonnes
id_col = 'Molecule ChEMBL ID'
stdval_col = 'Standard Value'

# ---------------------------
# Vérification fichier
# ---------------------------
if not os.path.exists(input_sdf):
    raise FileNotFoundError(f"❌ Fichier introuvable : {input_sdf}")
if os.path.getsize(input_sdf) == 0:
    raise ValueError(f"❌ Le fichier est vide : {input_sdf}")

# ---------------------------
# Lecture SDF
# ---------------------------
suppl = SDMolSupplier(input_sdf, removeHs=False)
mols = [mol for mol in suppl if mol is not None]

# ---------------------------
# Filtrage molécules invalides
# ---------------------------
valid_mols = []
invalid_count = 0
for mol in mols:
    if mol is None:
        invalid_count += 1
        continue
    try:
        Chem.SanitizeMol(mol)
        valid_mols.append(mol)
    except:
        invalid_count += 1
        continue

log(f"⚠️ Molécules ignorées : {invalid_count}")
log(f"✅ Molécules valides : {len(valid_mols)}")
mols = valid_mols

# ---------------------------
# Extraction des propriétés
# ---------------------------
records = []
for mol in mols:
    props = mol.GetPropsAsDict()
    if id_col not in props or stdval_col not in props:
        log(f"⚠️ Molécule ignorée (propriétés manquantes)")
        continue
    try:
        std_val = float(props[stdval_col])
    except:
        log(f"⚠️ Valeur Standard Value invalide: {props[stdval_col]}")
        continue
    if 'Smiles' not in props:
        props['Smiles'] = Chem.MolToSmiles(mol)
    props[stdval_col] = std_val
    records.append(props)

df = pd.DataFrame(records)
before_dup = len(df)
df = df.drop_duplicates(subset=['Smiles']).reset_index(drop=True)
after_dup = len(df)
df = df[df[stdval_col] <= 10].reset_index(drop=True)
log(f"⚡ {before_dup - after_dup} doublons supprimés, {len(df)} molécules restantes après filtrage Standard Value ≤10.")

# ---------------------------
# Génération conformers et calculs 3D complets
# ---------------------------
output_sdf_writer = SDWriter(output_sdf_file)
results = []
total = len(df)
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

# ---------------------------
# Correction getattr pour colonnes avec espaces
# ---------------------------
for idx, row in enumerate(df.itertuples(index=False, name=None), 1):
    mol_id = row[df.columns.get_loc(id_col)]
    std_val = row[df.columns.get_loc(stdval_col)]
    smiles = row[df.columns.get_loc('Smiles')]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log(f"[{idx}/{total}] ❌ SMILES invalide pour {mol_id}")
        continue
    mol = Chem.AddHs(mol)
    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, maxAttempts=1000, randomSeed=42)
    except Exception as e:
        log(f"[{idx}/{total}] ❌ Génération conformers échouée {mol_id}: {e}")
        continue

    energies = []
    for cid in conf_ids:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid) if props else AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            ff.Minimize(maxIts=1000, energyTol=1e-5)
            energies.append((cid, ff.CalcEnergy()))
        except Exception as e:
            log(f"[{idx}/{total}] ⚠️ Optimisation échouée {mol_id} Conf {cid}: {e}")

    if not energies:
        log(f"[{idx}/{total}] ⚠️ Aucun conformer optimisé pour {mol_id}")
        continue

    energies.sort(key=lambda x: x[1])
    min_energy = energies[0][1]
    top_confs = [(cid, e) for cid, e in energies if (e - min_energy) <= 10.0]

    valid_confs = []
    for cid, energy in top_confs:
        valid, rmsd_val, min_bond_dist = validate_conformer_geometry(mol, cid)
        log(f"[{idx}/{total}] {mol_id} - Conf {cid} valid: {valid}")
        if valid:
            valid_confs.append((cid, energy, rmsd_val, min_bond_dist))
        if len(valid_confs) >= 5:
            break

    for cid, energy, rmsd_val, min_bond_dist in valid_confs:
        conformer = mol.GetConformer(cid)
        coords = conformer.GetPositions()
        centroid = np.mean(coords, axis=0)
        min_xyz = coords.min(axis=0)
        max_xyz = coords.max(axis=0)
        bbox_dims = max_xyz - min_xyz
        volume = bbox_dims[0] * bbox_dims[1] * bbox_dims[2]
        surface = rdMolDescriptors.CalcLabuteASA(mol, includeHs=True, force=True)

        for col_idx, col_name in enumerate(df.columns):
            mol.SetProp(col_name, str(row[col_idx]))

        mol.SetProp('Conf_ID', str(cid))
        mol.SetProp('Energy', f"{energy:.2f}")
        mol.SetProp('Relative_Energy', f"{energy - min_energy:.2f}")
        mol.SetProp('Volume', f"{volume:.2f}")
        mol.SetProp('Surface', f"{surface:.2f}")
        mol.SetProp('Centroid_XYZ', ','.join([f"{x:.2f}" for x in centroid]))
        mol.SetProp('RMSD_to_conf0', f"{rmsd_val:.2f}")
        mol.SetProp('Min_Bond_Distance', f"{min_bond_dist:.2f}")
        mol.SetProp('Valid_Conformer', 'True')

        # Features pharmacophoriques complètes
        feat_list = factory.GetFeaturesForMol(mol)
        pharma_feats = [f for f in feat_list if f.GetFamily() in ['HBA','HBD','Aromatic','Hydrophobe','Acceptor','Donor']]
        feat_coords = [np.mean([conformer.GetAtomPosition(idx) for idx in f.GetAtomIds()], axis=0) for f in pharma_feats]
        feat_types = [f.GetFamily() for f in pharma_feats]

        num_hba = sum(1 for f in pharma_feats if f.GetFamily() == 'HBA')
        num_hbd = sum(1 for f in pharma_feats if f.GetFamily() == 'HBD')
        mol.SetProp('Num_HBA', str(num_hba))
        mol.SetProp('Num_HBD', str(num_hbd))
        num_aromatic = sum(1 for f in pharma_feats if f.GetFamily() == 'Aromatic')
        mol.SetProp('Num_Aromatic', str(num_aromatic))
        num_rings = Chem.GetSSSR(mol)
        mol.SetProp('Num_Rings', str(num_rings))

        distances, angles, dihedrals = [], [], []
        num_feats = len(feat_coords)
        for i in range(num_feats):
            for j in range(i+1, num_feats):
                distances.append((feat_types[i], feat_types[j], np.linalg.norm(feat_coords[i] - feat_coords[j])))
            for j in range(i+1, num_feats):
                for k in range(j+1, num_feats):
                    vec1 = feat_coords[i] - feat_coords[j]
                    vec2 = feat_coords[k] - feat_coords[j]
                    cos_theta = np.clip(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)), -1.0,1.0)
                    angle_deg = np.degrees(np.arccos(cos_theta))
                    angles.append((feat_types[i], feat_types[j], feat_types[k], angle_deg))
                for k in range(j+1, num_feats):
                    for l in range(k+1, num_feats):
                        p0,p1,p2,p3 = feat_coords[i], feat_coords[j], feat_coords[k], feat_coords[l]
                        b0 = p1 - p0
                        b1 = p2 - p1
                        b2 = p3 - p2
                        b1 /= np.linalg.norm(b1)
                        v = b0 - np.dot(b0,b1)*b1
                        w = b2 - np.dot(b2,b1)*b1
                        x = np.dot(v,w)
                        y = np.dot(np.cross(b1,v),w)
                        dihedrals.append((feat_types[i],feat_types[j],feat_types[k],feat_types[l],np.degrees(np.arctan2(y,x))))

        tpsa = rdMolDescriptors.CalcTPSA(mol)
        mol.SetProp('Num_Features', str(len(pharma_feats)))
        mol.SetProp('TPSA', f"{tpsa:.2f}")
        mol.SetProp('Distances_Features', str(distances))
        mol.SetProp('Angles_Features', str(angles))
        mol.SetProp('Dihedrals_Features', str(dihedrals))

        output_sdf_writer.write(mol, confId=cid)

        result_row = dict(zip(df.columns, row))
        result_row.update({
            'Conf_ID': cid,
            'Energy': energy,
            'Relative_Energy': energy - min_energy,
            'Volume': volume,
            'Surface': surface,
            'Centroid_XYZ': centroid.tolist(),
            'RMSD_to_conf0': rmsd_val,
            'Min_Bond_Distance': min_bond_dist,
            'Valid_Conformer': True,
            'Num_HBA': num_hba,
            'Num_HBD': num_hbd,
            'Num_Aromatic': num_aromatic,
            'Num_Rings': num_rings,
            'Distances_Features': distances,
            'Angles_Features': angles,
            'Dihedrals_Features': dihedrals
        })
        results.append(result_row)

output_sdf_writer.close()
df_results = pd.DataFrame(results)
df_results.to_excel(output_excel_file, index=False)
log(f"✅ Excel complet sauvegardé : {output_excel_file}")
log(f"🎉 Terminé : {total} molécules traitées. Fichiers sauvegardés.")
