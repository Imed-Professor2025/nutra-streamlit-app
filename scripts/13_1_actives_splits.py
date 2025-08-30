# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import SDWriter
from sklearn.model_selection import train_test_split

# =========================
# Vérification des arguments
# =========================
if len(sys.argv) < 3:
    raise ValueError("Usage: python 13_1.py <SDF_FILE> <OUTPUT_DIR>")

sdf_file = sys.argv[1]
out_dir = sys.argv[2]

# Crée le dossier de sortie si nécessaire
os.makedirs(out_dir, exist_ok=True)

# =========================
# Lecture des molécules
# =========================
supplier = Chem.ForwardSDMolSupplier(sdf_file, sanitize=True, removeHs=False)
mols = [mol for mol in supplier if mol is not None]

if not mols:
    raise ValueError(f"Aucune molécule valide trouvée dans {sdf_file}")

# =========================
# Split Train / Validation
# =========================
train_mols, valid_mols = train_test_split(mols, train_size=0.8, random_state=42)

# =========================
# Écriture fichiers SDF
# =========================
train_file = os.path.join(out_dir, f"{Path(sdf_file).stem}_train.sdf")
valid_file = os.path.join(out_dir, f"{Path(sdf_file).stem}_valid.sdf")

with SDWriter(train_file) as writer:
    for mol in train_mols:
        writer.write(mol)

with SDWriter(valid_file) as writer:
    for mol in valid_mols:
        writer.write(mol)
