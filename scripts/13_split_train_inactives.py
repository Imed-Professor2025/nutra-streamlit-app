# -*- coding: utf-8 -*-
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import SDWriter
from sklearn.model_selection import train_test_split
import sys

# ==============================
# Vérification des arguments
# ==============================
if len(sys.argv) < 3:
    raise ValueError("Usage: python 13_split.py <SDF_FILE> <OUTPUT_DIR>")

sdf_file = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Lecture SDF robuste
# ==============================
supplier = Chem.ForwardSDMolSupplier(sdf_file, sanitize=True, removeHs=False)
mols = [mol for mol in supplier if mol is not None]

if len(mols) == 0:
    raise ValueError(f"Aucune molécule valide trouvée dans {sdf_file}")

# ==============================
# Split train/validation
# ==============================
train_ratio = 0.8
train_mols, valid_mols = train_test_split(mols, train_size=train_ratio, random_state=42)

if len(train_mols) == 0 or len(valid_mols) == 0:
    raise ValueError("Split impossible : trop peu de molécules valides")

# ==============================
# Écriture fichiers SDF dans OUTPUT_DIR
# ==============================
train_file = os.path.join(OUTPUT_DIR, f"{Path(sdf_file).stem}_train.sdf")
valid_file = os.path.join(OUTPUT_DIR, f"{Path(sdf_file).stem}_valid.sdf")

with SDWriter(train_file) as writer:
    for mol in train_mols:
        writer.write(mol)

with SDWriter(valid_file) as writer:
    for mol in valid_mols:
        writer.write(mol)
