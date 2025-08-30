# -*- coding: utf-8 -*-
import os
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === CONFIG ===
INPUT_SDF = "./input_3DQSAR/actives3D_descriptors.sdf"  # fichier SDF d'entrée
OUTPUT_DIR = "./input_3DQSAR"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(OUTPUT_DIR, "actives_input_train_model.sdf")
VALID_FILE = os.path.join(OUTPUT_DIR, "actives_validation_3D_model.sdf")
TRAIN_RATIO = 0.8

# === Lecture des molécules ===
print(f"Lecture du fichier SDF : {INPUT_SDF}")
supplier = SDMolSupplier(INPUT_SDF, removeHs=False)
mols = [mol for mol in tqdm(supplier, desc="Chargement des molécules") if mol is not None]
print(f"Total molécules chargées : {len(mols)}")

# === Split 80/20 ===
train_mols, valid_mols = train_test_split(mols, train_size=TRAIN_RATIO, random_state=42)

# === Écriture des fichiers SDF ===
print(f"Écriture des molécules d'entraînement ({len(train_mols)}) dans : {TRAIN_FILE}")
with SDWriter(TRAIN_FILE) as writer:
    for mol in train_mols:
        writer.write(mol)

print(f"Écriture des molécules de validation ({len(valid_mols)}) dans : {VALID_FILE}")
with SDWriter(VALID_FILE) as writer:
    for mol in valid_mols:
        writer.write(mol)

print("\nSplit terminé avec succès !")
