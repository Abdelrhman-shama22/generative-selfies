from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
import pandas as pd

sdf_file = "gdb9.sdf"
csv_file = "gdb9.sdf.csv"

properties_df = pd.read_csv(r"C:\abdo123\gdb9\gdb9.sdf.csv")
supplier = Chem.SDMolSupplier(r"C:\abdo123\gdb9\gdb9.sdf", sanitize=False)


smiles_data = []
invalid_count = 0
counter = 0

for mol in supplier:
    if mol is None:
        continue
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if mol_id is None:
            continue
        smiles_data.append([mol_id, smiles])
        counter += 1
        if counter % 10000 == 0:
            print(f"Processed {counter} molecules...")
    except:
        invalid_count += 1
        continue

smiles_df = pd.DataFrame(smiles_data, columns=["mol_id", "isomeric_smiles"])
properties_df["mol_id"] = properties_df["mol_id"].astype(str)
final_df = pd.merge(properties_df, smiles_df, on="mol_id", how="inner")
final_df.to_csv("final_dataset.csv", index=False)

print(final_df.head())
print(f"\n    final_dataset.csv")
print(f"ت not_valid {invalid_count} ")

