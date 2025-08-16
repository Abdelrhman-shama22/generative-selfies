import pandas as pd
import selfies as sf


df = pd.read_csv(r"C:\abdo123\gdb9\final_dataset.csv")


df['selfies'] = df['isomeric_smiles'].apply(lambda x: sf.encoder(x))



df.to_csv(r"C:\abdo123\gdb9\final_dataset_with_selfies.csv", index=False)
