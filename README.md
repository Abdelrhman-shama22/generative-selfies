# generative-selfies
A generative model project for creating molecular SELFIES using the GDP_9 dataset.
## 📂 Dataset
The original dataset (`GDP_9.sdf`) is too large to be uploaded to GitHub.  
You can download it from [Google Drive](https://drive.google.com/file/d/1vtCJJbdy_n-sOGQsPSoubUD2GPf6yktx/view?usp=sharing).
## ⚙️ Requirements
pip install rdkit pandas selfies tensorflow numpy.
Check generated results in generated_selfies.csv .
---
In this project, we extracted SMILES representations from the GDP9.sdf dataset using the RDKit library.
To ensure the validity of the molecules, we converted the SMILES into SELFIES representations.
These SELFIES were then used to train a VAE model with LSTM and GRU architectures, in order to generate new SELFIES.
The generated molecules can be used in drug discovery and pharmaceutical research.


