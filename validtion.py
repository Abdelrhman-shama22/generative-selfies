import pandas as pd
import selfies as sf

file_path = r"C:\abdo123\gdb9\cleaned_selfies.csv"

df = pd.read_csv(file_path)

valid_count = 0
for s in df['generated_selfies']:
    try:
        _ = sf.decoder(s)  
        valid_count += 1
    except Exception:
        pass

print(f"num of valid selfies: {valid_count} from {len(df)}")
