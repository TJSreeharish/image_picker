import os
import pandas as pd

csv_path = "/workspace/P/dataset/manifest.csv"
selected_dir = "/workspace/P/SELECTED"

# 1. Read the CSV
df = pd.read_csv(csv_path)

# 2. Get list of file names inside /workspace/P/SELECTED
selected_files = set(os.listdir(selected_dir))

# 3. String manipulation: split by '/' and pick the last element (e.g., 'DSC00126.JPG')
filename = df["filepath"].str.split("/").str[-1]

# 4. Check if the filename exists in selected_files; map True -> 1, False -> 0
df["label_image"] = filename.isin(selected_files).astype(int)

# 5. Save the updated CSV
df.to_csv("/workspace/P/dataset/manifest.csv", index=False)

print(df.head())