import os
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import open_clip

# --- Paths ---
ALL_DIR = "/workspace/P/all_images"
SELECTED_DIR = "/workspace/P/SELECTED"
OUTPUT_DIR = "/workspace/P/dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- 1. Load CLIP ViT-B/32 & Aesthetic Predictor ---
print(f"Loading CLIP ViT-B/32 on {DEVICE}...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=DEVICE
)
clip_model.eval()

cache_dir = os.path.expanduser("~/.cache/laion_aesthetic")
os.makedirs(cache_dir, exist_ok=True)
weight_path = os.path.join(cache_dir, "sa_0_4_vit_b_32_linear.pth")

if not os.path.exists(weight_path):
    print("Downloading LAION aesthetic predictor weights...")
    url = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth"
    urllib.request.urlretrieve(url, weight_path)

aesthetic_head = nn.Linear(512, 1)
aesthetic_head.load_state_dict(torch.load(weight_path, map_location=DEVICE))
aesthetic_head.to(DEVICE)
aesthetic_head.eval()

# --- 2. Gather Images Directly From Both Folders ---
samples = []

# Good images from SELECTED (label = 1)
for fname in sorted(os.listdir(SELECTED_DIR)):
    if os.path.splitext(fname.lower())[1] in VALID_EXTS:
        full_path = os.path.join(SELECTED_DIR, fname)
        samples.append((fname, full_path, 1))

# Bad images from all_images (label = 0)
for fname in sorted(os.listdir(ALL_DIR)):
    if os.path.splitext(fname.lower())[1] in VALID_EXTS:
        full_path = os.path.join(ALL_DIR, fname)
        samples.append((fname, full_path, 0))

good_count = sum(s[2] == 1 for s in samples)
bad_count = sum(s[2] == 0 for s in samples)

print(f"Total images gathered: {len(samples)}")
print(f"Good (SELECTED = 1): {good_count}")
print(f"Bad  (all_images = 0): {bad_count}")

# --- 3. Extract Features & Scores ---
embeddings_list = []
aesthetic_scores = []
processed_records = []
batch_size = 64

with torch.no_grad():
    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting Features"):
        batch = samples[i : i + batch_size]
        batch_tensors = []
        valid_batch_samples = []

        for fname, fpath, label in batch:
            try:
                img = Image.open(fpath).convert("RGB")
                batch_tensors.append(preprocess(img))
                valid_batch_samples.append((fname, fpath, label))
            except Exception as e:
                print(f"Skipping corrupt image {fname}: {e}")

        if not batch_tensors:
            continue

        images_tensor = torch.stack(batch_tensors).to(DEVICE)

        image_features = clip_model.encode_image(images_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        scores = aesthetic_head(image_features).squeeze(-1)

        embeddings_list.append(image_features.cpu().numpy().astype(np.float32))
        aesthetic_scores.extend(scores.cpu().numpy().astype(np.float32).tolist())
        processed_records.extend(valid_batch_samples)

# --- 4. Save Outputs ---
final_embeddings = np.vstack(embeddings_list)
final_scores = np.array(aesthetic_scores, dtype=np.float32)
final_labels = np.array([item[2] for item in processed_records], dtype=np.int32)
filenames = [item[0] for item in processed_records]
filepaths = [item[1] for item in processed_records]

npz_out = os.path.join(OUTPUT_DIR, "img_vit_dataset.npz")
np.savez_compressed(
    npz_out,
    embeddings=final_embeddings,
    aesthetic_scores=final_scores,
    labels=final_labels,
    filenames=np.array(filenames),
)

df = pd.DataFrame({
    "filename": filenames,
    "filepath": filepaths,
    "aesthetic_score": final_scores,
    "label": final_labels,
})
csv_out = os.path.join(OUTPUT_DIR, "dataset_manifest.csv")
df.to_csv(csv_out, index=False)

print("\nDone!")
print(f"Saved NPZ -> {npz_out} | Shape: {final_embeddings.shape}")
print(f"Saved CSV -> {csv_out}")
print("Final distribution:\n", df["label"].value_counts())