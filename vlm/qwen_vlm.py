import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

OUTPUT_DIR = "/workspace/P/dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NPZ_PATH = os.path.join(OUTPUT_DIR, "dataset_vlm.npz")
CSV_PATH = os.path.join(OUTPUT_DIR, "vlm_manifest.csv")

# --- OOM fix 1: 4-bit quantization ---
# 2B params in bf16 alone is ~4-5GB including the vision tower -- already
# tight on a 4GB card before any activation memory. 4-bit cuts weight
# memory roughly 4x, leaving real headroom for per-image activations.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

print(f"Loading {MODEL_ID} on {DEVICE} (4-bit)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer

id_selected = tokenizer.encode("SELECTED", add_special_tokens=False)[0]
id_rejected = tokenizer.encode("REJECTED", add_special_tokens=False)[0]

SELECTED_DIR = "/workspace/P/SELECTED"
ALL_DIR = "/workspace/P/all_images"
VALID_EXTS = {".jpg", ".jpeg", ".png"}

samples = []
for f in sorted(os.listdir(SELECTED_DIR)):
    if os.path.splitext(f.lower())[1] in VALID_EXTS:
        samples.append((os.path.join(SELECTED_DIR, f), 1, f))

for f in sorted(os.listdir(ALL_DIR)):
    if os.path.splitext(f.lower())[1] in VALID_EXTS:
        samples.append((os.path.join(ALL_DIR, f), 0, f))

prompt = (
    "Evaluate this image for photographic curation standards. "
    "Classify whether this image is SELECTED or REJECTED:\n"
    "Classification:"
)

llm_vectors = []
probs_selected = []
filenames = []
filepaths = []
labels = []

# --- OOM fix 2: smaller max_pixels ---
# 768*768 lets larger source images produce many more visual tokens than
# smaller ones, so peak memory varies per image -- this is very likely
# why it ran for a while before dying, rather than failing immediately.
# 448*448 is still plenty for the VLM to judge composition/content.
MAX_PIXELS = 448 * 448

model.eval()
with torch.no_grad():
    for fpath, label, fname in tqdm(samples, desc="Extracting VLM Context Embeddings & Logits"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": fpath, "max_pixels": MAX_PIXELS},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1][0, -1, :]
        llm_vectors.append(last_hidden_state.to(torch.float32).cpu().numpy())

        next_token_logits = outputs.logits[0, -1, :]
        candidate_logits = torch.tensor([next_token_logits[id_rejected], next_token_logits[id_selected]])
        p_selected = F.softmax(candidate_logits, dim=0)[1].item()

        probs_selected.append(p_selected)
        filenames.append(fname)
        filepaths.append(fpath)
        labels.append(label)

        # --- OOM fix 3: explicit cleanup every iteration ---
        # Frees this image's activation memory (hidden states, logits,
        # input tensors) before the next iteration allocates more, instead
        # of relying on Python's garbage collector to catch up eventually.
        del outputs, inputs, next_token_logits, candidate_logits, last_hidden_state
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

X_llm = np.vstack(llm_vectors).astype(np.float32)
y = np.array(labels, dtype=np.int32)
p_selected_arr = np.array(probs_selected, dtype=np.float32)
filenames_arr = np.array(filenames)

np.savez_compressed(
    NPZ_PATH,
    llm_vectors=X_llm,
    probs_selected=p_selected_arr,
    labels=y,
    filenames=filenames_arr,
)
print(f"\nSaved NPZ -> {NPZ_PATH} | Vectors Shape: {X_llm.shape}")

df_manifest = pd.DataFrame(
    {
        "filename": filenames,
        "filepath": filepaths,
        "prob_selected": p_selected_arr,
        "label": y,
    }
)
df_manifest.to_csv(CSV_PATH, index=False)
print(f"Saved CSV -> {CSV_PATH}")

auc = roc_auc_score(y, p_selected_arr)
fpr, tpr, thresholds = roc_curve(y, p_selected_arr)
best_idx = np.argmax(tpr - fpr)
best_thresh = thresholds[best_idx]
preds = (p_selected_arr >= best_thresh).astype(int)

print("\n" + "=" * 50)
print(f"ZERO-SHOT VLM ROC-AUC: {auc:.4f}")
print(f"Optimal Threshold: {best_thresh:.4f}")
print("=" * 50)
print(classification_report(y, preds, target_names=["REJECTED (0)", "SELECTED (1)"]))