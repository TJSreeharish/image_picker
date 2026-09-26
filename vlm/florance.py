import os
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CLIPImageProcessor,
    AutoTokenizer,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# 1. Device and Precision Setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "microsoft/Florence-2-large"

# 2. Patch config to prevent 5.17.0 forced_bos_token_id crash
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
if not hasattr(config, "forced_bos_token_id"):
    config.forced_bos_token_id = None
if hasattr(config, "text_config") and not hasattr(config.text_config, "forced_bos_token_id"):
    config.text_config.forced_bos_token_id = None

# 3. Load Model
print(f"Loading {model_id} on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)
model.eval()

# 4. Assemble Florence2Processor explicitly (bypasses AutoProcessor bug)
image_processor = CLIPImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
Florence2Processor = get_class_from_dynamic_module(
    "processing_florence2.Florence2Processor", model_id
)
processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)

# 5. Load and Prepare Image & Prompt
prompt = "<OD>"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

# 6. Run Generation
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )

# 7. Decode and Post-Process Output
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(
    generated_text,
    task=prompt,
    image_size=(image.width, image.height)
)

print("\n--- Detection Result ---")
print(parsed_answer)