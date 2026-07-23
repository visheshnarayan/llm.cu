"""Download Qwen3-1.7B weights and save as FP32 binary files"""

import os 
import numpy as np 
from transformers import AutoModelForCausalLM
import torch 

MODEL_ID = "Qwen/Qwen3-1.7B"
OUT_DIR = "weights"

def main(): 
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )

    for name, param in model.named_parameters():
        # flatten name: model.layers.0.self_attn.q_proj.weight -> model.layers.0.self_attn.q_proj.weight.bin
        filename = name + ".bin"
        filepath = os.path.join(OUT_DIR, filename)

        data = param.detach().cpu().numpy().astype(np.float32)
        data.tofile(filepath)
        print(f"    {name}: {list(data.shape)} -> {filepath}")

    print(f"\nExported {len(list(model.named_parameters()))} tensors to {OUT_DIR}/")

if __name__ == "__main__":
    main()

