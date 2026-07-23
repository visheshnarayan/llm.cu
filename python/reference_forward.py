"""Run Qwen3-1.7B forward pass and dump intermediate activations"""

import os 
import numpy as np 
import torch 
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-1.7B"
OUT_DIR = "data/reference"

def save(name, tensor): 
    arr = tensor.detach().cpu().float().numpy()
    arr.tofile(os.path.join(OUT_DIR, name + ".bin"))
    print(f"    {name}: {list(arr.shape)}")

def main(): 
    os.makedirs(OUT_DIR, exist_ok=True)

    # load input tokens 
    import struct
    with open("data/input_tokens.bin", "rb") as f: 
        n = struct.unpack("i", f.read(4))[0]
        token_ids = [struct.unpack("i", f.read(4))[0] for _ in range(n)]

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    print(f"Input: {token_ids}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()

    with torch.no_grad():
        # embeddings
        hidden = model.model.embed_tokens(input_ids)
        save("embed_out", hidden.squeeze(0))

        # each transformer layer 
        for i, layer in enumerate(model.model.layers): 
            # attn norm 
            normed = layer.input_layernorm(hidden)
            save(f"layer_{i:02d}_attn_norm", normed.squeeze(0))

            # full layer forward
            hidden = layer(hidden)[0]
            save(f"layer_{i:02d}_out", hidden.squeeze(0))

        # final norm 
        hidden = model.model.norm(hidden)
        save("final_norm", hidden.squeeze(0))

        # logits (tied weights) 
        logits = model.lm_head(hidden)
        save("logits", logits.squeeze(0))

    print(f"\nDumped activations to {OUT_DIR}/")

if __name__ == "__main__":
    main()

