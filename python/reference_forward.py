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
        MODEL_ID, dtype=torch.float32
    )
    model.eval()

    # register hooks to capture intermediate activations
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    # hook embedding
    model.model.embed_tokens.register_forward_hook(make_hook("embed_out"))

    # hook each layer's input layernorm and full layer output
    for i, layer in enumerate(model.model.layers):
        layer.input_layernorm.register_forward_hook(make_hook(f"layer_{i:02d}_attn_norm"))
        layer.register_forward_hook(make_hook(f"layer_{i:02d}_out"))

    # hook final norm
    model.model.norm.register_forward_hook(make_hook("final_norm"))

    with torch.no_grad():
        # run full forward pass (handles position embeddings internally)
        outputs = model(input_ids)
        logits = outputs.logits
        save("logits", logits.squeeze(0))

        # save hooked activations
        save("embed_out", activations["embed_out"].squeeze(0))
        for i in range(len(model.model.layers)):
            save(f"layer_{i:02d}_attn_norm", activations[f"layer_{i:02d}_attn_norm"].squeeze(0))
            # layer output is a tuple (hidden_states, ...)
            layer_out = activations[f"layer_{i:02d}_out"]
            if isinstance(layer_out, tuple):
                layer_out = layer_out[0]
            save(f"layer_{i:02d}_out", layer_out.squeeze(0))
        save("final_norm", activations["final_norm"].squeeze(0))

    print(f"\nDumped activations to {OUT_DIR}/")

if __name__ == "__main__":
    main()

