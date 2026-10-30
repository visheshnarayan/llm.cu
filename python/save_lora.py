"""
LoRA weight utilities: save/load binary files produced by llm_train,
and convert to/from HuggingFace PEFT format for verification.

Usage:
  # inspect what was saved
  python python/save_lora.py --list --step 100

  # convert to HF PEFT state_dict
  python python/save_lora.py --to-hf --step 100 --out weights/lora_hf_step100.pt

  # load a PEFT adapter and export to binary for the CUDA engine
  python python/save_lora.py --from-hf --adapter path/to/adapter --step 0
"""

import argparse
import os
import struct
import numpy as np

# Qwen3-1.7B dimensions (must match config.h)
NUM_LAYERS    = 28
NUM_HEADS     = 16
NUM_KV_HEADS  = 8
HEAD_DIM      = 128
HIDDEN_SIZE   = 2048
LORA_RANK     = 16

Q_DIM  = NUM_HEADS    * HEAD_DIM   # 2048
KV_DIM = NUM_KV_HEADS * HEAD_DIM   # 1024

ADAPTER_DIMS = {
    "wq": (Q_DIM,  HIDDEN_SIZE),
    "wk": (KV_DIM, HIDDEN_SIZE),
    "wv": (KV_DIM, HIDDEN_SIZE),
    "wo": (HIDDEN_SIZE, Q_DIM),
}

HF_NAME_MAP = {
    "wq": "self_attn.q_proj",
    "wk": "self_attn.k_proj",
    "wv": "self_attn.v_proj",
    "wo": "self_attn.o_proj",
}


def bin_path(step: int, layer: int, proj: str, matrix: str, weights_dir: str = "weights") -> str:
    return os.path.join(weights_dir, f"lora_step{step}_layer{layer}_{proj}_{matrix}.bin")


def load_bin(path: str, shape: tuple) -> np.ndarray:
    n = 1
    for d in shape:
        n *= d
    with open(path, "rb") as f:
        arr = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
    return arr.reshape(shape)


def save_bin(path: str, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(arr.astype(np.float32).tobytes())


def list_weights(step: int, weights_dir: str = "weights") -> None:
    total_params = 0
    for l in range(NUM_LAYERS):
        for proj, (d_out, d_in) in ADAPTER_DIMS.items():
            pa = bin_path(step, l, proj, "A", weights_dir)
            pb = bin_path(step, l, proj, "B", weights_dir)
            if os.path.exists(pa) and os.path.exists(pb):
                nA = LORA_RANK * d_in
                nB = d_out * LORA_RANK
                total_params += nA + nB
                print(f"  layer {l:2d}  {proj}  A={LORA_RANK}x{d_in}  B={d_out}x{LORA_RANK}")
            else:
                print(f"  layer {l:2d}  {proj}  MISSING")
    print(f"\nTotal LoRA parameters: {total_params:,}  ({total_params * 4 / 1e6:.1f} MB)")


def to_hf(step: int, out_path: str, weights_dir: str = "weights") -> None:
    """Build a PEFT-compatible state_dict and save with torch."""
    import torch
    state = {}
    for l in range(NUM_LAYERS):
        for proj, (d_out, d_in) in ADAPTER_DIMS.items():
            hf_proj = HF_NAME_MAP[proj]
            A = load_bin(bin_path(step, l, proj, "A", weights_dir), (LORA_RANK, d_in))
            B = load_bin(bin_path(step, l, proj, "B", weights_dir), (d_out, LORA_RANK))
            # PEFT stores lora_A as [r, d_in] and lora_B as [d_out, r]
            key_A = f"base_model.model.model.layers.{l}.{hf_proj}.lora_A.weight"
            key_B = f"base_model.model.model.layers.{l}.{hf_proj}.lora_B.weight"
            state[key_A] = torch.from_numpy(A)
            state[key_B] = torch.from_numpy(B)
    torch.save(state, out_path)
    print(f"Saved HF PEFT state_dict to {out_path}  ({len(state)} tensors)")


def from_hf(adapter_path: str, step: int, weights_dir: str = "weights") -> None:
    """Load a PEFT adapter and export binary files for the CUDA engine."""
    import torch
    state = torch.load(adapter_path, map_location="cpu")
    os.makedirs(weights_dir, exist_ok=True)
    for l in range(NUM_LAYERS):
        for proj, (d_out, d_in) in ADAPTER_DIMS.items():
            hf_proj = HF_NAME_MAP[proj]
            key_A = f"base_model.model.model.layers.{l}.{hf_proj}.lora_A.weight"
            key_B = f"base_model.model.model.layers.{l}.{hf_proj}.lora_B.weight"
            A = state[key_A].float().numpy()   # [r, d_in]
            B = state[key_B].float().numpy()   # [d_out, r]
            save_bin(bin_path(step, l, proj, "A", weights_dir), A)
            save_bin(bin_path(step, l, proj, "B", weights_dir), B)
    print(f"Exported {NUM_LAYERS * 4 * 2} binary files to {weights_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step",     type=int, default=0)
    p.add_argument("--list",     action="store_true")
    p.add_argument("--to-hf",   action="store_true")
    p.add_argument("--from-hf", action="store_true")
    p.add_argument("--adapter",  type=str, default="")
    p.add_argument("--out",      type=str, default="weights/lora_hf.pt")
    p.add_argument("--weights-dir", type=str, default="weights")
    args = p.parse_args()

    if args.list:
        list_weights(args.step, args.weights_dir)
    elif args.to_hf:
        to_hf(args.step, args.out, args.weights_dir)
    elif args.from_hf:
        if not args.adapter:
            print("--adapter required"); raise SystemExit(1)
        from_hf(args.adapter, args.step, args.weights_dir)
    else:
        p.print_help()
