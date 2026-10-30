import sys
import struct
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} \"your prompt here\"")
        print(f"       python {sys.argv[0]} --train <text_file>  (writes data/train_tokens.bin)")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if sys.argv[1] == "--train":
        if len(sys.argv) < 3:
            print("Usage: python tokenize_input.py --train <text_file>")
            sys.exit(1)
        with open(sys.argv[2]) as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
        out_path = "data/train_tokens.bin"
        with open(out_path, "wb") as f:
            f.write(struct.pack("i", len(token_ids)))
            for tid in token_ids:
                f.write(struct.pack("i", tid))
        print(f"Wrote {len(token_ids)} tokens to {out_path}")
    else:
        prompt = sys.argv[1]
        token_ids = tokenizer.encode(prompt)
        out_path = "data/input_tokens.bin"
        with open(out_path, "wb") as f:
            f.write(struct.pack("i", len(token_ids)))
            for tid in token_ids:
                f.write(struct.pack("i", tid))
        print(f"Wrote {len(token_ids)} tokens to {out_path}")
        print(f"Token IDs: {token_ids}")

if __name__ == "__main__":
    main()
