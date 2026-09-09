"""Decode token IDs back to text using Qwen3 tokenizer"""

import sys
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <token_id1> <token_id2> ...")
        sys.exit(1)

    token_ids = [int(t) for t in sys.argv[1:]]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(text)

if __name__ == "__main__":
    main()
