import sys 
import struct 
from transformers import AutoTokenizer 

MODEL_ID = "Qwen/Qwen3-1.7B"

def main(): 
    if len(sys.argv) < 2: 
        print(f"Usage: python {sys.argv[0]} \"your prompt here\"")
        sys.exit(1)

    prompt = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    token_ids = tokenizer.encode(prompt) 
     
    out_path = "data/input_tokens.bin"
    with open(out_path, "wb") as f: 
        # header: num of tokens as int32 
        f.write(struct.pack("i", len(token_ids)))
        # token ids as int32 array 
        for tid in token_ids: 
            f.write(struct.pack("i", tid))
    
    print(f"Wrote {len(token_ids)} tokens to {out_path}")
    print(f"Token IDs: {token_ids}")

if __name__ == "__main__":
    main()