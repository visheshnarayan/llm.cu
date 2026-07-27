"""Compare CUDA activations against reference outputs."""

import sys 
import os 
import numpy as np 

def compare(name, cuda_path, ref_path):
    cuda = np.fromfile(cuda_path, dtype=np.float32)
    ref = np.fromfile(ref_path, dtype=np.float32)

    if cuda.shape != ref.shape: 
        print(f"    FAIL {name}: shape mismatch cuda={cuda.shape} ref={ref.shape}")
        return False 
    
    max_abs = np.max(np.abs(cuda - ref)) 
    mean_abs = np.mean(np.abs(cuda - ref)) 

    # cosine sim 
    dot = np.dot(cuda, ref) 
    norm = np.linalg.norm(cuda) * np.linalg.norm(ref)
    cosine = dot / norm if norm > 0 else 0.0 

    passed = max_abs < 0.02 and cosine > 0.9999
    status = "PASS" if passed else "FAIL" 
    print(f"    {status} {name}: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} cosine={cosine:.6f}")
    return passed 

def main():
    cuda_dir = sys.argv[1] if len(sys.argv) > 1 else "data/cuda_out"
    ref_dir = sys.argv[2] if len(sys.argv) > 2 else "data/reference"

    ref_files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".bin"))
    total, passed = 0, 0 
    for f in ref_files: 
        cuda_path = os.path.join(cuda_dir, f)
        ref_path = os.path.join(ref_dir, f) 
        if not os.path.exists(cuda_path):
            print(f"    SKIP {f}: no CUDA output")
            continue 
        total += 1
        if compare(f, cuda_path, ref_path): 
            passed += 1
        
    print(f"\n{passed}/{total} passed")

if __name__ == "__main__":
    main()
