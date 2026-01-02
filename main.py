"""
    : Binary file to Music/Sound + Visual Generation using machine-code-ish patterns
    Author: <RezSat | Yehan Wasura>
    Email: wasurayehan@gmail.com
"""
from pathlib import Path
import argparse
import numpy as np

from dataclasses import dataclass

@dataclass
class Features:
    entropy_bits: float   # 0..8 for byte distribution
    mean: float           # 0..255
    std: float            # 0..~128
    nibble_hist: np.ndarray  # shape (16,), normalized


def windows(data: np.ndarray, window_bytes: int, stride_bytes: int):
    """
    window_bytes - or window_size is how much data we summarize at once
    stride_bytes - tells how far to move in each step

    For large files: increase the window_bytes or stride_bytes to reduce 
                     runtime and audio length
    """
    for start in range(0, len(data), stride_bytes):
        yield start, data[start:start+window_bytes]

def shannon_entropy_bits(chunk: np.ndarray) -> float:
    """
    Extracts simple features per window using shannon entropy.
    (entropy in bits for byte distribution)
    This will reveal sections in the executable.
    - High Entropy: compressed or encrypted or packed 
    - Low Entropy: structured or repeated patterns or padding

    """
    if chunk.size == 0:
        return 0.0
    counts = np.bincount(chunk, minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def nibble_histogram(chunk: np.ndarray) -> np.ndarray:
    """
    Checking nibbles (half bytes) to reveal repeating instruction/padding patterns.
    """
    if chunk.size == 0:
        return np.zeros(16, dtype=np.float64)
    lo = chunk & 0x0F
    hi = (chunk >> 4) & 0x0F
    h = np.bincount(lo, minlength=16).astype(np.float64) + np.bincount(hi, minlength=16).astype(np.float64)
    h /= h.sum() if h.sum() else 1.0
    return h

def main():
    ap = argparse.ArgumentParser(description="Turn a binary file into sound + visuals")
    ap.add_argument("path", type=Path, help="Input binary file")
    args = ap.parse_args()

    raw = args.path.read_bytes()
    if not raw:
        raise SystemExit("Input file is empty")

    data = np.frombuffer(raw, dtype=np.uint8)

    #testing stuff up:
    # basic iformation:
    print("bytes:", data.size, "min:", data.min(), "max:", data.max())

    #windows and strides:
    for i, (s, w) in enumerate(windows(data, 2048, 2048)):
        if i == 3: break
        print(s, len(w))
    
    #checking entropy
    ents = []
    for _, w in windows(data, 2048, 2048):
        ents.append(shannon_entropy_bits(w))
    print("entropy range:", min(ents), max(ents))

    #testing nibbles for first window
    print("Nibble hist first window:", nibble_histogram(data[:2048]))



if __name__ == "__main__":
    main()
