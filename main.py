"""
    : Binary file to Music/Sound + Visual Generation using machine-code-ish patterns
    Author: <RezSat | Yehan Wasura>
    Email: wasurayehan@gmail.com
"""
from pathlib import Path
import argparse
import numpy as np


def windows(data: np.ndarray, window_bytes: int, stride_bytes: int):
    """
    window_bytes - or window_size is how much data we summarize at once
    stride_bytes - tells how far to move in each step
    """
    for start in range(0, len(data), stride_bytes):
        yield start, data[start:start+window_bytes]

def main():
    ap = argparse.ArgumentParser(description="Turn a binary file into sound + visuals")
    ap.add_argument("path", type=Path, help="Input binary file")
    args = ap.parse_args()

    raw = args.path.read_bytes()
    if not raw:
        raise SystemExit("Input file is empty")

    data = np.frombuffer(raw, dtype=np.uint8)
    print("bytes:", data.size, "min:", data.min(), "max:", data.max())
    for i, (s, w) in enumerate(windows(data, 2048, 2048)):
        if i == 3: break
        print(s, len(w))


if __name__ == "__main__":
    main()
