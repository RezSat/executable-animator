"""
    : Binary file to Music/Sound Generation
    Author: <RezSat | Yehan Wasura>
    Email: wasurayehan@gmail.com
"""
from pathlib import Path
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    raw = args.path.read_bytes()
    data = np.frombuffer(raw, dtype=np.uint8)
    print("bytes:", data.size, "min:", data.min(), "max:", data.max())

if __name__ == "__main__":
    main()
