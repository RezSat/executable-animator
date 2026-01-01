"""
    : Binary file to Music/Sound + Visual Generation using machine-code-ish patterns
    Author: <RezSat | Yehan Wasura>
    Email: wasurayehan@gmail.com
"""
from pathlib import Path
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Turn a binary file into sound + visuals")
    ap.add_argument("path", type=Path, help="Input binary file")
    args = ap.parse_args()

    raw = args.path.read_bytes()
    if not raw:
        raise SystemExit("Input file is empty")

    data = np.frombuffer(raw, dtype=np.uint8)
    print("bytes:", data.size, "min:", data.min(), "max:", data.max())

if __name__ == "__main__":
    main()
