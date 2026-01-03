"""
    : Binary file to Music/Sound + Visual Generation using machine-code-ish patterns
    Author: <RezSat | Yehan Wasura>
    Email: wasurayehan@gmail.com
"""
from pathlib import Path
import argparse
import numpy as np
import math
import wave
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Features:
    entropy_bits: float   # 0..8 for byte distribution
    mean: float           # 0..255
    std: float            # 0..~128
    nibble_hist: np.ndarray  # shape (16,), normalized


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

def extract_features(data: np.ndarray, window_bytes: int, stride_bytes: int) -> tuple[list[Features], np.ndarray]:
    """
    This extract all the necessary features from the data
    window_bytes - or window_size is how much data we summarize at once
    stride_bytes - tells how far to move in each step

    For large files: increase the window_bytes or stride_bytes to reduce 
                     runtime and audio length
    """
    feats: list[Features] = []
    starts = np.arange(0, data.size, stride_bytes, dtype=np.int64)
    for s in starts:
        chunk = data[s:s + window_bytes]
        ent = shannon_entropy_bits(chunk)
        mu = float(chunk.mean()) if chunk.size else 0.0
        sd = float(chunk.std()) if chunk.size else 0.0
        nh = nibble_histogram(chunk)
        feats.append(Features(entropy_bits=ent, mean=mu, std=sd, nibble_hist=nh))
    return feats, starts

def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

def quantize_to_scale(midi_note: float, scale_semitones: list[int]) -> float:
    # snap midi_note to nearest note in repeating scale
    base = math.floor(midi_note / 12.0) * 12
    candidates = [base + s for s in scale_semitones] + [base + 12 + s for s in scale_semitones]
    return float(min(candidates, key=lambda x: abs(x - midi_note)))

def synth_note(freq_hz: float, dur_s: float, sr: int, amp: float, brightness: float) -> np.ndarray:
    """
    Simple FM-ish synth:
      - carrier freq = freq_hz
      - mod freq = 2x carrier
      - mod index scales with brightness
    brightness: 0..1 (higher = brighter / more sidebands)
    """
    n = max(1, int(sr * dur_s))
    t = np.arange(n, dtype=np.float64) / sr

    mod_freq = freq_hz * 2.0
    mod_index = 0.5 + 6.0 * float(np.clip(brightness, 0.0, 1.0))
    phase = 2.0 * np.pi * freq_hz * t + mod_index * np.sin(2.0 * np.pi * mod_freq * t)

    # soft saturating mix of sine + a tiny noise bed for texture
    tone = np.sin(phase)

    # Envelope (quick attack, exponential-ish decay)
    attack = max(1, int(0.02 * n))
    env = np.ones(n, dtype=np.float64)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env *= np.exp(-3.0 * t / max(dur_s, 1e-9))

    y = amp * env * tone
    return y

def sonify(features: list[Features], sr: int, note_dur: float,
           midi_low: int, midi_high: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenating notes turns whole file into audio.
    """
    # Pentatonic-ish scale tends to sound less random
    scale = [0, 2, 4, 7, 9]  # major pentatonic within an octave

    pitches = []
    audio = []
    for f in features:
        # pitch from mean (0..255)
        midi = midi_low + (midi_high - midi_low) * (f.mean / 255.0)
        midi_q = quantize_to_scale(midi, scale)
        freq = midi_to_hz(midi_q)

        # amp from entropy (0..8) — more "complex" regions get louder
        amp = 0.08 + 0.35 * (np.clip(f.entropy_bits, 0.0, 8.0) / 8.0)

        # brightness from std (0..~128)
        bright = np.clip(f.std / 90.0, 0.0, 1.0)

        note = synth_note(freq, note_dur, sr, amp, bright)
        audio.append(note)
        pitches.append(midi_q)

    y = np.concatenate(audio) if audio else np.zeros(1, dtype=np.float64)
    y = np.clip(y, -1.0, 1.0)
    return y, np.asarray(pitches, dtype=np.float64)

def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    y16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y16.tobytes())

def main():
    ap = argparse.ArgumentParser(description="Turn a binary file into sound + visuals")
    ap.add_argument("path", type=Path, help="Input binary file")
    args = ap.parse_args()

    raw = args.path.read_bytes()
    if not raw:
        raise SystemExit("Input file is empty")

    data = np.frombuffer(raw, dtype=np.uint8)

    feats, starts = extract_features(data, 2048, 2048)
    print("Windows:", len(feats), "Entropy first:", feats[0].entropy_bits)
    y, pitches = sonify(feats, 44100, 0.08, 36, 84)
    write_wav(Path("sample_output\out.wav"), y, 44100)
    print("Wrote out.wav")





if __name__ == "__main__":
    main()
