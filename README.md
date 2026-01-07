# Executable Animator

Turn any binary (EXE/DLL/SO/OBJ/firmware dump, etc.) into **sound + visuals** based on “machine-code-ish” byte patterns.

This project has two parts:

- **`main.py`** — the worker: reads a binary, extracts rolling byte features, generates:
  - a **WAV** (sonification)
  - a **PNG** dashboard (visual fingerprint)
- **`gui.py`** — a simple GUI wrapper around `main.py`:
  - file picker + **Generate**
  - **Play / Stop / Loop**
  - 3 visualizer presets (oscilloscope / spectrum bars / radial pulse)
  - zoom + pan PNG viewer with cursor anchored zoom

---

## Why this is interesting (at least for me)

Even without fully disassembling the binary, executables often contain recognizable *structural regions* (code, padding, string tables, resources, compressed/encrypted blobs, etc.).  
By summarizing byte windows with simple statistics (entropy, histograms, nibble patterns) and mapping them into audio + plots, you get a weird but useful "signature" you can compare across files.

---

## Output files

For each run you’ll get:

- `*.wav` — audio generated from window by window features
- `*.png` — dashboard image:
  - byte-image ("DNA strip")
  - entropy over time (+ mapped pitch)
  - byte histogram
  - nibble heatmap over time

The GUI saves outputs under `./outputs/` with a timestamped prefix.

---

## Install

> Python 3.10+ recommended. (mine is 3.14.0)

### 1) Create a venv (recommended)

```bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# Linux/macOS:
#   source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### Audio playback note (PortAudio)

`sounddevice` uses **PortAudio**.

- **Linux (Debian/Ubuntu)**:
  ```bash
  sudo apt-get install libportaudio2
  ```
- **macOS (Homebrew)**:
  ```bash
  brew install portaudio
  ```

`Pillow` is optional but recommended for smoother image zooming (it’s included in `requirements.txt`).

---

## Usage

### A) Run the GUI (recommended)

```bash
python gui.py
```

Optional: point the GUI at a specific worker path (I dont know why? but in case you have a different worker file):

```bash
python gui.py --main /path/to/main.py
```

**GUI workflow**
1. Browse → select a binary
2. Generate → creates WAV + PNG
3. PNG appears on the left (zoom/pan)
4. Visualizer runs on the right during playback
5. Tweak parameters → Regenerate

---

### B) Run the worker directly (CLI)

```bash
python main.py /path/to/file.bin -o out
# outputs: out.wav, out.png
```

Useful parameters:

```bash
python main.py input.exe -o out \
  --window_bytes 2048 \
  --stride_bytes 2048 \
  --sr 44100 \
  --note_dur 0.08 \
  --midi_low 36 \
  --midi_high 84
```

**Tuning tips**
- Large files: increase `--stride_bytes` (fewer windows → faster + shorter audio)
- More detail: decrease `--stride_bytes` (more windows → more detail + longer audio)
- Slower / more legible audio: increase `--note_dur`
- More “musical range”: widen `--midi_low` / `--midi_high`

---

## How it works (high level)

1. **Read bytes** as `uint8`
2. Slice into windows (`window_bytes`, step by `stride_bytes`)
3. Per window compute features like:
   - Shannon entropy (0–8 bits)
   - mean / std
   - nibble histogram (0–15 distribution)
4. **Sonify** each window into a short note:
   - mean → pitch (quantized to a pentatonic-ish scale)
   - entropy → loudness
   - std → "brightness" (FM-ish modulation index)
5. **Visualize**:
   - byte-image (reshape bytes into rows)
   - entropy trace (and pitch overlay)
   - byte histogram
   - nibble heatmap

---

## Project layout

```
.
├── gui.py # GUI wrapper + player + visualizer
├── main.py # Worker: generate WAV + PNG
├── requirements.txt
└── outputs/ # Generated files (created at runtime)
```

---

## Troubleshooting

### “Audio playback unavailable” / `sounddevice` errors
- Install PortAudio (see install section).
- Make sure your system has an active audio output device.

### PNG preview looks blocky when zooming
Install Pillow:
```bash
pip install pillow
```

### Matplotlib / Tk backend issues
This project uses `TkAgg`. If your environment lacks Tk support, install your OS Tk packages (Linux often needs `python3-tk`).