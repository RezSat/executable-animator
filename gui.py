"""
    : Binary file to Music/Sound + Visual Generation using machine-code-ish patterns
    Author: <RezSat | Yehan Wasura>
    Email: wasurayehan@gmail.com

    I use argparse so the GUI can be launched with different configurations 
    (ex:a custom path to main.py) without editing the source, 
    which makes the tool easier to test, share, and run in different environments. 
    It also helps me practice controlling other scripts via command-line arguments 
    (and honestly… because I’m bored  touch the main code and it’s fun).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import wave
import subprocess
from dataclasses import dataclass
from pathlib import Path
from fractions import Fraction
from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, IntVar, DoubleVar, BooleanVar,
    filedialog, ttk, messagebox, Canvas, Scrollbar
)

import numpy as np

# Matplotlib embed for Tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Optional smooth image scaling
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

# Modern audio playback
try:
    import sounddevice as sd
except Exception:
    sd = None

APP_TITLE = "Executable Animator"
OUTPUT_DIR = Path("outputs")


@dataclass
class RenderConfig:
    window_bytes: int
    stride_bytes: int
    sr: int
    note_dur: float
    midi_low: int
    midi_high: int


def human_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def run_main(main_py: Path, input_path: Path, out_prefix: Path, cfg: RenderConfig) -> tuple[Path, Path]:
    """Run main.py to generate out_prefix.wav and out_prefix.png"""
    cmd = [
        sys.executable, str(main_py),
        str(input_path),
        "-o", str(out_prefix),
        "--window_bytes", str(cfg.window_bytes),
        "--stride_bytes", str(cfg.stride_bytes),
        "--sr", str(cfg.sr),
        "--note_dur", str(cfg.note_dur),
        "--midi_low", str(cfg.midi_low),
        "--midi_high", str(cfg.midi_high),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "main failed.")
    wav = out_prefix.with_suffix(".wav")
    png = out_prefix.with_suffix(".png")
    if not wav.exists() or not png.exists():
        raise RuntimeError("main finished but outputs were not found.")
    return wav, png


def read_wav_mono_int16(path: Path) -> tuple[np.ndarray, int]:
    """Return (samples_int16, sample_rate). Assumes 16-bit PCM mono (or stereo averaged)."""
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    if sampwidth != 2:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth*8} bits (expected 16-bit).")

    x = np.frombuffer(frames, dtype=np.int16)

    if nch == 2:
        x = x.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif nch != 1:
        raise RuntimeError(f"Unsupported channel count: {nch}.")

    return x, sr


class Visualizer:
    """Matplotlib-based audio visualizer with 3 presets."""
    def __init__(self, parent: Frame):
        self.parent = parent
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self.preset = "Oscilloscope"
        self.samples = np.zeros(1, dtype=np.int16)
        self.sr = 44100

        self._artists = []
        self._init_plot()

    def set_audio(self, samples: np.ndarray, sr: int):
        self.samples = samples if samples.size else np.zeros(1, dtype=np.int16)
        self.sr = sr
        self._init_plot()

    def set_preset(self, preset: str):
        self.preset = preset
        self._init_plot()

    def _init_plot(self):
        self.fig.clf()
        polar = (self.preset == "Radial Pulse")
        self.ax = self.fig.add_subplot(111, polar=polar)
        self._artists = []

        if self.preset == "Oscilloscope":
            self.ax.set_title("Oscilloscope")
            self.ax.set_xlabel("Time (ms)")
            self.ax.set_ylabel("Amplitude")
            line, = self.ax.plot([], [])
            self._artists = [line]
            self.ax.set_xlim(0, 80)
            self.ax.set_ylim(-32768, 32767)

        elif self.preset == "Spectrum Bars":
            self.ax.set_title("Spectrum Bars")
            self.ax.set_xlabel("Frequency (Hz)")
            self.ax.set_ylabel("Magnitude (dB)")
            nb = 64
            freqs = np.linspace(0, self.sr / 2, nb)
            width = (self.sr / 2) / nb * 0.9
            bars = self.ax.bar(freqs, np.zeros_like(freqs), width=width, align="center")
            self._artists = list(bars)
            self.ax.set_xlim(0, self.sr / 2)
            self.ax.set_ylim(-90, 0)

        elif self.preset == "Radial Pulse":
            self.ax.set_title("Radial Pulse")
            theta = np.linspace(0, 2*np.pi, 128, endpoint=False)
            r = np.ones_like(theta) * 0.2
            line, = self.ax.plot(theta, r)
            self._artists = [line]
            self.ax.set_ylim(0, 1.5)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        else:
            self.ax.set_title("Visualizer")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update(self, t_seconds: float):
        if self.samples.size < 2:
            return

        idx = int(t_seconds * self.sr)
        idx = max(0, min(idx, self.samples.size - 1))

        if self.preset == "Oscilloscope":
            win_ms = 80.0
            win = int(self.sr * (win_ms / 1000.0))
            start = max(0, idx - win // 2)
            end = min(self.samples.size, start + win)
            seg = self.samples[start:end]
            if seg.size < 2:
                return
            x_ms = (np.arange(seg.size) / self.sr) * 1000.0
            line = self._artists[0]
            line.set_data(x_ms, seg)
            self.ax.set_xlim(0, max(1.0, float(x_ms[-1])))
            self.canvas.draw_idle()

        elif self.preset == "Spectrum Bars":
            win = 2048
            start = max(0, idx - win)
            seg = self.samples[start:idx].astype(np.float64)
            if seg.size < 256:
                return
            seg = seg * np.hanning(seg.size)
            spec = np.fft.rfft(seg)
            mag = np.abs(spec) + 1e-9
            mag_db = 20 * np.log10(mag / mag.max())
            freqs = np.fft.rfftfreq(seg.size, d=1/self.sr)

            nb = 64
            bins = np.linspace(0, freqs.max(), nb + 1)
            y = np.full(nb, -90.0, dtype=np.float64)
            for i in range(nb):
                mask = (freqs >= bins[i]) & (freqs < bins[i+1])
                if mask.any():
                    y[i] = float(np.max(mag_db[mask]))

            for bar, val in zip(self._artists, y):
                bar.set_height(val)
            self.canvas.draw_idle()

        elif self.preset == "Radial Pulse":
            win = 4096
            start = max(0, idx - win)
            seg = self.samples[start:idx].astype(np.float64)
            if seg.size < 512:
                return
            seg = seg * np.hanning(seg.size)
            rms = float(np.sqrt(np.mean(seg**2))) / 32768.0
            rms = min(1.0, max(0.0, rms))

            spec = np.abs(np.fft.rfft(seg)) + 1e-9
            spec /= spec.max()
            theta = np.linspace(0, 2*np.pi, 128, endpoint=False)
            s = np.interp(np.linspace(0, spec.size-1, 128), np.arange(spec.size), spec)
            r = 0.15 + 0.95 * (0.35*rms + 0.65*s)
            line = self._artists[0]
            line.set_data(theta, r)
            self.canvas.draw_idle()


class mainGui:
    def __init__(self, root: Tk, main_py: Path):
        self.root = root
        self.main_py = main_py

        self.root.title(APP_TITLE)
        self.root.geometry("1200x720")

        # State
        self.input_path = StringVar(value="")
        self.wav_path = StringVar(value="")
        self.png_path = StringVar(value="")
        self.status = StringVar(value="Pick a file to begin.")
        self.preset = StringVar(value="Oscilloscope")

        self.loop = BooleanVar(value=False)

        # Parameters
        self.window_bytes = IntVar(value=2048)
        self.stride_bytes = IntVar(value=2048)
        self.sr = IntVar(value=44100)
        self.note_dur = DoubleVar(value=0.08)
        self.midi_low = IntVar(value=36)
        self.midi_high = IntVar(value=84)

        # Audio playback state
        self._samples = np.zeros(1, dtype=np.int16)
        self._sr_loaded = 44100
        self._stream = None
        self._samples_f32 = np.zeros(1, dtype=np.float32)
        self._play_idx = 0
        self._play_finished = False
        self._is_playing = False

        # Image preview state
        self._zoom = 1.0
        self._zoom_min = 0.1
        self._zoom_max = 12.0
        self._img_orig_tk = None # tk.PhotoImage fallback
        self._img_orig_pil = None # PIL.Image if available
        self._img_disp = None # PhotoImage currently displayed
        self._canvas_img_id = None

        # UI
        self._build_ui()

        # Visualizer
        self.visualizer = Visualizer(self.right_panel)
        self.preset.trace_add("write", lambda *_: self.visualizer.set_preset(self.preset.get()))

        # Periodic UI update loop
        self._tick()

    def _build_ui(self):
        # Top controls
        top = Frame(self.root, padx=10, pady=8)
        top.pack(fill="x")

        Label(top, text="Input file:").grid(row=0, column=0, sticky="w")
        Entry(top, textvariable=self.input_path, width=80).grid(row=0, column=1, padx=6, sticky="we")

        Button(top, text="Browse…", command=self.on_browse).grid(row=0, column=2, padx=4)
        Button(top, text="Generate", command=self.on_generate).grid(row=0, column=3, padx=4)
        Button(top, text="Play", command=self.on_play).grid(row=0, column=4, padx=4)
        Button(top, text="Stop", command=self.on_stop).grid(row=0, column=5, padx=4)

        ttk.Checkbutton(top, text="Loop / Repeat", variable=self.loop).grid(row=0, column=6, padx=10)
        top.columnconfigure(1, weight=1)

        # Parameters row
        params = Frame(self.root, padx=10, pady=6)
        params.pack(fill="x")

        def add_spin(label, var, frm, row, col, from_, to_, inc, width=8):
            Label(frm, text=label).grid(row=row, column=col, sticky="w", padx=(0,4))
            sb = ttk.Spinbox(frm, textvariable=var, from_=from_, to=to_, increment=inc, width=width)
            sb.grid(row=row, column=col+1, sticky="w", padx=(0,12))
            return sb

        add_spin("window_bytes", self.window_bytes, params, 0, 0, 256, 65536, 256)
        add_spin("stride_bytes", self.stride_bytes, params, 0, 2, 256, 65536, 256)
        add_spin("sample_rate", self.sr, params, 0, 4, 8000, 96000, 1000)

        Label(params, text="note_dur").grid(row=0, column=6, sticky="w", padx=(0,4))
        ttk.Spinbox(params, textvariable=self.note_dur, from_=0.01, to=1.0, increment=0.01, width=8).grid(
            row=0, column=7, sticky="w", padx=(0,12)
        )

        add_spin("midi_low", self.midi_low, params, 0, 8, 0, 120, 1, width=6)
        add_spin("midi_high", self.midi_high, params, 0, 10, 0, 120, 1, width=6)

        Label(params, text="Visualizer preset").grid(row=0, column=12, sticky="w", padx=(12,4))
        ttk.OptionMenu(params, self.preset, self.preset.get(), "Oscilloscope", "Spectrum Bars", "Radial Pulse").grid(
            row=0, column=13, sticky="w"
        )

        # Output paths
        out = Frame(self.root, padx=10, pady=4)
        out.pack(fill="x")
        Label(out, text="WAV:").grid(row=0, column=0, sticky="w")
        Label(out, textvariable=self.wav_path, anchor="w").grid(row=0, column=1, sticky="we")
        Label(out, text="PNG:").grid(row=1, column=0, sticky="w")
        Label(out, textvariable=self.png_path, anchor="w").grid(row=1, column=1, sticky="we")
        out.columnconfigure(1, weight=1)

        # Status line
        status = Frame(self.root, padx=10, pady=4)
        status.pack(fill="x")
        Label(status, textvariable=self.status, anchor="w").pack(fill="x")

        # Main panes: left PNG preview + right visualizer
        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=10)

        self.left_panel = Frame(paned, bd=1, relief="solid")
        self.right_panel = Frame(paned, bd=1, relief="solid")
        paned.add(self.left_panel, weight=1)
        paned.add(self.right_panel, weight=1)

        self.left_panel.pack_propagate(False)
        self.right_panel.pack_propagate(False)

        # Left pane: zoom toolbar + scrollable canvas
        left_top = Frame(self.left_panel, padx=6, pady=6)
        left_top.pack(fill="x")

        Button(left_top, text="−", width=3, command=lambda: self._zoom_step(1/1.25)).pack(side="left")
        Button(left_top, text="+", width=3, command=lambda: self._zoom_step(1.25)).pack(side="left", padx=(4, 8))
        Button(left_top, text="Fit", command=self._zoom_fit).pack(side="left")
        Button(left_top, text="100%", command=lambda: self._set_zoom(1.0)).pack(side="left", padx=(6, 0))

        self._zoom_label = Label(left_top, text="Zoom: 100%")
        self._zoom_label.pack(side="right")

        canvas_frame = Frame(self.left_panel)
        canvas_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._img_canvas = Canvas(canvas_frame, highlightthickness=0, bg="black")
        xbar = Scrollbar(canvas_frame, orient="horizontal", command=self._img_canvas.xview)
        ybar = Scrollbar(canvas_frame, orient="vertical", command=self._img_canvas.yview)
        self._img_canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self._img_canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        self._img_canvas.create_text(10, 10, anchor="nw", fill="white", text="PNG preview will appear here.")

        # Pan: click-drag
        self._img_canvas.bind("<ButtonPress-1>", lambda e: self._img_canvas.scan_mark(e.x, e.y))
        self._img_canvas.bind("<B1-Motion>", lambda e: self._img_canvas.scan_dragto(e.x, e.y, gain=1))

        # Zoom: mouse wheel (Windows/macOS) + Button-4/5 (Linux)
        self._img_canvas.bind("<MouseWheel>", self._on_mousewheel) # Windows, macOS (delta)
        self._img_canvas.bind("<Button-4>", lambda e: self._zoom_step(1.25, center=(e.x, e.y))) # Linux up
        self._img_canvas.bind("<Button-5>", lambda e: self._zoom_step(1/1.25, center=(e.x, e.y))) # Linux down

        # Re-render when pane resizes (keeps scrollregion correct)
        self.left_panel.bind("<Configure>", lambda e: self._redraw_image())

        # Bottom controls
        bottom = Frame(self.root, padx=10, pady=8)
        bottom.pack(fill="x")
        Button(bottom, text="Play again", command=self.on_play).pack(side="left")
        Button(bottom, text="Regenerate (with current params)", command=self.on_generate).pack(side="left", padx=8)

        if sd is None:
            Label(bottom, text="(Install sounddevice for playback: pip install sounddevice)", fg="darkred").pack(side="right")
        if (Image is None) or (ImageTk is None):
            Label(bottom, text="(Tip: install Pillow for smooth zoom: pip install pillow)", fg="gray25").pack(side="right", padx=10)

    def on_browse(self):
        fp = filedialog.askopenfilename(title="Select binary / executable", filetypes=[("All files", "*.*")])
        if fp:
            self.input_path.set(fp)
            self.status.set("Selected file. Click Generate.")

    def _current_cfg(self) -> RenderConfig:
        return RenderConfig(
            window_bytes=int(self.window_bytes.get()),
            stride_bytes=int(self.stride_bytes.get()),
            sr=int(self.sr.get()),
            note_dur=float(self.note_dur.get()),
            midi_low=int(self.midi_low.get()),
            midi_high=int(self.midi_high.get()),
        )

    def on_generate(self):
        ip = Path(self.input_path.get().strip())
        if not ip.exists():
            messagebox.showerror("No file", "Please select an existing input file.")
            return

        cfg = self._current_cfg()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_base = ip.name.replace(os.sep, "_")
        out_prefix = OUTPUT_DIR / f"{safe_base}_{ts}"

        self.status.set("Generating WAV + PNG…")
        self.on_stop()

        def worker():
            try:
                wav, png = run_main(self.main_py, ip, out_prefix, cfg)
                samples, sr = read_wav_mono_int16(wav)

                def on_done():
                    self.wav_path.set(human_path(wav))
                    self.png_path.set(human_path(png))
                    self.status.set("Generated. Click Play.")
                    self._samples = samples
                    self._sr_loaded = sr
                    self.visualizer.set_audio(samples, sr)
                    self._load_png_preview(png)

                self.root.after(0, on_done)

            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_error(self, msg: str):
        self.status.set("Error.")
        messagebox.showerror("Error", msg)

    # -------------------------
    # Zoomable image preview
    # -------------------------
    def _load_png_preview(self, png: Path):
        """Load PNG and display it in the left canvas (zoomable & pannable)."""
        self._img_canvas.delete("all")
        self._img_orig_tk = None
        self._img_orig_pil = None
        self._img_disp = None
        self._canvas_img_id = None

        try:
            if Image is not None:
                self._img_orig_pil = Image.open(str(png)).convert("RGB")
            else:
                import tkinter as tk
                self._img_orig_tk = tk.PhotoImage(file=str(png))
        except Exception:
            self._img_canvas.create_text(
                10, 10, anchor="nw", fill="white",
                text=f"PNG generated:\n{human_path(png)}\n\n(Preview requires Tk PNG support; install Pillow for best results.)"
            )
            return

        # Start with "Fit" so we can immediately see the whole image
        self.root.after(0, self._zoom_fit)

    def _on_mousewheel(self, event):
        # Windows: event.delta is +/- 120 per notch; macOS often smaller deltas.
        if event.delta == 0:
            return
        factor = 1.25 if event.delta > 0 else 1/1.25
        self._zoom_step(factor, center=(event.x, event.y))

    def _zoom_step(self, factor: float, center=None):
        self._set_zoom(self._zoom * factor, center=center)

    def _set_zoom(self, z: float, center=None):
        """
        Set zoom level.
        If center is provided as (cx, cy) in *widget* coordinates, keep the image point under the cursor fixed.
        """
        z = float(np.clip(z, self._zoom_min, self._zoom_max))
        old_z = float(self._zoom)
        if abs(z - old_z) < 1e-9:
            return

        # Compute the image-space coordinate (in original pixels) under the cursor BEFORE zoom.
        anchor_orig = None
        if center is not None and old_z > 0:
            cx, cy = center
            # Canvas coords accounting for scroll
            x_canvas = float(self._img_canvas.canvasx(cx))
            y_canvas = float(self._img_canvas.canvasy(cy))
            anchor_orig = (x_canvas / old_z, y_canvas / old_z, float(cx), float(cy))

        self._zoom = z
        self._zoom_label.config(text=f"Zoom: {int(self._zoom * 100)}%")
        self._redraw_image()

        # After redraw, scroll so that the same original image point stays under the cursor.
        if anchor_orig is not None and self._img_disp is not None:
            ox, oy, cx, cy = anchor_orig
            new_x = ox * self._zoom
            new_y = oy * self._zoom

            canvas_w = max(1, self._img_canvas.winfo_width())
            canvas_h = max(1, self._img_canvas.winfo_height())
            scroll_w = max(1, int(self._img_disp.width()))
            scroll_h = max(1, int(self._img_disp.height()))

            # Desired top-left in scroll coords
            left = new_x - cx
            top = new_y - cy

            max_left = max(0, scroll_w - canvas_w)
            max_top = max(0, scroll_h - canvas_h)

            left = float(np.clip(left, 0, max_left))
            top = float(np.clip(top, 0, max_top))

            self._img_canvas.xview_moveto(left / scroll_w if scroll_w else 0.0)
            self._img_canvas.yview_moveto(top / scroll_h if scroll_h else 0.0)

    def _zoom_fit(self):
        """Fit image into the visible canvas area."""
        # Need the canvas size
        cw = max(1, self._img_canvas.winfo_width())
        ch = max(1, self._img_canvas.winfo_height())

        iw, ih = self._img_size()
        if iw <= 0 or ih <= 0:
            return

        # Leave a small margin
        margin = 20
        z = min((cw - margin) / iw, (ch - margin) / ih)
        z = max(self._zoom_min, min(self._zoom_max, z))
        self._zoom = z
        self._zoom_label.config(text=f"Zoom: {int(self._zoom * 100)}%")
        self._redraw_image()
        # Reset scroll to top-left for predictable behavior
        self._img_canvas.xview_moveto(0)
        self._img_canvas.yview_moveto(0)

    def _img_size(self):
        if self._img_orig_pil is not None:
            return self._img_orig_pil.size
        if self._img_orig_tk is not None:
            return (self._img_orig_tk.width(), self._img_orig_tk.height())
        return (0, 0)

    def _redraw_image(self):
        """Redraw the image at the current zoom level and update scroll region."""
        if (self._img_orig_pil is None) and (self._img_orig_tk is None):
            return

        iw, ih = self._img_size()
        if iw <= 0 or ih <= 0:
            return

        zw = max(1, int(round(iw * self._zoom)))
        zh = max(1, int(round(ih * self._zoom)))

        # Render scaled image
        if self._img_orig_pil is not None and ImageTk is not None:
            # Smooth scaling
            try:
                img = self._img_orig_pil.resize((zw, zh), resample=Image.Resampling.LANCZOS)
            except Exception:
                img = self._img_orig_pil.resize((zw, zh))
            self._img_disp = ImageTk.PhotoImage(img)
        else:
            # Tk-only integer scaling using zoom/subsample approximation
            # Approximate zoom as a fraction n/d with small denominator.
            frac = Fraction(self._zoom).limit_denominator(8)
            n, d = frac.numerator, frac.denominator
            n = max(1, min(24, n))
            d = max(1, min(24, d))
            try:
                img = self._img_orig_tk.zoom(n, n).subsample(d, d)
            except Exception:
                img = self._img_orig_tk
            self._img_disp = img

        self._img_canvas.delete("all")
        self._canvas_img_id = self._img_canvas.create_image(0, 0, anchor="nw", image=self._img_disp)
        self._img_canvas.configure(scrollregion=(0, 0, self._img_disp.width(), self._img_disp.height()))

    # -------------------------
    # Audio + playback controls
    # -------------------------
    def on_play(self):
        #ran into all these nonsense errors so i added this
        if sd is None:
            messagebox.showerror(
                "Audio playback unavailable",
                "sounddevice is not installed (or PortAudio is missing).\n\n"
                "Install with:\n  pip install sounddevice\n\n"
                "On Linux you may also need:\n  sudo apt-get install libportaudio2"
            )
            return

        if self._samples.size < 2:
            wp = Path(self.wav_path.get().strip())
            if wp.exists():
                self._samples, self._sr_loaded = read_wav_mono_int16(wp)
                self.visualizer.set_audio(self._samples, self._sr_loaded)
            else:
                messagebox.showinfo("Nothing to play", "Generate first, then Play.")
                return

        self.on_stop()

        self._samples_f32 = (self._samples.astype(np.float32) / 32768.0)
        self._play_idx = 0
        self._play_finished = False
        self._is_playing = True
        self.status.set("Playing…")

        def callback(outdata, frames, time_info, status):
            start = self._play_idx
            end = start + frames
            chunk = self._samples_f32[start:end]
            if chunk.size < frames:
                outdata[:chunk.size, 0] = chunk
                outdata[chunk.size:, 0] = 0.0
                self._play_idx = start + chunk.size
                raise sd.CallbackStop
            outdata[:, 0] = chunk
            self._play_idx = end

        def finished_callback():
            self._play_finished = True

        try:
            self._stream = sd.OutputStream(
                samplerate=int(self._sr_loaded),
                channels=1,
                dtype="float32",
                callback=callback,
                finished_callback=finished_callback,
                blocksize=0,
            )
            self._stream.start()
        except Exception as e:
            self._is_playing = False
            self._stream = None
            self._show_error(str(e))

    def on_stop(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        self._is_playing = False
        self._play_finished = False

    def _tick(self):
        if self._is_playing and self._stream is not None:
            t = float(self._play_idx) / float(self._sr_loaded if self._sr_loaded else 1)
            self.visualizer.update(t)

            if self._play_finished or (not self._stream.active):
                self._is_playing = False
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

                if self.loop.get():
                    self.on_play()
                else:
                    self.status.set("Finished. Play again or tweak params & regenerate.")

        self.root.after(30, self._tick)


def locate_main_py() -> Path:
    # really want to keep the generation and gui seperately runnable scripts.
    here = Path(__file__).resolve().parent
    cand = here / "main.py"
    if cand.exists():
        return cand
    cand = Path.cwd() / "main.py"
    if cand.exists():
        return cand
    raise FileNotFoundError("Could not find main.py. Put gui.py next to main.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, default=None, help="Path to main.py (optional)")
    args = ap.parse_args()

    main_py = args.main if args.main else locate_main_py()

    root = Tk()
    mainGui(root, main_py)
    root.mainloop()


if __name__ == "__main__":
    main()
