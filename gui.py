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
import argparse
from pathlib import Path
import sys
from tkinter import Tk
from tkinter import Frame, Label, Button, Entry, StringVar, filedialog

APP_TITLE = "Executable Animator"

class EAGui:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x720")
        self.input_path = StringVar(value="")

        #state
        self.wav_path = StringVar(value="")
        self.png_path = StringVar(value="")
        self.status = StringVar(value="Pick a file to begin.")

        self._build_ui()

    def _build_ui(self):
        top = Frame(self.root, padx=10, pady=8)
        top.pack(fill="x")

        Label(top, text="Input file:").grid(row=0, column=0, sticky="w")
        Entry(top, textvariable=self.input_path, width=80).grid(row=0, column=1, padx=6, sticky="we")

        Button(top, text="Browse…", command=self.on_browse).grid(row=0, column=2, padx=4)

        top.columnconfigure(1, weight=1)

        #output paths
        out = Frame(self.root, padx=10, pady=4)
        out.pack(fill="x")
        Label(out, text="WAV:").grid(row=0, column=0, sticky="w")
        Label(out, textvariable=self.wav_path, anchor="w").grid(row=0, column=1, sticky="we")
        Label(out, text="PNG:").grid(row=1, column=0, sticky="w")
        Label(out, textvariable=self.png_path, anchor="w").grid(row=1, column=1, sticky="we")
        out.columnconfigure(1, weight=1)

        status = Frame(self.root, padx=10, pady=4)
        status.pack(fill="x")
        Label(status, textvariable=self.status, anchor="w").pack(fill="x")


    def on_browse(self):
        fp = filedialog.askopenfilename(title="Select binary / executable", filetypes=[("All files", "*.*")])
        if fp:
            self.input_path.set(fp)
            self.status.set("Selected file. Click Generate.")

def locate_main_py() -> Path:
    here = Path(__file__).resolve().parent
    cand = here / "main.py"
    if cand.exists():
        return cand
    cand = Path.cwd() / "main.py"
    if cand.exists():
        return cand
    raise FileNotFoundError("Could not find main.py. Put this GUI next to main.py.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, default=None)
    args = ap.parse_args()
    main_py = args.main if args.main else locate_main_py()

    root = Tk()
    EAGui(root)  # we'll pass main_py in the next commit
    root.mainloop()


if __name__ == "__main__":
    main()
