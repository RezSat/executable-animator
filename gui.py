"""
    : Binary file to Music/Sound + Visual Generation using machine-code-ish patterns
    Author: <RezSat | Yehan Wasura>
    Email: wasurayehan@gmail.com
"""
from tkinter import Tk
from tkinter import Frame, Label, Button, Entry, StringVar, filedialog

APP_TITLE = "Executable Animator"

class EAGui:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x720")
        self.input_path = StringVar(value="")
        self._build_ui()

    def _build_ui(self):
        top = Frame(self.root, padx=10, pady=8)
        top.pack(fill="x")

        Label(top, text="Input file:").grid(row=0, column=0, sticky="w")
        Entry(top, textvariable=self.input_path, width=80).grid(row=0, column=1, padx=6, sticky="we")

        Button(top, text="Browse…", command=self.on_browse).grid(row=0, column=2, padx=4)

        top.columnconfigure(1, weight=1)

    def on_browse(self):
        fp = filedialog.askopenfilename(title="Select binary / executable", filetypes=[("All files", "*.*")])
        if fp:
            self.input_path.set(fp)


def main():
    root = Tk()
    EAGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
