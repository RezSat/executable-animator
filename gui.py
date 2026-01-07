"""
    : Binary file to Music/Sound + Visual Generation using machine-code-ish patterns
    Author: <RezSat | Yehan Wasura>
    Email: wasurayehan@gmail.com
"""
from tkinter import Tk

APP_TITLE = "Executable Animator"

class EAGui:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x720")

def main():
    root = Tk()
    EAGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
