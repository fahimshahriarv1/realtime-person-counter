"""Realtime Person Counter & Namer

A simple Tkinter GUI that opens the webcam, detects faces in realtime,
tracks each one across frames, and counts how many people are currently
in view. The first time a face is seen it's "Unidentified" for a brief
moment; once detection is stable you're prompted to type a name for it,
which trains the recognizer so that person is remembered on future runs.

Run with:  python src/main.py
"""

import os
import sys
import tkinter as tk
from collections import deque
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_engine import FaceEngine  # noqa: E402
from tracker import CentroidTracker  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CASCADE_PATH = os.path.join(BASE_DIR, "assets", "haarcascade_frontalface_default.xml")

FRAME_INTERVAL_MS = 30
STABILITY_FRAMES = 10   # consecutive "unknown" frames before we prompt for a name
BUFFER_SIZE = 8         # face crops saved per new person for training


class TrackState:
    def __init__(self):
        self.name = None
        self.unknown_frames = 0
        self.prompted = False
        self.skipped = False
        self.buffer = deque(maxlen=BUFFER_SIZE)


class PersonCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Realtime Person Counter")
        self.root.geometry("980x580")
        self.root.minsize(820, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.engine = FaceEngine(DATA_DIR, CASCADE_PATH)
        self.tracker = CentroidTracker()
        self.track_state = {}

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open a webcam (device 0).")
            self.root.after(100, self.root.destroy)
            return

        self._build_ui()
        self.root.after(FRAME_INTERVAL_MS, self.update_frame)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        video_frame = ttk.Frame(main)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(main, width=260, padding=(12, 0, 0, 0))
        side.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(side, text="People in room", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.count_var = tk.StringVar(value="0")
        ttk.Label(side, textvariable=self.count_var, font=("Segoe UI", 40, "bold")).pack(
            anchor="w", pady=(0, 12)
        )

        ttk.Label(side, text="Currently visible:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        self.people_list = tk.Listbox(side, height=18, font=("Segoe UI", 10))
        self.people_list.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        ttk.Label(
            side,
            text="New faces are prompted for a name automatically\n"
            "once detection is stable.",
            foreground="#555555",
            wraplength=240,
            justify="left",
        ).pack(anchor="w")

    def prompt_for_name(self, state):
        name = simpledialog.askstring(
            "New person detected",
            "Enter a name for this person\n(leave blank or Cancel to skip):",
            parent=self.root,
        )
        if name and name.strip():
            self.engine.add_person(name.strip(), list(state.buffer))
            state.name = name.strip()
        else:
            state.skipped = True
        state.buffer.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(FRAME_INTERVAL_MS, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = self.engine.detect_faces(gray)
        tracked = self.tracker.update(rects)

        for tid in list(self.track_state.keys()):
            if tid not in tracked:
                del self.track_state[tid]

        current_names = []
        for tid, (x, y, w, h) in tracked.items():
            state = self.track_state.setdefault(tid, TrackState())
            crop = gray[y : y + h, x : x + w]

            if state.name is None and not state.skipped and crop.size > 0:
                name, _ = self.engine.predict(crop)
                if name:
                    state.name = name
                else:
                    state.buffer.append(crop.copy())
                    state.unknown_frames += 1
                    if state.unknown_frames >= STABILITY_FRAMES and not state.prompted:
                        state.prompted = True
                        self.prompt_for_name(state)

            if state.name:
                label = state.name
                color = (60, 180, 75)
                current_names.append(state.name)
            elif state.skipped:
                label = "Unnamed"
                color = (150, 150, 150)
            else:
                label = "Identifying..."
                color = (0, 165, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        self.count_var.set(str(len(tracked)))
        self.people_list.delete(0, tk.END)
        for tid, (x, y, w, h) in tracked.items():
            state = self.track_state.get(tid)
            display = state.name if state and state.name else f"Person #{tid} (unnamed)"
            self.people_list.insert(tk.END, display)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(FRAME_INTERVAL_MS, self.update_frame)

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    PersonCounterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
