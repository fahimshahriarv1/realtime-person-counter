"""Realtime Person Counter & Namer

A simple Tkinter GUI that opens the webcam, detects faces in realtime,
tracks each one across frames, and counts how many people are currently
in view. The first time a face is seen it's "Unidentified" for a brief
moment; once detection is stable you're prompted to type a name for it,
which trains the recognizer so that person is remembered on future runs.

The app tolerates a missing/disconnected camera: on startup it searches a
few device indices and keeps retrying on a timer until one responds, and
if the camera drops out mid-session it falls back to the same reconnect
loop instead of crashing.

Run with:  python src/main.py
"""

import os
import sys
import tkinter as tk
from collections import deque
from tkinter import simpledialog, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_engine import FaceEngine  # noqa: E402
from tracker import CentroidTracker  # noqa: E402
from notifications import notify  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CASCADE_PATH = os.path.join(BASE_DIR, "assets", "haarcascade_frontalface_default.xml")

FRAME_INTERVAL_MS = 30
STABILITY_FRAMES = 10   # consecutive "unknown" frames before we prompt for a name
BUFFER_SIZE = 8         # face crops saved per new person for training

CAMERA_INDICES_TO_TRY = [0, 1, 2]  # tried in order on every (re)connect attempt
CAMERA_RETRY_MS = 2000             # how often to retry when no camera is connected
READ_FAILURE_TOLERANCE = 8         # consecutive bad reads before declaring "disconnected"
PLACEHOLDER_SIZE = (860, 540)

ALERT_OFF = "Off"
ALERT_WATCHED_ARRIVES = "Watched person(s) appear"
ALERT_OTHER_ARRIVES = "Anyone not watched appears"
ALERT_MODES = [ALERT_OFF, ALERT_WATCHED_ARRIVES, ALERT_OTHER_ARRIVES]


class TrackState:
    def __init__(self):
        self.name = None
        self.unknown_frames = 0
        self.prompted = False
        self.skipped = False
        self.notified = False
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

        self.cap = None
        self.camera_index = None
        self.read_failures = 0
        self.reconnect_elapsed_ms = 0

        self._build_ui()
        self.attempt_connect(reset_status_on_fail=True)
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

        self.status_var = tk.StringVar(value="Connecting to camera...")
        self.status_label = ttk.Label(
            side, textvariable=self.status_var, font=("Segoe UI", 9), foreground="#b36b00"
        )
        self.status_label.pack(anchor="w", pady=(0, 8))

        ttk.Button(side, text="Retry camera now", command=self.retry_now).pack(
            anchor="w", pady=(0, 12)
        )

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
        self.people_list = tk.Listbox(side, height=16, font=("Segoe UI", 10))
        self.people_list.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        ttk.Label(
            side,
            text="New faces are prompted for a name automatically\n"
            "once detection is stable.",
            foreground="#555555",
            wraplength=240,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        notif_frame = ttk.LabelFrame(side, text="Notifications", padding=8)
        notif_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(notif_frame, text="Watched person(s):").pack(anchor="w")
        self.watched_listbox = tk.Listbox(
            notif_frame, selectmode=tk.EXTENDED, exportselection=False, height=5
        )
        self.watched_listbox.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(
            notif_frame,
            text="Ctrl/Shift-click to pick one or more people.",
            foreground="#555555",
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(notif_frame, text="Alert me when:").pack(anchor="w")
        self.alert_mode_var = tk.StringVar(value=ALERT_OFF)
        self.alert_mode_combo = ttk.Combobox(
            notif_frame,
            textvariable=self.alert_mode_var,
            values=ALERT_MODES,
            state="readonly",
        )
        self.alert_mode_combo.pack(fill=tk.X, pady=(2, 8))

        ttk.Button(
            notif_frame, text="Send test notification", command=self.send_test_notification
        ).pack(fill=tk.X)

        self._refresh_watched_person_options()
        self._show_placeholder("Connecting to camera...")

    def _show_placeholder(self, message):
        img = Image.new("RGB", PLACEHOLDER_SIZE, color=(32, 32, 32))
        draw = ImageDraw.Draw(img)
        text = f"\U0001F4F7  {message}"
        bbox = draw.textbbox((0, 0), text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((PLACEHOLDER_SIZE[0] - w) / 2, (PLACEHOLDER_SIZE[1] - h) / 2),
            text,
            fill=(200, 200, 200),
        )
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _set_status(self, text, ok):
        self.status_var.set(text)
        self.status_label.configure(foreground="#1a7f37" if ok else "#c0392b")

    def retry_now(self):
        self.reconnect_elapsed_ms = 0
        self.attempt_connect(reset_status_on_fail=True)

    def _refresh_watched_person_options(self):
        previously_selected = set(self._get_watched_names())
        names = sorted(set(self.engine.labels.values()))

        self.watched_listbox.delete(0, tk.END)
        for name in names:
            self.watched_listbox.insert(tk.END, name)
            if name in previously_selected:
                self.watched_listbox.selection_set(tk.END)

    def _get_watched_names(self):
        return {self.watched_listbox.get(i) for i in self.watched_listbox.curselection()}

    def send_test_notification(self):
        notify("Realtime Person Counter", "Notifications are working.")

    def maybe_notify(self, state):
        """Edge-triggered alert: fires once per track, the moment its
        identity is resolved (recognized, freshly named, or left unnamed)."""
        if state.notified:
            return
        state.notified = True

        mode = self.alert_mode_var.get()
        watched = self._get_watched_names()
        if mode == ALERT_OFF or not watched:
            return

        if mode == ALERT_WATCHED_ARRIVES:
            if state.name in watched:
                notify("Person detected", f"{state.name} has entered the room.")
        elif mode == ALERT_OTHER_ARRIVES:
            if state.name not in watched:
                who = state.name if state.name else "An unrecognized person"
                notify("Unexpected person detected", f"{who} has entered the room.")

    def attempt_connect(self, reset_status_on_fail=False):
        """Try each known camera index once. Returns True on success."""
        for idx in CAMERA_INDICES_TO_TRY:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else 0)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
                        self.camera_index = idx
                        self.read_failures = 0
                        self._set_status(f"Camera connected (device {idx})", ok=True)
                        return True
                if cap is not None:
                    cap.release()
            except cv2.error:
                if cap is not None:
                    cap.release()

        self.cap = None
        if reset_status_on_fail:
            self._set_status("No camera found. Retrying...", ok=False)
            self._show_placeholder("No camera found. Retrying...")
        return False

    def prompt_for_name(self, state):
        name = simpledialog.askstring(
            "New person detected",
            "Enter a name for this person\n(leave blank or Cancel to skip):",
            parent=self.root,
        )
        if name and name.strip():
            self.engine.add_person(name.strip(), list(state.buffer))
            state.name = name.strip()
            self._refresh_watched_person_options()
        else:
            state.skipped = True
        state.buffer.clear()

    def update_frame(self):
        if self.cap is None:
            self.reconnect_elapsed_ms += FRAME_INTERVAL_MS
            if self.reconnect_elapsed_ms >= CAMERA_RETRY_MS:
                self.reconnect_elapsed_ms = 0
                self.attempt_connect(reset_status_on_fail=True)
            self.count_var.set("0")
            self.people_list.delete(0, tk.END)
            self.root.after(FRAME_INTERVAL_MS, self.update_frame)
            return

        ret, frame = None, None
        try:
            ret, frame = self.cap.read()
        except cv2.error:
            ret = False

        if not ret or frame is None:
            self.read_failures += 1
            if self.read_failures >= READ_FAILURE_TOLERANCE:
                self.cap.release()
                self.cap = None
                self.read_failures = 0
                self.reconnect_elapsed_ms = 0
                self._set_status("Camera disconnected. Reconnecting...", ok=False)
                self._show_placeholder("Camera disconnected. Reconnecting...")
            self.root.after(FRAME_INTERVAL_MS, self.update_frame)
            return

        self.read_failures = 0

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = self.engine.detect_faces(gray)
        tracked = self.tracker.update(rects)

        for tid in list(self.track_state.keys()):
            if tid not in tracked:
                del self.track_state[tid]

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

            if state.name or state.skipped:
                self.maybe_notify(state)

            if state.name:
                label = state.name
                color = (60, 180, 75)
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
