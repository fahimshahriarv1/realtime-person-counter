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
ALERT_WATCHED_ARRIVES = "Only selected people notify"
ALERT_OTHER_ARRIVES = "Everyone except selected people notifies"
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

    def _build_scrollable_side(self, parent):
        """Side panel wrapped in a canvas+scrollbar so its content (which can
        exceed the window's height once shrunk) stays reachable by scrolling
        instead of being clipped."""
        container = ttk.Frame(parent, width=260)
        container.pack(side=tk.RIGHT, fill=tk.Y)
        container.pack_propagate(False)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas, padding=(12, 0, 0, 0))
        inner_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def sync_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_inner_width(event):
            canvas.itemconfig(inner_window, width=event.width)

        inner.bind("<Configure>", sync_scrollregion)
        canvas.bind("<Configure>", sync_inner_width)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        return inner

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        video_frame = ttk.Frame(main)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        side = self._build_scrollable_side(main)

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

        self.ask_for_name_var = tk.BooleanVar(value=self.engine.ask_for_name)
        ttk.Checkbutton(
            side,
            text="Ask for a name when a new person is detected",
            variable=self.ask_for_name_var,
            command=self._on_ask_for_name_toggle,
        ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            side,
            text="On: a dialog asks you to name each new face once "
            "detection is stable.\nOff: new faces are auto-labeled "
            '"Person N" with no prompt.',
            foreground="#555555",
            wraplength=240,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        notif_frame = ttk.LabelFrame(side, text="Notifications", padding=8)
        notif_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(notif_frame, text="Select people:").pack(anchor="w")
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

        self._build_tuning_frame(side)

        self._refresh_watched_person_options()
        self._show_placeholder("Connecting to camera...")

    def _build_tuning_frame(self, parent):
        self._tuning_widgets = []

        frame = ttk.LabelFrame(parent, text="Detection & Recognition Tuning", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        self._add_tuning_slider(
            frame,
            "min_neighbors",
            "Detection strictness",
            "Higher = fewer non-face objects mistaken for a person; too high can miss angled faces.",
            3,
            12,
        )
        self._add_tuning_slider(
            frame,
            "min_size",
            "Minimum face size (px)",
            "Higher = ignore small/distant blobs, a common source of false detections.",
            20,
            200,
        )
        self._add_tuning_slider(
            frame,
            "confidence_threshold",
            "Recognition strictness",
            "Lower = stricter name matching (fewer mix-ups between people), but may misjudge a known face as unrecognized.",
            30,
            150,
        )
        self._add_tuning_slider(
            frame,
            "scale_factor",
            "Detection scan detail",
            "Lower = finer-grained scanning (catches more faces, slower, slightly more false positives).",
            1.05,
            1.5,
            is_float=True,
        )

        ttk.Button(frame, text="Reset to defaults", command=self._reset_tuning).pack(
            fill=tk.X, pady=(6, 0)
        )

        return frame

    def _format_tuning_value(self, value, is_float):
        return f"{value:.2f}" if is_float else str(int(round(value)))

    def _add_tuning_slider(self, parent, attr_name, title, hint, frm, to, is_float=False):
        current = getattr(self.engine, attr_name)

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(6, 0))

        header = ttk.Frame(row)
        header.pack(fill=tk.X)
        ttk.Label(header, text=title, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        value_var = tk.StringVar(value=self._format_tuning_value(current, is_float))
        ttk.Label(header, textvariable=value_var, foreground="#555555").pack(side=tk.RIGHT)

        def coerce(raw):
            v = float(raw)
            return v if is_float else int(round(v))

        def on_move(raw):
            value = coerce(raw)
            value_var.set(self._format_tuning_value(value, is_float))
            setattr(self.engine, attr_name, value)

        def on_release(_event):
            self.engine.update_tuning(**{attr_name: getattr(self.engine, attr_name)})

        scale = ttk.Scale(row, from_=frm, to=to, orient=tk.HORIZONTAL, command=on_move)
        scale.set(current)
        scale.pack(fill=tk.X)
        scale.bind("<ButtonRelease-1>", on_release)

        ttk.Label(
            row,
            text=hint,
            foreground="#777777",
            font=("Segoe UI", 8),
            wraplength=220,
            justify="left",
        ).pack(anchor="w", pady=(0, 2))

        self._tuning_widgets.append((scale, value_var, attr_name, is_float))

    def _reset_tuning(self):
        self.engine.reset_tuning()
        for scale, value_var, attr_name, is_float in self._tuning_widgets:
            value = getattr(self.engine, attr_name)
            scale.set(value)
            value_var.set(self._format_tuning_value(value, is_float))

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

    def _on_ask_for_name_toggle(self):
        self.engine.set_ask_for_name(self.ask_for_name_var.get())

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

    def resolve_new_person(self, state):
        """Called once a track has held steady as 'unknown' long enough to
        act on. Either prompts for a name or auto-registers one, depending
        on the ask-for-name toggle."""
        if self.ask_for_name_var.get():
            self.prompt_for_name(state)
        else:
            state.name = self.engine.add_unnamed_person(list(state.buffer))
            state.buffer.clear()
            self._refresh_watched_person_options()

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
                        self.resolve_new_person(state)

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
