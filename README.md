# Realtime Person Counter

A simple desktop app that watches your webcam, counts how many people are
currently in the room, and lets you assign a name to each new face the
first time it's seen. Named people are recognized automatically after that,
even across app restarts.

## How it works

- **Detection**: OpenCV Haar cascade finds faces in each frame.
- **Tracking**: a lightweight centroid tracker assigns a stable ID to each
  face across frames, so "person #2" stays "person #2" while they move.
- **Recognition**: OpenCV's LBPH face recognizer matches faces against
  people you've already named. Chosen over `dlib`/`face_recognition`
  because it installs from a plain pip wheel on Windows — no C++ compiler
  needed.
- **Naming**: when a face is detected but doesn't match anyone known, and
  the detection has been stable for about a third of a second, a dialog
  pops up asking for a name. Enter one and the recognizer is trained on
  the spot; skip it and that face is labeled "Unnamed" for the session.
- **GUI**: Tkinter, showing the live annotated video feed, a running count
  of people currently visible, and a list of who's currently in frame.
- **Camera failsafe**: if no webcam is found (or it's a different device
  index), the app shows a status placeholder and retries every couple of
  seconds instead of crashing — same recovery path if the camera drops out
  mid-session. There's also a manual "Retry camera now" button.
- **Notifications**: a native desktop notification can fire when a specific
  "watched" person enters the room, or when anyone *other than* that person
  enters (an intruder-style alert). Cross-platform with no extra pip
  package — each OS's own notifier is used directly: Windows' toast API via
  a bundled PowerShell script (`assets/toast.ps1`), macOS's `osascript`,
  and Linux's `notify-send` (from libnotify, present on most desktop
  environments). This avoids depending on a notification library that
  might lack prebuilt wheels on newer Python versions.

## Setup

```
python -m venv .venv
.venv\Scripts\activate       # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

## Run

```
python src/main.py
```

A webcam window opens. When a new face is detected and holds steady, a
popup asks for their name — type it and press OK. That person is now
remembered (face images + trained model are stored under `data/`, which is
git-ignored since it's personal biometric data specific to your machine).

In the side panel, pick a **Watched person** and an **Alert me when**
mode ("Watched person appears" or "Someone else appears"), or use
"Send test notification" to confirm toasts are showing up on your machine.

## Notes & limitations

- Recognition accuracy from a small number of training samples under
  varying light is modest — it's built for the "who's in my room right
  now" use case, not access control.
- One webcam is used at a time, tried across a few device indices
  (`0`, `1`, `2`) on connect/reconnect.
- `data/` (trained faces and model) is excluded from version control by
  design; each installation builds its own local roster of named people.
- Notifications silently no-op if the platform's notifier isn't available
  (e.g. a minimal Linux install without `notify-send`) rather than
  erroring — the rest of the app is unaffected.
