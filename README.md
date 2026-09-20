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
  An "Ask for a name when a new person is detected" toggle turns this
  off — with it unchecked, new faces are silently auto-registered as
  "Person N" (still trained and recognized from then on) instead of
  interrupting you with a dialog.
- **GUI**: Tkinter, showing the live annotated video feed, a running count
  of people currently visible, and a list of who's currently in frame.
- **Camera failsafe**: if no webcam is found (or it's a different device
  index), the app shows a status placeholder and retries every couple of
  seconds instead of crashing — same recovery path if the camera drops out
  mid-session. There's also a manual "Retry camera now" button.
- **Notifications**: a native desktop notification can fire either way
  around a multi-select list of known people — "only selected people
  notify" (a whitelist) or "everyone except selected people notifies" (a
  blocklist, e.g. an intruder-style alert). Cross-platform with no extra
  pip package — each OS's own notifier is used directly: Windows' toast API via
  a bundled PowerShell script (`assets/toast.ps1`), macOS's `osascript`,
  and Linux's `notify-send` (from libnotify, present on most desktop
  environments). This avoids depending on a notification library that
  might lack prebuilt wheels on newer Python versions.
- **Tuning**: a "Detection & Recognition Tuning" panel with live sliders
  for the four knobs that most affect accuracy — detection strictness,
  minimum face size, recognition strictness, and scan detail (see
  Tuning below). Changes apply immediately and persist across restarts
  (`data/settings.json`); a "Reset to defaults" button is included.

## Quickest way to run it (Windows)

Double-click **`run.bat`**. The first launch creates the virtual environment
and installs dependencies automatically (takes a minute or two); every
launch after that starts the app straight away. If it exits with an error,
the window stays open so you can read it.

## Setup

```
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
PowerShell's default execution policy blocks `.venv\Scripts\activate`
(you'll see a `running scripts is disabled on this system` error), so the
command above calls the venv's `python.exe` directly instead — no policy
change needed. If you'd rather use plain `python`/`pip` with an activated
prompt, allow scripts for just this session first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
.venv\Scripts\python.exe src\main.py   # Windows, matching the Setup step above
python src/main.py                      # if you activated the venv, or on macOS/Linux
```

A webcam window opens. When a new face is detected and holds steady, a
popup asks for their name — type it and press OK. That person is now
remembered (face images + trained model are stored under `data/`, which is
git-ignored since it's personal biometric data specific to your machine).

In the side panel, Ctrl/Shift-click to pick one or more people under
**Select people**, then choose an **Alert me when** mode:
- **Only selected people notify** — a whitelist; everyone else is silent.
- **Everyone except selected people notifies** — a blocklist; the people
  you pick are exempt, anyone else (including unrecognized faces) alerts.

Use "Send test notification" any time to confirm toasts are showing up on
your machine.

## Tuning

If the app is missing real faces, mistaking objects for faces, or mixing
people up, adjust the sliders in **Detection & Recognition Tuning**:

| Slider | Raise it when... | Lower it when... |
| --- | --- | --- |
| Detection strictness | non-face objects keep getting boxed as a person | real faces at an angle aren't being detected |
| Minimum face size | small/distant background blobs are triggering false detections | people further from the camera aren't being picked up |
| Recognition strictness | two people keep getting mixed up | a known person keeps getting treated as a new/unrecognized face |
| Detection scan detail | detection needs to run faster (fewer scan steps) | faces are being missed and speed isn't a concern (finer-grained scan) |

Values apply live as you drag and are saved automatically once you release
the slider, so you can tune while watching the video feed react in real
time. "Reset to defaults" restores the original values.

## Notes & limitations

- Recognition accuracy from a small number of training samples under
  varying light is modest — it's built for the "who's in my room right
  now" use case, not access control.
- One webcam is used at a time, tried across a few device indices
  (`0`, `1`, `2`) on connect/reconnect.
- `data/` (trained faces, model, and tuning settings) is excluded from
  version control by design; each installation builds its own local
  roster of named people and tuning.
- Notifications silently no-op if the platform's notifier isn't available
  (e.g. a minimal Linux install without `notify-send`) rather than
  erroring — the rest of the app is unaffected.
