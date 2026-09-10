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

## Setup

```
python -m venv .venv
.venv\Scripts\activate       # Windows
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

## Notes & limitations

- Recognition accuracy from a small number of training samples under
  varying light is modest — it's built for the "who's in my room right
  now" use case, not access control.
- One webcam is used at a time (`cv2.VideoCapture(0)`).
- `data/` (trained faces and model) is excluded from version control by
  design; each installation builds its own local roster of named people.
