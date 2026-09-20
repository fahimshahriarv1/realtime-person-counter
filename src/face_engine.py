"""Face detection + recognition engine.

Detection: Haar cascade (bundled in assets/, since the current OpenCV wheel
for this Python version ships without its usual data files).

Recognition: OpenCV's LBPH (Local Binary Patterns Histograms) recognizer.
Chosen over dlib/face_recognition because it installs from a plain pip
wheel on Windows with no compiler toolchain required, which keeps this
project's setup to a single `pip install -r requirements.txt`.

Persistence layout under data/:
  data/labels.json          -> {"0": "Alice", "1": "Bob", ...}
  data/model.yml            -> trained LBPH model
  data/faces/<name>/*.png   -> stored training crops per person
"""

import json
import os

import cv2
import numpy as np

FACE_SIZE = (200, 200)

# Tuning defaults, all overridable at runtime via update_tuning() and
# persisted to data/settings.json:
#   scale_factor       Haar pyramid step. Closer to 1.0 = finer-grained
#                       scanning (catches more faces, slower, more false
#                       positives). Must be > 1.0.
#   min_neighbors       How many overlapping candidate detections Haar
#                       requires before accepting one as a face. Higher =
#                       fewer false positives (non-faces mistaken for
#                       faces), but can start missing real faces at odd
#                       angles.
#   min_size            Smallest detection (px) to accept. Raising this
#                       filters out small/distant blobs that are a common
#                       source of false positives.
#   confidence_threshold LBPH match distance ceiling; lower = stricter
#                       recognition (less likely to misidentify one person
#                       as another, but more likely to call a known person
#                       "unrecognized").
#   ask_for_name        Whether a newly detected, unrecognized person
#                       should be prompted for a name (True) or silently
#                       auto-registered as "Person N" (False).
DEFAULT_SCALE_FACTOR = 1.1
DEFAULT_MIN_NEIGHBORS = 5
DEFAULT_MIN_SIZE = 60
DEFAULT_CONFIDENCE_THRESHOLD = 75
DEFAULT_ASK_FOR_NAME = True


class FaceEngine:
    def __init__(self, data_dir, cascade_path):
        self.data_dir = data_dir
        self.faces_dir = os.path.join(data_dir, "faces")
        self.labels_path = os.path.join(data_dir, "labels.json")
        self.model_path = os.path.join(data_dir, "model.yml")
        self.settings_path = os.path.join(data_dir, "settings.json")

        os.makedirs(self.faces_dir, exist_ok=True)

        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load cascade from {cascade_path}")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = self._load_labels()
        self.trained = False

        if os.path.exists(self.model_path) and self.labels:
            self.recognizer.read(self.model_path)
            self.trained = True

        self._load_tuning()

    def _load_labels(self):
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {}

    def _save_labels(self):
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.labels.items()}, f, indent=2)

    def _load_tuning(self):
        settings = {}
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        self.scale_factor = settings.get("scale_factor", DEFAULT_SCALE_FACTOR)
        self.min_neighbors = settings.get("min_neighbors", DEFAULT_MIN_NEIGHBORS)
        self.min_size = settings.get("min_size", DEFAULT_MIN_SIZE)
        self.confidence_threshold = settings.get(
            "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.ask_for_name = settings.get("ask_for_name", DEFAULT_ASK_FOR_NAME)

    def _save_tuning(self):
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scale_factor": self.scale_factor,
                    "min_neighbors": self.min_neighbors,
                    "min_size": self.min_size,
                    "confidence_threshold": self.confidence_threshold,
                    "ask_for_name": self.ask_for_name,
                },
                f,
                indent=2,
            )

    def update_tuning(
        self, scale_factor=None, min_neighbors=None, min_size=None, confidence_threshold=None
    ):
        """Apply any given tuning values immediately and persist them.
        Omitted (None) values are left unchanged."""
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if min_neighbors is not None:
            self.min_neighbors = min_neighbors
        if min_size is not None:
            self.min_size = min_size
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        self._save_tuning()

    def reset_tuning(self):
        self.scale_factor = DEFAULT_SCALE_FACTOR
        self.min_neighbors = DEFAULT_MIN_NEIGHBORS
        self.min_size = DEFAULT_MIN_SIZE
        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self._save_tuning()

    def set_ask_for_name(self, value):
        """Toggle whether a newly detected, unrecognized person should be
        interactively named (True) or auto-registered with a generated
        'Person N' name (False). Persisted immediately."""
        self.ask_for_name = bool(value)
        self._save_tuning()

    def detect_faces(self, gray_frame):
        """Returns list of (x, y, w, h)."""
        return list(
            self.detector.detectMultiScale(
                gray_frame,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(self.min_size, self.min_size),
            )
        )

    def predict(self, face_gray):
        """face_gray: grayscale crop of a single face (any size).
        Returns (name_or_None, confidence_or_None). None name = unrecognized."""
        if not self.trained:
            return None, None
        face = cv2.resize(face_gray, FACE_SIZE)
        label_id, confidence = self.recognizer.predict(face)
        if confidence <= self.confidence_threshold and label_id in self.labels:
            return self.labels[label_id], confidence
        return None, confidence

    def _label_id_for_name(self, name):
        for label_id, existing_name in self.labels.items():
            if existing_name == name:
                return label_id
        return None

    def add_person(self, name, face_crops_gray):
        """Register (or extend) a person with one or more grayscale face crops,
        persist the training images, and retrain the recognizer."""
        name = name.strip()
        if not name:
            raise ValueError("Name must not be empty")

        label_id = self._label_id_for_name(name)
        if label_id is None:
            label_id = max(self.labels.keys(), default=-1) + 1
            self.labels[label_id] = name

        person_dir = os.path.join(self.faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        existing = len(os.listdir(person_dir))
        for i, crop in enumerate(face_crops_gray):
            resized = cv2.resize(crop, FACE_SIZE)
            cv2.imwrite(os.path.join(person_dir, f"img_{existing + i}.png"), resized)

        self._save_labels()
        self.retrain()

    def add_unnamed_person(self, face_crops_gray):
        """Like add_person(), but generates a 'Person N' name instead of
        taking one interactively -- used when ask_for_name is off. Returns
        the generated name."""
        next_id = max(self.labels.keys(), default=-1) + 1
        name = f"Person {next_id}"
        self.add_person(name, face_crops_gray)
        return name

    def retrain(self):
        """Rebuild the recognizer from every stored training image on disk.
        Dataset stays small for this use case, so a full retrain per new
        person is simpler and more robust than incremental update()."""
        images, ids = [], []
        for label_id, name in self.labels.items():
            person_dir = os.path.join(self.faces_dir, name)
            if not os.path.isdir(person_dir):
                continue
            for fname in os.listdir(person_dir):
                img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                ids.append(label_id)

        if not images:
            self.trained = False
            return

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(images, np.array(ids))
        self.recognizer.save(self.model_path)
        self.trained = True
