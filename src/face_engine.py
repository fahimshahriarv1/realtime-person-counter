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
CONFIDENCE_THRESHOLD = 75  # LBPH distance; lower = more confident match


class FaceEngine:
    def __init__(self, data_dir, cascade_path):
        self.data_dir = data_dir
        self.faces_dir = os.path.join(data_dir, "faces")
        self.labels_path = os.path.join(data_dir, "labels.json")
        self.model_path = os.path.join(data_dir, "model.yml")

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

    def _load_labels(self):
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {}

    def _save_labels(self):
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.labels.items()}, f, indent=2)

    def detect_faces(self, gray_frame):
        """Returns list of (x, y, w, h)."""
        return list(
            self.detector.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
        )

    def predict(self, face_gray):
        """face_gray: grayscale crop of a single face (any size).
        Returns (name_or_None, confidence_or_None). None name = unrecognized."""
        if not self.trained:
            return None, None
        face = cv2.resize(face_gray, FACE_SIZE)
        label_id, confidence = self.recognizer.predict(face)
        if confidence <= CONFIDENCE_THRESHOLD and label_id in self.labels:
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
