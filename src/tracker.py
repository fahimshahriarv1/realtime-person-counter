"""Lightweight centroid-distance tracker.

Assigns a stable integer ID to each detected face across frames so the rest
of the app can reason about "this same person" instead of raw per-frame
bounding boxes. Not a full multi-object tracker (no Kalman filter, no
re-identification) -- just nearest-centroid matching with a disappearance
grace period, which is plenty for a single-webcam, few-people scene.
"""

import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=75):
        self.next_id = 0
        self.objects = {}       # id -> centroid (x, y)
        self.bboxes = {}        # id -> (x, y, w, h)
        self.disappeared = {}   # id -> consecutive frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        oid = self.next_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.next_id += 1
        return oid

    def deregister(self, oid):
        del self.objects[oid]
        del self.bboxes[oid]
        del self.disappeared[oid]

    def update(self, rects):
        """rects: list of (x, y, w, h). Returns dict id -> (x, y, w, h)."""
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return dict(self.bboxes)

        input_centroids = np.array(
            [(x + w / 2.0, y + h / 2.0) for (x, y, w, h) in rects]
        )

        if len(self.objects) == 0:
            for i, rect in enumerate(rects):
                self.register(tuple(input_centroids[i]), rect)
            return dict(self.bboxes)

        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid] for oid in object_ids])

        distances = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = tuple(input_centroids[col])
            self.bboxes[oid] = rects[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distances.shape[0])) - used_rows
        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        unused_cols = set(range(distances.shape[1])) - used_cols
        for col in unused_cols:
            self.register(tuple(input_centroids[col]), rects[col])

        return dict(self.bboxes)
