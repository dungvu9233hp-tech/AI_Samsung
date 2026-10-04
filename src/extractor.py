"""Trích đặc trưng từ 1 frame: landmark (EAR/MAR/pose) + xác suất từ CNN (nếu có model)."""
import os
import time
import numpy as np
import cv2
from .detector import FaceDetector, crop_eye, crop_mouth, EYE_CROP_L, EYE_CROP_R
from .features import ear, mar, head_pose, L_EYE, R_EYE
from .classifier import ONNXClassifier

NAN = float("nan")


class FeatureExtractor:
    def __init__(self, models_dir="models", use_dl=True, dl_stride=2):
        self.det = FaceDetector()
        self.eye_clf = self.yawn_clf = None
        if use_dl:
            pe, py = os.path.join(models_dir, "eye.onnx"), os.path.join(models_dir, "yawn.onnx")
            if os.path.exists(pe): self.eye_clf = ONNXClassifier(pe)
            if os.path.exists(py): self.yawn_clf = ONNXClassifier(py)
        self.stride, self.n, self._cache = max(1, dl_stride), 0, (NAN, NAN)

    @property
    def has_dl(self):
        return self.eye_clf is not None or self.yawn_clf is not None

    def extract(self, frame):
        t0 = time.perf_counter()
        pts = self.det.process(frame)
        out = dict(face=pts is not None, ear=NAN, mar=NAN, pitch=NAN, yaw=NAN,
                   p_eye=NAN, p_yawn=NAN, t_lm=0.0, t_dl=0.0, pts=pts)
        self.n += 1
        if pts is None:
            self._cache = (NAN, NAN)
            out["t_lm"] = time.perf_counter() - t0
            return out
        h, w = frame.shape[:2]
        out["ear"] = float((ear(pts, L_EYE) + ear(pts, R_EYE)) / 2)
        out["mar"] = float(mar(pts))
        out["pitch"], out["yaw"], _ = head_pose(pts, w, h)
        out["t_lm"] = time.perf_counter() - t0

        if self.has_dl and (self.n % self.stride == 0 or np.isnan(self._cache[0]) and np.isnan(self._cache[1])):
            t1 = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pe = py = NAN
            if self.eye_clf is not None:
                crops = [c for c in (crop_eye(gray, pts, EYE_CROP_L), crop_eye(gray, pts, EYE_CROP_R)) if c is not None]
                if crops: pe = float(self.eye_clf.probs(crops).mean())
            if self.yawn_clf is not None:
                c = crop_mouth(gray, pts)
                if c is not None: py = float(self.yawn_clf.probs([c])[0])
            self._cache = (pe, py)
            out["t_dl"] = time.perf_counter() - t1
        out["p_eye"], out["p_yawn"] = self._cache
        return out
