"""MediaPipe FaceMesh wrapper + hàm crop mắt/miệng (DÙNG CHUNG cho huấn luyện và chạy thật)."""
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover
    raise SystemExit("Thiếu mediapipe: pip install mediapipe==0.10.14") from e

EYE_CROP_L = [362, 385, 387, 263, 373, 380]
EYE_CROP_R = [33, 160, 158, 133, 153, 144]
MOUTH_CROP = [61, 291, 0, 17, 13, 14, 78, 308]


class FaceDetector:
    def __init__(self, refine=True):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=refine,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)


def _square_crop(gray, pts, idx, pad, out):
    p = pts[idx]
    x0, y0 = p.min(0); x1, y1 = p.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * (1 + pad)
    x0, x1, y0, y1 = int(cx - s / 2), int(cx + s / 2), int(cy - s / 2), int(cy + s / 2)
    h, w = gray.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], (out, out), interpolation=cv2.INTER_AREA)


def crop_eye(gray, pts, idx, out=64, pad=1.0):
    return _square_crop(gray, pts, idx, pad, out)


def crop_mouth(gray, pts, out=64, pad=0.6):
    return _square_crop(gray, pts, MOUTH_CROP, pad, out)


if __name__ == "__main__":
    cap, det = cv2.VideoCapture(0), FaceDetector()
    while True:
        ok, frame = cap.read()
        if not ok: break
        pts = det.process(frame)
        if pts is not None:
            for x, y in pts[::5]:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) == 27: break
