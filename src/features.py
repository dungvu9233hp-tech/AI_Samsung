"""EAR, MAR, head pose từ 478 landmark của MediaPipe FaceMesh (toạ độ pixel)."""
import numpy as np
import cv2

# Thứ tự p1..p6 cho EAR
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 13, 14, 308]   # góc trái, môi trên trong, môi dưới trong, góc phải


def ear(pts, idx):
    p1, p2, p3, p4, p5, p6 = pts[idx]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-6)


def mar(pts):
    l, t, b, r = pts[MOUTH]
    return np.linalg.norm(t - b) / (np.linalg.norm(l - r) + 1e-6)


# Mô hình mặt 3D (hệ toạ độ camera: x phải, y xuống, z ra xa camera; mũi gần camera nhất).
# Thứ tự theo vị trí TRÊN ẢNH (frame không lật gương): trái ảnh = mắt phải của người (33, 61).
MODEL_3D = np.array([
    (0.0, 0.0, 0.0),        # mũi        (1)
    (0.0, 63.6, 12.5),      # cằm        (152)
    (-43.3, -32.7, 26.0),   # mắt ngoài trái-ảnh  (33)
    (43.3, -32.7, 26.0),    # mắt ngoài phải-ảnh  (263)
    (-28.9, 28.9, 24.1),    # miệng trái-ảnh      (61)
    (28.9, 28.9, 24.1),     # miệng phải-ảnh      (291)
], dtype=np.float64)
POSE_IDX = [1, 152, 33, 263, 61, 291]


def head_pose(pts, w, h):
    """Trả (pitch, yaw, roll) theo độ. Chiều dấu phụ thuộc camera -> dùng độ lệch tuyệt đối khi so ngưỡng."""
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts[POSE_IDX].astype(np.float64), cam,
                               np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(R)[0]
    pitch, yaw, roll = [float(a) for a in angles]
    if pitch > 90: pitch -= 180
    if pitch < -90: pitch += 180
    return pitch, yaw, roll
