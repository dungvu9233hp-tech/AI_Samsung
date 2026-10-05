"""Ứng dụng realtime. Chạy:  python -m src.app --mode fusion
Phím: q/ESC thoát | r hiệu chuẩn lại | m đổi mode (landmark/dl/fusion)"""
import argparse
import os
import time
from collections import deque
import cv2
import numpy as np
from .alert import Decider, EventLogger
from .config import Config
from .extractor import FeatureExtractor
from .sound import AlarmPlayer

COL = {"NORMAL": (80, 200, 80), "DISTRACTED": (0, 200, 255), "DROWSY": (0, 0, 230), "CALIBRATING": (200, 200, 200)}


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def draw_ui(frame, info, mode, fps, log_lines, ear_hist, muted=False):
    h, w = frame.shape[:2]
    col = COL[info["state"]]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 8)
    cv2.rectangle(frame, (0, 0), (w, 44), (30, 30, 30), -1)
    title = info["state"] if not info["noface"] else info["state"] + " (NO FACE)"
    cv2.putText(frame, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    cv2.putText(frame, f"{mode.upper()}  {fps:4.1f} FPS", (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if muted:
        cv2.putText(frame, "ALARM MUTED", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "s/SPACE: silence   r: recalibrate   m: mode   q: quit", (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if info["calibrating"]:
        cv2.putText(frame, f"Calibrating... look at camera naturally ({info['calib_progress']*100:.0f}%)",
                    (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    lines = [f"EAR {fmt(info['ear'])} (thr {fmt(info['ear_thr'])})  eyes {'CLOSED' if info['closed'] else 'open'}",
             f"MAR {fmt(info['mar'])}   yawns(5min) {info['yawns']}",
             f"PERCLOS {info['perclos']*100:4.1f}%   closed {info['closed_dur']:.1f}s",
             f"pitch {fmt(info['pitch_dev'],0)}  yaw {fmt(info['yaw_dev'],0)}  nod {info['nod_ratio']*100:3.0f}%",
             f"CNN eye {fmt(info['p_eye'])}  yawn {fmt(info['p_yawn'])}"]
    y = h - 34 - 22 * (len(lines) + len(log_lines))
    ov = frame.copy()
    cv2.rectangle(ov, (6, y - 20), (360, h - 8), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    for s in lines + ["-- log --"][:1 if log_lines else 0] + list(log_lines):
        cv2.putText(frame, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22
    if len(ear_hist) > 2:
        a = np.array(ear_hist); lo, hi = 0.0, max(0.45, a.max())
        xs = np.linspace(w - 190, w - 10, len(a)); ys = 100 - (a - lo) / (hi - lo) * 50
        cv2.polylines(frame, [np.stack([xs, ys], 1).astype(np.int32)], False, (255, 255, 0), 1)
        cv2.putText(frame, "EAR", (w - 190, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


def open_camera(index):
    """Windows hay lỗi với backend mặc định (MSMF): thử DSHOW rồi MSMF, rồi các chỉ số camera khác."""
    import sys
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for idx in [index] + [i for i in range(4) if i != index]:
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"[camera] Mở được camera {idx} (backend {be})")
                    return cap
            cap.release()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["auto", "landmark", "dl", "fusion"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None, help="chạy trên file video thay vì webcam")
    ap.add_argument("--models", default="models")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--dl-stride", type=int, default=2)
    ap.add_argument("--calib", type=float, default=10.0)
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--ear-factor", type=float, default=0.60, help="nhắm nếu EAR < factor*EAR_mở (thấp = phải nhắm hẳn; cao = nhíu mắt cũng tính)")
    ap.add_argument("--p-thr", type=float, default=0.5, help="ngưỡng xác suất nhắm của CNN (mode dl)")
    ap.add_argument("--w-dl", type=float, default=0.5, help="trọng số CNN trong fusion (0=chỉ landmark, 1=chỉ CNN)")
    ap.add_argument("--log", default="logs/alerts.csv")
    a = ap.parse_args()

    ext = FeatureExtractor(a.models, use_dl=a.mode != "landmark", dl_stride=a.dl_stride)
    mode = a.mode if a.mode != "auto" else ("fusion" if ext.has_dl else "landmark")
    if mode != "landmark" and not ext.has_dl:
        print("Không thấy model trong", a.models, "-> dùng landmark"); mode = "landmark"
    modes = ["landmark"] + (["dl", "fusion"] if ext.has_dl else [])
    cfg = Config(calib_s=a.calib, ear_factor=a.ear_factor, p_closed_thr=a.p_thr, w_dl=a.w_dl)
    dec = Decider(mode, cfg)
    sound, logger = AlarmPlayer(not a.no_sound), EventLogger(a.log)

    cap = cv2.VideoCapture(a.video) if a.video else open_camera(a.camera)
    if cap is None or not cap.isOpened():
        raise SystemExit("Không mở được camera/video. Đóng Zoom/Teams/Chrome đang dùng camera, "
                         "và bật Settings > Privacy > Camera > 'Let desktop apps access your camera'.")
    fps, t_prev, ear_hist, mute_until = 0.0, time.time(), deque(maxlen=150), 0.0
    vfps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if a.video else None   # video: dùng đồng hồ của VIDEO
    n_frame = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame.shape[1] != a.width:
            frame = cv2.resize(frame, (a.width, int(frame.shape[0] * a.width / frame.shape[1])))
        now = time.time()
        t = (n_frame / vfps) if a.video else now      # video file: thời gian theo frame, không theo tốc độ xử lý
        n_frame += 1
        f = ext.extract(frame)
        info = dec.update(t, f)
        if f["face"]:
            ear_hist.append(f["ear"])
        muted = now < mute_until
        sound.set(info["state"] == "DROWSY" and not muted)
        logger.update(info)
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3) if fps else 1 / max(now - t_prev, 1e-3)
        t_prev = now
        if not a.video:
            frame = cv2.flip(frame, 1)   # chỉ lật để HIỂN THỊ kiểu gương; xử lý đã làm trên ảnh gốc
        draw_ui(frame, info, mode, fps, logger.recent, ear_hist, muted)
        cv2.imshow("Drowsy Guard", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")): break
        if k == ord("r"): dec.reset()
        if k in (ord("s"), ord(" ")):
            dec.acknowledge(); mute_until = time.time() + 10   # tắt còi chủ động, tạm im 10 s
        if k == ord("m"):
            mode = modes[(modes.index(mode) + 1) % len(modes)]
            dec.mode = mode
    sound.set(False); logger.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
