"""Gán nhãn trạng thái theo đoạn thời gian.
  python tools/label_tool.py data/own/P01_microsleep_dark.mp4
Phím: 1=NORMAL 2=DISTRACTED 3=DROWSY (đổi nhãn từ thời điểm hiện tại) | SPACE tạm dừng | z làm lại từ đầu | q lưu & thoát
Kết quả nối vào data/own/labels.csv (video,t_start,t_end,label). Phần đầu video mặc định NORMAL."""
import csv, os, sys
import cv2

NAMES = {ord("1"): "NORMAL", ord("2"): "DISTRACTED", ord("3"): "DROWSY"}
COL = {"NORMAL": (80, 200, 80), "DISTRACTED": (0, 200, 255), "DROWSY": (0, 0, 230)}


def run(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    segs, cur, start, i, paused = [], "NORMAL", 0.0, 0, False
    while True:
        if not paused:
            ok, fr = cap.read()
            if not ok: break
            i += 1
        t = i / fps
        show = fr.copy()
        cv2.rectangle(show, (0, 0), (show.shape[1] - 1, show.shape[0] - 1), COL[cur], 8)
        cv2.putText(show, f"{cur}  t={t:5.1f}s {'[PAUSED]' if paused else ''}", (12, 30), 0, 0.8, COL[cur], 2)
        cv2.imshow("label", show); k = cv2.waitKey(int(1000 / fps) if not paused else 30) & 0xFF
        if k in NAMES and NAMES[k] != cur:
            if t > start: segs.append((start, t, cur))
            cur, start = NAMES[k], t
        elif k == ord(" "): paused = not paused
        elif k == ord("z"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); segs, cur, start, i = [], "NORMAL", 0.0, 0
        elif k in (27, ord("q")): break
    segs.append((start, i / fps, cur)); cap.release(); cv2.destroyAllWindows()
    return [s for s in segs if s[1] - s[0] > 0.2]


if __name__ == "__main__":
    vid = sys.argv[1]; out = os.path.join(os.path.dirname(vid), "labels.csv")
    segs = run(vid); new = not os.path.exists(out)
    rows = []
    if not new:   # bỏ nhãn cũ của video này
        rows = [r for r in csv.reader(open(out)) if r and r[0] != os.path.basename(vid) and r[0] != "video"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video", "t_start", "t_end", "label"]); w.writerows(rows)
        for a, b, l in segs: w.writerow([os.path.basename(vid), round(a, 2), round(b, 2), l])
    print("Đã lưu", len(segs), "đoạn ->", out)
