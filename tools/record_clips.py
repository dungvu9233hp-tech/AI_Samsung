"""Quay clip dữ liệu tự quay. Chạy: python tools/record_clips.py --subject P01 --scenario microsleep --tags glasses,dark
Mỗi clip BẮT ĐẦU bằng >=5 s nhìn thẳng bình thường (dùng để hiệu chuẩn). SPACE bắt đầu/dừng, q thoát."""
import argparse, os, time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--subject", required=True); ap.add_argument("--scenario", required=True)
ap.add_argument("--tags", default="", help="glasses,dark,mask (cách nhau dấu phẩy)")
ap.add_argument("--camera", type=int, default=0); ap.add_argument("--out", default="data/own")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
name = "_".join([a.subject, a.scenario] + [t for t in a.tags.split(",") if t]) + ".mp4"
path = os.path.join(a.out, name)
cap = cv2.VideoCapture(a.camera); cap.set(3, 640); cap.set(4, 480)
vw, t0 = None, 0
while True:
    ok, fr = cap.read()
    if not ok: break
    show = fr.copy()
    if vw is not None:
        vw.write(fr); cv2.circle(show, (20, 20), 8, (0, 0, 255), -1)
        cv2.putText(show, f"REC {time.time()-t0:4.1f}s  (SPACE = stop)", (36, 26), 0, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(show, f"{name}  SPACE = start", (10, 26), 0, 0.6, (255, 255, 255), 2)
    cv2.imshow("record", show); k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        if vw is None:
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (fr.shape[1], fr.shape[0])); t0 = time.time()
        else:
            break
    if k in (27, ord("q")): break
if vw: vw.release(); print("Đã lưu", path)
cap.release(); cv2.destroyAllWindows()
