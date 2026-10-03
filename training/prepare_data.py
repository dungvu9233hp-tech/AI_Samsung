"""Chuẩn bị dữ liệu, CHIA THEO SUBJECT (không theo ảnh).

  # 1) MRL Eye Dataset (Kaggle: imadeddinedjerarda/mrl-eye-dataset)
  python training/prepare_data.py mrl --dir data/raw/mrl --out data/processed/eye_split.csv --holdout P07,P08

  # 2) Dataset yawn dạng thư mục yawn/ và no_yawn/ (Kaggle 'yawn_eye dataset')
  python training/prepare_data.py yawn_folder --dir data/raw/yawn --out data/processed/yawn_split.csv

  # 3) Trích crop từ VIDEO TỰ QUAY (nhãn yếu, bán tự động) - rồi gộp vào bước 1/2 bằng --own
  python training/prepare_data.py own --videos data/own --out_dir data/processed/own
  python training/prepare_data.py mrl --dir data/raw/mrl --own data/processed/own/eye_own.csv ...

Quy ước tên video: P01_normal_glasses_dark.mp4  (subject là phần trước dấu '_' đầu tiên)
--holdout: các subject tự quay dùng làm TEST cuối -> tuyệt đối không đưa vào train/val.
"""
import argparse, glob, os, sys, random
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def assign_split(df, seed=42, holdout=(), ratios=(0.7, 0.15, 0.15)):
    subs = sorted(set(df.subject) - set(holdout))
    random.Random(seed).shuffle(subs)
    n = len(subs); a = int(round(n * ratios[0])); b = a + int(round(n * ratios[1]))
    m = {s: "train" for s in subs[:a]}
    m.update({s: "val" for s in subs[a:b]}); m.update({s: "test" for s in subs[b:]})
    m.update({s: "test" for s in holdout})
    df["split"] = df.subject.map(m)
    return df


def cap_per_subject(df, k, seed=42):
    if not k: return df
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in df.groupby(["subject", "label"]).groups.items():
        idx = list(idx)
        keep.extend(rng.choice(idx, min(len(idx), k), replace=False))
    return df.loc[keep].reset_index(drop=True)


def cmd_mrl(a):
    rows = []
    for p in glob.glob(os.path.join(a.dir, "**", "*.*"), recursive=True):
        if not p.lower().endswith((".png", ".jpg", ".jpeg")): continue
        parts = os.path.basename(p).split("_")
        if len(parts) < 5 or not parts[0].startswith("s"): continue
        # s0001_imgid_gender_glasses_eyestate(0=closed,1=open)_reflect_light_sensor
        rows.append(dict(path=os.path.abspath(p), label=1 if parts[4] == "0" else 0,
                         subject=parts[0], source="mrl"))
    df = pd.DataFrame(rows)
    if df.empty: raise SystemExit("Không tìm thấy ảnh MRL theo quy ước tên file.")
    df = cap_per_subject(df, a.cap)
    if a.own:
        own = pd.read_csv(a.own); own["source"] = "own"; own["subject"] = "own_" + own.subject.astype(str)
        df = pd.concat([df, own[["path", "label", "subject", "source"]]], ignore_index=True)
    hold = ["own_" + h for h in a.holdout.split(",") if h]
    df = assign_split(df, a.seed, hold)
    save(df, a.out)


def cmd_yawn_folder(a):
    rows = []
    for d in os.listdir(a.dir):
        full = os.path.join(a.dir, d)
        if not os.path.isdir(full): continue
        name = d.lower()
        if "yawn" not in name: continue
        label = 0 if name.startswith(("no", "not")) else 1
        for p in glob.glob(os.path.join(full, "**", "*.*"), recursive=True):
            if p.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append(dict(path=os.path.abspath(p), label=label, source="folder"))
    df = pd.DataFrame(rows)
    if df.empty: raise SystemExit("Cần thư mục con tên 'yawn' và 'no_yawn'.")
    # Không có mã subject -> nhóm theo tiền tố tên file (cố gắng giảm rò rỉ)
    df["subject"] = df.path.map(lambda p: os.path.basename(p).split("_")[0].rstrip("0123456789"))
    if df.subject.nunique() < 10:
        print("[CẢNH BÁO] Không suy ra được subject -> chia ngẫu nhiên theo ảnh, có thể rò rỉ. Ghi rõ trong báo cáo!")
        df["subject"] = [f"img{i}" for i in range(len(df))]
    df = cap_per_subject(df, a.cap)
    if a.own:
        own = pd.read_csv(a.own); own["source"] = "own"; own["subject"] = "own_" + own.subject.astype(str)
        df = pd.concat([df, own[["path", "label", "subject", "source"]]], ignore_index=True)
    hold = ["own_" + h for h in a.holdout.split(",") if h]
    df = assign_split(df, a.seed, hold)
    save(df, a.out)


def cmd_own(a):
    import cv2
    from src.detector import FaceDetector, crop_eye, crop_mouth, EYE_CROP_L, EYE_CROP_R
    from src.features import ear, mar, L_EYE, R_EYE
    det = FaceDetector()
    os.makedirs(os.path.join(a.out_dir, "eye"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "yawn"), exist_ok=True)
    eye_rows, yawn_rows = [], []
    for vp in sorted(glob.glob(os.path.join(a.videos, "*.mp4")) + glob.glob(os.path.join(a.videos, "*.avi"))):
        name = os.path.splitext(os.path.basename(vp))[0]; subj = name.split("_")[0]
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = []; i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % a.step == 0:
                pts = det.process(fr)
                if pts is not None:
                    frames.append((i, fr, pts, (ear(pts, L_EYE) + ear(pts, R_EYE)) / 2, mar(pts)))
            i += 1
        cap.release()
        if not frames: continue
        n0 = [f[3] for f in frames if f[0] / fps < a.calib_s]
        e_open = np.median(n0 if n0 else [f[3] for f in frames])
        for idx, fr, pts, e, m in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            lab = 1 if e < 0.70 * e_open else (0 if e > 0.85 * e_open else None)   # vùng mơ hồ -> bỏ
            if lab is not None:
                for side, ids in (("L", EYE_CROP_L), ("R", EYE_CROP_R)):
                    c = crop_eye(gray, pts, ids)
                    if c is None: continue
                    p = os.path.join(a.out_dir, "eye", f"{name}_{idx}_{side}.png"); cv2.imwrite(p, c)
                    eye_rows.append(dict(path=os.path.abspath(p), label=lab, subject=subj))
            ylab = 1 if m > 0.6 else (0 if m < 0.35 else None)
            if ylab is not None:
                c = crop_mouth(gray, pts)
                if c is not None:
                    p = os.path.join(a.out_dir, "yawn", f"{name}_{idx}.png"); cv2.imwrite(p, c)
                    yawn_rows.append(dict(path=os.path.abspath(p), label=ylab, subject=subj))
        print(name, "ok")
    pd.DataFrame(eye_rows).to_csv(os.path.join(a.out_dir, "eye_own.csv"), index=False)
    pd.DataFrame(yawn_rows).to_csv(os.path.join(a.out_dir, "yawn_own.csv"), index=False)
    print("Nhãn là NHÃN YẾU (từ EAR/MAR). Hãy xem lướt ảnh trong", a.out_dir, "và xoá ảnh sai.")


def save(df, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("Subjects:", df.groupby("split").subject.nunique().to_dict(), "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in (("mrl", cmd_mrl), ("yawn_folder", cmd_yawn_folder)):
        p = sp.add_parser(n); p.add_argument("--dir", required=True); p.add_argument("--out", required=True)
        p.add_argument("--own", default=None); p.add_argument("--holdout", default="")
        p.add_argument("--cap", type=int, default=300, help="tối đa ảnh/lớp/subject (0 = không giới hạn)")
        p.add_argument("--seed", type=int, default=42); p.set_defaults(fn=fn)
    p = sp.add_parser("own"); p.add_argument("--videos", default="data/own"); p.add_argument("--out_dir", default="data/processed/own")
    p.add_argument("--step", type=int, default=3); p.add_argument("--calib_s", type=float, default=5.0); p.set_defaults(fn=cmd_own)
    a = ap.parse_args(); a.fn(a)