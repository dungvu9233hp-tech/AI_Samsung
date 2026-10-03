"""Đánh giá offline trên video tự quay có nhãn.
  python eval/evaluate.py --videos data/own --labels data/own/labels.csv --test_subjects P07,P08
  python eval/evaluate.py ... --sweep --val_subjects P05,P06      # quét ngưỡng trên VAL
Bước nặng (MediaPipe + CNN) chạy 1 lần/video rồi cache (.npz); các mode/ngưỡng chỉ replay logic -> rất nhanh."""
import argparse, glob, itertools, os, sys, json
import cv2, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.alert import Decider
from src.config import Config

CLASSES = ["NORMAL", "DISTRACTED", "DROWSY"]
RES = os.path.join(ROOT, "eval", "results")
CACHE = os.path.join(ROOT, "eval", "cache")
KEYS = ["t", "face", "ear", "mar", "pitch", "yaw", "p_eye", "p_yawn", "t_lm", "t_dl"]


def extract_video(path, models, width=640):
    cp = os.path.join(CACHE, os.path.splitext(os.path.basename(path))[0] + ".npz")
    if os.path.exists(cp):
        return dict(np.load(cp))
    from src.extractor import FeatureExtractor
    ext = FeatureExtractor(models, use_dl=True, dl_stride=1)     # stride 1 = FPS trường hợp xấu nhất
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rec = {k: [] for k in KEYS}; i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[1] != width: fr = cv2.resize(fr, (width, int(fr.shape[0] * width / fr.shape[1])))
        f = ext.extract(fr); f["t"] = i / fps; i += 1
        for k in KEYS: rec[k].append(f[k])
    cap.release()
    d = {k: np.array(v, dtype=np.float64) for k, v in rec.items()}
    d["has_dl"] = np.array(ext.has_dl)
    os.makedirs(CACHE, exist_ok=True); np.savez(cp, **d)
    return d


def replay(d, mode, cfg):
    dec = Decider(mode, cfg); n = len(d["t"]); states = []
    for i in range(n):
        f = {k: d[k][i] for k in KEYS}; f["face"] = bool(f["face"])
        states.append(dec.update(float(d["t"][i]), f)["state"])
    return np.array(states)


def labels_for(df, vid):
    return [(r.t_start, r.t_end, r.label) for r in df[df.video == vid].itertuples()]


def true_at(segs, t):
    for a, b, l in segs:
        if a <= t < b: return l
    return None


def tags_of(name):
    s = name.lower()
    return dict(glasses="glasses" in s, dark="dark" in s, mask="mask" in s)


def evaluate_video(d, segs, mode, cfg, calib):
    states = replay(d, mode, cfg); t = d["t"]
    truth = np.array([true_at(segs, x) or "" for x in t])
    keep = (t >= calib) & (truth != "")
    pred = np.where(states == "CALIBRATING", "NORMAL", states)
    # lấy mẫu 1 Hz
    idx = [int(np.searchsorted(t, s)) for s in np.arange(calib, t[-1], 1.0)]
    idx = [i for i in idx if i < len(t) and keep[i]]
    # báo động sai: cạnh lên vào DROWSY khi nhãn thật là NORMAL
    d_on = (pred == "DROWSY"); rise = np.where(d_on & ~np.r_[False, d_on[:-1]])[0]
    fa = sum(1 for i in rise if keep[i] and truth[i] == "NORMAL")
    lat, missed = [], 0
    for a, b, l in segs:
        if l != "DROWSY": continue
        w = np.where((t >= a) & (t <= b) & d_on)[0]
        if len(w): lat.append(float(t[w[0]] - a))
        else: missed += 1
    hours = max(keep.sum() / max(1.0 / np.median(np.diff(t)), 1e-6), 1e-6) / 3600
    return dict(y=truth[idx], p=pred[idx], fa=fa, hours=hours, lat=lat, missed=missed, idx=idx,
                t_lm=float(np.mean(d["t_lm"])), t_dl=float(np.mean(d["t_dl"])))


def summarize(parts, mode):
    y = np.concatenate([p["y"] for p in parts]); p = np.concatenate([p["p"] for p in parts])
    if len(y) == 0: return None, None
    rep = classification_report(y, p, labels=CLASSES, output_dict=True, zero_division=0)
    hours = sum(x["hours"] for x in parts); lat = [l for x in parts for l in x["lat"]]
    t_lm = np.mean([x["t_lm"] for x in parts]); t_dl = np.mean([x["t_dl"] for x in parts])
    t_tot = t_lm + (t_dl if mode != "landmark" else 0)
    row = dict(mode=mode, n_windows=len(y), accuracy=round(float((y == p).mean()), 4),
               recall_drowsy=round(rep["DROWSY"]["recall"], 4), precision_drowsy=round(rep["DROWSY"]["precision"], 4),
               f1_drowsy=round(rep["DROWSY"]["f1-score"], 4), f1_macro=round(rep["macro avg"]["f1-score"], 4),
               false_alarms_per_hour=round(sum(x["fa"] for x in parts) / hours, 2),
               median_latency_s=round(float(np.median(lat)), 2) if lat else None,
               missed_drowsy_segments=sum(x["missed"] for x in parts), fps=round(1 / t_tot, 1))
    return row, confusion_matrix(y, p, labels=CLASSES)


def plot_cm(cm, mode):
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks(range(3), CLASSES, rotation=20); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title(mode); fig.tight_layout()
    fig.savefig(os.path.join(RES, f"confusion_{mode}.png"), dpi=150); plt.close(fig)


def pick(args, subjects):
    vids = sorted(glob.glob(os.path.join(args.videos, "*.mp4")) + glob.glob(os.path.join(args.videos, "*.avi")))
    if subjects:
        vids = [v for v in vids if os.path.basename(v).split("_")[0] in subjects]
    return vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/own"); ap.add_argument("--labels", default="data/own/labels.csv")
    ap.add_argument("--models", default="models"); ap.add_argument("--calib", type=float, default=5.0)
    ap.add_argument("--test_subjects", default=""); ap.add_argument("--val_subjects", default="")
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--sweep_mode", default="landmark")
    ap.add_argument("--err_mode", default="fusion"); ap.add_argument("--ear_factor", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(RES, "errors"), exist_ok=True)
    df = pd.read_csv(a.labels)
    cfg = Config(calib_s=a.calib)
    if a.ear_factor: cfg.ear_factor = a.ear_factor

    if a.sweep:
        vids = pick(a, set(filter(None, a.val_subjects.split(","))))
        data = {v: (extract_video(v, a.models), labels_for(df, os.path.basename(v))) for v in vids}
        rows = []
        for ef, pt, cd in itertools.product([0.65, 0.7, 0.75, 0.8, 0.85], [0.2, 0.3, 0.4], [1.0, 1.5, 2.0]):
            c = Config(calib_s=a.calib, ear_factor=ef, perclos_thr=pt, closed_dur_s=cd)
            r, _ = summarize([evaluate_video(d, s, a.sweep_mode, c, a.calib) for d, s in data.values()], a.sweep_mode)
            if r: rows.append(dict(ear_factor=ef, perclos_thr=pt, closed_dur_s=cd, **r))
        sw = pd.DataFrame(rows); sw.to_csv(os.path.join(RES, "sweep.csv"), index=False)
        plt.figure(figsize=(5, 4)); plt.scatter(sw.false_alarms_per_hour, sw.recall_drowsy)
        plt.xlabel("false alarms / hour"); plt.ylabel("recall DROWSY"); plt.title("Trade-off (val)")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(RES, "tradeoff.png"), dpi=150)
        print(sw.sort_values(["recall_drowsy", "false_alarms_per_hour"], ascending=[False, True]).head(10).to_string())
        return

    vids = pick(a, set(filter(None, a.test_subjects.split(","))))
    if not vids: raise SystemExit("Không có video phù hợp.")
    data = {v: (extract_video(v, a.models), labels_for(df, os.path.basename(v))) for v in vids}
    has_dl = all(bool(d["has_dl"]) for d, _ in data.values())
    modes = ["landmark"] + (["dl", "fusion"] if has_dl else [])
    if not has_dl: print("Chưa có model .onnx -> chỉ đánh giá landmark.")
    table, cond_rows = [], []
    for mode in modes:
        parts = {v: evaluate_video(d, s, mode, cfg, a.calib) for v, (d, s) in data.items()}
        row, cm = summarize(list(parts.values()), mode)
        if row is None: continue
        table.append(row); plot_cm(cm, mode)
        for cond in ("glasses", "dark", "mask"):
            for val in (True, False):
                sel = [p for v, p in parts.items() if tags_of(os.path.basename(v))[cond] == val]
                r, _ = summarize(sel, mode) if sel else (None, None)
                if r: cond_rows.append(dict(condition=f"{cond}={val}", **r))
        if mode == a.err_mode or (a.err_mode not in modes and mode == "landmark"):
            for v, p in parts.items():
                cap, n = cv2.VideoCapture(v), 0
                if not cap.isOpened(): continue
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                for i, (yy, pp) in zip(p["idx"], zip(p["y"], p["p"])):
                    if yy != pp and n < 20:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(data[v][0]["t"][i] * fps)); ok, fr = cap.read()
                        if ok:
                            cv2.imwrite(os.path.join(RES, "errors", f"{mode}_{os.path.basename(v)[:-4]}_t{data[v][0]['t'][i]:.0f}_true-{yy}_pred-{pp}.jpg"), fr); n += 1
                cap.release()
    pd.DataFrame(table).to_csv(os.path.join(RES, "results.csv"), index=False)
    pd.DataFrame(cond_rows).to_csv(os.path.join(RES, "per_condition.csv"), index=False)
    json.dump(cfg.to_dict(), open(os.path.join(RES, "config_used.json"), "w"), indent=1)
    print(pd.DataFrame(table).to_string(index=False)); print("\nTheo điều kiện:"); print(pd.DataFrame(cond_rows).to_string(index=False))


if __name__ == "__main__":
    main()
