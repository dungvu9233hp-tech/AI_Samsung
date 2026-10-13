"""Thống kê và đánh giá chất lượng các tập dữ liệu mắt: CEW, bus (COCO Roboflow), MRL.
Nhãn: 1 = nhắm (closed), 0 = mở (open). Truyền dataset nào thì phân tích dataset đó.

  python training/dataset_stats.py --cew data/raw/cew --bus data/raw/bus --mrl data/raw/mrl
  python training/dataset_stats.py --cew data/raw/cew --bus data/raw/bus --mrl data/raw/mrl \
         --model models/eye.onnx --thr 0.5          # thêm đánh giá model trên từng dataset

Với mỗi dataset: số ảnh theo lớp, số nhóm (người/khuôn mặt/khối khung), kích thước ảnh, độ sáng/tương phản/độ nét,
kiểm tra "đường tắt" (một đặc trưng đơn giản có tách được hai lớp không), ảnh trùng lặp, và kiểm tra riêng
(MRL: kính/ánh sáng/phản chiếu; bus: rò rỉ giữa split gốc; CEW: số khuôn mặt).
Kết quả lưu ở eval/results/dataset_stats/.
"""
import argparse, glob, hashlib, json, os, re
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- quét metadata (không đọc pixel)
def scan_cew(root):
    cl = glob.glob(os.path.join(root, "**", "closedEyes", "*.jpg"), recursive=True)
    op = glob.glob(os.path.join(root, "**", "openEyes", "*.jpg"), recursive=True)
    df = pd.DataFrame([dict(path=p, label=1) for p in sorted(cl)] + [dict(path=p, label=0) for p in sorted(op)])
    if df.empty:
        return df
    # ảnh mắt trái/phải của cùng một khuôn mặt -> cùng nhóm
    df["group"] = df.path.map(lambda p: re.sub(r"_[LR]$", "", os.path.splitext(os.path.basename(p))[0]))
    return df


def scan_mrl(root):
    rows = []
    for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        if not p.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        parts = os.path.splitext(os.path.basename(p))[0].split("_")
        if len(parts) < 5 or not parts[0].startswith("s"):
            continue
        g = lambda i: parts[i] if len(parts) > i else None
        # s0001_imgid_gender_glasses_eyestate(0=nhắm,1=mở)_reflection_lighting_sensor
        rows.append(dict(path=p, label=1 if parts[4] == "0" else 0, group=parts[0], gender=g(2), glasses=g(3),
                         reflection=g(5), lighting=g(6), sensor=g(7)))
    return pd.DataFrame(rows)


def scan_bus(root, block):
    rows = []
    for sp in ("train", "valid", "test"):
        jf = os.path.join(root, sp, "_annotations.coco.json")
        if not os.path.exists(jf):
            continue
        d = json.load(open(jf, encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            n = cats[a["category_id"]]
            if n not in ("closed_eye", "open_eye"):
                continue
            im = imgs[a["image_id"]]
            m = re.match(r"(\d+)", (im.get("extra") or {}).get("name", "") or im["file_name"])
            fno = int(m.group(1)) if m else -1
            rows.append(dict(path=os.path.join(root, sp, im["file_name"]), label=1 if n == "closed_eye" else 0,
                             group=f"b{fno // block}" if fno >= 0 else "b?", split=sp, frame=fno, bbox=a["bbox"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- đọc pixel và đặc trưng
def read_record(row, k):
    g = cv2.imread(row["path"], 0)
    if g is None:
        return None
    bb = row.get("bbox")
    if isinstance(bb, list):                      # bus: cắt vuông quanh hộp nhãn như crop_eye lúc chạy thật
        x, y, w, h = bb
        s, cx, cy = k * max(w, h), x + w / 2, y + h / 2
        x0, x1, y0, y1 = int(cx - s / 2), int(cx + s / 2), int(cy - s / 2), int(cy + s / 2)
        if x0 < 0 or y0 < 0 or x1 > g.shape[1] or y1 > g.shape[0] or x1 - x0 < 8:
            return None
        g = g[y0:y1, x0:x1]
    return dict(h=g.shape[0], w=g.shape[1], mean=float(g.mean()), std=float(g.std()),
                sharp=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                img=cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA),
                md5=hashlib.md5(g.tobytes()).hexdigest())


def auc_safe(y, x):
    return float(roc_auc_score(y, x)) if len(set(y)) == 2 else float("nan")


def file_md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def dup_stats(keys, labels):
    d = {}
    for k_, l in zip(keys, labels):
        d.setdefault(k_, []).append(int(l))
    groups = [v for v in d.values() if len(v) > 1]
    return dict(unique=len(d), dup_groups=len(groups), dup_rows=int(sum(len(v) - 1 for v in groups)),
                label_conflicts=int(sum(len(set(v)) > 1 for v in groups)))


# ---------------------------------------------------------------- phân tích một dataset
def analyze(name, meta, R, dups, out_dir):
    y = meta.label.values
    o = dict(dataset=name, n_total=len(meta), n_closed=int(y.sum()), n_open=int(len(y) - y.sum()))
    o["closed_ratio"] = round(o["n_closed"] / max(len(y), 1), 3)
    gs = meta.groupby("group").label.agg(["size", "mean"])
    o.update(n_groups=len(gs), per_group_min=int(gs["size"].min()), per_group_median=float(gs["size"].median()),
             per_group_max=int(gs["size"].max()), groups_one_class=int(((gs["mean"] == 0) | (gs["mean"] == 1)).sum()))
    yr = R.label.values
    o["n_loaded"] = len(R)
    for f in ("w", "h"):
        o[f"{f}_median"], o[f"{f}_min"] = float(R[f].median()), int(R[f].min())
    for f in ("mean", "std", "sharp"):
        for lab, tag in ((1, "closed"), (0, "open")):
            o[f"{f}_{tag}_median"] = round(float(R[R.label == lab][f].median()), 2)
        o[f"auc_{f}"] = round(auc_safe(yr, R[f].values), 3)
    o["auc_area"] = round(auc_safe(yr, (R.w * R.h).values), 3)
    o.update({f"dup_{k_}": v for k_, v in dups.items()})

    print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
    print(f"Tổng {o['n_total']} ảnh | nhắm {o['n_closed']} | mở {o['n_open']} | tỉ lệ nhắm {o['closed_ratio']:.1%}")
    print(f"Nhóm: {o['n_groups']} | ảnh/nhóm min {o['per_group_min']} - median {o['per_group_median']:.0f} - "
          f"max {o['per_group_max']} | nhóm chỉ có 1 lớp: {o['groups_one_class']}")
    print(f"Kích thước ảnh nạp: median {o['w_median']:.0f}x{o['h_median']:.0f}, nhỏ nhất {o['w_min']}x{o['h_min']} "
          f"(đã nạp {o['n_loaded']} ảnh)")
    print(f"{'đặc trưng':10s} {'nhắm(median)':>13s} {'mở(median)':>12s} {'AUC tách lớp':>13s}")
    for f, vi in (("mean", "độ sáng"), ("std", "tương phản"), ("sharp", "độ nét")):
        a = o[f"auc_{f}"]
        warn = "  ⚠ có thể là đường tắt" if abs(a - 0.5) > 0.2 else ""
        print(f"{vi:10s} {o[f'{f}_closed_median']:13.1f} {o[f'{f}_open_median']:12.1f} {a:13.3f}{warn}")
    print(f"AUC theo diện tích ảnh: {o['auc_area']:.3f}" + ("  ⚠ kích thước ảnh tách được lớp" if abs(o['auc_area'] - .5) > .2 else ""))
    print(f"Trùng lặp: {o['dup_unique']} ảnh khác nhau | {o['dup_dup_groups']} nhóm trùng "
          f"({o['dup_dup_rows']} ảnh thừa) | trùng nhưng khác nhãn: {o['dup_label_conflicts']}")

    # --- kiểm tra riêng từng dataset
    if name == "mrl":
        for c in ("glasses", "lighting", "reflection", "gender", "sensor"):
            if c in meta and meta[c].notna().any():
                t = meta.groupby(c).label.agg(n="size", closed_ratio="mean").round(3)
                print(f"\nMRL theo '{c}':\n{t.to_string()}")
        sub = meta.groupby("group").label.mean()
        print(f"\nMRL tỉ lệ nhắm theo người: min {sub.min():.2f} - median {sub.median():.2f} - max {sub.max():.2f}")
    if name == "bus":
        fr = meta[meta.frame >= 0].drop_duplicates("frame")[["frame", "split"]].sort_values("frame").values
        adj = sum(1 for a, b in zip(fr, fr[1:]) if a[1] != b[1] and b[0] - a[0] <= 2)
        o["bus_adjacent_cross_split"] = int(adj)
        blk = meta.groupby("group").label.mean()
        print(f"\nBUS: cặp khung liền kề (cách <=2) nằm ở split gốc khác nhau: {adj}/{max(len(fr) - 1, 1)}"
              f"  -> {'split gốc BỊ RÒ RỈ, đừng dùng' if adj > 0.1 * len(fr) else 'ổn'}")
        print(f"BUS: tỉ lệ nhắm theo khối khung: min {blk.min():.2f} - median {blk.median():.2f} - max {blk.max():.2f}")
        print("BUS: số hộp theo split gốc:", meta.split.value_counts().to_dict())
    if name == "cew":
        print(f"\nCEW: {o['n_groups']} nhóm (mắt trái/phải gộp theo khuôn mặt). Hai lớp lấy từ hai nguồn khác nhau "
              f"nên hãy chú ý các dòng '⚠ đường tắt' ở trên.")

    # --- ảnh mẫu
    rows = []
    for lab in (1, 0):
        sel = [im for im, l in zip(R.img, R.label) if l == lab][:24]
        sel += [np.zeros((64, 64), np.uint8)] * (24 - len(sel))
        rows += [np.hstack(sel[i * 12:(i + 1) * 12]) for i in range(2)]
    cv2.imwrite(os.path.join(out_dir, f"{name}_samples.png"), np.vstack(rows))
    return o


def group_ci(y, p, groups, thr, n=500, seed=0):
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(inv.max() + 1)]
    v = []
    for _ in range(n):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        v.append(f1_score(y[ii], p[ii] > thr, zero_division=0))
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cew"); ap.add_argument("--bus"); ap.add_argument("--mrl")
    ap.add_argument("--model", help="ONNX model mắt; nếu có thì đánh giá model trên từng dataset")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max_images", type=int, default=20000, help="tối đa ảnh nạp pixel mỗi dataset (lấy mẫu ngẫu nhiên)")
    ap.add_argument("--k", type=float, default=1.75, help="bus: cạnh crop = k * cạnh dài của hộp")
    ap.add_argument("--block", type=int, default=100, help="bus: số khung gốc mỗi nhóm")
    ap.add_argument("--skip_dups", action="store_true", help="bỏ md5 toàn bộ file (nhanh hơn với MRL)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval/results/dataset_stats")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    todo = [(n, p) for n, p in (("cew", a.cew), ("bus", a.bus), ("mrl", a.mrl)) if p]
    if not todo:
        raise SystemExit("Hãy truyền ít nhất một trong --cew --bus --mrl")
    sess = None
    if a.model:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name

    summary, model_rows, hist = [], [], {}
    for name, root in todo:
        meta = {"cew": lambda: scan_cew(root), "bus": lambda: scan_bus(root, a.block), "mrl": lambda: scan_mrl(root)}[name]()
        if meta.empty:
            print(f"\n[{name}] không tìm thấy dữ liệu trong {root}"); continue
        meta = meta.reset_index(drop=True)
        print(f"\n[{name}] quét xong {len(meta)} mục, đang nạp ảnh...")
        idx = np.random.default_rng(a.seed).choice(len(meta), min(a.max_images, len(meta)), replace=False)
        ms, recs = meta.iloc[np.sort(idx)].reset_index(drop=True), []
        for _, row in ms.iterrows():
            recs.append(read_record(row, a.k))
        ok = np.array([r is not None for r in recs])
        ms = ms[ok].reset_index(drop=True)
        R = pd.DataFrame([r for r in recs if r is not None])
        R["label"] = ms.label.values
        if name == "bus" or a.skip_dups:                       # bus: md5 của crop; --skip_dups: chỉ ảnh đã nạp
            dups = dup_stats(R.md5.tolist(), ms.label.tolist())
        else:
            dups = dup_stats([file_md5(p) for p in meta.path], meta.label)
        summary.append(analyze(name, meta, R, dups, a.out))
        hist[name] = R

        if sess is not None:
            X = np.stack(R.img.tolist()).astype(np.float32)[:, None] / 255.0
            lg = np.concatenate([sess.run(None, {inp: X[i:i + 512]})[0] for i in range(0, len(X), 512)])
            e = np.exp(lg - lg.max(1, keepdims=True)); p = e[:, 1] / e.sum(1)
            y = ms.label.values
            pr, rc, f1, _ = precision_recall_fscore_support(y, p > a.thr, average="binary", zero_division=0)
            ci = group_ci(y, p, ms.group.values, a.thr)
            row = dict(dataset=name, n=len(y), thr=a.thr, acc=float(((p > a.thr) == y).mean()), precision=float(pr),
                       recall_closed=float(rc), f1=float(f1), f1_ci_lo=ci[0], f1_ci_hi=ci[1], auc=auc_safe(y, p))
            model_rows.append(row)
            note = "TEST NGOÀI nếu model chưa từng thấy dataset này" if name == "cew" else \
                   "model có thể đã train trên dataset này -> KHÔNG phải test độc lập"
            print(f"\nModel {os.path.basename(a.model)} @thr {a.thr}: acc {row['acc']:.3f} | precision {pr:.3f} | "
                  f"recall(nhắm) {rc:.3f} | F1 {f1:.3f} [{ci[0]:.3f}-{ci[1]:.3f}] | AUC {row['auc']:.3f}\n  ({note})")
            if name == "mrl":
                for c in ("glasses", "lighting", "reflection"):
                    if c in ms and ms[c].notna().any():
                        t = pd.DataFrame(dict(v=ms[c], y=y, pred=(p > a.thr).astype(int)))
                        r = pd.DataFrame({v: dict(
                            n=len(d), recall_closed=d[d.y == 1].pred.mean() if (d.y == 1).any() else np.nan,
                            false_alarm=d[d.y == 0].pred.mean() if (d.y == 0).any() else np.nan)
                            for v, d in t.groupby("v")}).T.round(3)
                        print(f"  theo '{c}':\n{r.to_string()}")

    # --- biểu đồ độ sáng theo lớp
    if hist:
        fig, axs = plt.subplots(1, len(hist), figsize=(4.2 * len(hist), 3.2), squeeze=False)
        for ax, (n, R) in zip(axs[0], hist.items()):
            ax.hist(R[R.label == 1]["mean"], bins=40, alpha=.6, label="nhắm", density=True)
            ax.hist(R[R.label == 0]["mean"], bins=40, alpha=.6, label="mở", density=True)
            ax.set_title(n); ax.set_xlabel("độ sáng trung bình"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "brightness_hist.png"), dpi=130)

    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    pd.DataFrame(summary).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    if model_rows:
        pd.DataFrame(model_rows).to_csv(os.path.join(a.out, "model_eval.csv"), index=False)
    print(f"\nĐã lưu vào {a.out}: summary.json/csv, {', '.join(n + '_samples.png' for n in hist)}, brightness_hist.png"
          + (", model_eval.csv" if model_rows else ""))


if __name__ == "__main__":
    main()
