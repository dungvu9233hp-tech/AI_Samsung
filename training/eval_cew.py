"""Đánh giá model mắt (ONNX) trên CEW như TẬP TEST NGOÀI.
CEW không được dùng để train, không dùng để chọn ngưỡng, không dùng để chọn checkpoint.

  python training/eval_cew.py --dir data/raw/cew --model models/eye.onnx --thr 0.5
  python training/eval_cew.py --dir data/raw/cew --model models/eye.onnx --thr 0.43 --pads 0,0.5

--thr : ngưỡng chọn từ tập VAL của MRL (không chọn trên CEW).
--pads: tỉ lệ viền thêm mỗi cạnh. CEW crop rất sát mắt, còn crop_eye() lúc chạy thật dùng pad=1.0
        (mắt chỉ chiếm ~1/2 ảnh). pad=0.5 mô phỏng gần giống lúc chạy thật.
Nhãn: 1 = nhắm (closed), 0 = mở. Cùng quy ước với train.py.
"""
import argparse, glob, json, os, re
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

S = 64


def scan(root):
    closed = glob.glob(os.path.join(root, "**", "closedEyes", "*.jpg"), recursive=True)
    opened = glob.glob(os.path.join(root, "**", "openEyes", "*.jpg"), recursive=True)
    if not closed or not opened:
        raise SystemExit(f"Không thấy closedEyes/ và openEyes/ trong {root}")
    paths = sorted(closed) + sorted(opened)
    y = np.array([1] * len(closed) + [0] * len(opened))
    # ảnh mắt trái/phải của cùng một khuôn mặt -> cùng nhóm (để bootstrap không bị lạc quan)
    groups = np.array([re.sub(r"_[LR]$", "", os.path.splitext(os.path.basename(p))[0]) for p in paths])
    return paths, y, groups


def prep(path, pad):
    im = cv2.imread(path, 0)
    if im is None:
        return None
    if pad > 0:
        k = int(round(im.shape[0] * pad))
        im = cv2.copyMakeBorder(im, k, k, k, k, cv2.BORDER_REPLICATE)
    return cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)   # giống train.load()


def probs(sess, X):
    name, out = sess.get_inputs()[0].name, []
    for i in range(0, len(X), 512):
        x = X[i:i + 512].astype(np.float32)[:, None] / 255.0
        lg = sess.run(None, {name: x})[0]
        e = np.exp(lg - lg.max(1, keepdims=True))
        out.append(e[:, 1] / e.sum(1))
    return np.concatenate(out)


def f1_ci(y, p, groups, thr, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(inv.max() + 1)]
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        vals.append(f1_score(y[ii], p[ii] > thr, zero_division=0))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def montage(paths, out, cell=96, cols=10):
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    M = np.zeros((rows * cell, cols * cell), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.resize(cv2.imread(p, 0), (cell, cell), interpolation=cv2.INTER_CUBIC)
        M[(i // cols) * cell:(i // cols + 1) * cell, (i % cols) * cell:(i % cols + 1) * cell] = im
    cv2.imwrite(out, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="thư mục chứa closedEyes/ và openEyes/")
    ap.add_argument("--model", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--pads", default="0,0.5")
    ap.add_argument("--out", default="eval/results")
    a = ap.parse_args()

    paths, y, groups = scan(a.dir)
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    print(f"CEW: {len(y)} ảnh | nhắm {int(y.sum())} | mở {int((1 - y).sum())} | {len(set(groups))} khuôn mặt")

    os.makedirs(a.out, exist_ok=True)
    res = {}
    for pad in [float(x) for x in a.pads.split(",")]:
        X = [prep(p, pad) for p in paths]
        keep = np.array([x is not None for x in X])
        Xk = np.stack([x for x in X if x is not None])
        yk, gk, pk = y[keep], groups[keep], np.array(paths)[keep]
        p = probs(sess, Xk)
        pred = p > a.thr
        pr, rc, f1, _ = precision_recall_fscore_support(yk, pred, average="binary", zero_division=0)
        r = dict(pad=pad, thr=a.thr, n=int(len(yk)), acc=float((pred == yk).mean()),
                 precision=float(pr), recall_closed=float(rc), f1_closed=float(f1),
                 auc=float(roc_auc_score(yk, p)), f1_ci95=f1_ci(yk, p, gk, a.thr),
                 cm=confusion_matrix(yk, pred, labels=[0, 1]).tolist())
        res[f"pad_{pad}"] = r
        print(f"\n[pad={pad}] thr={a.thr}  acc {r['acc']:.3f} | precision {r['precision']:.3f} | "
              f"recall(nhắm) {r['recall_closed']:.3f} | F1 {r['f1_closed']:.3f} "
              f"(95% CI {r['f1_ci95'][0]:.3f}-{r['f1_ci95'][1]:.3f}) | AUC {r['auc']:.3f}")
        print("  confusion [hàng=thật, cột=đoán] (mở, nhắm):", r["cm"])
        fn = pk[(yk == 1) & ~pred][np.argsort(p[(yk == 1) & ~pred])][:40]       # nhắm nhưng đoán mở (nguy hiểm)
        fp = pk[(yk == 0) & pred][np.argsort(-p[(yk == 0) & pred])][:40]        # mở nhưng đoán nhắm
        montage(list(fn), os.path.join(a.out, f"cew_pad{pad}_missed_closed.png"))
        montage(list(fp), os.path.join(a.out, f"cew_pad{pad}_false_alarm.png"))

    json.dump(dict(model=a.model, results=res), open(os.path.join(a.out, "cew_eval.json"), "w"), indent=1)
    print("\nĐã lưu", os.path.join(a.out, "cew_eval.json"), "và ảnh lỗi cew_pad*_*.png")


if __name__ == "__main__":
    main()
