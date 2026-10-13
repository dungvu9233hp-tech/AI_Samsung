"""Biểu đồ đánh giá model mắt (ONNX): ROC, PR, chọn ngưỡng, phân bố xác suất, ma trận nhầm lẫn, F1 theo nguồn.

  python training/plot_eval.py --csv data/processed/eye_split.csv --model models/eye.onnx --out eval/plots
  python training/plot_eval.py --csv ... --model ... --cew eval/results/cew_eval.json     # thêm biểu đồ CEW

Ngưỡng tối ưu được chọn trên VAL (không chọn trên test/CEW), in ra để dùng cho eval_cew.py --thr.
Nhãn: 1 = nhắm, 0 = mở.
"""
import argparse, json, os
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
                             confusion_matrix, f1_score)

S = 64


def load(paths):
    out = np.zeros((len(paths), S, S), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(p, 0)
        out[i] = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA) if im is not None else 0
    return out


def probs(sess, X):
    name, out = sess.get_inputs()[0].name, []
    for i in range(0, len(X), 512):
        x = X[i:i + 512].astype(np.float32)[:, None] / 255.0
        lg = sess.run(None, {name: x})[0]
        e = np.exp(lg - lg.max(1, keepdims=True))
        out.append(e[:, 1] / e.sum(1))
    return np.concatenate(out)


def best_thr(y, p):
    ts = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, p > t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))]), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="eval/plots")
    ap.add_argument("--cew", default=None, help="cew_eval.json do eval_cew.py sinh ra")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv); df = df[df.path.map(os.path.exists)].reset_index(drop=True)
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    D = {}
    for sp in ("val", "test"):
        d = df[df.split == sp].reset_index(drop=True)
        D[sp] = (d, d.label.values, probs(sess, load(d.path.tolist())))
    yv, pv = D["val"][1], D["val"][2]
    yt, pt = D["test"][1], D["test"][2]
    thr, ts = best_thr(yv, pv)
    print(f"Ngưỡng tối ưu trên VAL (max F1): {thr:.2f}   (dùng cho eval_cew.py --thr {thr:.2f})")
    for sp, (_, y, p) in D.items():
        print(f"{sp}: AUC {roc_auc_score(y, p):.3f} | AP {average_precision_score(y, p):.3f} | "
              f"F1@{thr:.2f} {f1_score(y, p > thr):.3f} | F1@0.5 {f1_score(y, p > 0.5):.3f}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    col = {"val": "#378ADD", "test": "#D85A30"}
    for sp, (_, y, p) in D.items():
        fpr, tpr, _ = roc_curve(y, p); ax[0, 0].plot(fpr, tpr, c=col[sp], label=f"{sp} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p); ax[0, 1].plot(rc, pr, c=col[sp], label=f"{sp} (AP {average_precision_score(y, p):.3f})")
    ax[0, 0].plot([0, 1], [0, 1], "--", c="gray"); ax[0, 0].set_title("ROC"); ax[0, 0].set_xlabel("FPR"); ax[0, 0].set_ylabel("TPR (recall nhắm)"); ax[0, 0].legend()
    ax[0, 1].set_title("Precision-Recall"); ax[0, 1].set_xlabel("Recall"); ax[0, 1].set_ylabel("Precision"); ax[0, 1].legend()

    P = [((pv > t) & (yv == 1)).sum() / max((pv > t).sum(), 1) for t in ts]
    R = [((pv > t) & (yv == 1)).sum() / max((yv == 1).sum(), 1) for t in ts]
    F = [f1_score(yv, pv > t, zero_division=0) for t in ts]
    ax[0, 2].plot(ts, P, label="precision"); ax[0, 2].plot(ts, R, label="recall (nhắm)"); ax[0, 2].plot(ts, F, label="F1", lw=2, c="k")
    ax[0, 2].axvline(thr, ls="--", c="gray"); ax[0, 2].set_title(f"Chọn ngưỡng trên VAL (best = {thr:.2f})"); ax[0, 2].set_xlabel("ngưỡng"); ax[0, 2].legend()

    ax[1, 0].hist(pt[yt == 0], bins=40, alpha=.6, color="#1D9E75", label="mở (thật)")
    ax[1, 0].hist(pt[yt == 1], bins=40, alpha=.6, color="#E24B4A", label="nhắm (thật)")
    ax[1, 0].axvline(thr, ls="--", c="gray"); ax[1, 0].set_yscale("log"); ax[1, 0].set_title("Phân bố xác suất 'nhắm' (test, trục log)"); ax[1, 0].legend()

    cm = confusion_matrix(yt, pt > thr, labels=[0, 1]); cmn = cm / cm.sum(1, keepdims=True)
    ax[1, 1].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax[1, 1].text(j, i, f"{v}\n{cmn[i, j]*100:.1f}%", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
    ax[1, 1].set_xticks([0, 1], ["đoán mở", "đoán nhắm"]); ax[1, 1].set_yticks([0, 1], ["thật mở", "thật nhắm"])
    ax[1, 1].set_title(f"Ma trận nhầm lẫn (test, ngưỡng {thr:.2f})")

    dt = D["test"][0]
    if "source" in dt:
        src = sorted(dt.source.unique()); f1s = [f1_score(yt[dt.source.values == s], pt[dt.source.values == s] > thr, zero_division=0) for s in src]
        ns = [int((dt.source.values == s).sum()) for s in src]
        ax[1, 2].bar(src, f1s, color="#7F77DD")
        for i, (v, n) in enumerate(zip(f1s, ns)): ax[1, 2].text(i, v, f"{v:.3f}\nn={n}", ha="center", va="bottom")
        ax[1, 2].set_ylim(0, 1.15); ax[1, 2].set_title("F1 theo nguồn dữ liệu (test)")
    else:
        ax[1, 2].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "eval_overview.png"), dpi=150); plt.close(fig)

    if a.cew and os.path.exists(a.cew):
        r = json.load(open(a.cew, encoding="utf-8"))["results"]
        keys = list(r); x = np.arange(len(keys)); w = .27
        f1 = [r[k]["f1_closed"] for k in keys]; rc = [r[k]["recall_closed"] for k in keys]; pr = [r[k]["precision"] for k in keys]
        err = np.array([[r[k]["f1_closed"] - r[k]["f1_ci95"][0], r[k]["f1_ci95"][1] - r[k]["f1_closed"]] for k in keys]).T
        fig, ax2 = plt.subplots(figsize=(7, 4.2))
        ax2.bar(x - w, pr, w, label="precision", color="#85B7EB")
        ax2.bar(x, rc, w, label="recall (nhắm)", color="#F0997B")
        ax2.bar(x + w, f1, w, yerr=err, capsize=4, label="F1 (CI 95%)", color="#534AB7")
        ax2.set_xticks(x, [k.replace("pad_", "pad=") for k in keys]); ax2.set_ylim(0, 1.05)
        ax2.set_title("CEW (test ngoài): ảnh hưởng của pad"); ax2.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "cew_summary.png"), dpi=150); plt.close(fig)
    print("Đã lưu biểu đồ vào", a.out)


if __name__ == "__main__":
    main()
