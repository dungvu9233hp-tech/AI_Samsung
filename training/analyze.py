"""Thống kê dataset + vẽ biểu đồ huấn luyện.

  python training/analyze.py --csv data/processed/eye_split.csv --metrics models/eye_metrics.json --out eval/analysis

Đầu ra (trong --out):
  dataset_overview.png  : số ảnh theo split x nhãn, số subject mỗi split, ảnh/subject, độ sáng
  samples.png           : ảnh mẫu mắt nhắm / mở (ngẫu nhiên)
  training_curves.png   : loss, val_acc, val_f1 theo epoch (cần --metrics)
Và in ra: bảng thống kê, kiểm tra rò rỉ subject giữa các split.
"""
import argparse, json, os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS = ["train", "val", "test"]


def stats(df):
    df = df[df.path.map(os.path.exists)].copy()
    print(f"Tổng: {len(df)} ảnh | {df.subject.nunique()} subject")
    print("\nSố ảnh theo split x nhãn (1 = nhắm/yawn):")
    t = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(SPLITS)
    t["tổng"] = t.sum(axis=1); t["% nhãn 1"] = (100 * t[1] / t["tổng"]).round(1)
    print(t)
    print("\nSố subject mỗi split:", df.groupby("split").subject.nunique().reindex(SPLITS).to_dict())
    if "source" in df:
        print("\nNguồn x split:")
        print(df.groupby(["source", "split"]).size().unstack(fill_value=0).reindex(columns=SPLITS))
    # rò rỉ: một subject không được nằm ở 2 split
    s = {sp: set(df[df.split == sp].subject) for sp in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = s[a] & s[b]
        print(f"Rò rỉ subject {a}/{b}: {len(both)}" + ("  <-- LỖI" if both else "  (ok)"))
    return df


def brightness(df, n=400, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for sp in SPLITS:
        d = df[df.split == sp]
        d = d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)] if len(d) else d
        v = []
        for p in d.path:
            im = cv2.imread(p, 0)
            if im is not None: v.append(float(im.mean()))
        out[sp] = v
    return out


def overview(df, out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    c = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(SPLITS)
    c.plot(kind="bar", stacked=True, ax=ax[0, 0], rot=0, color=["#5DCAA5", "#E24B4A"])
    ax[0, 0].set_title("Số ảnh theo split và nhãn"); ax[0, 0].legend(["0 (mở)", "1 (nhắm)"])
    for i, sp in enumerate(SPLITS):
        ax[0, 0].text(i, c.loc[sp].sum(), int(c.loc[sp].sum()), ha="center", va="bottom")
    ns = df.groupby("split").subject.nunique().reindex(SPLITS)
    ax[0, 1].bar(ns.index, ns.values, color="#7F77DD"); ax[0, 1].set_title("Số subject mỗi split")
    for i, v in enumerate(ns.values): ax[0, 1].text(i, v, int(v), ha="center", va="bottom")
    per = df.groupby("subject").size()
    ax[1, 0].hist(per.values, bins=30, color="#378ADD")
    ax[1, 0].set_title("Số ảnh mỗi subject"); ax[1, 0].set_xlabel("ảnh / subject")
    b = brightness(df)
    ax[1, 1].boxplot([b[s] for s in SPLITS], tick_labels=SPLITS)
    ax[1, 1].set_title("Độ sáng trung bình ảnh (mẫu 400/split)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def samples(df, out, k=12, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for lab in (0, 1):
        d = df[df.label == lab]
        ps = d.path.iloc[rng.choice(len(d), min(k, len(d)), replace=False)] if len(d) else []
        ims = [cv2.resize(cv2.imread(p, 0), (96, 96)) for p in ps if cv2.imread(p, 0) is not None]
        while len(ims) < k: ims.append(np.zeros((96, 96), np.uint8))
        rows.append(np.hstack(ims))
    M = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(12, 2.2)); ax.imshow(M, cmap="gray"); ax.axis("off")
    ax.set_title("Hàng trên: mở (0)  |  Hàng dưới: nhắm (1)", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def curves(metrics_path, out):
    m = json.load(open(metrics_path, encoding="utf-8")); h = pd.DataFrame(m["history"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(h.epoch, h.loss, marker="o"); ax[0].set_title("Train loss")
    ax[1].plot(h.epoch, h.val_acc, marker="o", color="#1D9E75"); ax[1].set_title("Val accuracy")
    ax[2].plot(h.epoch, h.val_f1, marker="o", color="#D85A30"); ax[2].set_title("Val F1")
    best = int(h.val_f1.idxmax()); ax[2].axvline(h.epoch[best], ls="--", c="gray")
    ax[2].text(h.epoch[best], h.val_f1.min(), f" best ep{h.epoch[best]}", color="gray")
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    r = m["report"]
    print("\nTest nội bộ (checkpoint tốt nhất):")
    for k, v in r.items():
        if isinstance(v, dict): print(f"  {k:10s} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1-score']:.3f}  n={int(v['support'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metrics", default=None, help="file *_metrics.json do train.py sinh ra")
    ap.add_argument("--out", default="eval/analysis")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = stats(pd.read_csv(a.csv))
    overview(df, os.path.join(a.out, "dataset_overview.png"))
    samples(df, os.path.join(a.out, "samples.png"))
    if a.metrics and os.path.exists(a.metrics):
        curves(a.metrics, os.path.join(a.out, "training_curves.png"))
    print("\nĐã lưu biểu đồ vào", a.out)


if __name__ == "__main__":
    main()
