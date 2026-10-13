"""Vẽ biểu đồ cho báo cáo về các tập dữ liệu mắt CEW, bus (COCO), MRL.
  python training/plot_datasets.py --cew data/raw/cew --bus data/raw/bus --mrl data/raw/mrl
  python training/plot_datasets.py --cew data/raw/cew --bus data/raw/bus --mrl data/raw/mrl \
         --model models/eye.onnx --thr 0.5        # thêm biểu đồ điểm số model và đường ROC
Cần training/dataset_stats.py cùng thư mục (dùng chung hàm đọc dữ liệu). Truyền dataset nào vẽ dataset đó.
Ảnh lưu ở eval/results/figures/ (PNG). Nhãn: 1 = nhắm, 0 = mở.
"""
import argparse, os, sys
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_stats import auc_safe, read_record, scan_bus, scan_cew, scan_mrl

C_CLOSED, C_OPEN = "#D55E00", "#0072B2"
NAMES = {"cew": "CEW", "bus": "Xe buýt", "mrl": "MRL"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})


def fmt(n):                                   # 84898 -> 84.898
    return f"{int(n):,}".replace(",", ".")


def save(fig, out, name, dpi):
    fig.tight_layout()
    p = os.path.join(out, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  đã lưu", p)


def load(name, root, a):
    scan = {"cew": lambda: scan_cew(root), "bus": lambda: scan_bus(root, a.block), "mrl": lambda: scan_mrl(root)}[name]
    meta = scan().reset_index(drop=True)
    if meta.empty:
        return None
    idx = np.sort(np.random.default_rng(a.seed).choice(len(meta), min(a.max_images, len(meta)), replace=False))
    ms = meta.iloc[idx].reset_index(drop=True)
    recs = [read_record(r, a.k) for _, r in ms.iterrows()]
    ok = np.array([r is not None for r in recs])
    ms = ms[ok].reset_index(drop=True)
    R = pd.DataFrame([r for r in recs if r is not None])
    R["label"], R["group"] = ms.label.values, ms.group.values
    print(f"[{name}] {len(meta)} mục, nạp {len(R)} ảnh")
    return meta, ms, R


# ---------------------------------------------------------------- các biểu đồ
def fig_overview(D, out, dpi):
    names = list(D); x = np.arange(len(names)); w = .38
    cl = [int(D[n][0].label.sum()) for n in names]
    op = [len(D[n][0]) - c for n, c in zip(names, cl)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(x - w / 2, cl, w, color=C_CLOSED, label="Nhắm")
    ax[0].bar(x + w / 2, op, w, color=C_OPEN, label="Mở")
    for i, (c, o) in enumerate(zip(cl, op)):
        ax[0].text(i - w / 2, c, fmt(c), ha="center", va="bottom", fontsize=8)
        ax[0].text(i + w / 2, o, fmt(o), ha="center", va="bottom", fontsize=8)
    ax[0].set_yscale("log"); ax[0].set_ylim(10, max(cl + op) * 4)
    ax[0].set_xticks(x); ax[0].set_xticklabels([NAMES[n] for n in names])
    ax[0].set_ylabel("Số ảnh (thang log)"); ax[0].set_title("Quy mô và số ảnh mỗi lớp"); ax[0].legend()
    ratio = [100 * c / (c + o) for c, o in zip(cl, op)]
    ax[1].bar(x, ratio, color="#666", width=.55); ax[1].axhline(50, ls="--", color="k", lw=1)
    for i, r in enumerate(ratio):
        ax[1].text(i, r / 2, f"{r:.1f}%", ha="center", va="center", fontsize=10, color="white", weight="bold")
    ax[1].set_ylim(0, 100); ax[1].set_xticks(x); ax[1].set_xticklabels([NAMES[n] for n in names])
    ax[1].set_ylabel("Tỉ lệ ảnh mắt nhắm (%)"); ax[1].set_title("Cân bằng lớp (đường nét đứt = 50%)")
    ax[2].boxplot([D[n][2].w.values for n in names], showfliers=False)
    ax[2].set_xticks(range(1, len(names) + 1)); ax[2].set_xticklabels([NAMES[n] for n in names])
    ax[2].set_ylabel("Chiều rộng ảnh gốc (px)"); ax[2].set_title("Độ phân giải ảnh gốc")
    save(fig, out, "fig1_tong_quan.png", dpi)


FEATS = [("mean", "Độ sáng trung bình", False), ("std", "Tương phản (độ lệch chuẩn)", False),
         ("sharp", "Độ nét: log10(phương sai Laplacian)", True)]


def fig_features(D, out, dpi):
    names = list(D)
    fig, axs = plt.subplots(3, len(names), figsize=(4.6 * len(names), 9), squeeze=False)
    for j, n in enumerate(names):
        R = D[n][2]
        for i, (f, label, lg) in enumerate(FEATS):
            ax, v = axs[i][j], R[f].values.astype(float)
            auc = auc_safe(R.label.values, v)
            if lg:
                v = np.log10(v + 1)
            bins = np.linspace(np.percentile(v, .5), np.percentile(v, 99.5), 40)
            for lab, col, tag in ((1, C_CLOSED, "Nhắm"), (0, C_OPEN, "Mở")):
                ax.hist(v[R.label.values == lab], bins=bins, alpha=.55, color=col, density=True, label=tag)
            ax.set_title(f"{NAMES[n]}  (AUC={auc:.2f})" if i == 0 else f"AUC={auc:.2f}", fontsize=10)
            ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel("Mật độ")
            if i == 0 and j == 0:
                ax.legend()
    fig.suptitle("Phân bố đặc trưng ảnh theo lớp (AUC: khả năng dùng riêng đặc trưng này để tách hai lớp, 0,5 = không tách được)",
                 fontsize=11, y=1.0)
    save(fig, out, "fig2_dac_trung_anh.png", dpi)


def fig_groups(D, out, dpi):
    cand = [n for n in D if D[n][0].groupby("group").size().nunique() > 1]
    if not cand:
        return
    fig, axs = plt.subplots(2, len(cand), figsize=(6.5 * len(cand), 7), squeeze=False)
    for j, n in enumerate(cand):
        meta = D[n][0]
        g = meta.groupby("group").label.agg(size="size", ratio="mean")
        s = g.sort_values("size", ascending=False)
        axs[0][j].bar(range(len(s)), s["size"], color="#666")
        axs[0][j].set_title(f"{NAMES[n]}: số ảnh mỗi nhóm ({len(g)} nhóm)")
        axs[0][j].set_xlabel("Nhóm (sắp xếp giảm dần)"); axs[0][j].set_ylabel("Số ảnh")
        r = g.sort_values("ratio")
        axs[1][j].bar(range(len(r)), 100 * r["ratio"], color=C_CLOSED)
        axs[1][j].axhline(100 * meta.label.mean(), ls="--", color="k", lw=1, label="Tỉ lệ chung")
        axs[1][j].set_title(f"{NAMES[n]}: tỉ lệ ảnh nhắm trong từng nhóm")
        axs[1][j].set_xlabel("Nhóm (sắp xếp tăng dần)"); axs[1][j].set_ylabel("% ảnh nhắm"); axs[1][j].legend()
    save(fig, out, "fig3_phan_bo_nhom.png", dpi)


# Ý nghĩa mã trong tên file MRL (theo mô tả dataset, hãy đối chiếu lại với trang Kaggle)
MRL_MAP = {"glasses": ("Kính", {"0": "Không", "1": "Có"}),
           "lighting": ("Ánh sáng", {"0": "Kém", "1": "Tốt"}),
           "reflection": ("Phản chiếu", {"0": "Không", "1": "Ít", "2": "Nhiều"}),
           "gender": ("Giới tính", {"0": "Nam", "1": "Nữ"}),
           "sensor": ("Cảm biến", {"01": "01", "02": "02", "03": "03"})}


def fig_mrl(D, out, dpi):
    if "mrl" not in D:
        return
    meta = D["mrl"][0]
    cols = [c for c in MRL_MAP if c in meta and meta[c].notna().any()]
    fig, axs = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3.8), squeeze=False)
    for ax, c in zip(axs[0], cols):
        title, mp = MRL_MAP[c]
        t = meta.groupby(c).label.agg(n="size", closed="sum").sort_index()
        t["open"] = t.n - t.closed
        x = np.arange(len(t))
        ax.bar(x, t.closed, color=C_CLOSED, label="Nhắm")
        ax.bar(x, t.open, bottom=t.closed, color=C_OPEN, label="Mở")
        for i, (n_, cl_) in enumerate(zip(t.n, t.closed)):
            ax.text(i, n_, f"{100 * cl_ / n_:.0f}%\nnhắm", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([mp.get(str(k), str(k)) for k in t.index])
        ax.set_title(title); ax.set_ylim(top=t.n.max() * 1.2)
        if c == cols[0]:
            ax.set_ylabel("Số ảnh"); ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("MRL: phân bố theo điều kiện chụp (nhãn trên cột = tỉ lệ ảnh nhắm)", y=1.02)
    save(fig, out, "fig4_mrl_dieu_kien.png", dpi)


def fig_bus(D, out, dpi):
    if "bus" not in D:
        return
    meta = D["bus"][0]
    meta = meta[meta.frame >= 0]
    fr = meta.drop_duplicates("frame")[["frame", "split"]].sort_values("frame")
    v = fr.values
    adj = sum(1 for a, b in zip(v, v[1:]) if a[1] != b[1] and b[0] - a[0] <= 2)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    pos, col = {"train": 2, "valid": 1, "test": 0}, {"train": "#888888", "valid": "#E69F00", "test": "#CC79A7"}
    for sp in ("train", "valid", "test"):
        d = fr[fr.split == sp]
        ax[0].scatter(d.frame, [pos[sp]] * len(d), marker="|", s=120, color=col[sp], label=f"{sp} ({len(d)})")
    ax[0].set_ylim(-0.6, 3.0); ax[0].set_yticks([0, 1, 2]); ax[0].set_yticklabels(["test", "valid", "train"]); ax[0].legend(ncol=3, loc="upper right")
    ax[0].set_title(f"Xe buýt: split gốc của Roboflow xen kẽ theo khung hình ({adj}/{len(v) - 1} cặp khung liền kề nằm ở split khác nhau)")
    b = meta.assign(blk=meta.frame // 100).groupby("blk").label.mean()
    ax[1].bar(b.index * 100 + 50, 100 * b.values, width=90, color=C_CLOSED)
    ax[1].axhline(100 * meta.label.mean(), ls="--", color="k", lw=1)
    ax[1].set_ylabel("% mắt nhắm / khối 100 khung"); ax[1].set_xlabel("Chỉ số khung hình gốc")
    ax[1].set_title("Tỉ lệ mắt nhắm thay đổi mạnh theo đoạn thời gian (các khung cạnh nhau có nhãn giống nhau)")
    save(fig, out, "fig5_bus_roi_ri.png", dpi)


def fig_auc(D, out, dpi):
    rows = {}
    for n, (_, _, R) in D.items():
        y = R.label.values
        rows[n] = [auc_safe(y, R["mean"]), auc_safe(y, R["std"]), auc_safe(y, R["sharp"]), auc_safe(y, (R.w * R.h))]
    labels = ["Độ sáng", "Tương phản", "Độ nét", "Diện tích ảnh"]
    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(4); w = .8 / len(rows)
    for i, (n, v) in enumerate(rows.items()):
        b = ax.bar(x + i * w - .4 + w / 2, v, w, label=NAMES[n])
        for xi, vi in zip(x + i * w - .4 + w / 2, v):
            ax.text(xi, vi + .01, f"{vi:.2f}", ha="center", fontsize=7)
    ax.axhline(.5, color="k", lw=1); ax.axhspan(.3, .7, color="#999", alpha=.12)
    ax.set_ylim(.2, .85); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("AUC khi dùng riêng đặc trưng")
    ax.set_title("Kiểm tra 'đường tắt' (vùng xám: |AUC − 0,5| ≤ 0,2; càng xa 0,5 càng đáng ngờ)"); ax.legend()
    save(fig, out, "fig6_duong_tat.png", dpi)


def fig_samples(D, out, dpi, per=10):
    n = len(D)
    fig, axs = plt.subplots(2 * n, per, figsize=(per * 1.0, 2 * n * 1.15), squeeze=False)
    for i, (name, (_, _, R)) in enumerate(D.items()):
        for r, (lab, tag) in enumerate(((1, "Nhắm"), (0, "Mở"))):
            imgs = [im for im, l in zip(R.img, R.label) if l == lab][:per]
            for c in range(per):
                ax = axs[2 * i + r][c]; ax.axis("off")
                if c < len(imgs):
                    ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=255)
            axs[2 * i + r][0].text(-0.15, 0.5, f"{NAMES[name]}\n{tag}", transform=axs[2 * i + r][0].transAxes,
                                   ha="right", va="center", fontsize=9)
    fig.suptitle("Ảnh mẫu (đã đưa về 64x64 xám, đúng như đầu vào của model)", y=1.0)
    save(fig, out, "fig7_anh_mau.png", dpi)


def fig_model(D, out, dpi, model, thr):
    import onnxruntime as ort
    from sklearn.metrics import roc_auc_score
    s = ort.InferenceSession(model, providers=["CPUExecutionProvider"]); inp = s.get_inputs()[0].name
    names = list(D)
    fig, axs = plt.subplots(1, len(names) + 1, figsize=(4.4 * (len(names) + 1), 3.8))
    for ax, n in zip(axs, names):
        R = D[n][2]; X = np.stack(R.img.tolist()).astype(np.float32)[:, None] / 255.0
        lg = np.concatenate([s.run(None, {inp: X[i:i + 512]})[0] for i in range(0, len(X), 512)])
        e = np.exp(lg - lg.max(1, keepdims=True)); p = e[:, 1] / e.sum(1); y = R.label.values
        for lab, col, tag in ((1, C_CLOSED, "Nhắm (thật)"), (0, C_OPEN, "Mở (thật)")):
            ax.hist(p[y == lab], bins=30, range=(0, 1), alpha=.6, color=col, label=tag)
        ax.axvline(thr, ls="--", color="k", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("Xác suất 'nhắm' do model dự đoán"); ax.set_title(NAMES[n])
        if n == names[0]:
            ax.set_ylabel("Số ảnh (log)"); ax.legend(fontsize=8)
        fpr, tpr, _ = roc_curve(y, p)
        axs[-1].plot(fpr, tpr, label=f"{NAMES[n]} (AUC={roc_auc_score(y, p):.3f})")
    axs[-1].plot([0, 1], [0, 1], "k:", lw=1); axs[-1].set_xlabel("Tỉ lệ báo nhầm (mắt mở bị đoán nhắm)")
    axs[-1].set_ylabel("Tỉ lệ phát hiện đúng mắt nhắm"); axs[-1].set_title("Đường ROC"); axs[-1].legend(fontsize=8)
    fig.suptitle(f"Điểm số của {os.path.basename(model)} (đường nét đứt = ngưỡng {thr}). "
                 "Lưu ý: chỉ dataset model chưa từng thấy mới là test độc lập.", y=1.03, fontsize=10)
    save(fig, out, "fig8_diem_so_model.png", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cew"); ap.add_argument("--bus"); ap.add_argument("--mrl")
    ap.add_argument("--model"); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max_images", type=int, default=20000)
    ap.add_argument("--k", type=float, default=1.75); ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default="eval/results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = {}
    for n, root in (("cew", a.cew), ("bus", a.bus), ("mrl", a.mrl)):
        if root:
            r = load(n, root, a)
            if r is None:
                print(f"[{n}] không tìm thấy dữ liệu trong {root}")
            else:
                D[n] = r
    if not D:
        raise SystemExit("Hãy truyền ít nhất một trong --cew --bus --mrl")
    print("Đang vẽ...")
    fig_overview(D, a.out, a.dpi); fig_features(D, a.out, a.dpi); fig_groups(D, a.out, a.dpi)
    fig_mrl(D, a.out, a.dpi); fig_bus(D, a.out, a.dpi); fig_auc(D, a.out, a.dpi); fig_samples(D, a.out, a.dpi)
    if a.model:
        fig_model(D, a.out, a.dpi, a.model, a.thr)
    print("Xong. Ảnh nằm trong", a.out)


if __name__ == "__main__":
    main()
