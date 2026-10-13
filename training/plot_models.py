"""So sánh các model mắt (ONNX): bảng số liệu + biểu đồ để đưa vào báo cáo.
Nhãn: 1 = nhắm, 0 = mở.

  python training/plot_models.py \
      --models old=models/eye_old.onnx small=models/eye_small.onnx mbv3=models/eye_mbv3.onnx \
      --metrics small=models/eye_small_metrics.json mbv3=models/eye_mbv3_metrics.json \
      --cew data/raw/cew \
      --split_csv data/processed/eye_split.csv --img_dirs data/raw/mrl data/processed/bus/eye \
      --thr 0.5 --out eval/results/model_compare

Các tập đánh giá:
  CEW            : test ngoài, cả 3 model chưa từng thấy.
  Bus (khối test): các khối khung có split='test' trong eye_split.csv. Model mới giữ riêng khối này (nhưng cùng
                   tài xế có thể xuất hiện ở khối train); model cũ chưa từng thấy dataset xe buýt.
  MRL (test)     : các người có split='test' trong eye_split.csv. Model mới giữ riêng; model cũ CÓ THỂ ĐÃ HỌC
                   (split cũ khác) nên so sánh ở tập này KHÔNG công bằng với model mới.
Tên file trong CSV được dò theo tên (basename) trong --img_dirs, nên CSV tạo trên Colab vẫn dùng được trên máy.
Key model: old / small / mbv3 có nhãn tiếng Việt sẵn, key khác hiển thị đúng tên key.
"""
import argparse, json, os, sys, time
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_stats import scan_cew

C_CLOSED, C_OPEN = "#D55E00", "#0072B2"
LABELS = {"old": "Cũ: SmallCNN (MRL)", "small": "Mới: SmallCNN (MRL+bus)", "mbv3": "Mới: MobileNetV3 (MRL+bus)"}
COLORS = {"old": "#8C8C8C", "small": "#009E73", "mbv3": "#CC79A7"}
SHORT = {"old": "Cũ-Small", "small": "Mới-Small", "mbv3": "Mới-MobileNet"}
TRAIN = {"old": "MRL"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})


def fmt(n):
    return f"{int(n):,}".replace(",", ".")


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, key, path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 2          # giống classifier.py
        self.key, self.path = key, path
        self.label = LABELS.get(key, key)
        self.short = SHORT.get(key, key)
        self.color = COLORS.get(key, plt.cm.tab10(len(COLORS) % 10))
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.size_kb = os.path.getsize(path) / 1024
        try:
            import onnx
            self.params = sum(int(np.prod(i.dims)) for i in onnx.load(path).graph.initializer)
        except Exception:
            self.params = None

    def probs(self, X):
        out = []
        for i in range(0, len(X), 256):
            lg = self.s.run(None, {self.inp: X[i:i + 256].astype(np.float32)[:, None] / 255.0})[0]
            e = np.exp(lg - lg.max(1, keepdims=True)); out.append(e[:, 1] / e.sum(1))
        return np.concatenate(out)

    def latency_ms(self, n=200):
        x = (np.random.default_rng(0).random((2, 1, 64, 64)) * 255).astype(np.float32) / 255.0   # 2 mắt / khung
        for _ in range(20):
            self.s.run(None, {self.inp: x})
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); self.s.run(None, {self.inp: x}); t.append((time.perf_counter() - t0) * 1000)
        return float(np.median(t))

    @property
    def arch(self):
        big = (self.params or 0) > 500_000 or self.size_kb > 2000
        return "MobileNetV3-small" if big else "SmallCNN"


# ---------------------------------------------------------------- dữ liệu đánh giá
def load_gray(p):
    im = cv2.imread(p, 0)
    return None if im is None else cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)


def make_set(files, y, groups, note):
    X, keep = [], []
    for i, f in enumerate(files):
        g = load_gray(f)
        if g is not None:
            X.append(g); keep.append(i)
    keep = np.array(keep, dtype=int)
    X = np.stack(X)
    return dict(X=X, y=np.asarray(y)[keep], groups=np.asarray(groups)[keep],
                bright=X.reshape(len(X), -1).mean(1), note=note)


def build_sets(a):
    S = {}
    if a.cew:
        df = scan_cew(a.cew)
        if df.empty:
            print("[CEW] không tìm thấy dữ liệu")
        else:
            S["CEW"] = make_set(df.path, df.label, df.group, "test ngoài")
    if a.split_csv:
        df = pd.read_csv(a.split_csv); df = df[df.split == "test"].copy()
        df["bn"] = df.path.map(lambda p: p.replace("\\", "/").split("/")[-1])
        df["src"] = np.where(df.bn.str.startswith("bus_"), "bus", "mrl")
        need = set(df.bn[~df.path.map(os.path.exists)])
        idx = {}
        for d in a.img_dirs or []:
            for root, _, fs in os.walk(d):
                for f in fs:
                    if f in need and f not in idx:
                        idx[f] = os.path.join(root, f)
        df["file"] = [r.path if os.path.exists(r.path) else idx.get(r.bn) for r in df.itertuples()]
        for src, name, note in (("bus", "Bus (khối test)", "held-out theo khối"), ("mrl", "MRL (test)", "held-out theo người")):
            d = df[df.src == src]
            miss = int(d.file.isna().sum())
            if len(d) == 0:
                continue
            if miss:
                print(f"[{name}] thiếu {miss}/{len(d)} ảnh (không thấy trong --img_dirs)" + (" -> bỏ qua tập này" if miss == len(d) else ""))
            d = d[d.file.notna()]
            if len(d):
                S[name] = make_set(d.file, d.label, d.subject, note + (f" (thiếu {miss})" if miss else ""))
        if "Bus (khối test)" in S and "MRL (test)" in S and all(S[k]["note"].find("thiếu") < 0 for k in ("Bus (khối test)", "MRL (test)")):
            b, m = S["Bus (khối test)"], S["MRL (test)"]
            S["Test CSV (MRL+bus)"] = dict(X=np.concatenate([b["X"], m["X"]]), y=np.concatenate([b["y"], m["y"]]),
                                           groups=np.concatenate([b["groups"], m["groups"]]),
                                           bright=np.concatenate([b["bright"], m["bright"]]), note="đúng split lúc train")
    return S


# ---------------------------------------------------------------- số liệu
def group_ci(y, p, groups, thr, n=400, seed=0):
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(inv.max() + 1)]
    v = []
    for _ in range(n):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        v.append(f1_score(y[ii], p[ii] > thr, zero_division=0))
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def metrics(y, p, groups, thr):
    pred = p > thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    lo, hi = group_ci(y, p, groups, thr)
    fc, tc, _ = roc_curve(y, p) if len(set(y)) == 2 else ([0, 1], [0, 1], None)
    return dict(r5=float(np.interp(0.05, fc, tc)), n=len(y), acc=(tp + tn) / len(y), precision=pr, recall=rc, fpr=fp / max(fp + tn, 1), f1=f1,
                f1_lo=lo, f1_hi=hi, auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def save(fig, out, name, dpi):
    fig.tight_layout()
    p = os.path.join(out, name); fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print("  đã lưu", p)


# ---------------------------------------------------------------- biểu đồ
def fig_overview(M, lat, out, dpi):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    names = [m.label for m in M]; x = np.arange(len(M)); cols = [m.color for m in M]
    ax[0].bar(x, [m.params or 0 for m in M], color=cols); ax[0].set_yscale("log"); ax[0].set_title("Số tham số (log)")
    for i, m in enumerate(M):
        if m.params: ax[0].text(i, m.params, fmt(m.params), ha="center", va="bottom", fontsize=8)
    ax[1].bar(x, [m.size_kb for m in M], color=cols); ax[1].set_title("Dung lượng file (KB)")
    for i, m in enumerate(M): ax[1].text(i, m.size_kb, f"{m.size_kb:,.0f}".replace(",", "."), ha="center", va="bottom", fontsize=8)
    ax[2].bar(x, [lat[m.key] for m in M], color=cols); ax[2].set_title("Độ trễ CPU (ms / khung, 2 crop mắt)")
    for i, m in enumerate(M): ax[2].text(i, lat[m.key], f"{lat[m.key]:.1f} ms", ha="center", va="bottom", fontsize=8)
    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels([n.replace(": ", ":\n") for n in names], fontsize=8)
    save(fig, out, "m1_tong_quan_model.png", dpi)


def fig_curves(M, js, out, dpi):
    have = [m for m in M if m.key in js and js[m.key].get("history")]
    if not have:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in have:
        h = pd.DataFrame(js[m.key]["history"]); e = h.epoch + 1
        ax[0].plot(e, h.loss, "-o", ms=3, color=m.color, label=m.label)
        ax[1].plot(e, h.val_f1, "-o", ms=3, color=m.color, label=m.label)
        b = h.val_f1.idxmax(); ax[1].plot(e[b], h.val_f1[b], "*", ms=13, color=m.color, mec="k")
        ax[1].annotate(f"{h.val_f1[b]:.3f}", (e[b], h.val_f1[b]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax[0].set_title("Loss huấn luyện theo epoch"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(fontsize=8)
    ax[1].set_title("F1 (lớp 'nhắm') trên tập val; ★ = epoch được chọn"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("F1 val")
    save(fig, out, "m2_duong_cong_huan_luyen.png", dpi)


def fig_confusion(M, S, R, out, dpi, thr):
    fig, axs = plt.subplots(len(S), len(M), figsize=(3.3 * len(M), 3.1 * len(S)), squeeze=False)
    for i, sn in enumerate(S):
        for j, m in enumerate(M):
            r = R[sn][m.key]["m"]; cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            ax = axs[i][j]; ax.grid(False); ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (a_, b_), v in np.ndenumerate(cm):
                pc = 100 * v / cm.sum(1)[a_]
                ax.text(b_, a_, f"{v}\n{pc:.1f}%", ha="center", va="center", color="white" if pc > 55 else "black", fontsize=9)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["mở", "nhắm"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["mở", "nhắm"])
            if i == 0: ax.set_title(m.label, fontsize=9)
            if j == 0: ax.set_ylabel(f"{sn}\nthật")
            if i == len(S) - 1: ax.set_xlabel("dự đoán")
    fig.suptitle(f"Ma trận nhầm lẫn (ngưỡng {thr}; phần trăm tính theo từng lớp thật)", y=1.0)
    save(fig, out, "m3_ma_tran_nham_lan.png", dpi)


def fig_roc(M, S, R, out, dpi, thr):
    fig, axs = plt.subplots(1, len(S), figsize=(4.4 * len(S), 4.2), squeeze=False)
    for ax, sn in zip(axs[0], S):
        for m in M:
            d = R[sn][m.key]; fpr, tpr, _ = roc_curve(d["y"], d["p"])
            ax.plot(fpr, tpr, color=m.color, label=f"{m.short} (AUC={d['m']['auc']:.3f})")
            ax.plot(d["m"]["fpr"], d["m"]["recall"], "o", color=m.color, mec="k")
        ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_title(sn); ax.set_xlabel("Tỉ lệ báo nhầm"); ax.legend(fontsize=7, loc="lower right")
        if ax is axs[0][0]: ax.set_ylabel("Tỉ lệ phát hiện đúng mắt nhắm")
    fig.suptitle(f"Đường ROC (chấm = điểm hoạt động ở ngưỡng {thr})", y=1.02)
    save(fig, out, "m4_roc.png", dpi)


def fig_scores(M, S, R, out, dpi, thr):
    fig, axs = plt.subplots(len(S), len(M), figsize=(3.6 * len(M), 2.9 * len(S)), squeeze=False)
    for i, sn in enumerate(S):
        for j, m in enumerate(M):
            d = R[sn][m.key]; ax = axs[i][j]
            for lab, col, tag in ((1, C_CLOSED, "Nhắm (thật)"), (0, C_OPEN, "Mở (thật)")):
                ax.hist(d["p"][d["y"] == lab], bins=30, range=(0, 1), alpha=.6, color=col, label=tag)
            ax.axvline(thr, ls="--", color="k", lw=1); ax.set_yscale("log")
            if i == 0: ax.set_title(m.label, fontsize=9)
            if j == 0: ax.set_ylabel(f"{sn}\nsố ảnh (log)")
            if i == len(S) - 1: ax.set_xlabel("Xác suất 'nhắm'")
            if i == 0 and j == 0: ax.legend(fontsize=7)
    fig.suptitle("Phân bố điểm số của model theo lớp thật (đường nét đứt = ngưỡng)", y=1.0)
    save(fig, out, "m5_phan_bo_diem_so.png", dpi)


def fig_bars(M, S, R, out, dpi):
    fig, axs = plt.subplots(1, len(S), figsize=(4.6 * len(S), 4), squeeze=False)
    keys = [("precision", "Precision"), ("recall", "Recall (nhắm)"), ("f1", "F1")]
    w = .8 / len(M)
    for ax, sn in zip(axs[0], S):
        for k, m in enumerate(M):
            r = R[sn][m.key]["m"]; x = np.arange(3) + k * w - .4 + w / 2; v = [r[a_] for a_, _ in keys]
            ax.bar(x, v, w, color=m.color, label=m.label)
            ax.errorbar(x[2], r["f1"], yerr=[[r["f1"] - r["f1_lo"]], [r["f1_hi"] - r["f1"]]], color="k", capsize=3, lw=1)
            for xi, vi in zip(x, v): ax.text(xi, vi + .015, f"{vi:.2f}", ha="center", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels([b for _, b in keys]); ax.set_ylim(0, 1.12); ax.set_title(sn)
        if ax is axs[0][0]: ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("So sánh chỉ số (thanh đen trên F1 = khoảng tin cậy 95% theo nhóm)", y=1.02)
    save(fig, out, "m6_so_sanh_chi_so.png", dpi)


def fig_threshold(M, S, R, out, dpi, thr):
    grid = np.linspace(0.02, 0.98, 49)
    fig, axs = plt.subplots(1, len(S), figsize=(4.6 * len(S), 4), squeeze=False)
    for ax, sn in zip(axs[0], S):
        for m in M:
            d = R[sn][m.key]; y, p = d["y"], d["p"]
            ax.plot(grid, [((p > t) & (y == 1)).sum() / max((y == 1).sum(), 1) for t in grid], "-", color=m.color, label=m.label)
            ax.plot(grid, [((p > t) & (y == 0)).sum() / max((y == 0).sum(), 1) for t in grid], "--", color=m.color)
        ax.axvline(thr, color="k", lw=1, ls=":"); ax.set_title(sn); ax.set_xlabel("Ngưỡng")
        if ax is axs[0][0]: ax.set_ylabel("Liền = recall nhắm | Nét đứt = báo nhầm"); ax.legend(fontsize=7)
    fig.suptitle("Đánh đổi theo ngưỡng (chỉ để quan sát, KHÔNG chọn ngưỡng trên tập test)", y=1.02)
    save(fig, out, "m7_nguong.png", dpi)


def fig_brightness(M, S, R, out, dpi, thr):
    names = [n for n in S if n in ("CEW", "Bus (khối test)")]
    if not names:
        return
    fig, axs = plt.subplots(2, len(names), figsize=(5.2 * len(names), 6.4), squeeze=False)
    for j, sn in enumerate(names):
        for i, (lab, title) in enumerate(((1, "Recall mắt nhắm"), (0, "Báo nhầm ở mắt mở"))):
            ax = axs[i][j]; w = .8 / len(M)
            for k, m in enumerate(M):
                d = R[sn][m.key]; sel = d["y"] == lab; q = np.quantile(d["bright"][sel], [0, .25, .5, .75, 1.0])
                v = []
                for qi in range(4):
                    s_ = sel & (d["bright"] >= q[qi]) & (d["bright"] <= q[qi + 1]); v.append((d["p"][s_] > thr).mean())
                x = np.arange(4) + k * w - .4 + w / 2; ax.bar(x, v, w, color=m.color, label=m.label)
                for xi, vi in zip(x, v): ax.text(xi, vi + .01, f"{vi:.2f}", ha="center", fontsize=7)
            ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\ntối nhất", "Q2", "Q3", "Q4\nsáng nhất"])
            ax.set_title(f"{sn}: {title}"); ax.set_ylim(0, 1.45 if lab == 1 else None)
            if j == 0 and i == 0: ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Kết quả theo mức sáng của ảnh (chia tứ phân vị trong từng lớp thật)", y=1.0)
    save(fig, out, "m8_theo_do_sang.png", dpi)


def table_image(df, out, dpi):
    cols = ["Tập đánh giá", "Model", "n", "Acc", "Precision", "Recall nhắm", "F1 [KTC 95%]", "AUC", "Báo nhầm", "Recall@báo nhầm 5%"]
    t = df[cols].copy()
    t["Tập đánh giá"] = [v if (i == 0 or v != df["Tập đánh giá"].iloc[i - 1]) else "" for i, v in enumerate(t["Tập đánh giá"])]
    fig, ax = plt.subplots(figsize=(14, 0.42 * (len(t) + 2))); ax.axis("off")
    tb = ax.table(cellText=t.values, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.5)
    tb.auto_set_column_width(list(range(len(cols))))
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#D9E2F3"); cell.set_text_props(weight="bold")
        elif c == 0 and t.iloc[r - 1, 0]:
            cell.set_text_props(weight="bold")
    save(fig, out, "bang_chi_so.png", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="key=duong_dan.onnx (vd old=models/eye_old.onnx)")
    ap.add_argument("--metrics", nargs="*", default=[], help="key=file_metrics.json để vẽ đường cong huấn luyện")
    ap.add_argument("--cew"); ap.add_argument("--split_csv"); ap.add_argument("--img_dirs", nargs="*", default=[])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default="eval/results/model_compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    M = [Model(k, p) for k, p in (s.split("=", 1) for s in a.models)]
    js = {k: json.load(open(p, encoding="utf-8")) for k, p in (s.split("=", 1) for s in a.metrics if "=" in s and s.split("=", 1)[1])}
    print("Đang nạp tập đánh giá..."); S = build_sets(a)
    if not S:
        raise SystemExit("Không có tập đánh giá nào (kiểm tra --cew, --split_csv, --img_dirs)")
    for n, s in S.items():
        print(f"  {n}: {len(s['y'])} ảnh (nhắm {int(s['y'].sum())}) | {s['note']}")

    R = {}
    for sn, s in S.items():
        R[sn] = {}
        for m in M:
            p = m.probs(s["X"]); R[sn][m.key] = dict(p=p, y=s["y"], bright=s["bright"], m=metrics(s["y"], p, s["groups"], a.thr))

    lat = {m.key: m.latency_ms() for m in M}
    # ---- bảng 1: thông tin model
    t1 = pd.DataFrame([dict(Model=m.label, **{"Kiến trúc": m.arch, "Tham số": m.params if m.params else "n/a",
                          "Dung lượng (KB)": round(m.size_kb), "Độ trễ (ms/khung)": round(lat[m.key], 2),
                          "Dữ liệu huấn luyện": TRAIN.get(m.key, "MRL + bus")}) for m in M])
    t1.to_csv(os.path.join(a.out, "bang1_thong_tin_model.csv"), index=False, encoding="utf-8-sig")
    # ---- bảng 2: chỉ số
    rows = []
    for sn in S:
        for m in M:
            r = R[sn][m.key]["m"]
            rows.append({"Tập đánh giá": sn, "Model": m.label, "n": r["n"], "Acc": f"{r['acc']:.3f}",
                         "Precision": f"{r['precision']:.3f}", "Recall nhắm": f"{r['recall']:.3f}",
                         "F1 [KTC 95%]": f"{r['f1']:.3f} [{r['f1_lo']:.3f}-{r['f1_hi']:.3f}]",
                         "AUC": f"{r['auc']:.3f}", "Báo nhầm": f"{r['fpr']:.3f}", "Recall@báo nhầm 5%": f"{r['r5']:.3f}",
                         "Bỏ sót (FN)": r["fn"], "Báo nhầm (FP)": r["fp"]})
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(a.out, "bang2_chi_so.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(a.out, "bang_chi_so.md"), "w", encoding="utf-8") as f:
        try:
            md1, md2 = t1.to_markdown(index=False), t2.to_markdown(index=False)
        except ImportError:                       # chưa cài tabulate -> ghi dạng văn bản thường
            md1, md2 = t1.to_string(index=False), t2.to_string(index=False)
        f.write(f"Ngưỡng {a.thr}\n\n" + md1 + "\n\n" + md2 + "\n\nGhi chú tập đánh giá:\n")
        for n, s in S.items(): f.write(f"- {n}: {s['note']}\n")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    print("\n" + t1.to_string(index=False)); print("\n" + t2.drop(columns=["Bỏ sót (FN)", "Báo nhầm (FP)"]).to_string(index=False))

    # ---- đối chiếu với JSON lúc train (nếu có đủ tập test của CSV)
    if "Test CSV (MRL+bus)" in S:
        for m in M:
            if m.key in js:
                r = R["Test CSV (MRL+bus)"][m.key]["m"]; cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
                same = cm == js[m.key]["cm"]
                print(f"[đối chiếu] {m.label}: ma trận nhầm lẫn {'KHỚP' if same else 'KHÁC'} với {m.key} metrics.json lúc train "
                      f"({cm} so với {js[m.key]['cm']})")

    print("\nĐang vẽ...")
    fig_overview(M, lat, a.out, a.dpi); fig_curves(M, js, a.out, a.dpi)
    fig_confusion(M, S, R, a.out, a.dpi, a.thr); fig_roc(M, S, R, a.out, a.dpi, a.thr)
    fig_scores(M, S, R, a.out, a.dpi, a.thr); fig_bars(M, S, R, a.out, a.dpi)
    fig_threshold(M, S, R, a.out, a.dpi, a.thr); fig_brightness(M, S, R, a.out, a.dpi, a.thr)
    table_image(t2, a.out, a.dpi)
    print("Xong. Kết quả trong", a.out)


if __name__ == "__main__":
    main()
