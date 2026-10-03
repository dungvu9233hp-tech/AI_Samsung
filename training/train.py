"""Huấn luyện bộ phân loại 64x64 xám (eye: closed=1 | yawn: yawn=1) và export ONNX.
  python training/train.py --csv data/processed/eye_split.csv --out models/eye.onnx --epochs 15
  python training/train.py --csv ... --arch mobilenet --out models/eye_mbv3.onnx
"""
import argparse, json, os, random
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = 64


def load(paths):
    out = np.zeros((len(paths), S, S), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(p, 0)
        out[i] = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA) if im is not None else 0
    return out


def augment(img, rng):
    M = cv2.getRotationMatrix2D((S / 2, S / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    M[:, 2] += rng.uniform(-3, 3, 2)
    img = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5: img = cv2.flip(img, 1)
    x = img.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-40, 40)   # độ sáng/tương phản
    x = np.clip(x, 0, 255)
    if rng.random() < 0.3: x = 255 * (x / 255) ** rng.uniform(0.6, 1.6)           # gamma (thiếu sáng)
    if rng.random() < 0.3: x = cv2.GaussianBlur(x, (3, 3), 0)
    if rng.random() < 0.3: x = np.clip(x + rng.normal(0, 8, x.shape), 0, 255)
    return x.astype(np.uint8)


class DS(Dataset):
    def __init__(self, X, y, train): self.X, self.y, self.train = X, y, train
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        im = self.X[i]
        if self.train: im = augment(im, np.random.default_rng())
        return torch.from_numpy(im[None].astype(np.float32) / 255.0), int(self.y[i])


def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 4 * 4, 2))
    def forward(self, x): return self.c(self.f(x))


class MobileNetWrap(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision
        try: m = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
        except Exception: m = torchvision.models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
        self.m = m
    def forward(self, x):
        x = F.interpolate(x, size=96, mode="bilinear", align_corners=False)
        return self.m(((x - 0.5) / 0.25).repeat(1, 3, 1, 1))


@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(dev)).softmax(1)[:, 1].cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="small", choices=["small", "mobilenet"])
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos_name", default="closed")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(a.csv); D = {}
    for sp in ("train", "val", "test"):
        d = df[(df.split == sp) & df.path.map(os.path.exists)]; print(sp, len(d), d.label.value_counts().to_dict())
        D[sp] = (load(d.path.tolist()), d.label.values)
    tr = DataLoader(DS(*D["train"], True), a.bs, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(DS(*D["val"], False), 256); te = DataLoader(DS(*D["test"], False), 256)

    model = (SmallCNN() if a.arch == "small" else MobileNetWrap()).to(dev)
    cnt = np.bincount(D["train"][1], minlength=2); w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, hist = -1, []
    for ep in range(a.epochs):
        model.train(); tl = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); l = lossf(model(x), y); l.backward(); opt.step(); tl += l.item()
        sched.step()
        p, y = predict(model, va, dev); f1 = f1_score(y, p > 0.5, zero_division=0)
        acc = float(((p > 0.5) == y).mean()); hist.append(dict(epoch=ep, loss=tl / len(tr), val_acc=acc, val_f1=f1))
        print(f"ep{ep:02d} loss {tl/len(tr):.4f} val_acc {acc:.4f} val_f1({a.pos_name}) {f1:.4f}")
        if f1 > best: best = f1; torch.save(model.state_dict(), "best.pt")

    model.load_state_dict(torch.load("best.pt")); p, y = predict(model, te, dev); pred = p > 0.5
    rep = classification_report(y, pred, target_names=["neg", a.pos_name], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred)
    print(classification_report(y, pred, target_names=["neg", a.pos_name], zero_division=0)); print(cm)
    base = os.path.splitext(a.out)[0]; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(report=rep, cm=cm.tolist(), history=hist, args=vars(a)), open(base + "_metrics.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(3.6, 3.4)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1], ["neg", a.pos_name]); ax.set_yticks([0, 1], ["neg", a.pos_name])
    ax.set_xlabel("pred"); ax.set_ylabel("true"); fig.tight_layout(); fig.savefig(base + "_cm.png", dpi=150)

    model.eval().cpu(); dummy = torch.randn(1, 1, S, S)
    kw = dict(input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "b"}, "logits": {0: "b"}}, opset_version=17)
    try: torch.onnx.export(model, dummy, a.out, dynamo=False, **kw)
    except TypeError: torch.onnx.export(model, dummy, a.out, **kw)
    try:   # kiểm tra ONNX khớp PyTorch
        import onnxruntime as ort
        s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        o = s.run(None, {"x": dummy.numpy()})[0]; print("ONNX max diff:", float(np.abs(o - model(dummy).detach().numpy()).max()))
    except Exception as e: print("Bỏ qua kiểm tra ONNX:", e)
    print("Đã lưu", a.out)


if __name__ == "__main__":
    main()