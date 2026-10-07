"""Cắt crop mắt (64x64 xám) từ dataset COCO 'drowsy' (Roboflow, camera trong xe) -> CSV cho prepare_data.py.

  python training/prepare_bus.py --dir data/raw/bus --out_dir data/processed/bus
  python training/prepare_data.py mrl --dir data/raw/mrl --own data/processed/bus/eye_bus.csv \
         --out data/processed/eye_split.csv --cap 300

- GỘP cả 3 split của Roboflow rồi chia lại: split gốc rò rỉ (khung liền kề nằm ở split khác nhau).
- 'subject' = khối khung liên tiếp (mặc định 100 khung) -> prepare_data chia theo khối, giảm (không loại
  hẳn) rò rỉ giữa các khung gần nhau. Dataset không có mã tài xế nên không thể làm tốt hơn.
- Crop: hình vuông tâm ở tâm hộp, cạnh = k * max(w,h). k=1.75 ~ khớp crop_eye(pad=1.0) lúc chạy thật
  (đã đo: độ rộng mắt theo landmark ~0.87 * độ rộng hộp nhãn).
- Chỉ lấy MẮT. open_mouth quá ít (~220 hộp) và chưa chắc là ngáp nên bỏ qua.
Nhãn: 1 = closed_eye, 0 = open_eye.
"""
import argparse, glob, json, os, re
import cv2
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="thư mục giải nén chứa train/ valid/ test/")
    ap.add_argument("--out_dir", default="data/processed/bus")
    ap.add_argument("--k", type=float, default=1.75)
    ap.add_argument("--block", type=int, default=100, help="số khung gốc mỗi nhóm 'subject'")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out_dir, "eye"), exist_ok=True)
    rows = []
    for sp in ("train", "valid", "test"):
        jf = os.path.join(a.dir, sp, "_annotations.coco.json")
        if not os.path.exists(jf):
            print("thiếu", jf); continue
        d = json.load(open(jf, encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {i["id"]: i for i in d["images"]}
        by_img = {}
        for an in d["annotations"]:
            if cats[an["category_id"]] in ("closed_eye", "open_eye"):
                by_img.setdefault(an["image_id"], []).append(an)
        for iid, ans in by_img.items():
            im = imgs[iid]
            g = cv2.imread(os.path.join(a.dir, sp, im["file_name"]), 0)
            if g is None: continue
            m = re.match(r"(\d+)", im.get("extra", {}).get("name", "") or im["file_name"])
            fno = int(m.group(1)) if m else 0
            subj = f"bus_b{fno // a.block}"
            for j, an in enumerate(ans):
                x, y, w, h = an["bbox"]
                s = a.k * max(w, h); cx, cy = x + w / 2, y + h / 2
                x0, x1, y0, y1 = int(cx - s / 2), int(cx + s / 2), int(cy - s / 2), int(cy + s / 2)
                if x0 < 0 or y0 < 0 or x1 > g.shape[1] or y1 > g.shape[0] or x1 - x0 < 8:
                    continue
                c = cv2.resize(g[y0:y1, x0:x1], (64, 64), interpolation=cv2.INTER_AREA)
                p = os.path.abspath(os.path.join(a.out_dir, "eye", f"bus_{fno}_{iid}_{j}.png"))
                cv2.imwrite(p, c)
                rows.append(dict(path=p, label=1 if cats[an["category_id"]] == "closed_eye" else 0, subject=subj))
    df = pd.DataFrame(rows)
    out = os.path.join(a.out_dir, "eye_bus.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} crop | nhắm {int(df.label.sum())} | mở {int((1 - df.label).sum())} | {df.subject.nunique()} nhóm -> {out}")


if __name__ == "__main__":
    main()
