# Drowsy Guard – cảnh báo ngủ gật realtime (MediaPipe + CNN/ONNX)

## 1. Cài đặt (Python 3.10–3.12)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.sound                # sinh assets/alarm.wav
```

## 2. Chạy ngay (chưa cần model – chỉ nhánh landmark)
```bash
python -m src.app --mode landmark
```
Ngồi tự nhiên nhìn camera 10 s để hiệu chuẩn. Phím: `r` hiệu chuẩn lại · `m` đổi mode · `q` thoát.
Có file `models/eye.onnx` (và tuỳ chọn `models/yawn.onnx`) thì dùng `--mode dl` hoặc `--mode fusion`.
Thử trên video: `python -m src.app --video data/own/P01_xxx.mp4`.
Nhật ký cảnh báo: `logs/alerts.csv`.

## 3. Quy trình dữ liệu → huấn luyện → đánh giá
```bash
# a) Quay dữ liệu (mỗi clip mở đầu >= 5 s bình thường), có phiếu đồng ý của người tham gia
python tools/record_clips.py --subject P01 --scenario microsleep --tags glasses,dark
# b) Gán nhãn theo đoạn (phím 1/2/3)
python tools/label_tool.py data/own/P01_microsleep_glasses_dark.mp4
# c) Trích crop từ video tự quay (nhãn yếu) rồi chuẩn bị split THEO SUBJECT
python training/prepare_data.py own --videos data/own --out_dir data/processed/own
python training/prepare_data.py mrl --dir data/raw/mrl --own data/processed/own/eye_own.csv \
       --holdout P07,P08 --out data/processed/eye_split.csv
# d) Huấn luyện: mở training/train_colab.ipynb trên Colab (GPU), tải models/*.onnx về thư mục models/
# e) Quét ngưỡng trên VAL, rồi đánh giá TEST một lần
python eval/evaluate.py --sweep --val_subjects P05,P06
python eval/evaluate.py --test_subjects P07,P08
```
Kết quả: `eval/results/results.csv`, `per_condition.csv`, `confusion_*.png`, `tradeoff.png`, `errors/`.

## 4. Lưu ý
- Subject test (`--holdout`) KHÔNG được xuất hiện trong train/val của CNN.
- Dữ liệu tự quay là giả lập (nhắm mắt có chủ đích), chưa thử trên xe thật.
- Không push dữ liệu người thật lên Git công khai.
