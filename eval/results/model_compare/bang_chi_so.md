Ngưỡng 0.5

                     Model         Kiến trúc Tham số  Dung lượng (KB)  Độ trễ (ms/khung) Dữ liệu huấn luyện
        Cũ: SmallCNN (MRL)          SmallCNN     n/a              246               0.42                MRL
   Mới: SmallCNN (MRL+bus)          SmallCNN     n/a              246               0.44          MRL + bus
Mới: MobileNetV3 (MRL+bus) MobileNetV3-small     n/a             5949               1.80          MRL + bus

Tập đánh giá                      Model    n   Acc Precision Recall nhắm        F1 [KTC 95%]   AUC Báo nhầm Recall@báo nhầm 5%  Bỏ sót (FN)  Báo nhầm (FP)
         CEW         Cũ: SmallCNN (MRL) 4846 0.863     0.945       0.767 0.847 [0.834-0.860] 0.955    0.043              0.779          555            107
         CEW    Mới: SmallCNN (MRL+bus) 4846 0.888     0.943       0.821 0.878 [0.865-0.890] 0.972    0.048              0.824          426            118
         CEW Mới: MobileNetV3 (MRL+bus) 4846 0.924     0.908       0.943 0.925 [0.916-0.934] 0.979    0.093              0.889          137            229
  MRL (test)         Cũ: SmallCNN (MRL) 2380 0.914     0.882       0.997 0.936 [0.878-0.996] 0.985    0.227              0.961            5            200
  MRL (test)    Mới: SmallCNN (MRL+bus) 2380 0.994     0.997       0.993 0.995 [0.990-0.999] 0.999    0.006              0.998           10              5
  MRL (test) Mới: MobileNetV3 (MRL+bus) 2380 0.994     0.999       0.991 0.995 [0.989-0.999] 0.999    0.001              0.997           14              1

Ghi chú tập đánh giá:
- CEW: test ngoài
- MRL (test): held-out theo người
