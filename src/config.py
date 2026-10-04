from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- mắt ---
    ear_factor: float = 0.75       # nhắm nếu EAR < ear_factor * EAR_mở (hiệu chuẩn cá nhân)
    p_closed_thr: float = 0.5      # ngưỡng xác suất 'nhắm' của CNN (mode dl)
    w_dl: float = 0.5              # trọng số nhánh DL trong mode fusion
    vote_n: int = 5                # majority vote n frame gần nhất
    # --- miệng / ngáp ---
    mar_thr: float = 0.6
    yawn_hold_s: float = 1.0       # miệng mở liên tục >= giây này mới tính 1 lần ngáp
    # --- đầu ---
    nod_deg: float = 20.0          # lệch pitch so với tư thế chuẩn
    yaw_deg: float = 35.0
    # --- cửa sổ & luật trạng thái ---
    window_s: float = 30.0
    min_window_s: float = 5.0      # chưa đủ cửa sổ thì chưa dùng PERCLOS
    closed_dur_s: float = 1.5      # nhắm liên tục -> DROWSY
    perclos_thr: float = 0.30
    perclos_yawn_thr: float = 0.15
    nod_ratio_thr: float = 0.30
    yawn_count_drowsy: int = 3     # trong 5 phút
    yawn_count_distracted: int = 2
    drowsy_hold_s: float = 2.0     # giữ trạng thái DROWSY tối thiểu (chống nhấp nháy)
    noface_s: float = 2.0          # mất mặt quá lâu -> DISTRACTED
    calib_s: float = 10.0

    def to_dict(self):
        return asdict(self)
