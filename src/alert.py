"""Logic cảnh báo: hiệu chuẩn, cửa sổ trượt (PERCLOS), máy trạng thái, nhật ký.
Mọi hàm nhận thời gian t từ bên ngoài -> chạy được cả realtime lẫn replay offline."""
import csv
import os
import datetime
from collections import deque
import numpy as np
from .config import Config

STATES = ["NORMAL", "DISTRACTED", "DROWSY"]


class Calibrator:
    def __init__(self, seconds=10.0):
        self.seconds, self.t0 = seconds, None
        self.ears, self.pitches, self.yaws = [], [], []
        self.done, self.ear_open, self.pitch0, self.yaw0 = False, None, 0.0, 0.0

    def progress(self, t):
        return 0.0 if self.t0 is None else min(1.0, (t - self.t0) / self.seconds)

    def update(self, t, e, pitch, yaw):
        if self.t0 is None:
            self.t0 = t
        self.ears.append(e); self.pitches.append(pitch); self.yaws.append(yaw)
        if t - self.t0 >= self.seconds and len(self.ears) >= 10:
            self.ear_open = float(np.median(self.ears))
            self.pitch0 = float(np.median(self.pitches))
            self.yaw0 = float(np.median(self.yaws))
            self.done = True


class DrowsinessLogic:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.buf = deque()                 # (t, closed, nod)
        self.closed_since = None
        self.yawn_times = deque()
        self._y_since, self._y_counted = None, False
        self.state, self._drowsy_until = "NORMAL", -1e9

    def _ratios(self):
        b, tot, c, n = self.buf, 0.0, 0.0, 0.0
        for i in range(len(b) - 1):
            dt = min(b[i + 1][0] - b[i][0], 0.5)
            tot += dt
            c += dt * b[i][1]
            n += dt * b[i][2]
        return (c / tot, n / tot, tot) if tot > 0 else (0.0, 0.0, 0.0)

    def update(self, t, closed, yawn_raw, pitch_dev, yaw_dev):
        c = self.cfg
        nod = abs(pitch_dev) > c.nod_deg
        self.buf.append((t, bool(closed), nod))
        while self.buf and t - self.buf[0][0] > c.window_s:
            self.buf.popleft()

        if closed:
            if self.closed_since is None: self.closed_since = t
        else:
            self.closed_since = None
        closed_dur = (t - self.closed_since) if self.closed_since is not None else 0.0

        if yawn_raw:
            if self._y_since is None: self._y_since = t
            if not self._y_counted and t - self._y_since >= c.yawn_hold_s:
                self.yawn_times.append(t); self._y_counted = True
        else:
            self._y_since, self._y_counted = None, False
        while self.yawn_times and t - self.yawn_times[0] > 300:
            self.yawn_times.popleft()

        perclos, nod_ratio, covered = self._ratios()
        ready = covered >= c.min_window_s
        ny = len(self.yawn_times)

        drowsy = closed_dur > c.closed_dur_s or (ready and perclos > c.perclos_thr) \
            or (ready and ny >= c.yawn_count_drowsy and perclos > c.perclos_yawn_thr)
        if drowsy:
            self._drowsy_until = t + c.drowsy_hold_s
        if drowsy or t < self._drowsy_until:
            self.state = "DROWSY"
        elif abs(yaw_dev) > c.yaw_deg or (ready and nod_ratio > c.nod_ratio_thr) \
                or ny >= c.yawn_count_distracted:
            self.state = "DISTRACTED"
        else:
            self.state = "NORMAL"
        return dict(state=self.state, perclos=perclos, closed_dur=closed_dur,
                    yawns=ny, nod_ratio=nod_ratio)


class Decider:
    """Từ đặc trưng 1 frame -> trạng thái. mode: landmark | dl | fusion"""
    def __init__(self, mode="landmark", cfg: Config = None):
        self.mode, self.cfg = mode, cfg or Config()
        self.reset()

    def reset(self):
        self.cal = Calibrator(self.cfg.calib_s)
        self.logic = DrowsinessLogic(self.cfg)
        self.votes = deque(maxlen=self.cfg.vote_n)
        self.noface_since, self.state, self.closed = None, "CALIBRATING", False

    def _closed_raw(self, f):
        c, eo = self.cfg, self.cal.ear_open
        lm = f["ear"] < c.ear_factor * eo
        p = f["p_eye"]
        if self.mode == "landmark" or np.isnan(p):
            return lm
        if self.mode == "dl":
            return p > c.p_closed_thr
        s_lm = min(max((1 - f["ear"] / eo) / (1 - c.ear_factor), 0.0), 1.0)
        return (1 - c.w_dl) * s_lm + c.w_dl * p > 0.5

    def _yawn_raw(self, f):
        py, m = f["p_yawn"], f["mar"]
        if self.mode == "landmark" or np.isnan(py):
            return m > self.cfg.mar_thr
        if self.mode == "dl":
            return py > 0.5
        return m > self.cfg.mar_thr and py > 0.3

    def _info(self, f, **kw):
        d = dict(state=self.state, calibrating=not self.cal.done, calib_progress=0.0,
                 noface=not f["face"], ear=f["ear"], mar=f["mar"], p_eye=f["p_eye"], p_yawn=f["p_yawn"],
                 closed=self.closed, perclos=0.0, closed_dur=0.0, yawns=0, nod_ratio=0.0,
                 pitch_dev=0.0, yaw_dev=0.0, ear_thr=float("nan"))
        d.update(kw)
        return d

    def update(self, t, f):
        if not f["face"]:
            if self.noface_since is None: self.noface_since = t
            if self.cal.done and t - self.noface_since > self.cfg.noface_s and self.state != "DROWSY":
                self.state = "DISTRACTED"
            return self._info(f, calib_progress=self.cal.progress(t))
        self.noface_since = None
        if not self.cal.done:
            self.cal.update(t, f["ear"], f["pitch"], f["yaw"])
            self.state = "CALIBRATING" if not self.cal.done else "NORMAL"
            return self._info(f, calib_progress=self.cal.progress(t))
        self.votes.append(self._closed_raw(f))
        self.closed = sum(self.votes) > len(self.votes) / 2
        r = self.logic.update(t, self.closed, self._yawn_raw(f),
                              f["pitch"] - self.cal.pitch0, f["yaw"] - self.cal.yaw0)
        self.state = r.pop("state")
        return self._info(f, calib_progress=1.0, pitch_dev=f["pitch"] - self.cal.pitch0,
                          yaw_dev=f["yaw"] - self.cal.yaw0,
                          ear_thr=self.cfg.ear_factor * self.cal.ear_open, **r)


class EventLogger:
    """Ghi CSV khi trạng thái thay đổi; giữ vài dòng gần nhất để hiển thị."""
    def __init__(self, path="logs/alerts.csv"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["timestamp", "state", "ear", "mar", "perclos", "pitch_dev"])
        self.last, self.recent = None, deque(maxlen=5)

    def update(self, info):
        s = info["state"]
        if s != self.last and s != "CALIBRATING":
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), s,
                             round(info["ear"], 3), round(info["mar"], 3),
                             round(info["perclos"], 3), round(info["pitch_dev"], 1)])
            self.f.flush()
            self.recent.append(f"{ts}  {s}")
        self.last = s

    def close(self):
        self.f.close()
