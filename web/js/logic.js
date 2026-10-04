// Bản port của src/alert.py (Python). Thời gian t (giây) truyền từ ngoài vào -> test được bằng dữ liệu giả.
export const DEFAULTS = {
  earFactor: 0.75, voteN: 5, marThr: 0.6, yawnHoldS: 1.0, nodDeg: 20, yawDeg: 35,
  windowS: 30, minWindowS: 5, closedDurS: 1.5, perclosThr: 0.30, perclosYawnThr: 0.15,
  nodRatioThr: 0.30, yawnCountDrowsy: 3, yawnCountDistracted: 2, drowsyHoldS: 2.0,
  nofaceS: 2.0, calibS: 10,
};
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

export class Calibrator {
  constructor(sec) { this.sec = sec; this.t0 = null; this.e = []; this.p = []; this.y = []; this.done = false; this.earOpen = 0; this.pitch0 = 0; this.yaw0 = 0; }
  progress(t) { return this.t0 === null ? 0 : Math.min(1, (t - this.t0) / this.sec); }
  update(t, ear, pitch, yaw) {
    if (this.t0 === null) this.t0 = t;
    this.e.push(ear); this.p.push(pitch); this.y.push(yaw);
    if (t - this.t0 >= this.sec && this.e.length >= 10) {
      this.earOpen = median(this.e); this.pitch0 = median(this.p); this.yaw0 = median(this.y); this.done = true;
    }
  }
}

export class DrowsinessLogic {
  constructor(cfg) { this.cfg = cfg; this.buf = []; this.closedSince = null; this.yawns = []; this.yStart = null; this.yCounted = false; this.state = 'NORMAL'; this.drowsyUntil = -1e9; }
  ratios() {
    const b = this.buf; let tot = 0, c = 0, n = 0;
    for (let i = 0; i < b.length - 1; i++) {
      const dt = Math.min(b[i + 1].t - b[i].t, 0.5); tot += dt;
      if (b[i].closed) c += dt; if (b[i].nod) n += dt;
    }
    return tot > 0 ? { perclos: c / tot, nodRatio: n / tot, covered: tot } : { perclos: 0, nodRatio: 0, covered: 0 };
  }
  update(t, closed, yawnRaw, pitchDev, yawDev) {
    const c = this.cfg;
    this.buf.push({ t, closed, nod: Math.abs(pitchDev) > c.nodDeg });
    while (this.buf.length && t - this.buf[0].t > c.windowS) this.buf.shift();
    if (closed) { if (this.closedSince === null) this.closedSince = t; } else this.closedSince = null;
    const closedDur = this.closedSince === null ? 0 : t - this.closedSince;
    if (yawnRaw) {
      if (this.yStart === null) this.yStart = t;
      if (!this.yCounted && t - this.yStart >= c.yawnHoldS) { this.yawns.push(t); this.yCounted = true; }
    } else { this.yStart = null; this.yCounted = false; }
    while (this.yawns.length && t - this.yawns[0] > 300) this.yawns.shift();
    const { perclos, nodRatio, covered } = this.ratios(), ready = covered >= c.minWindowS, ny = this.yawns.length;
    // chỉ kích hoạt khi mắt ĐANG nhắm -> mở mắt là còi tắt sau drowsyHoldS
    const drowsy = closedDur > c.closedDurS || (closed && ready && perclos > c.perclosThr) ||
      (closed && ready && ny >= c.yawnCountDrowsy && perclos > c.perclosYawnThr);
    if (drowsy) this.drowsyUntil = t + c.drowsyHoldS;
    if (drowsy || t < this.drowsyUntil) this.state = 'DROWSY';
    else if (Math.abs(yawDev) > c.yawDeg || (ready && nodRatio > c.nodRatioThr) || ny >= c.yawnCountDistracted) this.state = 'DISTRACTED';
    else this.state = 'NORMAL';
    return { state: this.state, perclos, closedDur, yawns: ny, nodRatio };
  }
}

export class Decider {
  constructor(cfg = {}) { this.cfg = { ...DEFAULTS, ...cfg }; this.reset(); }
  reset() { this.cal = new Calibrator(this.cfg.calibS); this.logic = new DrowsinessLogic(this.cfg); this.votes = []; this.nofaceSince = null; this.state = 'CALIBRATING'; this.closed = false; }
  acknowledge() { this.logic = new DrowsinessLogic(this.cfg); this.votes = []; this.closed = false; this.state = this.cal.done ? 'NORMAL' : 'CALIBRATING'; }
  info(f, o = {}) {
    return { state: this.state, calibrating: !this.cal.done, calibProgress: 0, noface: !f.face, ear: f.ear, mar: f.mar, closed: this.closed,
      perclos: 0, closedDur: 0, yawns: 0, nodRatio: 0, pitchDev: 0, yawDev: 0, earThr: NaN, ...o };
  }
  update(t, f) {
    const c = this.cfg;
    if (!f.face) {
      if (this.nofaceSince === null) this.nofaceSince = t;
      if (this.cal.done && t - this.nofaceSince > c.nofaceS && this.state !== 'DROWSY') this.state = 'DISTRACTED';
      return this.info(f, { calibProgress: this.cal.progress(t) });
    }
    this.nofaceSince = null;
    if (!this.cal.done) {
      this.cal.update(t, f.ear, f.pitch, f.yaw);
      this.state = this.cal.done ? 'NORMAL' : 'CALIBRATING';
      return this.info(f, { calibProgress: this.cal.progress(t) });
    }
    this.votes.push(f.ear < c.earFactor * this.cal.earOpen);
    if (this.votes.length > c.voteN) this.votes.shift();
    this.closed = this.votes.filter(Boolean).length > this.votes.length / 2;
    const pd = f.pitch - this.cal.pitch0, yd = f.yaw - this.cal.yaw0;
    const r = this.logic.update(t, this.closed, f.mar > c.marThr, pd, yd);
    this.state = r.state;
    return this.info(f, { ...r, calibProgress: 1, pitchDev: pd, yawDev: yd, earThr: c.earFactor * this.cal.earOpen });
  }
}
