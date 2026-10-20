// Nhánh DL trên trình duyệt: ONNX Runtime Web chạy model mắt (cùng model với bản Python).
// Cắt ảnh mắt GIỐNG HỆT src/detector.py::_square_crop (pad=1.0, 64x64, xám, /255).
import { L_EYE, R_EYE } from './features.js';

export const ORT_VERSION = '1.18.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const S = 64;

// ---------- Hàm thuần (test được bằng Node) ----------
// lm: landmark chuẩn hoá 0..1 -> toạ độ pixel; trả {x0,y0,x1,y1} hoặc null nếu crop tràn ảnh / quá nhỏ
export function cropRect(lm, idx, w, h, pad = 1.0) {
  let xa = Infinity, ya = Infinity, xb = -Infinity, yb = -Infinity;
  for (const i of idx) {
    const x = lm[i].x * w, y = lm[i].y * h;
    if (x < xa) xa = x; if (x > xb) xb = x; if (y < ya) ya = y; if (y > yb) yb = y;
  }
  const cx = (xa + xb) / 2, cy = (ya + yb) / 2, s = Math.max(xb - xa, yb - ya) * (1 + pad);
  const x0 = Math.trunc(cx - s / 2), x1 = Math.trunc(cx + s / 2), y0 = Math.trunc(cy - s / 2), y1 = Math.trunc(cy + s / 2);
  if (x0 < 0 || y0 < 0 || x1 > w || y1 > h || x1 - x0 < 8 || y1 - y0 < 8) return null;
  return { x0, y0, x1, y1 };
}

// RGBA (ImageData.data) -> xám 0..1. Dùng đúng công thức số nguyên của cv2.cvtColor(BGR2GRAY): (4899R+9617G+1868B+8192)>>14
export function rgbaToGray01(rgba, out, offset = 0) {
  for (let i = 0, j = offset; i < rgba.length; i += 4, j++)
    out[j] = ((4899 * rgba[i] + 9617 * rgba[i + 1] + 1868 * rgba[i + 2] + 8192) >> 14) / 255;
  return out;
}

// logits [n,2] (phẳng) -> xác suất lớp 1 (closed) của từng dòng
export function softmaxClosed(logits, n) {
  const p = [];
  for (let r = 0; r < n; r++) {
    const a = logits[2 * r], b = logits[2 * r + 1], m = Math.max(a, b), ea = Math.exp(a - m), eb = Math.exp(b - m);
    p.push(eb / (ea + eb));
  }
  return p;
}

// ---------- Nạp ONNX Runtime Web (bản 1.18 chỉ có UMD -> chèn thẻ <script>, dùng window.ort) ----------
function loadOrtScript() {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  return new Promise((ok, fail) => {
    const s = document.createElement('script');
    s.src = ORT_BASE + 'ort.min.js';
    s.onload = () => (globalThis.ort ? ok(globalThis.ort) : fail(new Error('ort không có trên window')));
    s.onerror = () => fail(new Error('Không tải được ONNX Runtime Web'));
    document.head.appendChild(s);
  });
}

export class EyeModel {
  constructor(ort, session) {
    this.ort = ort; this.session = session;
    this.inName = session.inputNames[0]; this.outName = session.outputNames[0];
    this.cv = null; this.cx = null;
  }

  // ort: truyền vào khi test (Node); trên trình duyệt để trống -> tự nạp từ CDN
  static async create(modelUrl, ort = null) {
    ort = ort || await loadOrtScript();
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = 1;   // Netlify không bật cross-origin isolation -> không có đa luồng
    const buf = await (await fetch(modelUrl)).arrayBuffer();
    const session = await ort.InferenceSession.create(new Uint8Array(buf), { executionProviders: ['wasm'] });
    return new EyeModel(ort, session);
  }

  _canvas() {
    if (!this.cv) {
      this.cv = document.createElement('canvas'); this.cv.width = this.cv.height = S;
      this.cx = this.cv.getContext('2d', { willReadFrequently: true });
      this.cx.imageSmoothingEnabled = true; this.cx.imageSmoothingQuality = 'high';
    }
    return this.cx;
  }

  // Trả xác suất "nhắm" = min(hai mắt) (nháy một mắt không tính). NaN nếu không cắt được mắt nào.
  async predict(video, lm) {
    const w = video.videoWidth, h = video.videoHeight, cx = this._canvas();
    const rects = [L_EYE, R_EYE].map(ix => cropRect(lm, ix, w, h)).filter(Boolean);
    if (!rects.length) return NaN;
    const data = new Float32Array(rects.length * S * S);
    rects.forEach((r, k) => {
      cx.drawImage(video, r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0, 0, 0, S, S);
      rgbaToGray01(cx.getImageData(0, 0, S, S).data, data, k * S * S);
    });
    const out = await this.session.run({ [this.inName]: new this.ort.Tensor('float32', data, [rects.length, 1, S, S]) });
    return Math.min(...softmaxClosed(out[this.outName].data, rects.length));
  }
}
