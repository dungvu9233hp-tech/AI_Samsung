// EAR / MAR / góc đầu. Landmark MediaPipe là toạ độ chuẩn hoá 0..1 -> đổi sang pixel để tỉ lệ đúng.
export const L_EYE = [362, 385, 387, 263, 373, 380];
export const R_EYE = [33, 160, 158, 133, 153, 144];
export const MOUTH = [78, 13, 14, 308];
const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const px = (lm, i, w, h) => ({ x: lm[i].x * w, y: lm[i].y * h });

export function ear(lm, idx, w, h) {
  const p = idx.map(i => px(lm, i, w, h));
  return (d(p[1], p[5]) + d(p[2], p[4])) / (2 * d(p[0], p[3]) + 1e-6);
}
export function mar(lm, w, h) {
  const [l, t, b, r] = MOUTH.map(i => px(lm, i, w, h));
  return d(t, b) / (d(l, r) + 1e-6);
}
// m: ma trận 4x4 cột-trước (facialTransformationMatrixes[0].data). Công thức không phụ thuộc scale.
export function poseFromMatrix(m) {
  const R = (r, c) => m[c * 4 + r];
  const deg = 180 / Math.PI;
  return {
    pitch: Math.atan2(R(2, 1), R(2, 2)) * deg,
    yaw: Math.atan2(-R(2, 0), Math.hypot(R(0, 0), R(1, 0))) * deg,
  };
}
