import assert from 'node:assert/strict';
import { Decider } from '../js/logic.js';
import { poseFromMatrix } from '../js/features.js';

const F = (o = {}) => ({ face: true, ear: 0.30, mar: 0.2, pitch: 0, yaw: 0, ...o });
function sim(script, fps = 20) {
  const d = new Decider(), out = []; 
  for (const [dur, o] of script) for (let i = 0; i < dur * fps; i++) { const t = out.length / fps; out.push([t, d.update(t, F(o)).state]); }
  return { d, out, at: (a, b) => new Set(out.filter(([t]) => t >= a && t < b).map(x => x[1])) };
}
// 1. chớp mắt thường không báo
let s = sim([[15, {}], ...Array(8).fill([[0.2, { ear: 0.08 }], [1.0, {}]]).flat()]);
assert.ok(![...s.at(0, 100)].includes('DROWSY'), 'chớp mắt bị báo nhầm');
// 2. nhắm 3 s -> DROWSY; mở mắt -> hết báo
s = sim([[15, {}], [3, { ear: 0.08 }], [6, {}]]);
assert.ok(s.at(16.6, 18).has('DROWSY') && s.at(21, 24).size === 1 && s.at(21, 24).has('NORMAL'), 'DROWSY/recover');
// 3. quay đầu -> DISTRACTED ; tắt chủ động
s = sim([[15, {}], [3, { yaw: 60 }]]);
assert.equal(s.out.at(-1)[1], 'DISTRACTED');
s.d.acknowledge(); assert.equal(s.d.state, 'NORMAL');
// 4. mất mặt lâu -> DISTRACTED
s = sim([[12, {}], [4, { face: false }]]); assert.equal(s.out.at(-1)[1], 'DISTRACTED');
// 5. góc đầu từ ma trận (có scale)
const rad = Math.PI / 180, mk = (p, y, sc) => { const cp = Math.cos(p * rad), sp = Math.sin(p * rad), cy = Math.cos(y * rad), sy = Math.sin(y * rad);
  const R = [[cy, sy * sp, sy * cp], [0, cp, -sp], [-sy, cy * sp, cy * cp]]; const m = new Array(16).fill(0); m[15] = 1;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[c * 4 + r] = R[r][c] * sc; return m; };
for (const [p, y] of [[0, 0], [20, 0], [-25, 10], [5, 35], [-10, -40]]) {
  const o = poseFromMatrix(mk(p, y, 2.5)); assert.ok(Math.abs(o.pitch - p) < 0.01 && Math.abs(o.yaw - y) < 0.01, `pose ${p},${y} -> ${o.pitch},${o.yaw}`);
}
// 6. nheo mắt (EAR còn 70% mắt mở) không bị tính là nhắm
s = sim([[15, {}], [20, { ear: 0.21 }]]);
assert.ok(!s.at(0, 100).has('DROWSY'), 'nheo mắt bị báo nhầm');
// 7. SAU một lần báo, chớp mắt 0.2 s lặp lại không được báo lại (lỗi trước đây)
s = sim([[15, {}], [3, { ear: 0.08 }], [3, {}], ...Array(6).fill([[0.2, { ear: 0.08 }], [1.2, {}]]).flat()]);
assert.ok(!s.at(21.2, 100).has('DROWSY'), 'chớp mắt sau lần báo đầu vẫn bị báo');
// 8. nhưng nhắm thật lần nữa (>1.5 s) thì vẫn báo
s = sim([[15, {}], [3, { ear: 0.08 }], [3, {}], [2, { ear: 0.08 }]]);
assert.ok(s.at(22.6, 100).has('DROWSY'), 'nhắm thật lần 2 mà không báo');
console.log('TẤT CẢ TEST ĐẠT');
