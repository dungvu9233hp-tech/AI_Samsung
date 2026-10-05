import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { Decider } from "./logic.js";
import { ear, mar, poseFromMatrix, L_EYE, R_EYE } from "./features.js";

const $ = id => document.getElementById(id);
const LABEL = { CALIBRATING: 'Đang hiệu chuẩn…', NORMAL: 'Bình thường', DISTRACTED: 'Mất tập trung', DROWSY: 'BUỒN NGỦ – NGHỈ NGAY!' };
const MODEL_REMOTE = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const load = (k, d) => { try { return { ...d, ...JSON.parse(localStorage.getItem(k) || '{}') }; } catch { return d; } };
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

let settings = load('dg.settings.v2', { sens: 0.60, sound: true, vib: true, dots: false });
let events = (() => { try { return JSON.parse(localStorage.getItem('dg.events') || '[]'); } catch { return []; } })();
const dec = new Decider({ earFactor: settings.sens });
let landmarker, stream, running = false, lastVT = -1, muteUntil = 0, lastState = null, wake = null, fpsAcc = 0, fpsN = 0, fpsT = performance.now(), sessionStart = 0;
const v = $('v'), cv = $('c'), cx = cv.getContext('2d');

// ---------- Âm thanh ----------
let actx, alarmTimer = null;
function beep() {
  if (!actx) return;
  const o = actx.createOscillator(), g = actx.createGain();
  o.frequency.value = 1000; o.connect(g); g.connect(actx.destination);
  g.gain.setValueAtTime(0.0001, actx.currentTime); g.gain.exponentialRampToValueAtTime(0.5, actx.currentTime + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime + 0.25); o.start(); o.stop(actx.currentTime + 0.27);
  if (settings.vib && navigator.vibrate) navigator.vibrate([200, 100, 200]);
}
function setAlarm(on) {
  if (on && !alarmTimer) { beep(); alarmTimer = setInterval(beep, 700); }
  if (!on && alarmTimer) { clearInterval(alarmTimer); alarmTimer = null; navigator.vibrate && navigator.vibrate(0); }
}

// ---------- Khởi động ----------
async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  let path = MODEL_REMOTE;
  try { if ((await fetch('models/face_landmarker.task', { method: 'HEAD' })).ok) path = 'models/face_landmarker.task'; } catch {}
  const mk = delegate => FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: path, delegate }, runningMode: 'VIDEO', numFaces: 1,
    outputFaceBlendshapes: false, outputFacialTransformationMatrixes: true });
  try { return await mk('GPU'); } catch { return await mk('CPU'); }
}
async function start() {
  $('err').textContent = '';
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)(); await actx.resume();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    v.srcObject = stream; await v.play();
    $('bStart').disabled = true; $('bStart').textContent = 'Đang tải mô hình…';
    landmarker = landmarker || await createLandmarker();
    try { wake = await navigator.wakeLock?.request('screen'); } catch {}
    dec.reset(); lastState = null; sessionStart = Date.now(); running = true; $('start').style.display = 'none'; loop();
  } catch (e) {
    $('err').textContent = e.name === 'NotAllowedError' ? 'Bạn cần cho phép truy cập camera.' :
      location.protocol === 'http:' && location.hostname !== 'localhost' ? 'Camera chỉ hoạt động trên HTTPS.' : 'Lỗi: ' + (e.message || e);
    $('bStart').disabled = false; $('bStart').textContent = 'Bắt đầu';
  }
}
function stop() {
  running = false; setAlarm(false); stream?.getTracks().forEach(t => t.stop()); wake?.release?.(); wake = null;
  $('start').style.display = 'flex'; $('bStart').disabled = false; $('bStart').textContent = 'Bắt đầu lại'; logSummary();
}

// ---------- Vòng lặp ----------
function loop() {
  if (!running) return;
  if (v.readyState >= 2 && v.currentTime !== lastVT) {
    lastVT = v.currentTime;
    const t0 = performance.now(), res = landmarker.detectForVideo(v, t0), w = v.videoWidth, h = v.videoHeight;
    let f = { face: false, ear: NaN, mar: NaN, pitch: NaN, yaw: NaN };
    const lm = res.faceLandmarks?.[0];
    if (lm) {
      const pose = res.facialTransformationMatrixes?.[0] ? poseFromMatrix(res.facialTransformationMatrixes[0].data) : { pitch: 0, yaw: 0 };
      // ear = mắt MỞ HƠN trong hai mắt: phải nhắm cả hai mới tính là nhắm (nháy một mắt không báo)
      f = { face: true, ear: Math.max(ear(lm, L_EYE, w, h), ear(lm, R_EYE, w, h)), mar: mar(lm, w, h), ...pose };
    }
    dec.cfg.earFactor = settings.sens;
    const info = dec.update(t0 / 1000, f);
    draw(lm); render(info);
    fpsAcc += 1; if (performance.now() - fpsT > 1000) { $('fps').textContent = fpsAcc + ' FPS'; fpsAcc = 0; fpsT = performance.now(); }
  }
  requestAnimationFrame(loop);
}
function draw(lm) {
  if (cv.width !== v.videoWidth) { cv.width = v.videoWidth; cv.height = v.videoHeight; }
  cx.clearRect(0, 0, cv.width, cv.height);
  if (!lm || !settings.dots) return;
  cx.fillStyle = '#34c759';
  for (const i of [...L_EYE, ...R_EYE, 78, 13, 14, 308]) cx.fillRect(lm[i].x * cv.width - 2, lm[i].y * cv.height - 2, 4, 4);
}
const pct = x => (x * 100).toFixed(0) + '%', n2 = x => Number.isFinite(x) ? x.toFixed(2) : '--';
function render(i) {
  $('app').dataset.s = i.state; $('stateText').textContent = LABEL[i.state];
  $('hint').textContent = i.calibrating ? 'Ngồi tự nhiên, nhìn thẳng vào camera…' : i.noface ? 'Không thấy khuôn mặt' : '';
  $('bar').style.display = i.calibrating ? 'block' : 'none'; $('bar').firstChild.style.width = pct(i.calibProgress);
  $('metrics').innerHTML = `EAR (2 mắt) <b>${n2(i.ear)}</b> / ngưỡng <b>${n2(i.earThr)}</b> · mắt <b>${i.closed ? 'NHẮM' : 'mở'}</b><br>` +
    `MAR <b>${n2(i.mar)}</b> · ngáp (5 phút) <b>${i.yawns}</b><br>PERCLOS <b>${pct(i.perclos)}</b> · nhắm <b>${i.closedDur.toFixed(1)}s</b><br>` +
    `Cúi <b>${Number.isFinite(i.pitchDev) ? i.pitchDev.toFixed(0) : '--'}°</b> · Quay <b>${Number.isFinite(i.yawDev) ? i.yawDev.toFixed(0) : '--'}°</b>`;
  setAlarm(i.state === 'DROWSY' && settings.sound && performance.now() > muteUntil);
  if (i.state !== lastState && i.state !== 'CALIBRATING') {
    if (lastState !== null) { events.push({ t: Date.now(), s: i.state, perclos: +i.perclos.toFixed(3), ear: +(i.ear || 0).toFixed(3) }); events = events.slice(-500); save('dg.events', events); }
    lastState = i.state;
  }
}

// ---------- Nhật ký / cài đặt ----------
function logSummary() { renderHistory(); }
function renderHistory() {
  const d = events.filter(e => e.s === 'DROWSY').length, k = events.filter(e => e.s === 'DISTRACTED').length;
  $('sum').innerHTML = `<p>Tổng: <b>${d}</b> lần buồn ngủ · <b>${k}</b> lần mất tập trung</p>`;
  $('evs').innerHTML = events.slice(-30).reverse().map(e => `<div class="ev ${e.s}"><span>${new Date(e.t).toLocaleString('vi-VN')}</span><span>${LABEL[e.s]}</span></div>`).join('') || '<p style="color:var(--mut)">Chưa có cảnh báo nào.</p>';
}
const panel = (id, on) => $(id).classList.toggle('on', on);
$('bStart').onclick = start; $('bStop').onclick = stop;
$('bMute').onclick = () => { dec.acknowledge(); muteUntil = performance.now() + 10000; setAlarm(false); };
$('bCal').onclick = () => { dec.reset(); lastState = null; };
$('bHist').onclick = () => { renderHistory(); panel('pHist', true); }; $('bSet').onclick = () => panel('pSet', true);
document.querySelectorAll('[data-close]').forEach(b => b.onclick = () => { panel('pHist', false); panel('pSet', false); });
$('bClear').onclick = () => { events = []; save('dg.events', events); renderHistory(); };
$('sSens').value = settings.sens; $('sSound').checked = settings.sound; $('sVib').checked = settings.vib; $('sDots').checked = settings.dots;
$('sSens').oninput = e => { settings.sens = +e.target.value; save('dg.settings.v2', settings); };
for (const [id, k] of [['sSound', 'sound'], ['sVib', 'vib'], ['sDots', 'dots']]) $(id).onchange = e => { settings[k] = e.target.checked; save('dg.settings.v2', settings); };
document.addEventListener('visibilitychange', async () => { if (running && document.visibilityState === 'visible') { try { wake = await navigator.wakeLock?.request('screen'); } catch {} } });
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
