// Service worker: dùng offline được, nhưng LUÔN lấy bản mới của app khi có mạng.
const CACHE = 'drowsy-guard-v3';
const SHELL = ['./', 'index.html', 'js/app.js', 'js/logic.js', 'js/features.js', 'js/dl.js', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png'];
self.addEventListener('install', e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))); self.clients.claim(); });
const put = (req, r) => { if (r.ok) { const copy = r.clone(); caches.open(CACHE).then(c => c.put(req, copy)); } return r; };
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const own = new URL(req.url).origin === self.location.origin;
  if (own && !/\.(task|onnx)(\?|$)/.test(req.url)) {
    // File của app: ưu tiên mạng (có bản mới là dùng ngay), mất mạng thì dùng cache
    e.respondWith(fetch(req, { cache: 'no-cache' }).then(r => put(req, r)).catch(() => caches.match(req)));
  } else {
    // Thư viện CDN và mô hình nặng (.task/.onnx): ưu tiên cache cho nhanh. Đổi model -> tăng số v ở CACHE
    e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(r => put(req, r))));
  }
});