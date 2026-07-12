// Trains the 784-64-10 dense net on real MNIST and exports quantised weights
// as JSON (weights.json) ready to be inlined into index.html.
//
// Usage: node train.js <mnist-data-dir> <out-weights.json>
// The data dir must contain train-images.gz, train-labels.gz, t10k-images.gz, t10k-labels.gz
// (IDX format, gzipped, from the standard MNIST distribution).

'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const DATA_DIR = process.argv[2] || '.';
const OUT = process.argv[3] || 'weights.json';

// ---------- IDX loading ----------
function loadImages(file) {
  const buf = zlib.gunzipSync(fs.readFileSync(file));
  const magic = buf.readUInt32BE(0);
  if (magic !== 0x00000803) throw new Error('bad image magic in ' + file);
  const n = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  if (rows !== 28 || cols !== 28) throw new Error('expected 28x28');
  const out = new Uint8Array(buf.buffer, buf.byteOffset + 16, n * 784);
  return { n, pixels: new Uint8Array(out) }; // copy so gc can drop buf
}
function loadLabels(file) {
  const buf = zlib.gunzipSync(fs.readFileSync(file));
  const magic = buf.readUInt32BE(0);
  if (magic !== 0x00000801) throw new Error('bad label magic in ' + file);
  const n = buf.readUInt32BE(4);
  return new Uint8Array(buf.buffer, buf.byteOffset + 8, n).slice();
}

console.log('Loading MNIST...');
const train = loadImages(path.join(DATA_DIR, 'train-images.gz'));
const trainLab = loadLabels(path.join(DATA_DIR, 'train-labels.gz'));
const test = loadImages(path.join(DATA_DIR, 't10k-images.gz'));
const testLab = loadLabels(path.join(DATA_DIR, 't10k-labels.gz'));
console.log(`train=${train.n} test=${test.n}`);

// ---------- network: 784 -> 64 (ReLU) -> 10 (softmax) ----------
const IN = 784, HID = 64, OUTC = 10;

// deterministic RNG (mulberry32) for reproducible training
let seed = 42;
function rand() {
  seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function gauss() { // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const W1 = new Float32Array(HID * IN);
const b1 = new Float32Array(HID);
const W2 = new Float32Array(OUTC * HID);
const b2 = new Float32Array(OUTC);
for (let i = 0; i < W1.length; i++) W1[i] = gauss() * Math.sqrt(2 / IN);      // He init
for (let i = 0; i < W2.length; i++) W2[i] = gauss() * Math.sqrt(1 / HID);     // Xavier-ish

// momentum buffers
const vW1 = new Float32Array(W1.length), vb1 = new Float32Array(b1.length);
const vW2 = new Float32Array(W2.length), vb2 = new Float32Array(b2.length);

// scratch
const x = new Float32Array(IN);
const h = new Float32Array(HID);
const p = new Float32Array(OUTC);
const dh = new Float32Array(HID);
const gW1 = new Float32Array(W1.length), gb1 = new Float32Array(b1.length);
const gW2 = new Float32Array(W2.length), gb2 = new Float32Array(b2.length);

// load sample i from a pixel store into x, with optional integer shift (augmentation)
function loadX(pixels, i, dx, dy) {
  const base = i * 784;
  if (dx === 0 && dy === 0) {
    for (let k = 0; k < 784; k++) x[k] = pixels[base + k] / 255;
    return;
  }
  x.fill(0);
  for (let r = 0; r < 28; r++) {
    const sr = r - dy;
    if (sr < 0 || sr > 27) continue;
    for (let c = 0; c < 28; c++) {
      const sc = c - dx;
      if (sc < 0 || sc > 27) continue;
      x[r * 28 + c] = pixels[base + sr * 28 + sc] / 255;
    }
  }
}

function forward() {
  for (let j = 0; j < HID; j++) {
    let s = b1[j];
    const off = j * IN;
    for (let k = 0; k < IN; k++) s += W1[off + k] * x[k];
    h[j] = s > 0 ? s : 0; // ReLU
  }
  let maxz = -Infinity;
  for (let c = 0; c < OUTC; c++) {
    let s = b2[c];
    const off = c * HID;
    for (let j = 0; j < HID; j++) s += W2[off + j] * h[j];
    p[c] = s;
    if (s > maxz) maxz = s;
  }
  let sum = 0;
  for (let c = 0; c < OUTC; c++) { p[c] = Math.exp(p[c] - maxz); sum += p[c]; }
  for (let c = 0; c < OUTC; c++) p[c] /= sum;
}

// accumulate gradients for one sample (softmax + cross-entropy: dz2 = p - y)
function backward(label) {
  dh.fill(0);
  for (let c = 0; c < OUTC; c++) {
    const dz = p[c] - (c === label ? 1 : 0);
    gb2[c] += dz;
    const off = c * HID;
    for (let j = 0; j < HID; j++) {
      gW2[off + j] += dz * h[j];
      dh[j] += dz * W2[off + j];
    }
  }
  for (let j = 0; j < HID; j++) {
    if (h[j] <= 0) continue; // ReLU gradient
    const dz = dh[j];
    gb1[j] += dz;
    const off = j * IN;
    for (let k = 0; k < IN; k++) gW1[off + k] += dz * x[k];
  }
}

function sgdStep(lr, batch) {
  const mom = 0.9, inv = 1 / batch;
  for (let i = 0; i < W1.length; i++) { vW1[i] = mom * vW1[i] - lr * gW1[i] * inv; W1[i] += vW1[i]; }
  for (let i = 0; i < b1.length; i++) { vb1[i] = mom * vb1[i] - lr * gb1[i] * inv; b1[i] += vb1[i]; }
  for (let i = 0; i < W2.length; i++) { vW2[i] = mom * vW2[i] - lr * gW2[i] * inv; W2[i] += vW2[i]; }
  for (let i = 0; i < b2.length; i++) { vb2[i] = mom * vb2[i] - lr * gb2[i] * inv; b2[i] += vb2[i]; }
  gW1.fill(0); gb1.fill(0); gW2.fill(0); gb2.fill(0);
}

function evaluate(pixels, labels, n, shift) {
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const dx = shift ? Math.floor(rand() * 5) - 2 : 0;
    const dy = shift ? Math.floor(rand() * 5) - 2 : 0;
    loadX(pixels, i, dx, dy);
    forward();
    let best = 0;
    for (let c = 1; c < OUTC; c++) if (p[c] > p[best]) best = c;
    if (best === labels[i]) correct++;
  }
  return correct / n;
}

// ---------- training loop ----------
const EPOCHS = 8, BATCH = 64;
const order = new Int32Array(train.n);
for (let i = 0; i < train.n; i++) order[i] = i;

for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  // shuffle (Fisher-Yates with our RNG)
  for (let i = train.n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }
  const lr = 0.1 * Math.pow(0.7, epoch - 1);
  let loss = 0, inBatch = 0;
  const t0 = Date.now();
  for (let s = 0; s < train.n; s++) {
    const i = order[s];
    // augmentation: random shift up to +/-2 px, applied to 60% of samples
    let dx = 0, dy = 0;
    if (rand() < 0.6) { dx = Math.floor(rand() * 5) - 2; dy = Math.floor(rand() * 5) - 2; }
    loadX(train.pixels, i, dx, dy);
    forward();
    loss += -Math.log(Math.max(p[trainLab[i]], 1e-12));
    backward(trainLab[i]);
    if (++inBatch === BATCH) { sgdStep(lr, BATCH); inBatch = 0; }
  }
  if (inBatch > 0) sgdStep(lr, inBatch);
  const acc = evaluate(test.pixels, testLab, 2000, false); // quick check on first 2k test
  console.log(`epoch ${epoch}: loss=${(loss / train.n).toFixed(4)} lr=${lr.toFixed(4)} quickTestAcc=${(acc * 100).toFixed(2)}% (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
}

const accClean = evaluate(test.pixels, testLab, test.n, false);
const accShift = evaluate(test.pixels, testLab, test.n, true);
console.log(`FLOAT32 test accuracy (10000 held-out): clean=${(accClean * 100).toFixed(2)}% shifted±2px=${(accShift * 100).toFixed(2)}%`);

// ---------- int8 quantisation (per-output-row scale) ----------
function quantise(W, rows, cols) {
  const q = new Int8Array(rows * cols);
  const scales = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let m = 0;
    for (let c = 0; c < cols; c++) m = Math.max(m, Math.abs(W[r * cols + c]));
    const s = m / 127 || 1e-8;
    scales[r] = s;
    for (let c = 0; c < cols; c++) q[r * cols + c] = Math.max(-127, Math.min(127, Math.round(W[r * cols + c] / s)));
  }
  return { q, scales };
}
const q1 = quantise(W1, HID, IN);
const q2 = quantise(W2, OUTC, HID);

// dequantise back into W1/W2 so we can measure the app's exact accuracy
for (let r = 0; r < HID; r++) for (let c = 0; c < IN; c++) W1[r * IN + c] = q1.q[r * IN + c] * q1.scales[r];
for (let r = 0; r < OUTC; r++) for (let c = 0; c < HID; c++) W2[r * HID + c] = q2.q[r * HID + c] * q2.scales[r];
const accQ = evaluate(test.pixels, testLab, test.n, false);
const accQShift = evaluate(test.pixels, testLab, test.n, true);
console.log(`INT8-quantised test accuracy (what the app ships): clean=${(accQ * 100).toFixed(2)}% shifted±2px=${(accQShift * 100).toFixed(2)}%`);

// ---------- pick one correctly-classified sample per digit for the app ----------
const samples = [];
for (let d = 0; d < 10; d++) {
  let found = -1;
  for (let i = 0; i < test.n; i++) {
    if (testLab[i] !== d) continue;
    loadX(test.pixels, i, 0, 0);
    forward();
    let best = 0;
    for (let c = 1; c < OUTC; c++) if (p[c] > p[best]) best = c;
    if (best === d && p[best] > 0.9) { found = i; break; }
  }
  if (found < 0) throw new Error('no confident sample for digit ' + d);
  samples.push(Buffer.from(test.pixels.subarray(found * 784, found * 784 + 784)).toString('base64'));
}

// ---------- export ----------
const round = (arr, dp) => Array.from(arr, v => +v.toFixed(dp));
const model = {
  arch: [IN, HID, OUTC],
  w1: Buffer.from(q1.q.buffer, q1.q.byteOffset, q1.q.length).toString('base64'),
  s1: round(q1.scales, 7),
  b1: round(b1, 5),
  w2: Buffer.from(q2.q.buffer, q2.q.byteOffset, q2.q.length).toString('base64'),
  s2: round(q2.scales, 7),
  b2: round(b2, 5),
  samples,
};
fs.writeFileSync(OUT, JSON.stringify(model));
console.log(`wrote ${OUT} (${(fs.statSync(OUT).size / 1024).toFixed(1)} KB)`);
