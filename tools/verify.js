// Verifies the SHIPPED app: extracts the model JSON and the actual forward /
// normalise / dequant source code out of index.html, runs them in Node, and
// checks the maths against held-out MNIST data.
//
// Usage: node verify.js <path-to-index.html> <mnist-data-dir>

'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const vm = require('vm');

const HTML = process.argv[2] || '../index.html';
const DATA_DIR = process.argv[3] || '../../mnist-data';

// ---------- pull code + weights out of the shipped index.html ----------
const html = fs.readFileSync(HTML, 'utf8');

function extractFunction(name) {
  const sig = 'function ' + name;
  const start = html.indexOf(sig);
  if (start < 0) throw new Error('cannot find ' + sig + ' in ' + HTML);
  let i = html.indexOf('{', start), depth = 0;
  for (; i < html.length; i++) {
    if (html[i] === '{') depth++;
    else if (html[i] === '}') { depth--; if (depth === 0) break; }
  }
  return html.slice(start, i + 1);
}

const modelMatch = html.match(/const MODEL = (\{.*?\});\n/s);
if (!modelMatch) throw new Error('cannot find MODEL JSON in ' + HTML);
const MODEL = JSON.parse(modelMatch[1]);

// Build a sandbox mirroring the app's environment, then run the app's own code
const sandbox = {
  atob: (s) => Buffer.from(s, 'base64').toString('binary'),
  Float32Array, Int8Array, Uint8Array, Math, Infinity,
  MODEL, IN: 784, HID: 64, OUT: 10,
};
vm.createContext(sandbox);
const shippedSrc = [
  extractFunction('b64ToInt8'),
  extractFunction('dequant'),
  'const W1 = dequant(MODEL.w1, MODEL.s1, HID, IN);',
  'const B1 = Float32Array.from(MODEL.b1);',
  'const W2 = dequant(MODEL.w2, MODEL.s2, OUT, HID);',
  'const B2 = Float32Array.from(MODEL.b2);',
  'const hidden = new Float32Array(HID);',
  'const probs = new Float32Array(OUT);',
  extractFunction('forward'),
  extractFunction('normalise'),
  '({ forward, normalise, probs, hidden });',
].join('\n');
const { forward, normalise, probs } = vm.runInContext(shippedSrc, sandbox);
console.log('extracted shipped code: b64ToInt8, dequant, forward, normalise + MODEL (' +
  MODEL.arch.join('-') + ' net)');

// ---------- held-out data ----------
function loadImages(file) {
  const buf = zlib.gunzipSync(fs.readFileSync(file));
  if (buf.readUInt32BE(0) !== 0x00000803) throw new Error('bad image magic');
  return { n: buf.readUInt32BE(4), pixels: new Uint8Array(buf.buffer, buf.byteOffset + 16).slice() };
}
function loadLabels(file) {
  const buf = zlib.gunzipSync(fs.readFileSync(file));
  if (buf.readUInt32BE(0) !== 0x00000801) throw new Error('bad label magic');
  return new Uint8Array(buf.buffer, buf.byteOffset + 8).slice();
}
const test = loadImages(path.join(DATA_DIR, 't10k-images.gz'));
const testLab = loadLabels(path.join(DATA_DIR, 't10k-labels.gz'));
console.log('held-out set: ' + test.n + ' MNIST test images (never used in training)\n');

let failures = 0;
function check(name, ok, detail) {
  console.log((ok ? 'PASS' : 'FAIL') + '  ' + name + (detail ? ' (' + detail + ')' : ''));
  if (!ok) failures++;
}

// deterministic RNG for reproducible augmentation
let seed = 123;
function rand() {
  seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const x = new Float32Array(784);
function loadX(i, dx, dy) {
  x.fill(0);
  const base = i * 784;
  for (let r = 0; r < 28; r++) {
    const sr = r - dy;
    if (sr < 0 || sr > 27) continue;
    for (let c = 0; c < 28; c++) {
      const sc = c - dx;
      if (sc < 0 || sc > 27) continue;
      x[r * 28 + c] = test.pixels[base + sr * 28 + sc] / 255;
    }
  }
}
function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

// ---------- check 1: independent forward implementation agrees ----------
// A second implementation, written differently (nested reduce, float64),
// guards against transcription bugs in the shipped code.
function forwardIndependent(inp) {
  const relu = (v) => Math.max(0, v);
  const h = MODEL.s1.map((_, j) => j).map((j) => {
    const q = Buffer.from(MODEL.w1, 'base64');
    let z = MODEL.b1[j];
    for (let k = 0; k < 784; k++) {
      const raw = q[j * 784 + k];
      z += (raw > 127 ? raw - 256 : raw) * Math.fround(MODEL.s1[j]) * inp[k];
    }
    return relu(z);
  });
  const q2 = Buffer.from(MODEL.w2, 'base64');
  const z2 = MODEL.b2.map((b, c) => {
    let z = b;
    for (let j = 0; j < 64; j++) {
      const raw = q2[c * 64 + j];
      z += (raw > 127 ? raw - 256 : raw) * Math.fround(MODEL.s2[c]) * h[j];
    }
    return z;
  });
  const m = Math.max(...z2);
  const e = z2.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
let maxDiff = 0;
for (let i = 0; i < 50; i++) {
  loadX(i, 0, 0);
  forward(x);
  const ref = forwardIndependent(x);
  for (let c = 0; c < 10; c++) maxDiff = Math.max(maxDiff, Math.abs(ref[c] - probs[c]));
}
check('shipped forward pass matches independent float64 implementation',
  maxDiff < 1e-4, 'max probability difference over 50 images = ' + maxDiff.toExponential(2));

// ---------- check 2: softmax is a real probability distribution ----------
let maxSumErr = 0, allInRange = true;
for (let i = 0; i < test.n; i++) {
  loadX(i, 0, 0);
  forward(x);
  let s = 0;
  for (let c = 0; c < 10; c++) {
    s += probs[c];
    if (probs[c] < 0 || probs[c] > 1) allInRange = false;
  }
  maxSumErr = Math.max(maxSumErr, Math.abs(s - 1));
}
check('softmax outputs sum to 1 on all 10000 held-out images',
  maxSumErr < 1e-6, 'max |sum - 1| = ' + maxSumErr.toExponential(2));
check('all probabilities within [0, 1]', allInRange);

// ---------- check 3: held-out accuracy, clean ----------
let correct = 0;
for (let i = 0; i < test.n; i++) {
  loadX(i, 0, 0);
  forward(x);
  if (argmax(probs) === testLab[i]) correct++;
}
const accClean = correct / test.n;
check('held-out accuracy (clean) >= 95%', accClean >= 0.95, (accClean * 100).toFixed(2) + '%');

// ---------- check 4: held-out accuracy under augmentation (random +/-2 px shifts) ----------
correct = 0;
for (let i = 0; i < test.n; i++) {
  loadX(i, Math.floor(rand() * 5) - 2, Math.floor(rand() * 5) - 2);
  forward(x);
  if (argmax(probs) === testLab[i]) correct++;
}
const accShift = correct / test.n;
check('held-out accuracy (augmented, shifted +/-2 px) >= 90%', accShift >= 0.90, (accShift * 100).toFixed(2) + '%');

// ---------- check 5: the app's normalise() recovers badly off-centre digits ----------
// Simulates a user drawing in a corner of the canvas: shift digits by up to
// +/-4 px, then run the shipped preprocessing before the forward pass.
correct = 0;
const N5 = 2000;
for (let i = 0; i < N5; i++) {
  loadX(i, Math.floor(rand() * 9) - 4, Math.floor(rand() * 9) - 4);
  const xn = normalise(x);
  forward(xn);
  if (argmax(probs) === testLab[i]) correct++;
}
const accNorm = correct / N5;
check('accuracy on off-centre digits AFTER the app\'s normalise() >= 95%',
  accNorm >= 0.95, (accNorm * 100).toFixed(2) + '% on ' + N5 + ' digits shifted up to +/-4 px');

// ---------- check 6: the 10 embedded sample digits classify correctly ----------
let sampleOk = true;
const details = [];
MODEL.samples.forEach((b64, digit) => {
  const px = Buffer.from(b64, 'base64');
  for (let k = 0; k < 784; k++) x[k] = px[k] / 255;
  forward(x);
  const pred = argmax(probs);
  details.push(digit + '->' + pred + ' (' + (probs[pred] * 100).toFixed(0) + '%)');
  if (pred !== digit || probs[pred] < 0.9) sampleOk = false;
});
check('all 10 embedded sample digits predicted correctly with > 90% confidence',
  sampleOk, details.join(' '));

// ---------- check 7: sample-click path (upscale 10x -> 10x10 box downsample -> normalise) ----------
// Mirrors what the app does when a user taps a sample thumbnail.
correct = 0;
MODEL.samples.forEach((b64, digit) => {
  const px = Buffer.from(b64, 'base64');
  // nearest-neighbour 10x upscale then 10x10 average is identity for a 28x28
  // image, but normalise() still runs (this checks it doesn't wreck anything).
  for (let k = 0; k < 784; k++) x[k] = px[k] / 255;
  const xn = normalise(x);
  forward(xn);
  if (argmax(probs) === digit) correct++;
});
check('sample-click pipeline (downsample + normalise + forward) classifies all 10 samples',
  correct === 10, correct + '/10');

console.log('\n' + (failures === 0 ? 'ALL CHECKS PASSED' : failures + ' CHECK(S) FAILED'));
process.exit(failures === 0 ? 0 : 1);
