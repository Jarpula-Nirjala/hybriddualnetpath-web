/**
 * In-browser inference for HybridDualPatNet (EfficientNetV2-S + Swin-Tiny fusion).
 *
 * Mirrors web/inference/predictor.py from the Django app:
 *   cv2 RGB -> resize 224 (INTER_LINEAR) -> ImageNet normalise -> model -> softmax
 *   Grad-CAM++ on local_encoder.conv_head for the predicted class -> JET overlay.
 *
 * The model is split into three ONNX graphs (see build_models.py):
 *   backbone: image -> A, D, z, g      head: z, g -> logits      cam: A, D, dlogit/dz -> 7x7 cam
 * d logit / d z is computed with central finite differences on the (small) head graph.
 */
(function () {
  const MODEL_DIR = 'models/v1/';
  const CLASS_NAMES = ['AD', 'CONTROL', 'PD'];
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];
  const SIZE = 224;
  const FD_EPS = 1e-2;

  ort.env.wasm.wasmPaths = new URL('js/ort/', document.baseURI).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  /* ---------- weight loading ---------- */

  let halfTable = null;
  function fp16Table() {
    if (halfTable) return halfTable;
    halfTable = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const f = h & 0x3ff;
      if (e === 0) halfTable[h] = s * Math.pow(2, -14) * (f / 1024);
      else if (e === 31) halfTable[h] = f ? NaN : s * Infinity;
      else halfTable[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return halfTable;
  }

  async function fetchBytes(url, onBytes) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to download ${url} (${res.status})`);
    if (!res.body || !onBytes) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const parts = [];
    let total = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      total += value.length;
      onBytes(value.length);
    }
    const out = new Uint8Array(total);
    let p = 0;
    for (const v of parts) { out.set(v, p); p += v.length; }
    return out;
  }

  async function loadPacked(name, onBytes) {
    const man = await (await fetch(MODEL_DIR + name + '.json')).json();
    const [graph, ...chunks] = await Promise.all([
      fetchBytes(MODEL_DIR + man.model),
      ...man.chunks.map(c => fetchBytes(MODEL_DIR + c, onBytes)),
    ]);
    const blob = new Uint8Array(man.packed);
    let p = 0;
    for (const c of chunks) { blob.set(c, p); p += c.length; }

    // Rebuild the original external-data file: fp16 segments expand back to fp32.
    const raw = new Uint8Array(man.size);
    const table = fp16Table();
    p = 0;
    for (const [off, len, isHalf] of man.segs) {
      if (isHalf) {
        const n = len / 4;
        const halves = new Uint16Array(blob.slice(p, p + n * 2).buffer);
        const floats = new Float32Array(n);
        for (let i = 0; i < n; i++) floats[i] = table[halves[i]];
        raw.set(new Uint8Array(floats.buffer), off);
        p += n * 2;
      } else {
        raw.set(blob.subarray(p, p + len), off);
        p += len;
      }
    }
    return ort.InferenceSession.create(graph, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      externalData: [{ path: man.external, data: raw }],
    });
  }

  let sessions = null;
  let loading = null;
  const TOTAL_BYTES = 74e6; // approx size of all weight chunks, for the progress bar

  function loadModels(onProgress) {
    if (sessions) return Promise.resolve(sessions);
    if (loading) return loading;
    let got = 0;
    const tick = n => { got += n; onProgress && onProgress(Math.min(got / TOTAL_BYTES, 0.99)); };
    loading = (async () => {
      const [backbone, head, cam] = await Promise.all(['backbone', 'head', 'cam'].map(n => loadPacked(n, tick)));
      sessions = { backbone, head, cam };
      onProgress && onProgress(1);
      return sessions;
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }

  /* ---------- image preprocessing (cv2-compatible) ---------- */

  async function decodeToRGBA(file) {
    const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const c = document.createElement('canvas');
    c.width = bmp.width; c.height = bmp.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    const data = ctx.getImageData(0, 0, c.width, c.height);
    bmp.close && bmp.close();
    return data;
  }

  // cv2.resize(..., INTER_LINEAR) coordinate mapping (half-pixel centres, clamped borders)
  function linearCoords(src, dst) {
    const scale = src / dst;
    const idx = new Int32Array(dst * 2);
    const w = new Float32Array(dst);
    for (let d = 0; d < dst; d++) {
      let f = (d + 0.5) * scale - 0.5;
      let s = Math.floor(f);
      f -= s;
      if (s < 0) { f = 0; s = 0; }
      if (s >= src - 1) { f = 0; s = src - 1; }
      idx[2 * d] = s;
      idx[2 * d + 1] = Math.min(s + 1, src - 1);
      w[d] = f;
    }
    return { idx, w };
  }

  function resizeRGB(img, dw, dh) {
    const { width: sw, height: sh, data } = img;
    const out = new Uint8ClampedArray(dw * dh * 3);
    if (sw === dw * 2 && sh === dh * 2) {
      // OpenCV switches to INTER_AREA for exact 2x downscale
      for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) for (let c = 0; c < 3; c++) {
        const a = ((2 * y) * sw + 2 * x) * 4 + c;
        out[(y * dw + x) * 3 + c] = Math.round((data[a] + data[a + 4] + data[a + sw * 4] + data[a + sw * 4 + 4]) / 4);
      }
      return out;
    }
    const X = linearCoords(sw, dw), Y = linearCoords(sh, dh);
    for (let y = 0; y < dh; y++) {
      const y0 = Y.idx[2 * y], y1 = Y.idx[2 * y + 1], fy = Y.w[y];
      for (let x = 0; x < dw; x++) {
        const x0 = X.idx[2 * x], x1 = X.idx[2 * x + 1], fx = X.w[x];
        for (let c = 0; c < 3; c++) {
          const top = data[(y0 * sw + x0) * 4 + c] * (1 - fx) + data[(y0 * sw + x1) * 4 + c] * fx;
          const bot = data[(y1 * sw + x0) * 4 + c] * (1 - fx) + data[(y1 * sw + x1) * 4 + c] * fx;
          out[(y * dw + x) * 3 + c] = Math.round(top * (1 - fy) + bot * fy);
        }
      }
    }
    return out;
  }

  function resizeFloat(src, sw, sh, dw, dh) {
    const out = new Float32Array(dw * dh);
    const X = linearCoords(sw, dw), Y = linearCoords(sh, dh);
    for (let y = 0; y < dh; y++) {
      const y0 = Y.idx[2 * y], y1 = Y.idx[2 * y + 1], fy = Y.w[y];
      for (let x = 0; x < dw; x++) {
        const x0 = X.idx[2 * x], x1 = X.idx[2 * x + 1], fx = X.w[x];
        const top = src[y0 * sw + x0] * (1 - fx) + src[y0 * sw + x1] * fx;
        const bot = src[y1 * sw + x0] * (1 - fx) + src[y1 * sw + x1] * fx;
        out[y * dw + x] = top * (1 - fy) + bot * fy;
      }
    }
    return out;
  }

  function toTensor(rgb) {
    const n = SIZE * SIZE;
    const t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
      t[c * n + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
    }
    return new ort.Tensor('float32', t, [1, 3, SIZE, SIZE]);
  }

  /* ---------- heatmap overlay (pytorch_grad_cam show_cam_on_image, JET) ---------- */

  function jet(v) {
    const c = x => Math.max(0, Math.min(1, x));
    return [c(1.5 - Math.abs(4 * v - 3)), c(1.5 - Math.abs(4 * v - 2)), c(1.5 - Math.abs(4 * v - 1))];
  }

  function overlay(rgb, cam224) {
    const n = SIZE * SIZE;
    const mix = new Float32Array(n * 3);
    let max = 0;
    for (let i = 0; i < n; i++) {
      const h = jet(Math.floor(255 * cam224[i]) / 255);
      for (let c = 0; c < 3; c++) {
        const v = h[c] + rgb[i * 3 + c] / 255;
        mix[i * 3 + c] = v;
        if (v > max) max = v;
      }
    }
    const out = new ImageData(SIZE, SIZE);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) out.data[i * 4 + c] = Math.floor((255 * mix[i * 3 + c]) / max);
      out.data[i * 4 + 3] = 255;
    }
    return out;
  }

  /* ---------- prediction ---------- */

  function softmax(l) {
    const m = Math.max(...l);
    const e = l.map(v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }

  async function predict(file, onStage) {
    const t0 = performance.now();
    const s = await loadModels(p => onStage && onStage('download', p));
    onStage && onStage('analyze');

    const rgb = resizeRGB(await decodeToRGBA(file), SIZE, SIZE);
    const bb = await s.backbone.run({ image: toTensor(rgb) });
    const { A, D, z, g } = bb;
    const logits = Array.from((await s.head.run({ z, g })).logits.data);
    const probs = softmax(logits);
    const idx = probs.indexOf(Math.max(...probs));

    onStage && onStage('explain');
    // central finite differences: d logit[idx] / d z, one batched head call
    const n = z.dims[1], gd = g.dims[1];
    const zz = new Float32Array(2 * n * n), gg = new Float32Array(2 * n * gd);
    for (let r = 0; r < 2 * n; r++) {
      zz.set(z.data, r * n);
      gg.set(g.data, r * gd);
    }
    for (let i = 0; i < n; i++) {
      zz[i * n + i] += FD_EPS;
      zz[(n + i) * n + i] -= FD_EPS;
    }
    const L = (await s.head.run({
      z: new ort.Tensor('float32', zz, [2 * n, n]),
      g: new ort.Tensor('float32', gg, [2 * n, gd]),
    })).logits.data;
    const gz = new Float32Array(n);
    for (let i = 0; i < n; i++) gz[i] = (L[i * 3 + idx] - L[(n + i) * 3 + idx]) / (2 * FD_EPS);

    const cam = (await s.cam.run({ A, D, gz: new ort.Tensor('float32', gz, [1, n]) })).cam;
    const [, ch, cw] = cam.dims;
    const heat = overlay(rgb, resizeFloat(cam.data, cw, ch, SIZE, SIZE));

    return {
      predictedClass: CLASS_NAMES[idx],
      probs: { AD: probs[0], CONTROL: probs[1], PD: probs[2] },
      heatmap: heat,
      seconds: (performance.now() - t0) / 1000,
    };
  }

  window.HDPN = { loadModels, predict };
})();
