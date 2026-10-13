# 🧠 HybridDualPatNet — Brain MRI Classifier (Web Demo)

**🔗 Live demo: https://hybriddualnetpath-web.onrender.com**

A web app that classifies brain MRI scans as **Alzheimer's Disease (AD)**, **Parkinson's Disease (PD)** or **Healthy Control** and shows a **Grad-CAM++ heatmap** of the brain regions that drove each prediction.

The model runs **entirely in your browser** with WebAssembly. Your scan is never uploaded to a server, and the site is static files only, so it can be hosted for free.

> Model training, the research paper and the full Django version live in the main repository:
> **[Jarpula-Nirjala/HybridDualNetPath](https://github.com/Jarpula-Nirjala/HybridDualNetPath)**

---

## ✨ Features

- **3-class prediction** (AD / PD / Healthy) with a confidence score for each class
- **Grad-CAM++ heatmap** on the EfficientNetV2-S path
- **Private:** inference happens on your device, so images never leave your computer
- **No server needed:** deploys as a static site on Render, Vercel, Netlify or Cloudflare Pages
- **"Try a sample MRI"** button for a quick demo
- Model downloads once (~75 MB) and is then cached by the browser

## 📊 Model

| | |
|---|---|
| Architecture | EfficientNetV2-S (local features) + Swin-Tiny (global context), fused with Cross-Fusion Attention |
| Test accuracy | **98.46%** |
| Macro AUC-ROC | **0.998** |
| Training data | 6K+ brain MRI images |
| Explainability | Grad-CAM++ on `local_encoder.conv_head` |

## ⚙️ How the in-browser version works

The original PyTorch model (204 MB) is converted to ONNX and split into three small graphs, which run with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/):

| Graph | Input → Output |
|---|---|
| `backbone` | image → CNN feature maps + Swin features |
| `head` | features → class scores |
| `cam` | feature maps + gradients → Grad-CAM++ map |

To keep the download small, Transformer layers are stored as int8 and all other weights as float16, which the browser expands back to float32 when it loads them. That brings the total to about 73 MB.

**Checked against the original PyTorch model on 10 real MRI scans:** the predicted class was the same every time, and confidence scores were within 3.5%.

**Differences from the Django version:** each image is analysed once (single pass, no test-time augmentation), and there are no user accounts or scan history.

## 📁 Project structure

```
├── index.html           # Single-page app
├── js/
│   ├── infer.js         # Preprocessing, inference, Grad-CAM++ overlay
│   ├── page.js          # Upload / results UI
│   └── ort/             # ONNX Runtime Web (self-hosted)
├── models/v1/           # Compressed model graphs + weight chunks
├── samples/             # Sample MRI for the demo button
├── static/              # Styles and background animation
├── tools/
│   └── build_models.py  # Converts final_model.pth → models/v1/
├── _headers             # Headers for Netlify / Cloudflare Pages
└── vercel.json          # Headers for Vercel
```

## 💻 Run locally

Any static file server works:

```bash
git clone https://github.com/Jarpula-Nirjala/hybriddualnetpath-web.git
cd hybriddualnetpath-web
python -m http.server 8080
```

Then open http://localhost:8080.

## 🚀 Deploy for free

**Render (Static Site)**
1. **New +** → **Static Site** → select this repo
2. **Build Command:** `echo done` · **Publish Directory:** `.`
3. **Settings → Headers:** add these for path `/*`, so the model can use several CPU cores (about 3× faster):
   - `Cross-Origin-Opener-Policy: same-origin`
   - `Cross-Origin-Embedder-Policy: credentialless`

**Vercel:** `npx vercel --prod`. Headers are already set in `vercel.json`.

**Netlify / Cloudflare Pages:** deploy the folder as-is. Headers are already set in `_headers`.

## 🔄 Rebuilding the model files

If you retrain the model, regenerate `models/v1/` with `tools/build_models.py`. It needs PyTorch, timm, onnx, onnxruntime and grad-cam, plus the `web/` folder from the main repository. The paths at the top of the script assume the Docker setup it was written for, so adjust them to your machine.

## ⚠️ Disclaimer

For **research and educational purposes only**. This is not a medical device and must not be used for diagnosis. Always consult a qualified healthcare professional.

## 👩‍💻 Author

**Jarpula Nirjala**, B.Tech Electronics & Communication Engineering, IIIT Surat (2022 – 2026)

[GitHub](https://github.com/Jarpula-Nirjala) · [LinkedIn](https://www.linkedin.com/in/nirjala-jarpula-749346321/) · [LeetCode](https://leetcode.com/u/nirjala8462/) · [GfG](https://auth.geeksforgeeks.org/user/nirjala8462/)
