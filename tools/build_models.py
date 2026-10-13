"""Build browser models for HybridDualPathModel.

backbone: image -> A (conv_head out, 1x1280x7x7), D (d act(bn2(A))/dA), z (proj_local Linear out, 1x512), g (swin feat, 1x768)
head:     z, g -> logits                      (dynamic batch: browser does central finite differences on z)
cam:      A, D, gz -> cam (7x7, min-max normalised Grad-CAM++ on local_encoder.conv_head)
Weights are int8 for MatMul/Gemm (dynamic quant), other floats stored as fp16, split in 20 MB chunks.
"""
import glob, json, os, shutil, sys
import cv2, numpy as np, onnx, onnxruntime as ort, torch, torch.nn as nn
from onnx.external_data_helper import convert_model_to_external_data
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, "/home/user/app/web")
from inference.architecture import HybridDualPathModel

OUT, SITE = "/out", "/out/site_models"
m = HybridDualPathModel(3)
ck = torch.load("/home/user/app/web/weights/final_model.pth", map_location="cpu")
m.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck)
m.eval()
le = m.local_encoder
assert isinstance(le.bn2.act, nn.SiLU)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = m

    def forward(self, x):
        le = self.m.local_encoder
        bn = le.bn2
        A = le.conv_head(le.blocks(le.bn1(le.conv_stem(x))))
        k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        zz = (A - bn.running_mean[None, :, None, None]) * k[None, :, None, None] + bn.bias[None, :, None, None]
        s = torch.sigmoid(zz)
        D = k[None, :, None, None] * s * (1 + zz * (1 - s))
        pooled = (zz * s).mean(dim=(2, 3))
        return A, D, self.m.fusion.proj_local[0](pooled), self.m.global_encoder(x)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = m

    def forward(self, z, g):
        f = self.m.fusion
        l = f.proj_local[2](f.proj_local[1](z)).unsqueeze(1)
        gp = f.proj_global(g).unsqueeze(1)
        a, _ = f.attn(l, gp, gp)
        a, l = a.squeeze(1), l.squeeze(1)
        gate = f.gate(torch.cat([a, l], dim=-1))
        return self.m.classifier(f.out_proj(gate * a + (1 - gate) * l))


class Cam(nn.Module):
    """Grad-CAM++ as pytorch_grad_cam 1.5.4 (GradCAMPlusPlus.get_cam_weights + relu + min-max)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("W", m.fusion.proj_local[0].weight.detach().clone())

    def forward(self, A, D, gz):
        gp = gz @ self.W  # d score / d pooled (1x1280)
        grads = gp[:, :, None, None] / (A.shape[2] * A.shape[3]) * D
        g2 = grads * grads
        g3 = g2 * grads
        sa = A.sum(dim=(2, 3))[:, :, None, None]
        aij = g2 / (2 * g2 + sa * g3 + 1e-6)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))
        w = (torch.relu(grads) * aij).sum(dim=(2, 3))
        cam = torch.relu((w[:, :, None, None] * A).sum(dim=1))
        cam = cam - cam.min()
        return cam / (1e-7 + cam.max())


bb, hd, cm = Backbone().eval(), Head().eval(), Cam().eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    A, D, z, g = bb(x)
    assert torch.allclose(hd(z, g), m(x), atol=1e-4)
print("decomposition matches original model")
torch.onnx.export(bb, x, f"{OUT}/bb.onnx", input_names=["image"], output_names=["A", "D", "z", "g"], opset_version=17)
torch.onnx.export(hd, (z, g), f"{OUT}/hd.onnx", input_names=["z", "g"], output_names=["logits"], opset_version=17,
                  dynamic_axes={"z": {0: "n"}, "g": {0: "n"}, "logits": {0: "n"}})
torch.onnx.export(cm, (A, D, torch.randn(1, 512)), f"{OUT}/cam.onnx", input_names=["A", "D", "gz"],
                  output_names=["cam"], opset_version=17)
quantize_dynamic(f"{OUT}/bb.onnx", f"{OUT}/bb_q.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])


def pack(src, name):
    mdl = onnx.load(src)
    convert_model_to_external_data(mdl, all_tensors_to_one_file=True, location=f"{name}.bin", size_threshold=1024)
    tmp = f"/tmp/pack_{name}"
    os.makedirs(tmp, exist_ok=True)
    onnx.save_model(mdl, f"{tmp}/{name}.onnx")
    raw = open(f"{tmp}/{name}.bin", "rb").read()
    segs, parts = [], []
    for t in mdl.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            continue
        kv = {e.key: e.value for e in t.external_data}
        off, ln = int(kv["offset"]), int(kv["length"])
        b = raw[off:off + ln]
        f16 = int(t.data_type == onnx.TensorProto.FLOAT)
        if f16:
            b = np.frombuffer(b, np.float32).astype(np.float16).tobytes()
        segs.append([off, ln, f16])
        parts.append(b)
    blob = b"".join(parts)
    shutil.copy(f"{tmp}/{name}.onnx", f"{SITE}/{name}.onnx")
    ch, chunks = 20 * 2**20, []
    for i in range(0, len(blob), ch):
        fn = f"{name}.w{i // ch}.bin"
        open(f"{SITE}/{fn}", "wb").write(blob[i:i + ch])
        chunks.append(fn)
    man = {"model": f"{name}.onnx", "external": f"{name}.bin", "size": len(raw), "chunks": chunks,
           "packed": len(blob), "segs": segs}
    json.dump(man, open(f"{SITE}/{name}.json", "w"))
    gsize = os.path.getsize(f"{SITE}/{name}.onnx") // 1024
    print(f"{name}: graph {gsize} KB, weights {len(blob) / 2**20:.1f} MB in {len(chunks)} chunks")
    return man


def session(man):
    """Rebuild exactly as the browser does: chunks -> fp16 expand -> external data."""
    blob = b"".join(open(f"{SITE}/{c}", "rb").read() for c in man["chunks"])
    raw, p = bytearray(man["size"]), 0
    for off, ln, f16 in man["segs"]:
        n = ln // 2 if f16 else ln
        raw[off:off + ln] = np.frombuffer(blob[p:p + n], np.float16).astype(np.float32).tobytes() if f16 else blob[p:p + n]
        p += n
    d = f"/tmp/re_{man['model']}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"{SITE}/{man['model']}", f"{d}/m.onnx")
    open(f"{d}/{man['external']}", "wb").write(raw)
    return ort.InferenceSession(f"{d}/m.onnx")


shutil.rmtree(SITE, ignore_errors=True)
os.makedirs(SITE)
S = {k: session(pack(src, k)) for k, src in
     [("backbone", f"{OUT}/bb_q.onnx"), ("head", f"{OUT}/hd.onnx"), ("cam", f"{OUT}/cam.onnx")]}

# ---- verify against PyTorch + pytorch_grad_cam on real scans ----
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
EPS = 1e-2
worst_p = worst_c = flips = 0
refs = {}
for f in sorted(glob.glob("/samples/*")):
    rgb = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (224, 224))
    xt = ((rgb / 255.0 - mean) / std).astype(np.float32).transpose(2, 0, 1)[None]
    with torch.no_grad():
        rp = torch.softmax(m(torch.from_numpy(xt)), 1)[0].numpy()
    ridx = int(rp.argmax())
    rcam = GradCAMPlusPlus(model=m, target_layers=[le.conv_head])(
        input_tensor=torch.from_numpy(xt), targets=[ClassifierOutputTarget(ridx)])[0]

    A, D, z, g = S["backbone"].run(None, {"image": xt})
    lg = S["head"].run(None, {"z": z, "g": g})[0][0]
    pr = np.exp(lg - lg.max())
    pr /= pr.sum()
    idx = int(pr.argmax())
    n = z.shape[1]
    Z = np.repeat(z, 2 * n, 0)
    Z[np.arange(n), np.arange(n)] += EPS
    Z[n + np.arange(n), np.arange(n)] -= EPS
    L = S["head"].run(None, {"z": Z, "g": np.repeat(g, 2 * n, 0)})[0][:, idx]
    gz = ((L[:n] - L[n:]) / (2 * EPS)).astype(np.float32)[None]
    cam = cv2.resize(S["cam"].run(None, {"A": A, "D": D, "gz": gz})[0][0], (224, 224))
    worst_p = max(worst_p, float(np.abs(pr - rp).max()))
    worst_c = max(worst_c, float(np.abs(cam - rcam).mean()))
    flips += int(idx != ridx)
    refs[os.path.basename(f)] = {"torch": [round(float(v), 4) for v in rp],
                                 "browser_model": [round(float(v), 4) for v in pr]}
print(f"vs PyTorch on {len(refs)} scans: max prob diff {worst_p:.4f}, mean heatmap diff {worst_c:.4f}, class flips {flips}")
json.dump(refs, open(f"{OUT}/refs.json", "w"), indent=1)
