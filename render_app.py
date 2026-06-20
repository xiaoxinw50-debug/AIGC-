from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from model_payload import ensure_model_artifacts
from origin_utils import load_bundle, predict_one


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ensure_model_artifacts(ROOT)
BINARY_BUNDLE = MODEL_DIR / "binary_model_bundle.joblib"
PLATFORM_BUNDLE = MODEL_DIR / "platform_model_bundle.joblib"
BINARY_TOP = MODEL_DIR / "binary_top_features.csv"
PLATFORM_TOP = MODEL_DIR / "platform_top_features.csv"
GENERATED_THRESHOLD = float(os.environ.get("AIGC_GENERATED_THRESHOLD", "0.75"))
REAL_THRESHOLD = float(os.environ.get("AIGC_REAL_THRESHOLD", "0.45"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

binary_bundle = load_bundle(BINARY_BUNDLE)
platform_bundle = load_bundle(PLATFORM_BUNDLE)

FEATURE_LABEL_MAP = {
    "is_png": "PNG 导出格式",
    "is_jpeg": "JPEG 导出格式",
    "is_webp": "WEBP 导出格式",
    "exif_present": "EXIF 是否存在",
    "info_key_count": "信息键数量",
    "band_count": "颜色通道数量",
    "has_alpha": "是否带 alpha 通道",
    "orig_width": "原始宽度",
    "orig_height": "原始高度",
    "aspect_ratio": "长宽比",
    "megapixels": "像素规模",
    "file_size_kb": "文件大小",
}


def read_top_features(path: Path, top_n: int = 6) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows[:top_n]:
        out.append(
            {
                "feature": FEATURE_LABEL_MAP.get(row.get("feature", ""), row.get("feature", "")),
                "importance": f"{float(row.get('importance', 0.0)):.3f}",
            }
        )
    return out


GLOBAL_BINARY_FEATURES = read_top_features(BINARY_TOP)
GLOBAL_PLATFORM_FEATURES = read_top_features(PLATFORM_TOP)


def platform_label_text(platform_id: str) -> str:
    mapping = {
        "PLT01": "文心一言",
        "PLT02": "通义千问",
        "PLT03": "即梦AI / 字节系文生图",
        "PLT05": "智谱GLM-Image / 清言相关文生图",
        "real": "真实图片",
        "uncertain": "需人工复核",
    }
    return mapping.get(platform_id, platform_id)


def format_signal_snapshot(snapshot: dict[str, float]) -> list[dict[str, str]]:
    return [
        {"label": "原始尺寸", "value": f"{int(snapshot['orig_width'])} × {int(snapshot['orig_height'])}"},
        {"label": "像素规模", "value": f"{snapshot['megapixels']:.3f} MP"},
        {"label": "文件大小", "value": f"{snapshot['file_size_kb']:.1f} KB"},
        {"label": "格式", "value": "PNG" if snapshot["is_png"] >= 0.5 else ("JPEG" if snapshot["is_jpeg"] >= 0.5 else ("WEBP" if snapshot["is_webp"] >= 0.5 else "其他"))},
        {"label": "Alpha 通道", "value": "是" if snapshot["has_alpha"] >= 0.5 else "否"},
        {"label": "EXIF", "value": "有" if snapshot["exif_present"] >= 0.5 else "无"},
        {"label": "信息键数量", "value": str(int(snapshot["info_key_count"]))},
    ]


HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIGC 图片识别与平台归因</title>
  <style>
    :root { --bg:#f5efe8; --panel:#fffdfb; --ink:#201916; --muted:#6e625b; --brand:#7f231d; --line:#e6d8cb; --accent:#c65a45; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; }
    .page { max-width:1120px; margin:0 auto; padding:28px 22px 40px; }
    .hero { display:grid; grid-template-columns:1.35fr .9fr; gap:18px; margin-bottom:18px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:22px; box-shadow:0 18px 40px rgba(32,25,22,.06); }
    h1 { margin:0 0 12px; font-size:42px; line-height:1.04; letter-spacing:-.02em; }
    h2 { margin:0 0 10px; font-size:18px; }
    p, li { color:var(--muted); }
    .eyebrow { color:var(--brand); font-weight:700; font-size:12px; letter-spacing:.08em; text-transform:uppercase; margin:0 0 10px; }
    .layout { display:grid; grid-template-columns:360px 1fr; gap:18px; margin-bottom:18px; }
    .drop { border:1px dashed #cfb39f; border-radius:16px; padding:20px; background:#fff; }
    input[type=file] { width:100%; margin:8px 0 14px; }
    button { appearance:none; border:0; background:var(--brand); color:#fff; border-radius:999px; padding:12px 16px; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.55; cursor:default; }
    .hint { font-size:14px; color:var(--muted); min-height:22px; }
    .preview { min-height:300px; display:flex; align-items:center; justify-content:center; background:#fbf7f2; border:1px solid var(--line); border-radius:16px; overflow:hidden; }
    .preview img { max-width:100%; max-height:420px; display:none; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .badge { display:inline-block; padding:6px 12px; border-radius:999px; background:#f3e1de; color:var(--brand); font-weight:700; }
    .metric { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:14px 0; }
    .metric div { border:1px solid var(--line); border-radius:14px; padding:12px 14px; background:#fff; }
    .metric span { display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }
    .prob { margin:0; display:grid; gap:10px; }
    .prob-row { display:grid; gap:6px; }
    .prob-head { display:flex; justify-content:space-between; gap:10px; }
    .track { height:8px; background:#efe5da; border-radius:999px; overflow:hidden; }
    .fill { height:100%; background:linear-gradient(90deg,var(--brand),var(--accent)); }
    .kv { display:grid; gap:10px; }
    .kv-row { display:flex; justify-content:space-between; gap:12px; padding:10px 12px; background:#fff; border:1px solid var(--line); border-radius:12px; }
    .notes { display:grid; gap:10px; }
    .notes div { padding:12px 14px; border-left:4px solid #d5b5ab; background:#fcf8f5; border-radius:10px; color:var(--muted); }
    .footer { margin-top:18px; display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    ul.clean { margin:0; padding-left:18px; }
    code { background:#f3e8df; border-radius:6px; padding:2px 6px; }
    @media (max-width: 980px) { .hero,.layout,.grid,.footer { grid-template-columns:1fr; } h1 { font-size:34px; } }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <article class="panel">
        <p class="eyebrow">AIGC 标识治理 Live Inference Demo</p>
        <h1>AI 图片识别与平台归因</h1>
        <p>先判断图片是否为 AI 生成，再在已采样平台 <code>文心一言</code>、<code>通义千问</code>、<code>即梦AI</code>、<code>智谱 GLM-Image</code> 中做来源归因。当前归因依据以文件层留痕为主，而非纯内容语义。</p>
      </article>
      <article class="panel">
        <h2>解释边界</h2>
        <ul class="clean">
          <li>平台归因主要利用尺寸、格式、Alpha、EXIF 和信息字段差异。</li>
          <li>当前结果应理解为采样平台条件下的归因原型。</li>
          <li>未采样平台、重编辑图、复杂传播图不在覆盖范围内。</li>
        </ul>
      </article>
    </section>

    <section class="layout">
      <article class="panel">
        <h2>上传检测</h2>
        <div class="drop">
          <input id="fileInput" type="file" accept="image/png,image/jpeg,image/webp">
          <button id="predictBtn" type="button">开始识别</button>
          <p id="statusText" class="hint">等待上传</p>
        </div>
        <h2 style="margin-top:18px;">文件层优先信号</h2>
        <ul class="clean">
          <li>图像格式：PNG / JPEG / WEBP</li>
          <li>原始尺寸、长宽比、像素规模</li>
          <li>是否存在 Alpha 通道</li>
          <li>EXIF 与信息键数量</li>
        </ul>
      </article>
      <article class="panel preview">
        <img id="previewImage" alt="预览图">
      </article>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>识别结果</h2>
        <div id="resultSummary" class="hint">尚未识别</div>
      </article>
      <article class="panel">
        <h2>平台概率</h2>
        <div id="platformProbabilities" class="hint">尚未识别</div>
      </article>
      <article class="panel">
        <h2>文件层信号</h2>
        <div id="signalSnapshot" class="hint">尚未识别</div>
      </article>
      <article class="panel">
        <h2>解释</h2>
        <div id="rationaleBox" class="hint">尚未识别</div>
      </article>
    </section>

    <section class="footer">
      <article class="panel">
        <h2>AI / real 阶段重点信号</h2>
        <ul class="clean">
          {% for row in global_binary_features %}
          <li>{{ row.feature }}（{{ row.importance }}）</li>
          {% endfor %}
        </ul>
      </article>
      <article class="panel">
        <h2>平台归因阶段重点信号</h2>
        <ul class="clean">
          {% for row in global_platform_features %}
          <li>{{ row.feature }}（{{ row.importance }}）</li>
          {% endfor %}
        </ul>
      </article>
    </section>
  </main>
  <script>
    const fileInput = document.getElementById('fileInput');
    const predictBtn = document.getElementById('predictBtn');
    const statusText = document.getElementById('statusText');
    const previewImage = document.getElementById('previewImage');
    const resultSummary = document.getElementById('resultSummary');
    const platformProbabilities = document.getElementById('platformProbabilities');
    const signalSnapshot = document.getElementById('signalSnapshot');
    const rationaleBox = document.getElementById('rationaleBox');
    const PLATFORM_NAME_MAP = { PLT01:'文心一言', PLT02:'通义千问', PLT03:'即梦AI / 字节系文生图', PLT05:'智谱GLM-Image / 清言相关文生图', real:'真实图片' };

    function setStatus(text) { statusText.textContent = text; }
    function esc(value) {
      return String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('\"','&quot;').replaceAll(\"'\", '&#39;');
    }

    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      resultSummary.innerHTML = '尚未识别';
      platformProbabilities.innerHTML = '尚未识别';
      signalSnapshot.innerHTML = '尚未识别';
      rationaleBox.innerHTML = '尚未识别';
      if (!file) { previewImage.style.display = 'none'; setStatus('等待上传'); return; }
      const reader = new FileReader();
      reader.onload = event => { previewImage.src = event.target.result; previewImage.style.display = 'block'; };
      reader.readAsDataURL(file);
      setStatus('已选择 ' + file.name);
    });

    async function predict() {
      const file = fileInput.files[0];
      if (!file) { setStatus('请先上传图片'); return; }
      predictBtn.disabled = true;
      setStatus('识别中...');
      try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/predict', { method:'POST', body:formData });
        const data = await response.json();
        if (!response.ok || data.status !== 'ok') throw new Error(data.message || '识别失败');
        renderResult(data.result);
        setStatus('识别完成');
      } catch (error) {
        setStatus('识别失败：' + error.message);
      } finally {
        predictBtn.disabled = false;
      }
    }

    function renderResult(result) {
      const isReal = result.binary_label === 'real';
      resultSummary.innerHTML = `
        <span class="badge">${isReal ? '真实图片' : 'AI 生成'}</span>
        <div class="metric">
          <div><span>AI 生成概率</span><strong>${(result.generated_probability * 100).toFixed(2)}%</strong></div>
          <div><span>最终判断</span><strong>${esc(result.platform_label_text)}</strong></div>
        </div>
        <div><strong>平台来源：</strong>${esc(result.platform_label_text)}</div>
      `;
      platformProbabilities.innerHTML = '<div class="prob">' + Object.entries(result.platform_probabilities).sort((a,b)=>b[1]-a[1]).map(([label, prob]) => `
        <div class="prob-row">
          <div class="prob-head"><span>${esc(PLATFORM_NAME_MAP[label] || label)}</span><strong>${(prob * 100).toFixed(2)}%</strong></div>
          <div class="track"><div class="fill" style="width:${(prob * 100).toFixed(2)}%"></div></div>
        </div>`).join('') + '</div>';
      signalSnapshot.innerHTML = '<div class="kv">' + result.signal_snapshot.map(item => `<div class="kv-row"><span>${esc(item.label)}</span><strong>${esc(item.value)}</strong></div>`).join('') + '</div>';
      rationaleBox.innerHTML = '<div class="notes">' + result.rationale.map(line => `<div>${esc(line)}</div>`).join('') + '</div>';
    }

    predictBtn.addEventListener('click', predict);
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(
        HTML,
        global_binary_features=GLOBAL_BINARY_FEATURES,
        global_platform_features=GLOBAL_PLATFORM_FEATURES,
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict_api():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "missing file"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"status": "error", "message": "empty filename"}), 400
    raw = file.read()
    if not raw:
        return jsonify({"status": "error", "message": "empty file"}), 400
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        return jsonify({"status": "error", "message": "unsupported or broken image"}), 400

    suffix = Path(file.filename).suffix or ".png"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            temp_path = Path(tmp.name)
        prediction = predict_one(
            temp_path,
            binary_bundle,
            platform_bundle,
            generated_threshold=GENERATED_THRESHOLD,
            real_threshold=REAL_THRESHOLD,
        )
    finally:
        if temp_path and temp_path.exists():
            os.unlink(temp_path)

    return jsonify(
        {
            "status": "ok",
            "result": {
                "binary_label": prediction["pred_binary_label"],
                "generated_probability": prediction["pred_prob_generated"],
                "platform_label": prediction["pred_final_label"],
                "platform_label_text": platform_label_text(prediction["pred_final_label"]),
                "platform_probabilities": prediction["pred_platform_probabilities"],
                "signal_snapshot": format_signal_snapshot(prediction["feature_snapshot"]),
                "rationale": [
                    f"当前模型先判断为 {prediction['pred_binary_label']}，AI 生成概率为 {prediction['pred_prob_generated']:.4f}。",
                    "平台归因当前主要依赖文件层和导出链路信号，而不是纯语义内容。",
                    "当前结果应理解为已采样平台条件下的归因原型，不应表述成开放世界稳定来源鉴定。",
                ],
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    app.run(host="0.0.0.0", port=port, debug=False)
