const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const batchPredictBtn = document.getElementById('batchPredictBtn');
const statusText = document.getElementById('statusText');
const previewImage = document.getElementById('previewImage');
const resultSummary = document.getElementById('resultSummary');
const platformProbabilities = document.getElementById('platformProbabilities');
const signalSnapshot = document.getElementById('signalSnapshot');
const rationaleBox = document.getElementById('rationaleBox');
const queueSummary = document.getElementById('queueSummary');
const batchResults = document.getElementById('batchResults');
const historyList = document.getElementById('historyList');
const exportHistoryBtn = document.getElementById('exportHistoryBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const exportBatchCsvBtn = document.getElementById('exportBatchCsvBtn');
const exportSingleReportBtn = document.getElementById('exportSingleReportBtn');
const dropzone = document.querySelector('.upload-dropzone');

const HISTORY_KEY = 'aigc-trace-history-v1';
const PLATFORM_NAME_MAP = {
  PLT01: '文心一言',
  PLT02: '通义千问',
  PLT03: '即梦AI / 字节系文生图',
  PLT05: '智谱GLM-Image / 清言相关文生图',
  real: '真实图片'
};
let lastBatchResults = [];
let lastSingleResult = null;
let lastSingleFileName = '';

function getSelectedFiles() {
  return Array.from(fileInput.files || []);
}

function setStatus(text) {
  statusText.textContent = text;
}

function isStaticPreviewMode() {
  return window.location.protocol === 'file:';
}

function clearResult() {
  resultSummary.innerHTML = '尚未识别';
  platformProbabilities.innerHTML = '尚未识别';
  signalSnapshot.innerHTML = '尚未识别';
  rationaleBox.innerHTML = '尚未识别';
  lastSingleResult = null;
  lastSingleFileName = '';
}

function updateQueueSummary() {
  const files = getSelectedFiles();
  const queueCount = files.length;
  const pendingCount = Math.max(queueCount - lastBatchResults.length, 0);
  queueSummary.innerHTML = `
    <div class="queue-chip">当前队列 ${queueCount}</div>
    <div class="queue-chip">待识别 ${queueCount === 0 ? 0 : pendingCount}</div>
  `;
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch {
    return [];
  }
}

function writeHistory(rows) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(rows));
}

function renderHistory() {
  const rows = readHistory();
  if (!rows.length) {
    historyList.innerHTML = '尚无历史记录';
    return;
  }
  historyList.innerHTML = `
    <div class="history-list">
      ${rows.slice(0, 8).map(row => `
        <div class="history-item">
          <div>
            <strong>${row.file_name}</strong>
            <span>${row.binary_label === 'real' ? '真实图片' : row.platform_label_text}</span>
          </div>
          <div class="history-meta">
            <span>${row.generated_probability_pct}</span>
            <span>${row.timestamp}</span>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function pushHistory(record) {
  const rows = readHistory();
  rows.unshift(record);
  writeHistory(rows.slice(0, 30));
  renderHistory();
}

function downloadText(filename, text, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function toHistoryRecord(fileName, result) {
  return {
    file_name: fileName,
    binary_label: result.binary_label,
    platform_label: result.platform_label,
    platform_label_text: result.platform_label_text,
    generated_probability: result.generated_probability,
    generated_probability_pct: `${(result.generated_probability * 100).toFixed(2)}%`,
    timestamp: new Date().toLocaleString('zh-CN')
  };
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function platformDisplayName(code) {
  return PLATFORM_NAME_MAP[code] || code;
}

function renderSingleReportHtml(fileName, result) {
  const platformRows = Object.entries(result.platform_probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([label, prob]) => `
      <tr>
        <td>${escapeHtml(platformDisplayName(label))}</td>
        <td>${(prob * 100).toFixed(2)}%</td>
      </tr>
    `)
    .join('');
  const signalRows = result.signal_snapshot
    .map(item => `
      <tr>
        <td>${escapeHtml(item.label)}</td>
        <td>${escapeHtml(item.value)}</td>
      </tr>
    `)
    .join('');
  const rationaleRows = result.rationale
    .map(line => `<li>${escapeHtml(line)}</li>`)
    .join('');

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>单张识别报告 - ${escapeHtml(fileName)}</title>
  <style>
    body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#fff7f3;color:#2b1b18;margin:0;padding:32px}
    .wrap{max-width:980px;margin:0 auto}
    .hero,.card{background:#fff;border:1px solid rgba(151,16,13,.1);border-radius:18px;box-shadow:0 18px 40px rgba(47,22,16,.08);padding:24px;margin-bottom:18px}
    h1,h2{margin:0 0 12px}
    .meta{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
    .pill{padding:8px 12px;border-radius:999px;background:rgba(151,16,13,.08);color:#97100d;font-weight:700}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px 12px;border-bottom:1px solid rgba(151,16,13,.08);text-align:left}
    th{color:#97100d}
    ul{margin:0;padding-left:20px;line-height:1.8}
    @media (max-width:900px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>AI 图片识别与平台归因单张报告</h1>
      <p>文件名：${escapeHtml(fileName)}</p>
      <div class="meta">
        <span class="pill">AI/real：${escapeHtml(result.binary_label)}</span>
        <span class="pill">最终标签：${escapeHtml(result.platform_label_text)}</span>
        <span class="pill">AI 概率：${(result.generated_probability * 100).toFixed(2)}%</span>
        <span class="pill">导出时间：${escapeHtml(new Date().toLocaleString('zh-CN'))}</span>
      </div>
    </section>
    <section class="grid">
      <article class="card">
        <h2>平台概率</h2>
        <table>
          <thead><tr><th>标签</th><th>概率</th></tr></thead>
          <tbody>${platformRows}</tbody>
        </table>
      </article>
      <article class="card">
        <h2>文件层信号</h2>
        <table>
          <thead><tr><th>信号</th><th>取值</th></tr></thead>
          <tbody>${signalRows}</tbody>
        </table>
      </article>
    </section>
    <section class="card">
      <h2>解释</h2>
      <ul>${rationaleRows}</ul>
    </section>
  </div>
</body>
</html>`;
}

function markDropzone(active) {
  if (active) {
    dropzone.classList.add('is-dragover');
  } else {
    dropzone.classList.remove('is-dragover');
  }
}

fileInput.addEventListener('change', () => {
  clearResult();
  lastBatchResults = [];
  renderBatchResults();
  updateQueueSummary();
  const file = getSelectedFiles()[0];
  if (!file) {
    previewImage.style.display = 'none';
    setStatus('等待上传');
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.style.display = 'block';
  };
  reader.readAsDataURL(file);
  const selectedFiles = getSelectedFiles();
  setStatus(`已选择 ${file.name}${selectedFiles.length > 1 ? ` 等 ${selectedFiles.length} 张图片` : ''}`);
  if (!isStaticPreviewMode() && selectedFiles.length > 1) {
    setStatus(`已选择 ${selectedFiles.length} 张图片，准备自动批量识别`);
    setTimeout(() => {
      if (getSelectedFiles().length > 1 && !batchPredictBtn.disabled) {
        batchPredictBtn.click();
      }
    }, 250);
  }
});

['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    markDropzone(true);
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, () => {
    markDropzone(false);
  });
});

if (isStaticPreviewMode()) {
  setStatus('当前是静态预览模式：可以查看 UI，但识别功能需要先启动本地后端服务。');
}

function renderResult(result) {
  const generatedPct = (result.generated_probability * 100).toFixed(2);
  const isReal = result.binary_label === 'real';
  resultSummary.innerHTML = `
    <div class="result-card">
      <span class="result-badge ${isReal ? 'is-real' : ''}">${isReal ? '真实图片' : 'AI 生成'}</span>
      <div class="result-metrics">
        <div class="metric-tile">
          <span>AI 生成概率</span>
          <strong>${generatedPct}%</strong>
        </div>
        <div class="metric-tile">
          <span>最终判断</span>
          <strong>${result.platform_label_text}</strong>
        </div>
      </div>
      <div><strong>平台来源：</strong>${escapeHtml(result.platform_label_text)}</div>
    </div>
  `;

  const probabilityRows = Object.entries(result.platform_probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([label, prob]) => `
      <div class="prob-row">
        <div class="prob-head"><span>${escapeHtml(platformDisplayName(label))}</span><strong>${(prob * 100).toFixed(2)}%</strong></div>
        <div class="prob-track"><div class="prob-fill" style="width:${(prob * 100).toFixed(2)}%"></div></div>
      </div>
    `)
    .join('');
  platformProbabilities.innerHTML = probabilityRows;

  signalSnapshot.innerHTML = `
    <div class="signal-list">
      ${result.signal_snapshot.map(item => `<div class="signal-item"><span>${item.label}</span><strong>${item.value}</strong></div>`).join('')}
    </div>
  `;

  rationaleBox.innerHTML = `
    <div class="rationale-list">
      ${result.rationale.map(line => `<div>${line}</div>`).join('')}
    </div>
  `;
}

async function requestPredict(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/predict', {
    method: 'POST',
    body: formData
  });
  const data = await response.json();
  if (!response.ok || data.status !== 'ok') {
    throw new Error(data.message || '识别失败');
  }
  return data.result;
}

function renderBatchResults() {
  if (!lastBatchResults.length) {
    batchResults.innerHTML = '尚未执行批量识别';
    return;
  }
  batchResults.innerHTML = `
    <div class="table-wrap">
      <table class="result-table">
        <thead>
          <tr>
            <th>文件</th>
            <th>AI/real</th>
            <th>最终标签</th>
            <th>AI 概率</th>
          </tr>
        </thead>
        <tbody>
          ${lastBatchResults.map(row => `
            <tr>
              <td>${row.file_name}</td>
              <td>${row.binary_label}</td>
              <td>${row.platform_label_text}</td>
              <td>${row.generated_probability_pct}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

predictBtn.addEventListener('click', async () => {
  if (isStaticPreviewMode()) {
    setStatus('当前是静态预览模式。请先运行 webapp/ai_platform_demo.py，再访问 http://127.0.0.1:8765');
    return;
  }
  const file = getSelectedFiles()[0];
  if (!file) {
    setStatus('请先上传图片');
    return;
  }
  predictBtn.disabled = true;
  batchPredictBtn.disabled = true;
  setStatus('识别中...');
  clearResult();

  try {
    const result = await requestPredict(file);
    renderResult(result);
    lastSingleResult = result;
    lastSingleFileName = file.name;
    pushHistory(toHistoryRecord(file.name, result));
    setStatus('识别完成');
  } catch (error) {
    setStatus(`识别失败：${error.message}`);
  } finally {
    predictBtn.disabled = false;
    batchPredictBtn.disabled = false;
  }
});

batchPredictBtn.addEventListener('click', async () => {
  if (isStaticPreviewMode()) {
    setStatus('当前是静态预览模式。请先运行 webapp/ai_platform_demo.py，再访问 http://127.0.0.1:8765');
    return;
  }
  const files = getSelectedFiles();
  if (!files.length) {
    setStatus('请先上传图片');
    return;
  }
  predictBtn.disabled = true;
  batchPredictBtn.disabled = true;
  lastBatchResults = [];
  renderBatchResults();

  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      setStatus(`批量识别中 ${index + 1}/${files.length}：${file.name}`);
      const result = await requestPredict(file);
      if (index === 0) {
        renderResult(result);
        lastSingleResult = result;
        lastSingleFileName = file.name;
      }
      const record = toHistoryRecord(file.name, result);
      lastBatchResults.push(record);
      pushHistory(record);
      renderBatchResults();
      updateQueueSummary();
    }
    setStatus(`批量识别完成，共 ${files.length} 张`);
  } catch (error) {
    setStatus(`批量识别失败：${error.message}`);
  } finally {
    predictBtn.disabled = false;
    batchPredictBtn.disabled = false;
  }
});

clearHistoryBtn.addEventListener('click', () => {
  writeHistory([]);
  renderHistory();
});

exportHistoryBtn.addEventListener('click', () => {
  const rows = readHistory();
  if (!rows.length) {
    setStatus('没有可导出的历史记录');
    return;
  }
  downloadText(
    'aigc_trace_history.json',
    JSON.stringify(rows, null, 2),
    'application/json;charset=utf-8'
  );
  setStatus('历史记录已导出');
});

exportBatchCsvBtn.addEventListener('click', () => {
  if (!lastBatchResults.length) {
    setStatus('没有可导出的批量结果');
    return;
  }
  const header = ['file_name', 'binary_label', 'platform_label', 'platform_label_text', 'generated_probability_pct'];
  const lines = [
    header.join(','),
    ...lastBatchResults.map(row => header.map(key => `"${String(row[key] ?? '').replaceAll('"', '""')}"`).join(','))
  ];
  downloadText('aigc_batch_results.csv', lines.join('\n'), 'text/csv;charset=utf-8');
  setStatus('批量结果已导出');
});

exportSingleReportBtn.addEventListener('click', () => {
  if (!lastSingleResult) {
    setStatus('没有可导出的单张识别结果');
    return;
  }
  const html = renderSingleReportHtml(lastSingleFileName || 'single_result', lastSingleResult);
  const safeName = (lastSingleFileName || 'single_result').replace(/[^\w.-]+/g, '_');
  downloadText(`aigc_single_report_${safeName}.html`, html, 'text/html;charset=utf-8');
  setStatus('单张报告已导出');
});

updateQueueSummary();
renderHistory();
