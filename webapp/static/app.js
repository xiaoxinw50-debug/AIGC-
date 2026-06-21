const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const batchPredictBtn = document.getElementById('batchPredictBtn');
const statusText = document.getElementById('statusText');
const previewImage = document.getElementById('previewImage');
const previewPlaceholder = document.getElementById('previewPlaceholder');
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
const fileMeta = document.getElementById('fileMeta');
const progressWrap = document.getElementById('progressWrap');
const progressBar = document.getElementById('progressBar');

const HISTORY_KEY = 'aigc-trace-history-v1';
const MAX_FILE_SIZE = 16 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const ACCEPTED_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp']);
const PLATFORM_NAME_MAP = {
  PLT01: '文心一言',
  PLT02: '通义千问',
  PLT03: '即梦AI / 字节系文生图',
  PLT05: '智谱GLM-Image / 清言相关文生图',
  real: '真实图片',
  uncertain: '需人工复核'
};

let lastBatchResults = [];
let lastSingleResult = null;
let lastSingleFileName = '';

function getSelectedFiles() {
  return Array.from(fileInput.files || []);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function platformDisplayName(code) {
  return PLATFORM_NAME_MAP[code] || code;
}

function asPercent(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '0.00%';
  return `${(number * 100).toFixed(digits)}%`;
}

function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(100, number * 100));
}

function setStatus(text, tone = 'neutral') {
  statusText.textContent = text;
  statusText.dataset.tone = tone;
}

function isStaticPreviewMode() {
  return window.location.protocol === 'file:';
}

function setProgress(value) {
  const clamped = Math.max(0, Math.min(100, value));
  progressBar.style.width = `${clamped}%`;
  progressWrap.classList.toggle('is-active', clamped > 0 && clamped < 100);
  progressWrap.setAttribute('aria-hidden', clamped > 0 && clamped < 100 ? 'false' : 'true');
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  batchPredictBtn.disabled = isLoading;
  predictBtn.textContent = isLoading ? '识别中...' : '识别当前图片';
  batchPredictBtn.textContent = isLoading ? '处理中...' : '批量识别全部';
}

function clearResult() {
  resultSummary.innerHTML = '尚未识别';
  platformProbabilities.innerHTML = '尚未识别';
  signalSnapshot.innerHTML = '尚未识别';
  rationaleBox.innerHTML = '尚未识别';
  lastSingleResult = null;
  lastSingleFileName = '';
  setProgress(0);
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

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function hasAcceptedExtension(fileName) {
  const lower = fileName.toLowerCase();
  return Array.from(ACCEPTED_EXTENSIONS).some(ext => lower.endsWith(ext));
}

function validateFiles(files) {
  if (!files.length) return { ok: true, message: '' };
  const invalidType = files.find(file => {
    const browserTypeOk = file.type ? ACCEPTED_TYPES.has(file.type) : false;
    return !browserTypeOk && !hasAcceptedExtension(file.name);
  });
  if (invalidType) {
    return { ok: false, message: `不支持 ${invalidType.name} 的文件类型，请上传 PNG、JPEG 或 WEBP。` };
  }
  const oversize = files.find(file => file.size > MAX_FILE_SIZE);
  if (oversize) {
    return { ok: false, message: `${oversize.name} 超过 16MB，请压缩后再上传。` };
  }
  return { ok: true, message: '' };
}

function renderFileMeta() {
  const files = getSelectedFiles();
  if (!files.length) {
    fileMeta.className = 'file-meta empty';
    fileMeta.innerHTML = '尚未选择文件';
    return;
  }
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  const first = files[0];
  const more = files.length > 1 ? `，另有 ${files.length - 1} 张` : '';
  fileMeta.className = 'file-meta';
  fileMeta.innerHTML = `
    <div>
      <strong>${escapeHtml(first.name)}${more}</strong>
      <span>${files.length} 张图片，合计 ${formatBytes(totalBytes)}</span>
    </div>
    <span class="file-meta-pill">${escapeHtml(first.type || first.name.split('.').pop() || 'image')}</span>
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

function decisionTone(row) {
  if (row.binary_label === 'real') return 'real';
  if (row.binary_label === 'uncertain') return 'uncertain';
  return 'ai';
}

function decisionAdvice(result) {
  if (result.binary_label === 'real') {
    return {
      title: '可作为真实图低风险结果',
      body: '当前不继续平台归因。若用于纠纷场景，仍建议保留原图文件、订单上下文和沟通记录。'
    };
  }
  if (result.binary_label === 'uncertain') {
    return {
      title: '进入人工复核，不硬判平台',
      body: '建议要求补充原始文件、多角度图片或订单证据。系统不会把边界图片直接判成某个平台生成。'
    };
  }
  return {
    title: '高置信 AI 生成，继续查看平台归因',
    body: '平台结果来自已采样四个平台的导出留痕差异，可作为初筛线索，不应表述为开放世界来源鉴定。'
  };
}

function decisionLabel(row) {
  return row.decision_text || (row.binary_label === 'real' ? '真实图片' : row.binary_label === 'uncertain' ? '需人工复核' : 'AI 生成');
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
            <strong>${escapeHtml(row.file_name)}</strong>
            <span>${escapeHtml(decisionLabel(row))}</span>
          </div>
          <div class="history-meta">
            <span>${escapeHtml(row.generated_probability_pct)}</span>
            <span>${escapeHtml(row.timestamp)}</span>
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
    decision_status: result.decision_status,
    decision_text: result.decision_text,
    platform_label: result.platform_label,
    platform_label_text: result.platform_label_text,
    generated_probability: result.generated_probability,
    generated_probability_pct: asPercent(result.generated_probability),
    timestamp: new Date().toLocaleString('zh-CN')
  };
}

function renderSingleReportHtml(fileName, result) {
  const advice = decisionAdvice(result);
  const platformRows = Object.entries(result.platform_probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .map(([label, prob]) => `
      <tr>
        <td>${escapeHtml(platformDisplayName(label))}</td>
        <td>${asPercent(prob)}</td>
      </tr>
    `)
    .join('');
  const platformBlock = platformRows
    ? `<table>
          <thead><tr><th>标签</th><th>概率</th></tr></thead>
          <tbody>${platformRows}</tbody>
        </table>`
    : '<p class="empty">当前未进入平台归因。只有 AI 生成概率达到阈值时，系统才输出四平台概率。</p>';
  const signalRows = (result.signal_snapshot || [])
    .map(item => `
      <tr>
        <td>${escapeHtml(item.label)}</td>
        <td>${escapeHtml(item.value)}</td>
      </tr>
    `)
    .join('');
  const rationaleRows = (result.rationale || [])
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
    .advice{padding:14px 16px;border-radius:14px;background:#fff9f5;border:1px solid rgba(151,16,13,.12);line-height:1.7}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px 12px;border-bottom:1px solid rgba(151,16,13,.08);text-align:left}
    th{color:#97100d}
    ul{margin:0;padding-left:20px;line-height:1.8}
    @media (max-width:900px){.grid{grid-template-columns:1fr}body{padding:18px}}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>AI 图片识别与平台归因单张报告</h1>
      <p>文件名：${escapeHtml(fileName)}</p>
      <div class="meta">
        <span class="pill">AI / real：${escapeHtml(decisionLabel(result))}</span>
        <span class="pill">最终标签：${escapeHtml(result.platform_label_text)}</span>
        <span class="pill">AI 概率：${asPercent(result.generated_probability)}</span>
        <span class="pill">导出时间：${escapeHtml(new Date().toLocaleString('zh-CN'))}</span>
      </div>
    </section>
    <section class="card advice">
      <h2>${escapeHtml(advice.title)}</h2>
      <p>${escapeHtml(advice.body)}</p>
    </section>
    <section class="grid">
      <article class="card">
        <h2>平台概率</h2>
        ${platformBlock}
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
  dropzone.classList.toggle('is-dragover', active);
}

function resetPreview() {
  previewImage.removeAttribute('src');
  previewImage.style.display = 'none';
  if (previewPlaceholder) {
    previewPlaceholder.style.display = 'grid';
  }
}

function renderPreview(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = event.target.result;
    previewImage.style.display = 'block';
    if (previewPlaceholder) {
      previewPlaceholder.style.display = 'none';
    }
  };
  reader.readAsDataURL(file);
}

function assignDroppedFiles(files) {
  if (!files.length) return;
  try {
    const dataTransfer = new DataTransfer();
    files.forEach(file => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
  } catch {
    setStatus('当前浏览器不支持拖拽批量赋值，请点击上传区域选择图片。', 'error');
  }
}

fileInput.addEventListener('change', () => {
  clearResult();
  lastBatchResults = [];
  renderBatchResults();
  updateQueueSummary();
  renderFileMeta();

  const selectedFiles = getSelectedFiles();
  const validation = validateFiles(selectedFiles);
  if (!validation.ok) {
    fileInput.value = '';
    renderFileMeta();
    updateQueueSummary();
    resetPreview();
    setStatus(validation.message, 'error');
    return;
  }

  const file = selectedFiles[0];
  if (!file) {
    resetPreview();
    setStatus('等待上传');
    return;
  }

  renderPreview(file);
  setStatus(`已选择 ${escapeHtml(file.name)}${selectedFiles.length > 1 ? ` 等 ${selectedFiles.length} 张图片` : ''}`, 'success');

  if (!isStaticPreviewMode() && selectedFiles.length > 1) {
    setStatus(`已选择 ${selectedFiles.length} 张图片，准备自动批量识别`, 'working');
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

dropzone.addEventListener('dragleave', () => {
  markDropzone(false);
});

dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  markDropzone(false);
  assignDroppedFiles(Array.from(event.dataTransfer?.files || []));
});

if (isStaticPreviewMode()) {
  setStatus('当前是静态预览模式：可以查看 UI，但识别功能需要先启动本地后端服务。', 'warning');
}

function renderResult(result) {
  const generatedPct = asPercent(result.generated_probability);
  const pointerLeft = clampPercent(result.generated_probability);
  const tone = decisionTone(result);
  const advice = decisionAdvice(result);
  const badgeText = decisionLabel(result);
  const thresholdText = result.thresholds
    ? `真实 ≤ ${(result.thresholds.real * 100).toFixed(0)}%，AI ≥ ${(result.thresholds.generated * 100).toFixed(0)}%`
    : '';
  const finalText = result.binary_label === 'generated'
    ? result.platform_label_text
    : badgeText;

  resultSummary.innerHTML = `
    <div class="result-card is-${tone}">
      <div class="result-head">
        <span class="result-badge is-${tone}">${escapeHtml(badgeText)}</span>
        <span class="result-subtle">${escapeHtml(thresholdText || '保守阈值已启用')}</span>
      </div>
      <div class="result-metrics">
        <div class="metric-tile">
          <span>AI 生成概率</span>
          <strong>${generatedPct}</strong>
        </div>
        <div class="metric-tile">
          <span>${result.binary_label === 'generated' ? '平台归因' : '处理结论'}</span>
          <strong>${escapeHtml(finalText)}</strong>
        </div>
      </div>
      <div class="probability-meter" aria-label="AI 生成概率区间">
        <div class="meter-track">
          <span class="meter-pointer" style="left:${pointerLeft}%"></span>
        </div>
        <div class="meter-labels">
          <span>真实区 ≤45%</span>
          <span>复核区</span>
          <span>AI 区 ≥75%</span>
        </div>
      </div>
      <div class="result-advice">
        <strong>${escapeHtml(advice.title)}</strong>
        <p>${escapeHtml(advice.body)}</p>
      </div>
    </div>
  `;

  const probabilityEntries = Object.entries(result.platform_probabilities || {});
  const probabilityRows = probabilityEntries
    .sort((a, b) => b[1] - a[1])
    .map(([label, prob]) => {
      const width = clampPercent(prob);
      return `
        <div class="prob-row">
          <div class="prob-head">
            <span>${escapeHtml(platformDisplayName(label))}</span>
            <strong>${asPercent(prob)}</strong>
          </div>
          <div class="prob-track"><div class="prob-fill" style="width:${width}%"></div></div>
        </div>
      `;
    })
    .join('');
  platformProbabilities.innerHTML = probabilityRows || `
    <div class="platform-gate">
      <strong>未进入平台归因</strong>
      <p>只有图片先被判为高置信 AI 生成时，系统才展示四平台概率；真实图和复核图不输出平台来源，避免制造过度结论。</p>
    </div>
  `;

  signalSnapshot.innerHTML = `
    <div class="signal-list">
      ${(result.signal_snapshot || []).map(item => `
        <div class="signal-item">
          <span>${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(item.value)}</strong>
        </div>
      `).join('')}
    </div>
  `;

  rationaleBox.innerHTML = `
    <div class="rationale-list">
      ${(result.rationale || []).map((line, index) => `
        <div><span>${index + 1}</span>${escapeHtml(line)}</div>
      `).join('')}
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

function renderBatchSummary() {
  const counts = lastBatchResults.reduce(
    (acc, row) => {
      acc[row.binary_label] = (acc[row.binary_label] || 0) + 1;
      return acc;
    },
    { generated: 0, real: 0, uncertain: 0 }
  );
  return `
    <div class="batch-summary">
      <div><strong>${lastBatchResults.length}</strong><span>已识别</span></div>
      <div><strong>${counts.generated || 0}</strong><span>AI 生成</span></div>
      <div><strong>${counts.real || 0}</strong><span>真实图片</span></div>
      <div><strong>${counts.uncertain || 0}</strong><span>人工复核</span></div>
    </div>
  `;
}

function renderBatchResults() {
  if (!lastBatchResults.length) {
    batchResults.innerHTML = '尚未执行批量识别';
    return;
  }
  batchResults.innerHTML = `
    ${renderBatchSummary()}
    <div class="table-wrap">
      <table class="result-table">
        <thead>
          <tr>
            <th>文件</th>
            <th>AI / real</th>
            <th>最终标签</th>
            <th>AI 概率</th>
          </tr>
        </thead>
        <tbody>
          ${lastBatchResults.map(row => `
            <tr>
              <td>${escapeHtml(row.file_name)}</td>
              <td><span class="table-badge is-${decisionTone(row)}">${escapeHtml(decisionLabel(row))}</span></td>
              <td>${escapeHtml(row.platform_label_text)}</td>
              <td>${escapeHtml(row.generated_probability_pct)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

predictBtn.addEventListener('click', async () => {
  if (isStaticPreviewMode()) {
    setStatus('当前是静态预览模式。请先运行 webapp/ai_platform_demo.py，再访问 http://127.0.0.1:8765', 'warning');
    return;
  }
  const files = getSelectedFiles();
  const validation = validateFiles(files);
  if (!validation.ok) {
    setStatus(validation.message, 'error');
    return;
  }
  const file = files[0];
  if (!file) {
    setStatus('请先上传图片', 'warning');
    return;
  }
  setLoading(true);
  setStatus('识别中：正在读取文件层留痕和图像统计特征。', 'working');
  setProgress(35);
  clearResult();

  try {
    const result = await requestPredict(file);
    setProgress(92);
    renderResult(result);
    lastSingleResult = result;
    lastSingleFileName = file.name;
    pushHistory(toHistoryRecord(file.name, result));
    setStatus('识别完成：结果已按结论、概率、证据和建议动作分层展示。', 'success');
  } catch (error) {
    setStatus(`识别失败：${error.message}`, 'error');
  } finally {
    setProgress(100);
    setTimeout(() => setProgress(0), 500);
    setLoading(false);
    updateQueueSummary();
  }
});

batchPredictBtn.addEventListener('click', async () => {
  if (isStaticPreviewMode()) {
    setStatus('当前是静态预览模式。请先运行 webapp/ai_platform_demo.py，再访问 http://127.0.0.1:8765', 'warning');
    return;
  }
  const files = getSelectedFiles();
  const validation = validateFiles(files);
  if (!validation.ok) {
    setStatus(validation.message, 'error');
    return;
  }
  if (!files.length) {
    setStatus('请先上传图片', 'warning');
    return;
  }
  setLoading(true);
  lastBatchResults = [];
  renderBatchResults();
  setProgress(5);

  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      setStatus(`批量识别中 ${index + 1}/${files.length}：${file.name}`, 'working');
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
      setProgress(((index + 1) / files.length) * 100);
    }
    setStatus(`批量识别完成，共 ${files.length} 张。可导出 CSV 用于答辩展示。`, 'success');
  } catch (error) {
    setStatus(`批量识别失败：${error.message}`, 'error');
  } finally {
    setTimeout(() => setProgress(0), 500);
    setLoading(false);
  }
});

clearHistoryBtn.addEventListener('click', () => {
  writeHistory([]);
  renderHistory();
  setStatus('历史记录已清空', 'success');
});

exportHistoryBtn.addEventListener('click', () => {
  const rows = readHistory();
  if (!rows.length) {
    setStatus('没有可导出的历史记录', 'warning');
    return;
  }
  downloadText(
    'aigc_trace_history.json',
    JSON.stringify(rows, null, 2),
    'application/json;charset=utf-8'
  );
  setStatus('历史记录已导出', 'success');
});

exportBatchCsvBtn.addEventListener('click', () => {
  if (!lastBatchResults.length) {
    setStatus('没有可导出的批量结果', 'warning');
    return;
  }
  const header = ['file_name', 'binary_label', 'decision_status', 'decision_text', 'platform_label', 'platform_label_text', 'generated_probability_pct'];
  const lines = [
    header.join(','),
    ...lastBatchResults.map(row => header.map(key => `"${String(row[key] ?? '').replaceAll('"', '""')}"`).join(','))
  ];
  downloadText('aigc_batch_results.csv', lines.join('\n'), 'text/csv;charset=utf-8');
  setStatus('批量结果已导出', 'success');
});

exportSingleReportBtn.addEventListener('click', () => {
  if (!lastSingleResult) {
    setStatus('没有可导出的单张识别结果', 'warning');
    return;
  }
  const html = renderSingleReportHtml(lastSingleFileName || 'single_result', lastSingleResult);
  const safeName = (lastSingleFileName || 'single_result').replace(/[^\w.-]+/g, '_');
  downloadText(`aigc_single_report_${safeName}.html`, html, 'text/html;charset=utf-8');
  setStatus('单张报告已导出', 'success');
});

updateQueueSummary();
renderFileMeta();
renderHistory();
