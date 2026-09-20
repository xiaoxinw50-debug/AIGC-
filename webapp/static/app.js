'use strict';

const byId = id => document.getElementById(id);
const fileInput = byId('fileInput');
const predictBtn = byId('predictBtn');
const clearFilesBtn = byId('clearFilesBtn');
const cancelBtn = byId('cancelBtn');
const statusText = byId('statusText');
const statusSpinner = byId('statusSpinner');
const previewImage = byId('previewImage');
const previewPlaceholder = byId('previewPlaceholder');
const previewCaption = byId('previewCaption');
const imageDimensions = byId('imageDimensions');
const fileMeta = byId('fileMeta');
const queueSummary = byId('queueSummary');
const progressWrap = byId('progressWrap');
const progressBar = byId('progressBar');
const resultsSection = byId('results');
const resultsTitle = byId('resultsTitle');
const reportMeta = byId('reportMeta');
const resultSummary = byId('resultSummary');
const platformProbabilities = byId('platformProbabilities');
const signalSnapshot = byId('signalSnapshot');
const rationaleBox = byId('rationaleBox');
const gateDetails = byId('gateDetails');
const usageGuide = byId('usageGuide');
const evidenceDetails = byId('evidenceDetails');
const usageDetails = byId('usageDetails');
const batchSection = byId('batchSection');
const batchResults = byId('batchResults');
const exportSingleReportBtn = byId('exportSingleReportBtn');
const exportBatchCsvBtn = byId('exportBatchCsvBtn');
const selectedPolicyLabel = byId('selectedPolicyLabel');
const helpDialog = byId('helpDialog');
const dropzone = document.querySelector('.upload-dropzone');
const policyInputs = Array.from(document.querySelectorAll('input[name="policyProfile"]'));
const MAX_FILE_SIZE = 16 * 1024 * 1024;
const MAX_BATCH_SIZE = 20;
const REQUEST_TIMEOUT_MS = 90000;
const PLATFORM_NAME_MAP = { PLT01: '文心一言', PLT02: '通义千问', PLT03: '即梦AI', PLT05: '智谱 GLM-Image' };
const POLICY_NAMES = { operational_low_false_ai: '低误判优先', balanced_review: '均衡复核', high_risk_after_sales_review: '高风险线索筛查' };
let policyData = { profiles: [], default_id: 'operational_low_false_ai' };
try { policyData = JSON.parse(byId('policyProfileData').textContent); } catch { /* Default policy remains usable when optional metadata is missing. */ }
let selectedFiles = [];
let lastSingleResult = null;
let lastSingleFileName = '';
let lastBatchResults = [];
let activeFile = null;
let previewUrl = '';
let isBusy = false;
let runRevision = 0;
let currentController = null;
const resultCache = new Map();
const completedFiles = new Set();
const failures = new Map();

function escapeHtml(value) {
  return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}
function asPercent(value, digits = 2) {
  return value != null && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : '--';
}
function platformDisplayName(code) { return PLATFORM_NAME_MAP[code] || '暂无法归因'; }
function platformAttributionAccepted(result) {
  return result.binary_label === 'generated' && result.platform_accepted === true && Object.hasOwn(PLATFORM_NAME_MAP, result.platform_label);
}
function decisionLabel(result) {
  return { real: '更接近真实图', generated: 'AI 生成倾向较高', uncertain: '需要进一步核验' }[result.binary_label] || '检测未完成';
}
function sourceLabel(result) {
  if (result.binary_label !== 'generated') return '未进行来源归因';
  return platformAttributionAccepted(result) ? platformDisplayName(result.platform_label) : '暂无法归因';
}
function decisionTone(result) { return { real: 'real', generated: 'ai', uncertain: 'uncertain' }[result.binary_label] || 'uncertain'; }
function getSelectedFiles() { return selectedFiles; }
function getSelectedPolicyProfile() { return policyInputs.find(input => input.checked)?.value || policyData.default_id; }
function getSelectedPolicyName() { return POLICY_NAMES[getSelectedPolicyProfile()] || '默认策略'; }
function isStaticPreviewMode() { return window.location.protocol === 'file:'; }
function formatBytes(bytes) { return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(2)} MB`; }
function validateFiles(files) {
  if (files.length > MAX_BATCH_SIZE) return { ok: false, message: `每批最多 20 张，请减少图片后重试。` };
  const invalid = files.find(file => !['image/png', 'image/jpeg', 'image/webp'].includes(file.type) && !(!file.type && /\.(png|jpe?g|webp)$/i.test(file.name)));
  if (invalid) return { ok: false, message: `${invalid.name} 格式不支持，请选择 PNG、JPEG 或 WEBP。` };
  const oversize = files.find(file => file.size > MAX_FILE_SIZE);
  if (oversize) return { ok: false, message: `${oversize.name} 超过 16 MB，请调整文件后重试。` };
  return { ok: true };
}
function setStatus(text, tone = 'neutral') { statusText.textContent = text; statusText.dataset.tone = tone; }
function updateQueueSummary() {
  queueSummary.hidden = !selectedFiles.length;
  const remaining = selectedFiles.filter(file => !completedFiles.has(file)).length;
  queueSummary.textContent = `${selectedFiles.length} 张图片 / 已完成 ${completedFiles.size} / 待检测 ${remaining}`;
}
function setProgress(done, total) {
  progressWrap.hidden = total <= 1 || !isBusy;
  const percent = total ? Math.round(done / total * 100) : 0;
  progressBar.style.width = `${percent}%`;
  progressWrap.setAttribute('aria-valuenow', String(percent));
}
function updateControls() {
  const hasFiles = selectedFiles.length > 0;
  predictBtn.disabled = isBusy || !hasFiles || isStaticPreviewMode();
  clearFilesBtn.disabled = isBusy || !hasFiles;
  clearFilesBtn.hidden = isBusy || !hasFiles;
  cancelBtn.hidden = !isBusy;
  cancelBtn.disabled = !isBusy || currentController?.signal.aborted === true;
  fileInput.disabled = isBusy;
  policyInputs.forEach(input => { input.disabled = isBusy; });
  statusSpinner.hidden = !isBusy;
  dropzone.classList.toggle('is-busy', isBusy);
  exportSingleReportBtn.disabled = isBusy || !lastSingleResult;
  exportBatchCsvBtn.disabled = isBusy || !lastBatchResults.length;
  selectedPolicyLabel.textContent = getSelectedPolicyName();
  const pending = selectedFiles.filter(file => !completedFiles.has(file)).length;
  predictBtn.innerHTML = isBusy ? '正在检测' : `${completedFiles.size && pending ? '重试未完成' : completedFiles.size && !pending ? '重新检测' : '开始检测'}${selectedFiles.length > 1 ? `（${pending || selectedFiles.length} 张）` : ''} <span aria-hidden="true">↗</span>`;
  resultsSection.setAttribute('aria-busy', String(isBusy));
}
function resetPreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = '';
  previewImage.removeAttribute('src');
  previewImage.hidden = true;
  previewPlaceholder.hidden = false;
  previewCaption.textContent = '待上传';
  imageDimensions.textContent = '原图比例预览';
}
function renderPreview(file) {
  if (activeFile !== file || !previewUrl) {
    resetPreview();
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
  }
  activeFile = file;
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
  previewCaption.textContent = file.name;
  previewImage.onload = () => { if (activeFile === file) imageDimensions.textContent = `${previewImage.naturalWidth} × ${previewImage.naturalHeight} px`; };
  previewImage.onerror = () => {
    if (activeFile !== file) return;
    previewImage.hidden = true;
    previewPlaceholder.hidden = false;
    setStatus('图片无法预览，请检查文件是否完整或改用其他图片。', 'error');
  };
}
function clearResult() {
  resultsSection.hidden = true;
  batchSection.hidden = true;
  evidenceDetails.open = false;
  usageDetails.open = false;
  lastSingleResult = null;
  lastSingleFileName = '';
  reportMeta.textContent = '';
  resultSummary.innerHTML = '';
  platformProbabilities.innerHTML = '';
}
function clearSelection() {
  runRevision += 1;
  currentController?.abort();
  selectedFiles = [];
  fileInput.value = '';
  resultCache.clear(); completedFiles.clear(); failures.clear();
  lastBatchResults = [];
  activeFile = null;
  resetPreview(); clearResult();
  fileMeta.className = 'file-meta empty';
  fileMeta.textContent = '尚未选择图片';
  updateQueueSummary(); updateControls();
  setStatus('选图仅在本地预览，点击检测后才提交分析。');
}
function selectFiles(files) {
  if (isBusy) return;
  const validation = validateFiles(files);
  if (!validation.ok) { fileInput.value = ''; setStatus(validation.message, 'error'); return; }
  clearSelection();
  selectedFiles = files;
  if (!files.length) return;
  const first = files[0];
  fileMeta.className = 'file-meta';
  fileMeta.innerHTML = `<strong>${escapeHtml(first.name)}${files.length > 1 ? `，另有 ${files.length - 1} 张` : ''}</strong><span>${formatBytes(files.reduce((sum, file) => sum + file.size, 0))}</span>`;
  renderPreview(first);
  updateQueueSummary(); updateControls();
  setStatus(isStaticPreviewMode() ? '当前为界面预览，在线服务连接后可开始检测。' : `已选择 ${files.length} 张图片，确认后点击开始检测。`);
}
function resultUsageGuide(result) {
  return [
    { title: '核验建议', body: result.binary_label === 'real' ? '低评分不构成真实性保证。敏感场景中，请结合拍摄原件与上下文继续核验。' : '建议补充原始文件、生成或拍摄记录及相关上下文，由人工完成最终核验。' },
    { title: '使用边界', body: platformAttributionAccepted(result) ? '来源判断限定于当前采样范围，不是生成平台的独立认证，也不能单独用于定责。' : '未接受的平台候选只提供排查线索，不应被当作已经确认的来源。' }
  ];
}
function technicalRationale(result) {
  return (result.rationale || []).map(line => String(line).replaceAll('·', ' ').replaceAll('默认展示档', '当前策略').replaceAll('运营低误伤', '低误判优先'));
}
function scoreEntries(result) {
  if (result.binary_label !== 'generated') return [];
  return Object.entries(result.platform_probabilities || {}).filter(([code, score]) => Object.hasOwn(PLATFORM_NAME_MAP, code) && typeof score === 'number' && Number.isFinite(score) && score >= 0 && score <= 1).sort((a, b) => b[1] - a[1]);
}
function sourceEvidence(result) {
  const thresholds = result.platform_open_set_thresholds || {};
  return [
    ['平台候选评分', asPercent(result.platform_confidence), `接受阈值 ${asPercent(thresholds.confidence)}`],
    ['前两名分差', asPercent(result.platform_margin), `接受阈值 ${asPercent(thresholds.margin)}`],
    ['已知来源评分', asPercent(result.platform_knownness_score), `接受阈值 ${asPercent(thresholds.knownness)}`],
    ['特征漂移评分', result.platform_drift_score != null && Number.isFinite(Number(result.platform_drift_score)) ? Number(result.platform_drift_score).toFixed(3) : '--', '反映与参考分布的差异'],
    ['漂移检查', result.platform_drift_flag === true ? '触发拒绝' : result.platform_drift_flag === false ? '未触发' : '未提供', '触发时不接受平台归因'],
    ['模型候选一致性', result.platform_model_agreement === true ? '一致' : result.platform_model_agreement === false ? '不一致' : '未提供', thresholds.require_agreement === true ? '当前要求模型候选一致' : thresholds.require_agreement === false ? '当前未要求一致' : '当前要求未提供'],
  ];
}
function renderResult(result) {
  resultsSection.hidden = false;
  const accepted = platformAttributionAccepted(result);
  const tone = decisionTone(result);
  const score = Number(result.generated_probability);
  reportMeta.textContent = result._client ? `${result._client.fileName} / ${new Date(result._client.createdAt).toLocaleString('zh-CN')}` : '';
  resultSummary.innerHTML = `<div class="result-card is-${tone}"><div class="verdict-top"><span class="result-kicker">图像判断</span><span class="result-status">检测完成</span></div><h3 class="verdict-title">${escapeHtml(decisionLabel(result))}</h3><div class="verdict-data"><div class="score-block"><span>AI 生成评分</span><strong>${asPercent(score)}</strong></div><div class="source-block"><span>来源判断</span><strong>${escapeHtml(sourceLabel(result))}</strong></div></div><div class="score-track" aria-label="AI 生成评分 ${asPercent(score)}"><span style="width:${Math.max(0, Math.min(100, score * 100))}%"></span></div><p class="result-caption">${result.binary_label === 'uncertain' ? '当前信号处于复核区间，暂不作确定判断。' : result.binary_label === 'real' ? '当前特征更接近真实图片，不继续推断生成平台。' : accepted ? '来源证据满足当前模型的接受条件。' : '生成评分较高，但来源证据尚不足以接受平台归因。'}</p></div>`;
  const entries = scoreEntries(result);
  platformProbabilities.innerHTML = entries.length ? `<div class="platform-decision ${accepted ? 'is-accepted' : 'is-rejected'}"><strong>${accepted ? `来源候选已接受：${escapeHtml(sourceLabel(result))}` : '来源证据不足，暂无法归因'}</strong></div><div class="probability-list">${entries.map(([code, probability]) => `<div class="prob-row"><div class="prob-head"><span>${escapeHtml(platformDisplayName(code))}</span><strong>${asPercent(probability)}</strong></div><div class="prob-track"><span class="prob-fill" style="width:${Number(probability) * 100}%"></span></div></div>`).join('')}</div><p class="candidate-note">候选分数不代表来源已被独立确认。</p>` : '<div class="source-empty"><span class="empty-rule"></span><strong>未进行来源归因</strong><p>仅对进入 AI 高评分区的图片进一步分析来源。</p></div>';
  signalSnapshot.innerHTML = `<dl class="signal-list">${(result.signal_snapshot || []).map(item => `<div class="signal-item"><dt>${escapeHtml(item.label)}</dt><dd>${escapeHtml(item.value)}</dd></div>`).join('')}</dl>`;
  rationaleBox.innerHTML = `<ol class="rationale-list">${technicalRationale(result).map(line => `<li>${escapeHtml(line)}</li>`).join('')}</ol>`;
  gateDetails.innerHTML = result.binary_label === 'generated' ? `<h3>来源接受条件与诊断</h3><dl class="gate-grid">${sourceEvidence(result).map(([label, value, note]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd><small>${escapeHtml(note)}</small></div>`).join('')}</dl><p>${escapeHtml((result.platform_rejection_explanations || []).join('；') || (accepted ? '当前服务返回来源归因已接受。' : '当前服务未提供通过接受条件的来源结论。'))}</p>` : '';
  usageGuide.innerHTML = resultUsageGuide(result).map(item => `<div><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.body)}</p></div>`).join('');
}
function renderLoading() {
  resultsSection.hidden = false;
  evidenceDetails.open = false;
  usageDetails.open = false;
  signalSnapshot.innerHTML = '';
  rationaleBox.innerHTML = '';
  gateDetails.innerHTML = '';
  usageGuide.innerHTML = '';
  reportMeta.textContent = '正在读取图像与文件信息';
  resultSummary.innerHTML = '<div class="skeleton" aria-hidden="true"><span></span><span></span><span></span></div>';
  platformProbabilities.innerHTML = '<div class="skeleton" aria-hidden="true"><span></span><span></span><span></span></div>';
}
function renderBatchResults() {
  batchSection.hidden = selectedFiles.length <= 1 || (!lastBatchResults.length && !failures.size && !isBusy);
  const rows = selectedFiles.map((file, index) => {
    const result = resultCache.get(file);
    const error = failures.get(file);
    return `<tr ${activeFile === file ? 'class="is-selected"' : ''}><td><button class="batch-select" type="button" data-file-index="${index}" ${isBusy || !result ? 'disabled' : ''}>${escapeHtml(file.name)}</button></td><td>${result ? `<span class="table-badge is-${decisionTone(result)}">${escapeHtml(decisionLabel(result))}</span>` : error ? '<span class="table-badge is-error">检测失败</span>' : '等待检测'}</td><td>${result ? escapeHtml(sourceLabel(result)) : error ? escapeHtml(error) : '--'}</td><td>${result ? asPercent(result.generated_probability) : '--'}</td></tr>`;
  }).join('');
  batchResults.innerHTML = `<div class="batch-summary"><span>已完成 <strong>${completedFiles.size} / ${selectedFiles.length}</strong></span>${failures.size ? `<span class="error-count">${failures.size} 张未完成，可重试</span>` : ''}</div><div class="table-wrap" tabindex="0" aria-label="批量检测结果，可横向滚动"><table class="result-table"><thead><tr><th>图片</th><th>图像判断</th><th>来源判断</th><th>AI 生成评分</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function validPrediction(result) {
  return result && ['real', 'generated', 'uncertain'].includes(result.binary_label) && typeof result.generated_probability === 'number' && Number.isFinite(result.generated_probability) && result.generated_probability >= 0 && result.generated_probability <= 1;
}
async function requestPredict(file, signal, profileId) {
  const controller = new AbortController();
  let timedOut = false;
  const relayAbort = () => controller.abort();
  signal.addEventListener('abort', relayAbort, { once: true });
  if (signal.aborted) controller.abort();
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, REQUEST_TIMEOUT_MS);
  try {
    const formData = new FormData();
    formData.append('file', file); formData.append('policy_profile', profileId);
    const response = await fetch('/api/predict', { method: 'POST', body: formData, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { throw new Error('服务暂时未返回有效结果，请稍后重试。'); }
    if (!response.ok || data.status !== 'ok') {
      const messages = {
        400: '图片无法读取，请确认文件完整，或重新导出为 PNG、JPEG、WEBP。',
        413: '图片超出服务接收限制，请缩小文件后重试。',
        422: '图片无法处理，请重新导出后重试。',
        429: '当前请求较多，请稍后再试。',
      };
      throw new Error(messages[response.status] || (response.status >= 500 ? '检测服务暂时不可用，请稍后重试。' : '图片未能完成检测，请稍后重试。'));
    }
    if (!validPrediction(data.result)) throw new Error('服务返回的检测数据不完整，请重试。');
    return data.result;
  } catch (error) {
    if (signal.aborted) throw Object.assign(new Error('已停止等待'), { name: 'AbortError' });
    if (timedOut) throw new Error('等待超时，服务可能正在启动或繁忙，请稍后重试。');
    if (error instanceof TypeError) throw new Error(navigator.onLine === false ? '网络已断开，请连接后重试。' : '无法连接检测服务，请检查网络后重试。');
    throw error;
  } finally { clearTimeout(timeout); signal.removeEventListener('abort', relayAbort); }
}
function selectResult(file, moveFocus = false) {
  const result = resultCache.get(file);
  if (!result) return;
  renderPreview(file); renderResult(result);
  lastSingleResult = result; lastSingleFileName = file.name;
  updateControls(); renderBatchResults();
  if (moveFocus) focusResults();
}
function focusResults() {
  resultsTitle.focus({ preventScroll: true });
  resultsSection.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
}
async function runDetection() {
  if (isBusy || !selectedFiles.length || isStaticPreviewMode()) return;
  if (completedFiles.size === selectedFiles.length) {
    resultCache.clear(); completedFiles.clear(); failures.clear(); lastBatchResults = [];
    lastSingleResult = null; lastSingleFileName = '';
  }
  const todo = selectedFiles.filter(file => !completedFiles.has(file));
  const revision = ++runRevision;
  const profileId = getSelectedPolicyProfile();
  currentController = new AbortController();
  const signal = currentController.signal;
  isBusy = true;
  updateControls(); updateQueueSummary(); setProgress(completedFiles.size, selectedFiles.length);
  if (!lastSingleResult || !resultCache.size) renderLoading();
  const slowTimer = setTimeout(() => { if (isBusy && revision === runRevision) setStatus('分析仍在进行，服务启动或大型文件可能需要更多时间。', 'working'); }, 15000);
  try {
    for (const file of todo) {
      if (signal.aborted || revision !== runRevision) break;
      failures.delete(file);
      setStatus(`正在分析 ${selectedFiles.indexOf(file) + 1} / ${selectedFiles.length}：${file.name}`, 'working');
      try {
        const raw = await requestPredict(file, signal, profileId);
        if (signal.aborted || revision !== runRevision) break;
        const result = { ...raw, _client: { fileName: file.name, fileSize: file.size, createdAt: new Date().toISOString(), reportId: `IMG-${crypto.randomUUID().slice(0, 8).toUpperCase()}` } };
        resultCache.set(file, result); completedFiles.add(file); failures.delete(file);
        lastBatchResults = selectedFiles.filter(item => resultCache.has(item)).map(item => resultCache.get(item));
        if (!lastSingleResult || !resultCache.has(activeFile)) selectResult(file);
      } catch (error) {
        if (signal.aborted || revision !== runRevision) break;
        failures.set(file, error.message);
      }
      updateQueueSummary(); setProgress(completedFiles.size, selectedFiles.length); renderBatchResults();
    }
    if (revision !== runRevision) return;
    if (signal.aborted) setStatus('已停止等待，已完成的结果仍可查看与导出。', 'warning');
    else if (failures.size) setStatus(`${failures.size} 张图片未完成。${selectedFiles.length === 1 ? failures.get(selectedFiles[0]) : '已完成结果保留，可重试未完成图片。'}`, 'error');
    else setStatus(`检测完成${selectedFiles.length > 1 ? `，共 ${completedFiles.size} 张` : ''}。可查看结果或导出报告。`, 'success');
    if (!completedFiles.size) clearResult();
  } finally {
    clearTimeout(slowTimer);
    if (revision === runRevision) {
      isBusy = false; currentController = null; progressWrap.hidden = true;
      updateControls(); updateQueueSummary(); renderBatchResults();
      if (lastSingleResult && !signal.aborted) focusResults();
    }
  }
}
function cancelDetection() {
  currentController?.abort();
  cancelBtn.disabled = true;
  setStatus('正在停止等待，已完成的结果将保留。', 'warning');
}
function downloadText(fileName, text, mimeType) {
  const url = URL.createObjectURL(new Blob([text], { type: mimeType }));
  const link = document.createElement('a'); link.href = url; link.download = fileName; link.hidden = true;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function safeCsvCell(value) {
  const text = String(value ?? '');
  return `"${(/^[\s]*[=+@-]/.test(text) ? "'" + text : text).replaceAll('"', '""')}"`;
}
function buildBatchCsv(results) {
  const rows = [['检测编号', '文件名', '图像判断', '来源判断', 'AI生成评分', '检测时间'], ...results.map(result => [result._client.reportId, result._client.fileName, decisionLabel(result), sourceLabel(result), asPercent(result.generated_probability), result._client.createdAt])];
  return '\uFEFF' + rows.map(row => row.map(safeCsvCell).join(',')).join('\r\n');
}
function renderSingleReportHtml(fileName, result) {
  const metadata = result._client || {};
  const rows = (result.signal_snapshot || []).map(item => `<tr><th>${escapeHtml(item.label)}</th><td>${escapeHtml(item.value)}</td></tr>`).join('');
  const platformRows = scoreEntries(result).map(([code, score]) => `<tr><th>${escapeHtml(platformDisplayName(code))}</th><td>${asPercent(score)}</td></tr>`).join('');
  return `<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>图像检测报告 ${escapeHtml(metadata.reportId || '')}</title><style>body{margin:0;background:#f7f4ee;color:#292723;font:14px/1.85 'Times New Roman','FangSong','STFangsong','仿宋',serif}.report{max-width:880px;margin:32px auto;padding:40px;background:#fffefd;border-top:4px solid #7b252b}header{border-bottom:1px solid #ddd5cb;padding-bottom:22px}h1,h2{font-family:'Times New Roman','FangSong','STFangsong','仿宋',serif;font-weight:600}h1{font-size:32px;margin:8px 0}h2{font-size:22px;margin:28px 0 12px}.kicker{color:#7b252b;font-size:12px;letter-spacing:.1em}.meta{font-size:12px;color:#706a62;overflow-wrap:anywhere}.conclusion{border-bottom:1px solid #ddd5cb;padding:22px 0}.conclusion h2{margin:0 0 10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}table{width:100%;border-collapse:collapse}th,td{padding:10px 8px;border-bottom:1px solid #e2dcd2;text-align:left;font-size:13px}th{font-weight:500;width:55%}.score{font:36px 'Times New Roman','FangSong','STFangsong','仿宋',serif;color:#7b252b}.usage{border-top:1px solid #ddd5cb;margin-top:28px;padding-top:16px;font-size:12px;color:#706a62}li{padding-left:8px;margin-bottom:10px}button{border:1px solid #7b252b;background:transparent;color:#7b252b;padding:9px 15px;cursor:pointer}.hash{overflow-wrap:anywhere;font-family:'Times New Roman','FangSong','STFangsong','仿宋',serif;font-size:11px}@media(max-width:650px){.report{margin:0;padding:24px}.grid{grid-template-columns:1fr}h1{font-size:26px}}@media print{body{background:white}.report{margin:0;padding:12px;border:0}button{display:none}h2,tr{break-inside:avoid}h2{break-after:avoid}}</style></head><body><main class="report"><header><span class="kicker">AIGC 标识治理研究</span><h1>图像检测报告</h1><p class="meta">检测编号：${escapeHtml(metadata.reportId || '未提供')}<br>文件名：${escapeHtml(fileName)}<br>检测时间：${escapeHtml(metadata.createdAt ? new Date(metadata.createdAt).toLocaleString('zh-CN') : '未提供')}</p><button onclick="window.print()" type="button">打印 / 保存为 PDF</button></header><section class="conclusion"><h2>${escapeHtml(decisionLabel(result))}</h2><p>AI 生成评分：<strong class="score">${asPercent(result.generated_probability)}</strong></p><p>来源判断：<strong>${escapeHtml(sourceLabel(result))}</strong></p></section><div class="grid"><section><h2>文件与图像信号</h2><table>${rows}</table></section><section><h2>平台候选评分</h2>${platformRows ? `<table>${platformRows}</table><p class="meta">候选评分不代表来源已经独立确认。${platformAttributionAccepted(result) ? '当前服务接受了来源候选。' : '当前服务未接受平台归因。'}</p>` : '<p>未进行来源归因。</p>'}</section></div>${result.binary_label === 'generated' ? `<section><h2>来源接受条件与诊断</h2><table>${sourceEvidence(result).map(([label, value, note]) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}<br><span class="meta">${escapeHtml(note)}</span></td></tr>`).join('')}</table><p class="meta">${escapeHtml((result.platform_rejection_explanations || []).join('；') || (platformAttributionAccepted(result) ? '当前服务返回来源归因已接受。' : '当前服务未接受平台归因。'))}</p></section>` : ''}<section><h2>判断依据</h2><ol>${technicalRationale(result).map(line => `<li>${escapeHtml(line)}</li>`).join('')}</ol><p class="meta">检测策略：${escapeHtml(POLICY_NAMES[result.policy_profile_id] || result.policy_profile_id || '未提供')}<br>真实区阈值：${asPercent(result.thresholds?.real)} / AI 区阈值：${asPercent(result.thresholds?.generated)}</p>${result.file_sha256 ? `<p class="meta">原始文件 SHA-256</p><p class="hash">${escapeHtml(result.file_sha256)}</p>` : ''}</section><aside class="usage"><h2>使用说明</h2><p>以上评分是模型输出，不是经校准的真实风险率。结果用于辅助核验，不构成真实性证明或独立的平台认证。</p>${resultUsageGuide(result).map(item => `<p><strong>${escapeHtml(item.title)}</strong>：${escapeHtml(item.body)}</p>`).join('')}</aside></main></body></html>`;
}

fileInput.addEventListener('change', () => selectFiles(Array.from(fileInput.files || [])));
predictBtn.addEventListener('click', runDetection);
clearFilesBtn.addEventListener('click', clearSelection);
cancelBtn.addEventListener('click', cancelDetection);
['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); if (!isBusy) dropzone.classList.add('is-dragover'); }));
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('is-dragover'));
dropzone.addEventListener('drop', event => { event.preventDefault(); dropzone.classList.remove('is-dragover'); if (!isBusy) selectFiles(Array.from(event.dataTransfer?.files || [])); });
policyInputs.forEach(input => input.addEventListener('change', () => {
  if (isBusy) return;
  runRevision += 1; resultCache.clear(); completedFiles.clear(); failures.clear(); lastBatchResults = [];
  clearResult(); updateQueueSummary(); updateControls();
  setStatus(`已切换为${getSelectedPolicyName()}，可重新开始检测。`);
}));
batchResults.addEventListener('click', event => { const button = event.target.closest('[data-file-index]'); if (!isBusy && button) selectResult(selectedFiles[Number(button.dataset.fileIndex)], true); });
exportSingleReportBtn.addEventListener('click', () => {
  if (!lastSingleResult || isBusy) return;
  downloadText(`图像检测报告_${lastSingleResult._client.reportId}.html`, renderSingleReportHtml(lastSingleFileName, lastSingleResult), 'text/html;charset=utf-8');
  setStatus('已发起报告下载，请在浏览器下载列表中查看。', 'success');
});
exportBatchCsvBtn.addEventListener('click', () => {
  if (!lastBatchResults.length || isBusy) return;
  downloadText('图像批量检测结果.csv', buildBatchCsv(lastBatchResults), 'text/csv;charset=utf-8');
  setStatus('已发起 CSV 下载，请在浏览器下载列表中查看。', 'success');
});
byId('helpBtn').addEventListener('click', () => helpDialog.showModal());
byId('footerHelpBtn').addEventListener('click', () => helpDialog.showModal());
byId('closeHelpBtn').addEventListener('click', () => helpDialog.close());
helpDialog.addEventListener('click', event => { if (event.target === helpDialog) { const bounds = helpDialog.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) helpDialog.close(); } });
window.addEventListener('pagehide', () => {
  runRevision += 1; currentController?.abort();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = '';
});
window.addEventListener('pageshow', event => {
  if (!event.persisted) return;
  isBusy = false; currentController = null; progressWrap.hidden = true;
  if (activeFile) renderPreview(activeFile);
  if (lastSingleResult) renderResult(lastSingleResult); else clearResult();
  updateControls(); updateQueueSummary(); renderBatchResults();
  if (selectedFiles.length) setStatus('已返回检测页，已完成结果仍可查看；未完成图片可继续检测。');
});
updateControls();
