const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../webapp/static/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../webapp/templates/index.html'), 'utf8');
const makeFile = (name = 'photo.png') => new File(['test image'], name, { type: 'image/png' });
const prediction = overrides => ({
  binary_label: 'generated', generated_probability: .92,
  platform_label: 'unknown_platform', platform_accepted: false,
  platform_probabilities: { PLT01: .7, PLT02: .1, PLT03: .1, PLT05: .1 },
  signal_snapshot: [{ label: '文件格式', value: 'PNG' }],
  rationale: ['文件信息与视觉特征联合分析。'], ...overrides,
  label_evidence: {
    explicit_label: { status: 'manual_review_required', text: '需要人工核验。' },
    implicit_label: { status: 'detected', detected_keys: ['AIGC'], text: '发现疑似 AIGC 隐式标识字段。' },
    generation_metadata_clues: { status: 'detected', detected_keys: ['Software'] },
    file_metadata: { format: 'PNG', info_key_count: 2, exif_key_count: 0 },
    standard_scope_note: '不构成标识合规认证。',
  },
  model_release: { open_set_version: 'platform_knownness_gate_v6', models: { binary: { file: 'binary.joblib', sha256: 'a'.repeat(64) } } },
  report_verification: { status: 'signed', report_id: 'AIGC-TEST-001', issued_at: '2026-09-27T00:00:00Z', algorithm: 'HMAC-SHA256', payload_sha256: 'b'.repeat(64), verification_url: 'https://example.test/verify#token=test' },
});
const reply = result => ({ ok: true, status: 200, json: async () => ({ status: 'ok', result }) });

function harness(options = {}) {
  const elements = new Map();
  const calls = [];
  const revoked = [];
  const windowListeners = new Map();
  let urlCount = 0;
  function element(id) {
    if (!elements.has(id)) {
      const listeners = new Map();
      const classes = new Set();
      elements.set(id, {
        id, listeners, textContent: '', innerHTML: '', value: '', files: [],
        style: {}, dataset: {}, hidden: false, disabled: false, open: false,
        classList: {
          add(...names) { names.forEach(name => classes.add(name)); },
          remove(...names) { names.forEach(name => classes.delete(name)); },
          toggle(name, force) { if (force === true) classes.add(name); else if (force === false) classes.delete(name); else if (classes.has(name)) classes.delete(name); else classes.add(name); },
          contains(name) { return classes.has(name); },
        },
        addEventListener(name, callback) { listeners.set(name, callback); },
        setAttribute(name, value) { this[name] = value; },
        removeAttribute(name) { delete this[name]; },
        focus() { this.focused = true; }, scrollIntoView() { this.scrolled = true; },
        remove() { this.removed = true; },
        click() { listeners.get('click')?.({ target: this }); },
        showModal() { this.open = true; }, close() { this.open = false; },
      });
    }
    return elements.get(id);
  }
  ['results', 'batchSection', 'queueSummary'].forEach(id => { element(id).hidden = true; });
  element('policyProfileData').textContent = JSON.stringify({ profiles: [], default_id: 'operational_low_false_ai' });
  const policies = ['operational_low_false_ai', 'balanced_review', 'high_risk_after_sales_review'].map((value, index) => Object.assign(element(value), { value, checked: index === 0 }));
  const context = vm.createContext({
    document: {
      getElementById: element,
      querySelector: selector => selector === '.upload-dropzone' ? element('dropzone') : selector === '.upload-panel' ? element('upload-panel') : null,
      querySelectorAll: () => policies,
      createElement: () => element(`created-${elements.size}`),
      body: { appendChild() {} },
    },
    window: { location: { protocol: options.protocol || 'http:' }, addEventListener: (name, callback) => windowListeners.set(name, callback), matchMedia: () => ({ matches: false }) },
    navigator: { onLine: true },
    localStorage: { getItem() { throw Error('Public detection must not read persisted history'); }, setItem() { throw Error('Public detection must not persist history'); } },
    URL: { createObjectURL: () => `blob:test-${++urlCount}`, revokeObjectURL: url => revoked.push(url) },
    fetch: async (...args) => { calls.push(args); return options.fetch ? options.fetch(...args) : reply(prediction()); },
    FormData, File, Blob, AbortController, crypto: webcrypto,
    setTimeout: (callback, delay) => setTimeout(callback, options.fastTimeout && delay === 90000 ? 1 : delay),
    clearTimeout, console,
  });
  vm.runInContext(source, context);
  return { context, element, policies, calls, revoked, windowListeners, state: expression => vm.runInContext(expression, context) };
}

test('initial page and file selection do not upload or fetch shared reviews', () => {
  const h = harness();
  assert.equal(h.element('predictBtn').disabled, true);
  h.context.selectFiles([makeFile()]);
  assert.equal(h.calls.length, 0);
  assert.equal(h.element('results').hidden, true);
  assert.equal(h.element('predictBtn').disabled, false);
  assert.equal(h.element('previewImage').hidden, false);
  assert.equal(h.element('upload-panel').classList.contains('has-selection'), true);
  assert.doesNotMatch(source, /localStorage\.|\/api\/reviews/);
});

test('validation enforces MIME, size and batch limits', () => {
  const h = harness();
  assert.equal(h.context.validateFiles([{ name: 'photo.jpg', type: '', size: 100 }]).ok, true);
  assert.equal(h.context.validateFiles([{ name: 'photo.png', type: 'application/pdf', size: 100 }]).ok, false);
  assert.equal(h.context.validateFiles([{ name: 'large.png', type: 'image/png', size: 17 * 1024 * 1024 }]).ok, false);
  assert.equal(h.context.validateFiles(Array.from({ length: 21 }, () => makeFile())).ok, false);
  h.context.selectFiles([makeFile('kept.png')]);
  h.context.selectFiles([{ name: 'bad.pdf', type: 'application/pdf', size: 10 }]);
  assert.equal(h.context.getSelectedFiles()[0].name, 'kept.png');
  assert.equal(h.element('statusText').dataset.tone, 'error');
});

test('platform verdict requires generated, known code and explicit acceptance', () => {
  const h = harness();
  for (const result of [prediction({ binary_label: 'real', platform_accepted: true, platform_label: 'PLT01' }), prediction({ platform_label: 'PLT01' }), prediction({ platform_accepted: 'true', platform_label: 'PLT01' }), prediction({ platform_accepted: true })]) {
    assert.equal(h.context.platformAttributionAccepted(result), false);
  }
  assert.equal(h.context.platformAttributionAccepted(prediction({ platform_accepted: true, platform_label: 'PLT01' })), true);
  assert.equal(h.context.platformDisplayName('PLT03'), '即梦AI');
});

test('malformed or non-finite model scores cannot become a verdict', () => {
  const h = harness();
  for (const overrides of [{ generated_probability: NaN }, { generated_probability: Infinity }, { generated_probability: '0.9' }, { generated_probability: 1.1 }, { binary_label: 'bad' }]) {
    assert.equal(h.context.validPrediction(prediction(overrides)), false);
  }
  assert.equal(h.context.scoreEntries(prediction({ platform_probabilities: { PLT01: null, PLT02: .5, evil: .5 } })).length, 1);
});

test('preview object URLs are released on replacement and clearing', () => {
  const h = harness();
  h.context.selectFiles([makeFile('one.png')]);
  h.context.selectFiles([makeFile('two.png')]);
  assert.equal(h.revoked.length, 1);
  h.context.clearSelection();
  assert.equal(h.revoked.length, 2);
  assert.equal(h.element('previewImage').hidden, true);
  assert.equal(h.element('predictBtn').disabled, true);
  assert.equal(h.element('upload-panel').classList.contains('has-selection'), false);
});

test('same filenames remain two independent jobs and batch selection updates preview', async () => {
  const h = harness();
  h.context.selectFiles([makeFile('same.png'), makeFile('same.png')]);
  await h.context.runDetection();
  assert.equal(h.calls.length, 2);
  assert.match(h.element('queueSummary').textContent, /已完成 2 \/ 待检测 0/);
  assert.equal(h.element('exportBatchCsvBtn').disabled, false);
  h.context.selectResult(h.context.getSelectedFiles()[1], true);
  assert.equal(h.element('resultsTitle').focused, true);
  assert.equal(h.element('results').scrolled, true);
});

test('fresh rerun replaces old verdict and clears stale technical details', async () => {
  let runs = 0;
  const h = harness({ fetch: async () => reply(prediction({ generated_probability: ++runs === 1 ? .92 : .12, binary_label: runs === 1 ? 'generated' : 'real' })) });
  h.context.selectFiles([makeFile()]);
  await h.context.runDetection();
  assert.match(h.element('resultSummary').innerHTML, /92.00%/);
  h.element('evidenceDetails').open = true;
  await h.context.runDetection();
  assert.match(h.element('resultSummary').innerHTML, /12.00%/);
  assert.doesNotMatch(h.element('resultSummary').innerHTML, /92.00%|skeleton/);
  assert.equal(h.element('evidenceDetails').open, false);
});

test('partial batch failure preserves completed jobs and retries only unfinished', async () => {
  let requests = 0;
  const h = harness({ fetch: async () => ++requests === 2 ? ({ ok: false, status: 503, json: async () => ({}) }) : reply(prediction()) });
  h.context.selectFiles([makeFile('one.png'), makeFile('two.png'), makeFile('three.png')]);
  await h.context.runDetection();
  assert.equal(h.state('completedFiles.size'), 2);
  assert.equal(h.state('failures.size'), 1);
  assert.match(h.element('predictBtn').innerHTML, /重试未完成/);
  await h.context.runDetection();
  assert.equal(requests, 4);
  assert.equal(h.state('completedFiles.size'), 3);
  assert.equal(h.state('failures.size'), 0);
});

test('busy run locks inputs and cancellation ignores a late server reply', async () => {
  let resolve;
  const h = harness({ fetch: () => new Promise(done => { resolve = done; }) });
  h.context.selectFiles([makeFile()]);
  const run = h.context.runDetection();
  assert.equal(h.element('fileInput').disabled, true);
  assert.equal(h.policies.every(input => input.disabled), true);
  assert.equal(h.element('cancelBtn').hidden, false);
  h.context.selectFiles([makeFile('ignored.png')]);
  assert.equal(h.context.getSelectedFiles()[0].name, 'photo.png');
  h.context.cancelDetection();
  resolve(reply(prediction()));
  await run;
  assert.equal(h.state('resultCache.size'), 0);
  assert.equal(h.element('results').hidden, true);
  assert.equal(h.element('predictBtn').disabled, false);
  assert.match(h.element('statusText').textContent, /已停止等待/);
});

test('HTML 500 and aborted timeout produce safe, recoverable messages', async () => {
  const h = harness({ fetch: async () => ({ ok: false, status: 500, json: async () => { throw Error('<private stack>'); } }) });
  await assert.rejects(h.context.requestPredict(makeFile(), new AbortController().signal, 'test'), /有效结果/);
  const timed = harness({ fastTimeout: true, fetch: (_, { signal }) => new Promise((_, reject) => signal.addEventListener('abort', () => reject(Object.assign(new Error(), { name: 'AbortError' })))) });
  await assert.rejects(timed.context.requestPredict(makeFile(), new AbortController().signal, 'test'), /等待超时/);
});

test('unreadable image error is translated rather than exposing backend wording', async () => {
  const h = harness({ fetch: async () => ({ ok: false, status: 400, json: async () => ({ status: 'error', message: 'unsupported or broken image' }) }) });
  h.context.selectFiles([makeFile()]);
  await h.context.runDetection();
  assert.match(h.element('statusText').textContent, /图片无法读取/);
  assert.doesNotMatch(h.element('statusText').textContent, /unsupported/);
  assert.equal(h.element('predictBtn').disabled, false);
  assert.equal(h.element('exportSingleReportBtn').disabled, true);
});

test('policy switch invalidates old results rather than relabelling them', async () => {
  const h = harness();
  h.context.selectFiles([makeFile()]);
  await h.context.runDetection();
  h.policies[0].checked = false;
  h.policies[1].checked = true;
  h.policies[1].listeners.get('change')();
  assert.equal(h.element('results').hidden, true);
  assert.equal(h.state('resultCache.size'), 0);
  assert.equal(h.element('selectedPolicyLabel').textContent, '均衡复核');
  assert.equal(h.element('exportSingleReportBtn').disabled, true);
});

test('page, HTML report and CSV share the same qualified verdict', () => {
  const h = harness();
  const result = prediction({ _client: { reportId: 'IMG-TEST', fileName: '=bad.png', createdAt: '2026-09-17T00:00:00Z' }, rationale: ['<unsafe>'] });
  h.context.renderResult(result);
  const report = h.context.renderSingleReportHtml('<photo>.png', result);
  const csv = h.context.buildBatchCsv([result]);
  for (const text of [h.element('resultSummary').innerHTML, report, csv]) {
    assert.match(text, /AI 生成倾向较高/);
    assert.match(text, /暂无法归因/);
  }
  assert.match(report, /&lt;photo&gt;\.png/);
  assert.doesNotMatch(report, /<unsafe>/);
  assert.match(csv, /"'=bad.png"/);
  assert.match(report, /签名有效/);
  assert.match(report, /在线验证报告/);
  assert.match(report, /verify#token=test/);
  assert.match(report, /隐式标识/);
  assert.match(report, /报告内容 SHA-256/);
  assert.match(csv, /已签发/);
});

test('all required HTML hooks exist once and public entry excludes internal modules', () => {
  const hookIds = [...new Set([...source.matchAll(/byId\('([^']+)'\)/g)].map(match => match[1]))];
  assert.ok(hookIds.length > 30);
  for (const id of hookIds) assert.equal((html.match(new RegExp(`id="${id}"`, 'g')) || []).length, 1, id);
  assert.doesNotMatch(html, /·|答辩演示定位|核心原则|已知限制|系统交付|样本数|historyList|reviewList|模型结果用于辅助核验|分析生成可能性/);
  const css = fs.readFileSync(path.join(__dirname, '../webapp/static/style.css'), 'utf8');
  assert.match(css, /'Times New Roman', 'FangSong', 'STFangsong', '仿宋'/);
  assert.doesNotMatch(css, /Georgia|PingFang SC|Microsoft YaHei|Songti SC|STSong|SimSun/);
  assert.match(css, /upload-panel:not\(\.has-selection\) \.preview-box/);
  assert.match(css, /result-table th:first-child, \.result-table td:first-child \{ position: sticky/);
  assert.match(html, /laboratory-20260927-verified-report-v1/);
  assert.match(html, /id="results"[^>]*hidden/);
  assert.match(html, /id="evidenceDetails"/);
  assert.match(html, /id="usageDetails"/);
});

test('static preview never offers a misleading online detection action', () => {
  const h = harness({ protocol: 'file:' });
  h.context.selectFiles([makeFile()]);
  assert.equal(h.element('predictBtn').disabled, true);
  assert.equal(h.calls.length, 0);
  assert.match(h.element('statusText').textContent, /界面预览/);
});

test('browser back-cache restore releases busy state and regenerates the preview URL', async () => {
  let resolve;
  const h = harness({ fetch: () => new Promise(done => { resolve = done; }) });
  h.context.selectFiles([makeFile()]);
  const oldUrl = h.element('previewImage').src;
  const run = h.context.runDetection();
  h.windowListeners.get('pagehide')();
  h.windowListeners.get('pageshow')({ persisted: true });
  assert.equal(h.element('predictBtn').disabled, false);
  assert.notEqual(h.element('previewImage').src, oldUrl);
  resolve(reply(prediction()));
  await run;
  assert.equal(h.state('resultCache.size'), 0);
});
