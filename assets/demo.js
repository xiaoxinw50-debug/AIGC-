const sampleSelector = document.getElementById('sampleSelector');
const previewImage = document.getElementById('previewImage');
const resultSummary = document.getElementById('resultSummary');
const platformProbabilities = document.getElementById('platformProbabilities');
const signalSnapshot = document.getElementById('signalSnapshot');
const rationaleBox = document.getElementById('rationaleBox');
const statusText = document.getElementById('statusText');
const fileInput = document.getElementById('fileInput');
const localPreviewBtn = document.getElementById('localPreviewBtn');
const resetSampleBtn = document.getElementById('resetSampleBtn');
const PLATFORM_NAME_MAP = {
  PLT01: '文心一言',
  PLT02: '通义千问',
  PLT03: '即梦AI / 字节系文生图',
  PLT05: '智谱GLM-Image / 清言相关文生图',
  real: '真实图片'
};

let samples = [];
let activeSampleId = null;

function setStatus(text) {
  statusText.textContent = text;
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

function resultUsageGuide(result) {
  if (result.binary_label === 'real') {
    return [
      ['页面交付', '输出为真实图低风险，不继续给平台来源。'],
      ['审核动作', '保留原图文件、订单上下文和沟通记录。'],
      ['对外说明', '当前未发现强 AI 生成留痕，本次不按 AI 假图处理。']
    ];
  }
  return [
    ['页面交付', `输出为高置信 AI 生成，并在已采样平台中给出 ${result.platform_label_text}。`],
    ['审核动作', '查看平台概率和文件层信号，结合原始文件、订单链路和多角度图片做最终核验。'],
    ['对外说明', '平台归因是已采样范围内的技术线索，不是开放世界定责结论。']
  ];
}

function renderResult(sample) {
  const result = sample.result;
  const generatedPct = (result.generated_probability * 100).toFixed(2);
  const isReal = result.binary_label === 'real';
  const usageGuide = resultUsageGuide(result);
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
          <strong>${escapeHtml(result.platform_label_text)}</strong>
        </div>
      </div>
      <div><strong>平台来源：</strong>${escapeHtml(result.platform_label_text)}</div>
      <div class="sample-note"><strong>样本说明：</strong>${escapeHtml(sample.summary)}</div>
      <div class="result-usage-grid">
        ${usageGuide.map(([title, body]) => `
          <div class="usage-card">
            <span>${escapeHtml(title)}</span>
            <p>${escapeHtml(body)}</p>
          </div>
        `).join('')}
      </div>
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
      ${result.signal_snapshot.map(item => `<div class="signal-item"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`).join('')}
    </div>
  `;

  rationaleBox.innerHTML = `
    <div class="rationale-list">
      ${result.rationale.map(line => `<div>${escapeHtml(line)}</div>`).join('')}
    </div>
  `;
}

function renderSelector() {
  sampleSelector.innerHTML = samples.map(sample => `
    <button
      type="button"
      class="sample-card ${sample.id === activeSampleId ? 'is-active' : ''}"
      data-sample-id="${sample.id}"
    >
      <strong>${escapeHtml(sample.title)}</strong>
      <span>${escapeHtml(sample.subtitle)}</span>
    </button>
  `).join('');

  sampleSelector.querySelectorAll('[data-sample-id]').forEach(button => {
    button.addEventListener('click', () => {
      const sample = samples.find(item => item.id === button.dataset.sampleId);
      if (!sample) return;
      activeSampleId = sample.id;
      previewImage.src = sample.image;
      previewImage.style.display = 'block';
      renderSelector();
      renderResult(sample);
      setStatus(`已切换到展示样本：${sample.title}`);
    });
  });
}

async function boot() {
  const response = await fetch('./assets/demo-data.json');
  const data = await response.json();
  samples = data.samples || [];
  if (!samples.length) {
    setStatus('没有可展示样本');
    return;
  }
  activeSampleId = samples[0].id;
  renderSelector();
  previewImage.src = samples[0].image;
  previewImage.style.display = 'block';
  renderResult(samples[0]);
}

localPreviewBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.style.display = 'block';
    setStatus(`已加载本地图片 ${file.name}。GitHub 展示版不会对本地上传图片做实时推理。`);
  };
  reader.readAsDataURL(file);
});

resetSampleBtn.addEventListener('click', () => {
  if (!samples.length) return;
  activeSampleId = samples[0].id;
  renderSelector();
  previewImage.src = samples[0].image;
  previewImage.style.display = 'block';
  renderResult(samples[0]);
  setStatus('已恢复默认展示样本。');
});

boot().catch((error) => {
  console.error(error);
  setStatus('展示版加载失败');
});
