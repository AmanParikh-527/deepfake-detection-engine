const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];

const state = {
  mode: 'upload',
  file: null,
  url: '',
  analyzing: false,
};

const fileInput = $('#file-input');
const dropZone = $('#drop-zone');
const fileSelection = $('#file-selection');
const mediaUrl = $('#media-url');
const urlInsight = $('#url-insight');
const analyzeButton = $('#analyze-button');
const message = $('#form-message');
const toast = $('#toast');

function prettyBytes(bytes) {
  if (!bytes) return '0 KB';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / (1024 ** index)).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function typeFromFile(file) {
  const type = file?.type || '';
  if (type.startsWith('video')) return { label: 'Video', code: 'VID', className: 'video' };
  if (type.startsWith('audio')) return { label: 'Audio', code: 'AUD', className: 'audio' };
  return { label: 'Image', code: 'IMG', className: 'image' };
}

function typeFromUrl(url) {
  const value = url.toLowerCase();
  if (value.match(/\.mp3|\.wav|\.m4a|\.ogg|soundcloud|spotify/)) return { label: 'Audio', code: 'AUD', className: 'audio' };
  if (value.match(/\.jpg|\.jpeg|\.png|\.webp|\.gif/)) return { label: 'Image', code: 'IMG', className: 'image' };
  return { label: 'Video', code: 'VID', className: 'video' };
}

function showToast(text) {
  toast.textContent = text;
  toast.classList.add('show');
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove('show'), 2600);
}

function switchIntake(mode) {
  state.mode = mode;
  $$('.intake-tab').forEach((tab) => {
    const active = tab.dataset.intake === mode;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', active);
  });
  $$('.intake-pane').forEach((pane) => pane.classList.toggle('active', pane.dataset.pane === mode));
  message.textContent = '';
}

function selectFile(file) {
  if (!file) return;
  if (!file.type.match(/^(image|video|audio)\//)) {
    message.textContent = 'Please choose an image, video, or audio file.';
    return;
  }
  state.file = file;
  const type = typeFromFile(file);
  $('#file-name').textContent = file.name;
  $('#file-meta').textContent = `${prettyBytes(file.size)} · ${type.label}`;
  $('#file-type-icon').textContent = type.code;
  fileSelection.hidden = false;
  message.textContent = '';
}

function clearFile() {
  state.file = null;
  fileInput.value = '';
  fileSelection.hidden = true;
}

function validateUrl(value) {
  if (!value.trim()) {
    urlInsight.className = 'url-insight';
    urlInsight.innerHTML = '<span class="status-orb"></span><span>Enter a public URL to validate its source and extract media.</span>';
    return false;
  }
  try {
    const parsed = new URL(value);
    if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('protocol');
    const hostname = parsed.hostname.replace(/^www\./, '');
    state.url = parsed.href;
    urlInsight.className = 'url-insight valid';
    urlInsight.innerHTML = `<span class="status-orb"></span><span><b>${hostname}</b> is publicly reachable. Source reputation and media metadata will be checked.</span>`;
    return true;
  } catch {
    state.url = '';
    urlInsight.className = 'url-insight invalid';
    urlInsight.innerHTML = '<span class="status-orb"></span><span>Please enter a complete public URL beginning with https://</span>';
    return false;
  }
}

function resetPipeline() {
  $$('.pipeline-steps li').forEach((step) => step.classList.remove('active', 'done'));
  $('#progress-fill').style.width = '0%';
  $('#progress-number').textContent = '00%';
}

function updatePipeline(progress, activeStep, text) {
  $('#progress-fill').style.width = `${progress}%`;
  $('#progress-number').textContent = `${String(progress).padStart(2, '0')}%`;
  $('#progress-heading').textContent = text;
  $('#progress-state').textContent = progress === 100 ? 'Report complete' : 'Analysis in progress';
  $$('.pipeline-steps li').forEach((step) => {
    const stepIndex = Number(step.dataset.step);
    step.classList.toggle('done', stepIndex < activeStep);
    step.classList.toggle('active', stepIndex === activeStep && progress < 100);
    if (progress === 100 && stepIndex === 4) step.classList.add('done');
  });
}

function sampleVerdict(name) {
  const seed = [...name].reduce((sum, char) => sum + char.charCodeAt(0), 0);
  const authentic = /original|authentic|nature|receipt|document|real/i.test(name) || seed % 7 === 0;
  if (authentic) return { synthetic: false, score: 88.6, confidence: 'HIGH CONFIDENCE' };
  return { synthetic: true, score: 81 + (seed % 144) / 10, confidence: 'HIGH CONFIDENCE' };
}

function setReport(inputName, mediaType, source) {
  const verdict = sampleVerdict(inputName);
  const title = $('#verdict-title');
  const riskFill = $('#risk-fill');
  const ring = $('.verdict-ring');
  const scoreText = verdict.score.toFixed(1);
  $('#confidence-value').textContent = `${scoreText}%`;
  $('#ring-value').innerHTML = `${Math.floor(verdict.score)}<span>.${scoreText.split('.')[1]}</span>`;
  $('#confidence-caption').textContent = verdict.confidence;
  $('#report-type').textContent = mediaType.toUpperCase();
  $('#report-time').textContent = mediaType === 'Video' ? '00:04.8' : mediaType === 'Audio' ? '00:03.1' : '00:02.6';
  riskFill.style.width = `${verdict.score}%`;
  ring.style.background = `conic-gradient(${verdict.synthetic ? 'var(--danger)' : 'var(--success)'} 0 ${verdict.score * 3.6}deg, rgba(255,255,255,.09) ${verdict.score * 3.6}deg)`;
  if (verdict.synthetic) {
    title.innerHTML = 'Likely <em>synthetic</em>';
    $('#verdict-description').textContent = 'Multiple model signals indicate that this media was likely generated or manipulated.';
    $('.verdict-card').style.borderLeftColor = 'var(--danger)';
    $('#confidence-value').style.color = 'var(--danger)';
  } else {
    title.innerHTML = 'Likely <em>authentic</em>';
    $('#verdict-description').textContent = 'The forensic models found no material signs of generation or intentional manipulation.';
    $('.verdict-card').style.borderLeftColor = 'var(--success)';
    $('#confidence-value').style.color = 'var(--success)';
  }
  addHistory(inputName, mediaType, source, verdict);
}

function iconFor(type) {
  if (type === 'Video') return '▶';
  if (type === 'Audio') return '◔';
  return '◈';
}

function addHistory(name, type, source, verdict) {
  const row = document.createElement('tr');
  const result = verdict.synthetic ? 'Synthetic' : 'Authentic';
  row.innerHTML = `<td><div class="media-name"><span class="media-icon ${type.toLowerCase()}">${iconFor(type)}</span><b>${name}</b></div></td><td><span class="source-badge">${source}</span></td><td>${type}</td><td><span class="result-tag ${verdict.synthetic ? 'synthetic' : 'authentic'}">${result}</span></td><td><b>${verdict.score.toFixed(1)}%</b></td><td>Just now</td>`;
  $('#history-body').prepend(row);
  $('#history-count').textContent = String($('#history-body').children.length).padStart(2, '0');
}

function runAnalysis() {
  if (state.analyzing) return;
  const usable = state.mode === 'upload' ? state.file : validateUrl(mediaUrl.value);
  if (!usable) {
    message.textContent = state.mode === 'upload' ? 'Drop a media file or browse your device to begin.' : 'Add a valid public media URL to begin.';
    return;
  }
  state.analyzing = true;
  const type = state.mode === 'upload' ? typeFromFile(state.file) : typeFromUrl(state.url);
  const inputName = state.mode === 'upload' ? state.file.name : new URL(state.url).hostname.replace(/^www\./, '');
  const source = state.mode === 'upload' ? 'Upload' : new URL(state.url).hostname.replace(/^www\./, '').split('.')[0].replace(/^./, (c) => c.toUpperCase());
  analyzeButton.disabled = true;
  analyzeButton.querySelector('span').textContent = 'Analyzing media';
  $('#pipeline-name').textContent = `${type.label} pipeline active`;
  resetPipeline();
  $('#progress-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
  const steps = [
    [18, 1, 'Validating media integrity'],
    [43, 2, 'Extracting forensic regions'],
    [73, 3, 'Running neural detection models'],
    [100, 4, 'Generating evidence report'],
  ];
  steps.forEach(([percent, step, copy], index) => {
    window.setTimeout(() => updatePipeline(percent, step, copy), 550 + index * 700);
  });
  window.setTimeout(() => {
    setReport(inputName, type.label, source);
    analyzeButton.disabled = false;
    analyzeButton.querySelector('span').textContent = 'Analyze media';
    state.analyzing = false;
    $('#pipeline-name').textContent = 'Analysis complete';
    $('#report').scrollIntoView({ behavior: 'smooth', block: 'start' });
    showToast('Forensic report generated successfully');
  }, 3450);
}

$$('.intake-tab').forEach((tab) => tab.addEventListener('click', () => switchIntake(tab.dataset.intake)));
fileInput.addEventListener('change', (event) => selectFile(event.target.files?.[0]));
$('#clear-file').addEventListener('click', clearFile);
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('dragover'); }));
dropZone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files?.[0]));
mediaUrl.addEventListener('input', () => validateUrl(mediaUrl.value));
$$('.source-chip').forEach((chip) => chip.addEventListener('click', () => { mediaUrl.value = chip.dataset.url; validateUrl(mediaUrl.value); }));
analyzeButton.addEventListener('click', runAnalysis);

$$('.evidence-tab').forEach((tab) => tab.addEventListener('click', () => {
  $$('.evidence-tab').forEach((button) => button.classList.toggle('active', button === tab));
  const isHeatmap = tab.dataset.view === 'heatmap';
  $('#media-frame').classList.toggle('heatmap', isHeatmap);
  $('#media-view-label').textContent = isHeatmap ? 'ATTENTION MAP' : 'ORIGINAL';
  $('#evidence-caption').innerHTML = isHeatmap
    ? '<span></span> Heatmap shows the regions that contributed most strongly to the verdict.'
    : '<span></span> Original frame selected. Switch to heatmap to see regions that influenced the result.';
}));

$('#clear-history').addEventListener('click', () => {
  $('#history-body').innerHTML = '';
  $('#history-count').textContent = '00';
  showToast('Detection history cleared');
});
$('#view-details').addEventListener('click', () => showToast('Full forensic evidence will open in the detailed report.'));
$('#download-report').addEventListener('click', () => showToast('Report download prepared (demo frontend).'));
$('#share-report').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(window.location.href + '#report'); showToast('Report link copied to clipboard'); }
  catch { showToast('Share link is ready: ' + window.location.href + '#report'); }
});
