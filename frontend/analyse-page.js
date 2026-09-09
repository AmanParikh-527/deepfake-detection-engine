const select = (q, scope = document) => scope.querySelector(q);
const selectAll = (q, scope = document) => [...scope.querySelectorAll(q)];
let pickedFile = null;

function readableSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
function classify(file) {
  if (file.type.startsWith('audio')) return ['AUD', 'Audio'];
  if (file.type.startsWith('video')) return ['VID', 'Video'];
  return ['IMG', 'Image'];
}
function selectFile(file) {
  if (!file || !file.type.match(/^(image|video|audio)\//)) return;
  pickedFile = file;
  const [code, type] = classify(file);
  select('#file-name').textContent = file.name;
  select('#file-meta').textContent = `${readableSize(file.size)} · ${type}`;
  select('#file-type-icon').textContent = code;
  select('#file-selection').hidden = false;
  select('#form-message').textContent = '';
}
function setProgress(value, active, label) {
  select('#progress-fill').style.width = `${value}%`;
  select('#progress-number').textContent = `${String(value).padStart(2, '0')}%`;
  select('#progress-heading').textContent = label;
  selectAll('.pipeline-steps li').forEach((item) => {
    const position = Number(item.dataset.step);
    item.classList.toggle('done', position < active || value === 100);
    item.classList.toggle('active', position === active && value < 100);
  });
}

selectAll('.intake-tab').forEach((tab) => tab.addEventListener('click', () => {
  selectAll('.intake-tab').forEach((item) => item.classList.toggle('active', item === tab));
  selectAll('.intake-pane').forEach((pane) => pane.classList.toggle('active', pane.dataset.pane === tab.dataset.intake));
}));
select('#file-input').addEventListener('change', (event) => selectFile(event.target.files[0]));
select('#clear-file').addEventListener('click', () => { pickedFile = null; select('#file-input').value = ''; select('#file-selection').hidden = true; });
['dragenter', 'dragover'].forEach((eventName) => select('#drop-zone').addEventListener(eventName, (event) => { event.preventDefault(); select('#drop-zone').classList.add('dragover'); }));
['dragleave', 'drop'].forEach((eventName) => select('#drop-zone').addEventListener(eventName, (event) => { event.preventDefault(); select('#drop-zone').classList.remove('dragover'); }));
select('#drop-zone').addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));
selectAll('.source-chip').forEach((chip) => chip.addEventListener('click', () => { select('#media-url').value = chip.dataset.url; }));
select('#analyze-button').addEventListener('click', () => {
  const onLinkTab = select('[data-pane="link"]').classList.contains('active');
  const hasUrl = /^https:\/\//.test(select('#media-url').value.trim());
  if ((!onLinkTab && !pickedFile) || (onLinkTab && !hasUrl)) { select('#form-message').textContent = onLinkTab ? 'Add a public URL to begin.' : 'Choose a media file to begin.'; return; }
  const button = select('#analyze-button');
  button.disabled = true; button.querySelector('span').textContent = 'Analyzing media';
  select('#pipeline-name').textContent = 'Examination in progress';
  select('#progress-section').scrollIntoView({behavior: 'smooth', block: 'center'});
  [[22, 1, 'Validating media'], [51, 2, 'Extracting relevant signals'], [78, 3, 'Reviewing authenticity'], [100, 4, 'Preparing your report']].forEach(([value, step, label], index) => setTimeout(() => setProgress(value, step, label), 400 + index * 700));
  setTimeout(() => { button.disabled = false; button.querySelector('span').textContent = 'Analyze another file'; select('#pipeline-name').textContent = 'Examination complete'; select('#progress-state').textContent = 'Report complete'; select('#complete-callout').hidden = false; select('#complete-callout').scrollIntoView({behavior: 'smooth', block: 'center'}); }, 3400);
});
