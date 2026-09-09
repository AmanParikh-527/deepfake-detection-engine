const historyBody = document.querySelector('tbody');
const historySummary = document.querySelector('.history-section .section-label p');

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
}

function renderHistory() {
  const reports = JSON.parse(localStorage.getItem('truesight:reports') || '[]');
  historySummary.textContent = `${reports.length} saved examination${reports.length === 1 ? '' : 's'}`;
  if (!reports.length) {
    historyBody.innerHTML = '<tr><td colspan="6">No completed examinations yet. Run an analysis to add it here.</td></tr>';
    return;
  }
  historyBody.innerHTML = reports.map((report, index) => {
    const synthetic = report.verdict === 'AI-Generated / Manipulated';
    const type = report.media_type || (report.source === 'Social link' ? 'Image' : 'Video');
    const icon = type === 'Audio' ? '◔' : type === 'Image' ? '◈' : '▶';
    const confidence = synthetic ? Number(report.visual_confidence_score || 0) : 100 - Number(report.visual_confidence_score || 0);
    return `<tr data-report-index="${index}"><td><div class="media-name"><span class="media-icon ${type.toLowerCase()}">${icon}</span><b>${escapeHtml(report.filename || 'Untitled')}</b></div></td><td><span class="source-badge">${escapeHtml(report.source || 'Upload')}</span></td><td>${type}</td><td><span class="result-tag ${synthetic ? 'synthetic' : 'authentic'}">${synthetic ? 'Synthetic' : 'Authentic'}</span></td><td><b>${confidence.toFixed(1)}%</b></td><td>${new Date(report.analyzedAt).toLocaleString()}</td></tr>`;
  }).join('');
  historyBody.querySelectorAll('tr[data-report-index]').forEach((row) => row.addEventListener('click', () => {
    sessionStorage.setItem('truesight:last-report', JSON.stringify(reports[Number(row.dataset.reportIndex)]));
    window.location.href = 'report.html';
  }));
}

window.addEventListener('storage', (event) => {
  if (event.key === 'truesight:reports') renderHistory();
});
renderHistory();
