document.querySelectorAll('.evidence-tab').forEach((tab) => tab.addEventListener('click', () => {
  document.querySelectorAll('.evidence-tab').forEach((item) => item.classList.toggle('active', item === tab));
  const heatmap = tab.dataset.view === 'heatmap';
  document.querySelector('#media-frame').classList.toggle('heatmap', heatmap);
  document.querySelector('#media-view-label').textContent = heatmap ? 'ATTENTION MAP' : 'ORIGINAL';
  document.querySelector('#evidence-caption').innerHTML = heatmap ? '<span></span> Heatmap shows the regions that contributed most to the verdict.' : '<span></span> Original frame selected. Switch to heatmap to see the strongest signals.';
}));

const savedReport = sessionStorage.getItem('truesight:last-report');
if (savedReport) {
  const report = JSON.parse(savedReport);
  const synthetic = report.verdict === 'AI-Generated / Manipulated';
  const modelScore = Number(report.visual_confidence_score || 0);
  const score = synthetic ? modelScore : 100 - modelScore;
  const scoreText = score.toFixed(1);
  const verdictTitle = document.querySelector('.verdict-copy h3');
  const verdictDescription = document.querySelector('.verdict-copy p');
  const scoreLabel = document.querySelector('.confidence-label b');
  const scoreRing = document.querySelector('.verdict-ring b');
  const riskFill = document.querySelector('.risk-track i');
  const verdictCard = document.querySelector('.verdict-card');

  verdictTitle.innerHTML = synthetic ? 'Likely <em>synthetic</em>' : 'Likely <em>authentic</em>';
  const mediaType = (report.media_type || 'Media').toLowerCase();
  verdictDescription.textContent = synthetic
    ? `The ${mediaType} model detected signals consistent with generated or manipulated media.`
    : `The ${mediaType} model found no material signs of generated or manipulated media.`;
  scoreLabel.textContent = `${scoreText}%`;
  scoreRing.innerHTML = `${Math.floor(score)}<span>.${scoreText.split('.')[1]}</span>`;
  riskFill.style.width = `${score}%`;
  verdictCard.style.borderLeftColor = synthetic ? 'var(--danger)' : 'var(--success)';
  scoreLabel.style.color = synthetic ? 'var(--danger)' : 'var(--success)';
  document.querySelector('.verdict-ring').style.background = `conic-gradient(${synthetic ? 'var(--danger)' : 'var(--success)'} 0 ${score * 3.6}deg, #edf0f3 ${score * 3.6}deg)`;

  const statValues = document.querySelectorAll('.quick-stat b');
  statValues[0].textContent = report.media_type || 'Media';
  statValues[1].textContent = `${report.frames_analyzed || 0} frames`;
  statValues[2].textContent = new Date(report.analyzedAt).toLocaleDateString();
  const metaValues = document.querySelectorAll('.verdict-meta b');
  metaValues[0].textContent = (report.media_type || 'Media').toUpperCase();
  metaValues[1].textContent = `${report.frames_analyzed || 0} FRAMES`;

  if (report.evidence_frame_base64) {
    const frame = document.querySelector('#media-frame');
    frame.style.backgroundImage = `url(data:image/jpeg;base64,${report.evidence_frame_base64})`;
    frame.style.backgroundPosition = 'center';
    frame.style.backgroundSize = 'cover';
  }
} else {
  document.querySelector('.verdict-copy h3').textContent = 'No report selected';
  document.querySelector('.verdict-copy p').textContent = 'Run an image, video, audio, or social-image analysis to view its forensic report.';
  document.querySelector('.confidence-label b').textContent = '—';
  document.querySelector('.risk-track i').style.width = '0%';
  document.querySelector('.verdict-ring b').textContent = '—';
  document.querySelector('.verdict-ring').style.background = 'conic-gradient(#edf0f3 0 360deg)';
  document.querySelectorAll('.quick-stat b').forEach((item) => { item.textContent = '—'; });
  document.querySelectorAll('.verdict-meta b').forEach((item) => { item.textContent = '—'; });
}

document.querySelector('.secondary-button')?.addEventListener('click', (event) => {
  event.preventDefault();
  const report = savedReport ? JSON.parse(savedReport) : {};
  const score = Number(report.visual_confidence_score || 0).toFixed(1);
  const lines = [
    'TRUESIGHT AI — FORENSIC REPORT',
    '',
    `Media: ${report.filename || 'Latest examination'}`,
    `Source: ${report.source || 'Upload'}`,
    `Verdict: ${report.verdict || 'No analysis available'}`,
    `Model fake-score: ${score}%`,
    `Frames analyzed: ${report.frames_analyzed || 0}`,
    `Generated: ${report.analyzedAt ? new Date(report.analyzedAt).toLocaleString() : new Date().toLocaleString()}`,
  ];
  const escapePdf = (value) => String(value).replace(/[^\x20-\x7E]/g, '?').replace(/[\\()]/g, '\\$&');
  const content = `BT /F1 16 Tf 50 760 Td ${lines.map((line, index) => `(${escapePdf(line)}) Tj${index < lines.length - 1 ? ' 0 -24 Td' : ''}`).join('\n')} ET`;
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>',
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
  ];
  let pdf = '%PDF-1.4\n';
  const offsets = [0];
  objects.forEach((object, index) => {
    offsets.push(pdf.length);
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });
  const xref = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  offsets.slice(1).forEach((offset) => { pdf += `${String(offset).padStart(10, '0')} 00000 n \n`; });
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`;
  const url = URL.createObjectURL(new Blob([pdf], { type: 'application/pdf' }));
  const download = document.createElement('a');
  download.href = url;
  download.download = 'truesight-forensic-report.pdf';
  download.click();
  URL.revokeObjectURL(url);
});
