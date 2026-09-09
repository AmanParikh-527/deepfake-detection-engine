document.querySelectorAll('.evidence-tab').forEach((tab) => tab.addEventListener('click', () => {
  document.querySelectorAll('.evidence-tab').forEach((item) => item.classList.toggle('active', item === tab));
  const heatmap = tab.dataset.view === 'heatmap';
  document.querySelector('#media-frame').classList.toggle('heatmap', heatmap);
  document.querySelector('#media-view-label').textContent = heatmap ? 'ATTENTION MAP' : 'ORIGINAL';
  document.querySelector('#evidence-caption').innerHTML = heatmap ? '<span></span> Heatmap shows the regions that contributed most to the verdict.' : '<span></span> Original frame selected. Switch to heatmap to see the strongest signals.';
}));
