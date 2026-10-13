/* Upload / analyze UI for the static HybridDualPatNet site. */
(function () {
  const MAX_MB = 10;
  const LABELS = { AD: "Alzheimer's Disease (AD)", PD: "Parkinson's Disease (PD)", CONTROL: 'Healthy Control' };
  const BADGES = { AD: 'badge-ad', PD: 'badge-pd', CONTROL: 'badge-control' };

  const $ = id => document.getElementById(id);
  const input = $('file-input'), dropZone = $('drop-zone'), analyzeBtn = $('analyze-btn'), sampleBtn = $('sample-btn');
  let currentFile = null, busy = false;

  function showError(msg) {
    $('error').textContent = msg;
    $('error').classList.toggle('d-none', !msg);
  }

  function setStatus(text, frac) {
    $('status').classList.remove('d-none');
    $('status-text').textContent = text;
    $('status-pct').textContent = frac == null ? '' : Math.round(frac * 100) + '%';
    $('status-bar').style.width = (frac == null ? 100 : frac * 100) + '%';
  }

  function handleFile(file) {
    if (!file) return;
    showError('');
    if (!['image/jpeg', 'image/png'].includes(file.type)) return showError('Only JPG and PNG images are allowed.');
    if (file.size > MAX_MB * 1024 * 1024) return showError(`File must be ${MAX_MB} MB or smaller.`);
    currentFile = file;
    $('file-name').textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    $('preview').src = URL.createObjectURL(file);
    $('preview').classList.remove('d-none');
    $('preview-placeholder').classList.add('d-none');
    analyzeBtn.disabled = busy;
  }

  function renderResult(res) {
    $('res-time').textContent = res.seconds.toFixed(1);
    const badge = $('res-badge');
    badge.className = `badge badge-glow fs-6 ${BADGES[res.predictedClass]}`;
    badge.textContent = LABELS[res.predictedClass];
    $('res-original').src = $('preview').src;
    $('res-heatmap').getContext('2d').putImageData(res.heatmap, 0, 0);
    $('res-conf').innerHTML = [['AD', 'conf-ad'], ['PD', 'conf-pd'], ['CONTROL', 'conf-control']].map(([k, cls]) => `
      <div class="conf-row">
        <div class="conf-label"><span>${LABELS[k]}</span><strong>${(res.probs[k] * 100).toFixed(1)}%</strong></div>
        <div class="conf-track"><div class="conf-fill ${cls}" data-w="${res.probs[k] * 100}"></div></div>
      </div>`).join('');
    $('results').classList.remove('d-none');
    requestAnimationFrame(() => document.querySelectorAll('.conf-fill').forEach(el => { el.style.width = el.dataset.w + '%'; }));
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function analyze() {
    if (!currentFile || busy) return;
    busy = true;
    analyzeBtn.disabled = sampleBtn.disabled = true;
    showError('');
    try {
      const res = await HDPN.predict(currentFile, (stage, p) => {
        if (stage === 'download') setStatus('Downloading AI model (one time)…', p);
        else if (stage === 'analyze') setStatus('Analyzing scan…', null);
        else if (stage === 'explain') setStatus('Generating Grad-CAM++ heatmap…', null);
      });
      setStatus('Done', 1);
      renderResult(res);
      return res;
    } catch (e) {
      console.error(e);
      showError('Analysis failed: ' + (e && e.message ? e.message : e));
      $('status').classList.add('d-none');
    } finally {
      busy = false;
      analyzeBtn.disabled = !currentFile;
      sampleBtn.disabled = false;
    }
  }

  async function loadSample() {
    const blob = await (await fetch('samples/sample-mri.jpg')).blob();
    handleFile(new File([blob], 'sample-mri.jpg', { type: 'image/jpeg' }));
  }

  dropZone.addEventListener('click', () => input.click());
  dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => handleFile(input.files[0]));
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  analyzeBtn.addEventListener('click', analyze);
  sampleBtn.addEventListener('click', async () => { await loadSample(); analyze(); });
})();
