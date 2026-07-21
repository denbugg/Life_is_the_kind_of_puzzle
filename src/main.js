import JSZip from 'jszip';
import './style.css';

const REFERENCE_FILES = [
  'img_000013.png', 'img_000275.png', 'img_000313.png', 'img_000697.png',
  'img_000809.png', 'img_000840.png', 'img_001737.png', 'img_001786.png',
  'img_001990.png', 'img_001997.png', 'img_002051.png', 'img_002127.png',
  'img_002198.png', 'img_002647.png', 'img_002775.png', 'img_002829.png',
  'img_002948.png', 'img_002950.png'
];

const state = { results: [], objectUrls: [], selected: null, busy: false };

document.querySelector('#app').innerHTML = `
  <header class="topbar">
    <a class="brand" href="#" aria-label="Puzzle Check — на главную">
      <span class="brand-mark"><i></i><i></i><i></i><i></i></span>
      <span>Puzzle Check</span>
    </a>
    <div class="top-meta"><span class="pulse"></span> локальный скорер <b>SSIM / v1.0</b></div>
  </header>

  <main>
    <section class="hero">
      <div class="eyebrow"><span>01</span> private leaderboard</div>
      <h1>Проверьте сборку.<br><em>До сабмита.</em></h1>
      <p class="lead">Сравниваем найденные по имени изображения с 18 проверенными эталонами и считаем средний structural similarity index.</p>
    </section>

    <section class="workspace">
      <div class="upload-column">
        <div class="section-label"><span>Загрузка</span><small>ZIP · PNG</small></div>
        <div id="dropzone" class="dropzone" tabindex="0" role="button" aria-label="Выбрать ZIP архив">
          <input id="fileInput" type="file" accept=".zip,application/zip" hidden />
          <div class="drop-icon"><span>↗</span></div>
          <h2>Перетащите ZIP сюда</h2>
          <p>или <button id="chooseButton" type="button">выберите файл</button> на компьютере</p>
          <div class="drop-note">Папки внутри архива не важны — матчинг идёт по имени файла</div>
        </div>
        <div id="fileCard" class="file-card hidden"></div>
      </div>

      <aside class="score-panel">
        <div class="section-label"><span>Результат</span><small id="scoreStatus">ожидает архив</small></div>
        <div class="score-wrap">
          <div class="score-caption">СРЕДНИЙ SSIM</div>
          <div id="scoreValue" class="score-value">—</div>
          <div class="score-scale"><span>0</span><div><i id="scoreBar"></i></div><span>1</span></div>
        </div>
        <div class="stats">
          <div><strong id="matchedCount">0</strong><span>совпало</span></div>
          <div><strong id="missingCount">18</strong><span>не найдено</span></div>
          <div><strong id="invalidCount">0</strong><span>с ошибкой</span></div>
        </div>
        <p class="formula">mean(SSIM<sub>RGB</sub>) · win 7×7 · range 255</p>
      </aside>
    </section>

    <section id="resultsSection" class="results-section hidden">
      <div class="results-head">
        <div>
          <div class="eyebrow"><span>02</span> matched samples</div>
          <h2>Разбор результата</h2>
        </div>
        <div class="legend"><span><i class="good"></i> ≥ 0.90</span><span><i class="mid"></i> 0.70–0.90</span><span><i class="bad"></i> &lt; 0.70</span></div>
      </div>
      <div id="comparison" class="comparison"></div>
      <div id="gallery" class="gallery"></div>
    </section>
  </main>

  <footer><span>PUZZLE CHALLENGE · INTERNAL TOOL</span><span>Обработка выполняется только в вашем браузере</span></footer>
`;

const $ = (selector) => document.querySelector(selector);
const dropzone = $('#dropzone');
const fileInput = $('#fileInput');

$('#chooseButton').addEventListener('click', (event) => { event.stopPropagation(); fileInput.click(); });
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => fileInput.files[0] && scoreArchive(fileInput.files[0]));
['dragenter', 'dragover'].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.remove('dragging');
}));
dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files[0];
  if (file) scoreArchive(file);
});

async function scoreArchive(file) {
  if (state.busy) return;
  if (!file.name.toLowerCase().endsWith('.zip')) return showFileError(file, 'Нужен файл в формате .zip');
  state.busy = true;
  releaseObjectUrls();
  state.results = [];
  updateFileCard(file, 'Распаковываем архив…', 4);
  setStatus('считаем', true);

  try {
    const zip = await JSZip.loadAsync(file);
    const pngEntries = Object.values(zip.files).filter((entry) => !entry.dir && entry.name.toLowerCase().endsWith('.png'));
    const byName = new Map();
    for (const entry of pngEntries) {
      const basename = entry.name.split('/').pop().toLowerCase();
      if (!byName.has(basename)) byName.set(basename, entry);
    }

    for (let index = 0; index < REFERENCE_FILES.length; index += 1) {
      const name = REFERENCE_FILES[index];
      const entry = byName.get(name.toLowerCase());
      updateFileCard(file, `Сравниваем ${index + 1} из ${REFERENCE_FILES.length}…`, 8 + ((index + 1) / REFERENCE_FILES.length) * 88);
      if (!entry) {
        state.results.push({ name, status: 'missing' });
        continue;
      }

      try {
        const predictionBlob = await entry.async('blob');
        const predictionUrl = keepUrl(URL.createObjectURL(predictionBlob));
        const referenceUrl = `/reference/${name}`;
        const [reference, prediction] = await Promise.all([loadPixels(referenceUrl), loadPixels(predictionUrl)]);
        if (reference.width !== prediction.width || reference.height !== prediction.height) {
          throw new Error(`${prediction.width}×${prediction.height}, ожидалось ${reference.width}×${reference.height}`);
        }
        const score = structuralSimilarity(reference, prediction);
        state.results.push({ name, status: 'ok', score, referenceUrl, predictionUrl });
      } catch (error) {
        state.results.push({ name, status: 'invalid', error: error.message });
      }
      await nextFrame();
    }

    renderResults(file, pngEntries.length);
  } catch (error) {
    showFileError(file, `Не удалось прочитать архив: ${error.message}`);
    setStatus('ошибка');
  } finally {
    state.busy = false;
  }
}

function structuralSimilarity(a, b) {
  const { width: w, height: h } = a;
  if (w < 7 || h < 7) throw new Error('изображение меньше окна SSIM 7×7');
  const channels = 3;
  const area = 49;
  const covarianceNorm = area / (area - 1);
  const c1 = (0.01 * 255) ** 2;
  const c2 = (0.03 * 255) ** 2;
  let total = 0;

  for (let channel = 0; channel < channels; channel += 1) {
    const sumsA = integralImage(a.data, w, h, channel, false);
    const sumsB = integralImage(b.data, w, h, channel, false);
    const squaresA = integralImage(a.data, w, h, channel, true);
    const squaresB = integralImage(b.data, w, h, channel, true);
    const products = integralProduct(a.data, b.data, w, h, channel);
    let channelSum = 0;
    let count = 0;

    for (let y = 0; y <= h - 7; y += 1) {
      for (let x = 0; x <= w - 7; x += 1) {
        const sumA = boxSum(sumsA, w + 1, x, y, 7);
        const sumB = boxSum(sumsB, w + 1, x, y, 7);
        const meanA = sumA / area;
        const meanB = sumB / area;
        const varianceA = Math.max(0, (boxSum(squaresA, w + 1, x, y, 7) / area - meanA * meanA) * covarianceNorm);
        const varianceB = Math.max(0, (boxSum(squaresB, w + 1, x, y, 7) / area - meanB * meanB) * covarianceNorm);
        const covariance = (boxSum(products, w + 1, x, y, 7) / area - meanA * meanB) * covarianceNorm;
        channelSum += ((2 * meanA * meanB + c1) * (2 * covariance + c2)) /
          ((meanA * meanA + meanB * meanB + c1) * (varianceA + varianceB + c2));
        count += 1;
      }
    }
    total += channelSum / count;
  }
  return total / channels;
}

function integralImage(data, w, h, channel, square) {
  const stride = w + 1;
  const integral = new Float64Array(stride * (h + 1));
  for (let y = 1; y <= h; y += 1) {
    let rowSum = 0;
    for (let x = 1; x <= w; x += 1) {
      const value = data[((y - 1) * w + x - 1) * 4 + channel];
      rowSum += square ? value * value : value;
      integral[y * stride + x] = integral[(y - 1) * stride + x] + rowSum;
    }
  }
  return integral;
}

function integralProduct(a, b, w, h, channel) {
  const stride = w + 1;
  const integral = new Float64Array(stride * (h + 1));
  for (let y = 1; y <= h; y += 1) {
    let rowSum = 0;
    for (let x = 1; x <= w; x += 1) {
      const offset = ((y - 1) * w + x - 1) * 4 + channel;
      rowSum += a[offset] * b[offset];
      integral[y * stride + x] = integral[(y - 1) * stride + x] + rowSum;
    }
  }
  return integral;
}

function boxSum(integral, stride, x, y, size) {
  const x2 = x + size;
  const y2 = y + size;
  return integral[y2 * stride + x2] - integral[y * stride + x2] - integral[y2 * stride + x] + integral[y * stride + x];
}

async function loadPixels(url) {
  const image = new Image();
  image.decoding = 'async';
  image.src = url;
  await image.decode();
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  return { width: canvas.width, height: canvas.height, data: context.getImageData(0, 0, canvas.width, canvas.height).data };
}

function renderResults(file, archivePngCount) {
  const valid = state.results.filter((item) => item.status === 'ok');
  const missing = state.results.filter((item) => item.status === 'missing');
  const invalid = state.results.filter((item) => item.status === 'invalid');
  const mean = valid.length ? valid.reduce((sum, item) => sum + item.score, 0) / valid.length : null;
  $('#matchedCount').textContent = valid.length;
  $('#missingCount').textContent = missing.length;
  $('#invalidCount').textContent = invalid.length;
  $('#scoreValue').textContent = mean === null ? '—' : mean.toFixed(6);
  $('#scoreBar').style.width = `${Math.max(0, Math.min(1, mean ?? 0)) * 100}%`;
  setStatus(valid.length ? `готово · ${valid.length}/18` : 'нет совпадений');
  updateFileCard(file, `${archivePngCount} PNG в архиве · ${valid.length} использовано`, 100, true);

  $('#resultsSection').classList.remove('hidden');
  $('#gallery').innerHTML = state.results.map((item, index) => {
    if (item.status !== 'ok') return `<div class="sample-card unavailable"><div class="thumb-placeholder">${item.status === 'missing' ? 'не найдено' : 'ошибка'}</div><div class="sample-info"><span>${item.name}</span><b>—</b></div>${item.error ? `<small>${escapeHtml(item.error)}</small>` : ''}</div>`;
    return `<button class="sample-card ${index === state.selected ? 'selected' : ''}" data-index="${index}">
      <img src="${item.predictionUrl}" alt="Предсказание ${item.name}" />
      <div class="sample-info"><span>${item.name}</span><b class="${scoreClass(item.score)}">${item.score.toFixed(4)}</b></div>
    </button>`;
  }).join('');

  $('#gallery').querySelectorAll('button').forEach((button) => button.addEventListener('click', () => selectResult(Number(button.dataset.index))));
  const bestIndex = state.results.findIndex((item) => item.status === 'ok');
  if (bestIndex >= 0) selectResult(bestIndex);
  $('#resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function selectResult(index) {
  const item = state.results[index];
  if (!item || item.status !== 'ok') return;
  state.selected = index;
  document.querySelectorAll('.sample-card').forEach((card) => card.classList.toggle('selected', Number(card.dataset.index) === index));
  $('#comparison').innerHTML = `
    <div class="compare-visual">
      <div class="compare-image base"><img src="${item.referenceUrl}" alt="Правильный ответ ${item.name}" /></div>
      <div class="compare-image reveal" id="reveal"><img src="${item.predictionUrl}" alt="Результат ${item.name}" /></div>
      <div class="divider" id="divider"><i></i></div>
      <input id="compareRange" type="range" min="0" max="100" value="50" aria-label="Сравнить эталон и результат" />
      <span class="visual-label left">ЭТАЛОН</span><span class="visual-label right">ВАШ РЕЗУЛЬТАТ</span>
    </div>
    <div class="compare-copy">
      <div class="compare-kicker">Детальное сравнение</div>
      <h3>${item.name}</h3>
      <div class="detail-score"><span>SSIM</span><strong class="${scoreClass(item.score)}">${item.score.toFixed(6)}</strong></div>
      <p>Двигайте разделитель, чтобы увидеть расхождения. Слева — правильная сборка, справа — изображение из загруженного архива.</p>
      <div class="quality"><span>Оценка</span><b>${qualityLabel(item.score)}</b></div>
    </div>`;
  $('#compareRange').addEventListener('input', (event) => {
    const value = event.target.value;
    $('#reveal').style.clipPath = `inset(0 0 0 ${value}%)`;
    $('#divider').style.left = `${value}%`;
  });
}

function updateFileCard(file, text, progress, done = false) {
  const card = $('#fileCard');
  card.classList.remove('hidden', 'error');
  card.innerHTML = `<div class="file-type">ZIP</div><div class="file-copy"><b>${escapeHtml(file.name)}</b><span>${text}</span><div class="progress"><i style="width:${progress}%"></i></div></div><div class="file-size">${formatBytes(file.size)}${done ? '<i>✓</i>' : ''}</div>`;
}

function showFileError(file, message) {
  updateFileCard(file, message, 0);
  $('#fileCard').classList.add('error');
}

function setStatus(text, loading = false) {
  $('#scoreStatus').textContent = text;
  $('#scoreStatus').classList.toggle('loading', loading);
}

function scoreClass(score) { return score >= 0.9 ? 'good-text' : score >= 0.7 ? 'mid-text' : 'bad-text'; }
function qualityLabel(score) { return score >= 0.95 ? 'Почти идеально' : score >= 0.9 ? 'Отлично' : score >= 0.7 ? 'Есть расхождения' : 'Требует внимания'; }
function formatBytes(bytes) { return bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`; }
function nextFrame() { return new Promise((resolve) => requestAnimationFrame(resolve)); }
function keepUrl(url) { state.objectUrls.push(url); return url; }
function releaseObjectUrls() { state.objectUrls.forEach(URL.revokeObjectURL); state.objectUrls = []; }
function escapeHtml(value) { return value.replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char])); }
