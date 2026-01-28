/**
    Панель детекции и трекинга людей: загрузка видео, калибровка, WebSocket-стриминг.
 */
const els = {
  btnUpload: document.getElementById('btnUpload'),
  btnResetPoints: document.getElementById('btnResetPoints'),
  btnCalibrateStart: document.getElementById('btnCalibrateStart'),
  btnRefreshHistory: document.getElementById('btnRefreshHistory'),
  file: document.getElementById('videoFile'),
  jobPill: document.getElementById('jobPill'),
  pointsText: document.getElementById('pointsText'),
  canvas: document.getElementById('calibCanvas'),
  imgVideo: document.getElementById('imgVideo'),
  imgMap: document.getElementById('imgMap'),
  pillVideo: document.getElementById('pillVideo'),
  pillMap: document.getElementById('pillMap'),
  statStatus: document.getElementById('statStatus'),
  statFps: document.getElementById('statFps'),
  statPeople: document.getElementById('statPeople'),
  statPeopleMax: document.getElementById('statPeopleMax'),
  statFrame: document.getElementById('statFrame'),
  statTotal: document.getElementById('statTotal'),
  historyBody: document.getElementById('historyBody'),
  btnFsVideo: document.getElementById('btnFsVideo'),
  btnFsMap: document.getElementById('btnFsMap'),
  btnZoomOut: document.getElementById('btnZoomOut'),
  btnZoomReset: document.getElementById('btnZoomReset'),
  btnZoomIn: document.getElementById('btnZoomIn'),
  boxVideo: document.getElementById('boxVideo'),
  boxMap: document.getElementById('boxMap'),
};

const ctx = els.canvas.getContext('2d');
let jobId = null;
let frame0Img = null;
/** Точки калибровки в координатах canvas: массив {x, y} */
let points = [];
let ws = null;
let mapZoom = 1.0;

/** Устанавливает текст и стиль «таблетки» статуса (good/warn/bad). */
function setPill(el, text, kind = '') {
  el.textContent = text;
  el.className = 'pill' + (kind ? ' ' + kind : '');
}

/** Ограничивает масштаб карты и применяет transform к imgMap. */
function setMapZoom(z) {
  mapZoom = Math.max(0.5, Math.min(4.0, z));
  els.imgMap.style.transform = `scale(${mapZoom})`;
}

/** Переключает полноэкранный режим для переданного элемента. */
async function toggleFullscreen(el) {
  if (!document.fullscreenElement) {
    await el.requestFullscreen?.();
  } else {
    await document.exitFullscreen?.();
  }
}

/**
 * Отрисовка кадра калибровки и точек на canvas.
 * Если frame0 не загружен — рисует заглушку. Обновляет кнопки и текст точек.
 */
function drawCalibration() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (frame0Img) {
    ctx.drawImage(frame0Img, 0, 0, els.canvas.width, els.canvas.height);
  } else {
    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '14px Arial';
    ctx.fillText('Загрузите видео, чтобы отобразить первый кадр для калибровки.', 16, 28);
  }
  /* Отрисовка точек и подписей TL, TR, BR, BL */
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#22c55e';
  ctx.fillStyle = '#22c55e';
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e7eefc';
    ctx.font = '12px Arial';
    const labels = ['TL', 'TR', 'BR', 'BL'];
    ctx.fillText(labels[i] || String(i + 1), p.x + 8, p.y - 8);
    ctx.fillStyle = '#22c55e';
  });
  if (points.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    if (points.length === 4) ctx.closePath();
    ctx.stroke();
  }
  els.pointsText.textContent = 'Points: ' + JSON.stringify(points.map(p => ({ x: Math.round(p.x), y: Math.round(p.y) })));
  els.btnResetPoints.disabled = !jobId || points.length === 0;
  els.btnCalibrateStart.disabled = !jobId || points.length !== 4;
}

/**
 * Перевод координат из canvas (отображаемый первый кадр) в координаты исходного изображения.
 * Сервер использует тот же frame0; масштаб учитывается через naturalWidth/naturalHeight.
 */
function canvasToImageCoords(px, py) {
  const scaleX = frame0Img.naturalWidth / els.canvas.width;
  const scaleY = frame0Img.naturalHeight / els.canvas.height;
  return { x: px * scaleX, y: py * scaleY };
}

/** Загрузка видео на сервер (POST /api/v1/jobs), получение job_id и URL первого кадра. */
async function uploadJob() {
  const f = els.file.files[0];
  if (!f) {
    alert('Сначала выберите видео');
    return;
  }

  setPill(els.jobPill, 'Загрузка…', 'warn');
  points = [];
  drawCalibration();
  if (ws) {
    ws.close();
    ws = null;
  }
  els.imgVideo.removeAttribute('src');
  els.imgMap.removeAttribute('src');
  setPill(els.pillVideo, 'Ожидание');
  setPill(els.pillMap, 'Ожидание');

  const fd = new FormData();
  fd.append('file', f);

  try {
    const res = await fetch('/api/v1/jobs', { method: 'POST', body: fd });

    if (!res.ok) {
      setPill(els.jobPill, 'Ошибка загрузки', 'bad');
      const text = await res.text();
      alert('Ошибка загрузки: ' + text);
      return;
    }

    const data = await res.json();
    jobId = data.job_id;
    setPill(els.jobPill, 'Job: ' + jobId.slice(0, 8), 'good');

    /* Загрузка первого кадра для калибровки */
    const img = new Image();
    img.onload = () => {
      frame0Img = img;
      drawCalibration();
    };
    img.src = data.frame0_url + '?t=' + Date.now();
    refreshHistory();
  } catch (error) {
    setPill(els.jobPill, 'Ошибка загрузки', 'bad');
    alert('Ошибка загрузки: ' + error.message);
  }
}

/**
 * Калибровка и старт: отправка 4 точек и (опционально) реальных размеров,
 * затем POST калибровки и POST start, подключение WebSocket.
 */
async function calibrateAndStart() {
  if (!jobId || points.length !== 4) return;

  const pts = points.map(p => canvasToImageCoords(p.x, p.y));
  const scaleType = document.querySelector('input[name="scaleType"]:checked').value;
  const knowsRealDimensions = scaleType === 'real';

  const calibrationData = {
    points: pts,
    knows_real_dimensions: knowsRealDimensions,
  };

  if (knowsRealDimensions) {
    const width = parseFloat(document.getElementById('realWidth').value);
    const height = parseFloat(document.getElementById('realHeight').value);

    if (!width || !height || width <= 0 || height <= 0) {
      alert('Введите положительные значения ширины и высоты в метрах');
      return;
    }

    calibrationData.real_width_m = width;
    calibrationData.real_height_m = height;
  }

  const res = await fetch(`/api/v1/jobs/${jobId}/calibrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(calibrationData),
  });

  if (!res.ok) {
    const errorText = await res.text();
    alert('Ошибка калибровки: ' + errorText);
    return;
  }

  const res2 = await fetch(`/api/v1/jobs/${jobId}/start`, { method: 'POST' });
  if (!res2.ok) {
    const errorText = await res2.text();
    alert('Ошибка запуска: ' + errorText);
    return;
  }

  connectWS();
  setPill(els.pillVideo, 'Стриминг…', 'warn');
  setPill(els.pillMap, 'Стриминг…', 'warn');
  els.statStatus.textContent = 'processing';
}

/** Подключение WebSocket к /ws/jobs/{jobId}: видео, карта, статистика, done/error. */
function connectWS() {
  if (!jobId) return;
  if (ws) ws.close();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/jobs/${jobId}`);

  ws.onmessage = (ev) => {
    let msg = null;
    try {
      msg = JSON.parse(ev.data);
    } catch (e) {
      return;
    }

    if (msg.type === 'video_frame') {
      if (msg.jpeg_b64) els.imgVideo.src = 'data:image/jpeg;base64,' + msg.jpeg_b64;
    } else if (msg.type === 'map_frame') {
      if (msg.jpeg_b64) els.imgMap.src = 'data:image/jpeg;base64,' + msg.jpeg_b64;
    } else if (msg.type === 'stats') {
      els.statStatus.textContent = msg.status ?? 'processing';
      els.statFps.textContent = (msg.processing_fps ?? 0).toFixed(1);
      els.statPeople.textContent = msg.people ?? 0;
      els.statPeopleMax.textContent = msg.people_max ?? 0;
      els.statFrame.textContent = msg.frame ?? 0;
      els.statTotal.textContent = msg.total_frames ?? '?';
    } else if (msg.type === 'done') {
      setPill(els.pillVideo, 'Готово', 'good');
      setPill(els.pillMap, 'Готово', 'good');
      els.statStatus.textContent = 'done';
      refreshHistory();
    } else if (msg.type === 'error') {
      setPill(els.pillVideo, 'Ошибка', 'bad');
      setPill(els.pillMap, 'Ошибка', 'bad');
      els.statStatus.textContent = 'error';
      alert(msg.message || 'Ошибка задания');
      refreshHistory();
    }
  };
}

/** Загрузка списка заданий с API и заполнение таблицы истории. */
async function refreshHistory() {
  try {
    const res = await fetch('/api/v1/history?limit=25');
    if (!res.ok) return;
    const data = await res.json();
    const items = data.items || [];
    if (!items.length) {
      els.historyBody.innerHTML = '<tr><td colspan="6" class="muted">Заданий пока нет</td></tr>';
      return;
    }
    els.historyBody.innerHTML = items
      .map((r) => {
        const d = r.downloads || {};
        const links = [];
        if (d.feed) links.push(`<a href="${d.feed}" target="_blank">feed.mp4</a>`);
        if (d.map) links.push(`<a href="${d.map}" target="_blank">map.mp4</a>`);
        if (d.tracks_xlsx) links.push(`<a href="${d.tracks_xlsx}" target="_blank">tracks.xlsx</a>`);
        if (d.report_pdf) links.push(`<a href="${d.report_pdf}" target="_blank">report.pdf</a>`);
        const status = r.status || '—';
        const statusPill = status === 'done' ? 'good' : status === 'error' ? 'bad' : 'warn';
        return `
          <tr>
            <td><span class="small">${String(r.job_id).slice(0, 8)}</span></td>
            <td>${r.filename || ''}</td>
            <td><span class="pill ${statusPill}">${status}</span></td>
            <td>${r.frames_processed ?? ''}</td>
            <td>${r.people_max ?? ''}</td>
            <td>${links.length ? links.join(' | ') : '<span class="muted">—</span>'}</td>
          </tr>
        `;
      })
      .join('');
  } catch (error) {
    /* Игнорируем ошибку обновления истории */
  }
}

/* Переключение типа шкалы: показ/скрытие полей реальных размеров и текст кнопки калибровки */
document.querySelectorAll('input[name="scaleType"]').forEach((radio) => {
  radio.addEventListener('change', function () {
    const dimensionsInput = document.getElementById('realDimensionsInput');
    dimensionsInput.style.display = this.value === 'real' ? 'block' : 'none';
    const calibrateBtn = document.getElementById('btnCalibrateStart');
    if (this.value === 'real') {
      calibrateBtn.textContent = 'Калибровать (реальный масштаб) и старт';
    } else {
      calibrateBtn.textContent = 'Калибровать (пиксельный масштаб) и старт';
    }
  });
});

/* Обработчики кнопок */
els.btnUpload.addEventListener('click', () =>
  uploadJob().catch((e) => {
    alert('Ошибка загрузки: ' + e.message);
  })
);

els.btnResetPoints.addEventListener('click', () => {
  points = [];
  drawCalibration();
});

els.btnCalibrateStart.addEventListener('click', () =>
  calibrateAndStart().catch((e) => {
    alert('Ошибка калибровки: ' + e.message);
  })
);

els.btnRefreshHistory.addEventListener('click', () => refreshHistory().catch(() => {}));

els.btnFsVideo.addEventListener('click', () => toggleFullscreen(els.boxVideo).catch(() => {}));
els.btnFsMap.addEventListener('click', () => toggleFullscreen(els.boxMap).catch(() => {}));

els.btnZoomIn.addEventListener('click', () => setMapZoom(mapZoom * 1.2));
els.btnZoomOut.addEventListener('click', () => setMapZoom(mapZoom / 1.2));
els.btnZoomReset.addEventListener('click', () => setMapZoom(1.0));

/* Масштаб карты колёсиком мыши */
els.boxMap.addEventListener(
  'wheel',
  (ev) => {
    ev.preventDefault();
    const delta = Math.sign(ev.deltaY);
    if (delta > 0) setMapZoom(mapZoom / 1.12);
    else setMapZoom(mapZoom * 1.12);
  },
  { passive: false }
);

/* Клик по canvas калибровки: добавление точки TL → TR → BR → BL */
els.canvas.addEventListener('click', (ev) => {
  if (!jobId || !frame0Img) return;
  if (points.length >= 4) return;
  const rect = els.canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (els.canvas.width / rect.width);
  const y = (ev.clientY - rect.top) * (els.canvas.height / rect.height);
  points.push({ x, y });
  drawCalibration();
});

/* Инициализация: отрисовка калибровки, загрузка истории, сброс масштаба карты */
drawCalibration();
refreshHistory();
setMapZoom(1.0);

document.addEventListener('DOMContentLoaded', function () {
  const realRadio = document.getElementById('scaleReal');
  const pixelRadio = document.getElementById('scalePixel');
  const dimensionsInput = document.getElementById('realDimensionsInput');

  if (dimensionsInput) {
    dimensionsInput.style.display = realRadio && realRadio.checked ? 'block' : 'none';
  }

  const calibrateBtn = document.getElementById('btnCalibrateStart');
  if (calibrateBtn) {
    calibrateBtn.textContent =
      pixelRadio && pixelRadio.checked
        ? 'Калибровать (пиксельный масштаб) и старт'
        : 'Калибровать (реальный масштаб) и старт';
  }
});
