/**
 * YOLO Video Game Observer — Frontend Application
 * Features: live detection, pause/resume, frame browser, search, heatmap,
 *           confidence histogram, object presence timeline, annotations, export
 */

// === DOM refs ===
const $ = id => document.getElementById(id);
const videoSelect = $('videoSelect'), fileInput = $('fileInput');
const fpsSlider = $('fpsSlider'), fpsValue = $('fpsValue');
const confSlider = $('confSlider'), confValue = $('confValue');
const maxFramesInput = $('maxFrames');
const startBtn = $('startBtn'), pauseBtn = $('pauseBtn'), stopBtn = $('stopBtn');
const canvas = $('frameCanvas'), ctx = canvas.getContext('2d');
const dropZone = $('drop-zone');
const frameCounter = $('frameCounter'), frameTimestamp = $('frameTimestamp');
const sceneStatus = $('sceneStatus'), detectionCount = $('detectionCount');
const progressBar = $('progress-bar');
const statusText = $('statusText'), connectionStatus = $('connectionStatus');
const sceneTimeline = $('scene-timeline'), objectChart = $('object-chart');
const detectionLog = $('detection-log');
const searchInput = $('searchInput'), searchBtn = $('searchBtn'), clearSearchBtn = $('clearSearchBtn');
const searchResults = $('search-results');
const heatmapToggle = $('heatmapToggle'), annotateToggle = $('annotateToggle');
const frameScrubberDiv = $('frame-scrubber'), frameScrubber = $('frameScrubber'), scrubberInfo = $('scrubberInfo');
const filmstripDiv = $('filmstrip');
const annotationBar = $('annotation-bar'), annotationLabel = $('annotationLabel');
const confidenceHistCanvas = $('confidenceHistogram');
const presenceCanvas = $('presenceTimeline');
const exportCsvBtn = $('exportCsvBtn'), exportFrameBtn = $('exportFrameBtn');

// Stat DOM
const statInference = $('statInference'), statP95 = $('statP95');
const statThroughput = $('statThroughput'), statMemory = $('statMemory');
const statChanges = $('statChanges'), statSkipped = $('statSkipped');
const statSkipRate = $('statSkipRate'), statFrames = $('statFrames');

// === State ===
let ws = null;
let isProcessing = false, isPaused = false;
let selectedVideoPath = '', currentSessionId = null;
let estimatedFrames = 0;
let frameHistory = [];       // Array of full frame messages
const FRAME_HISTORY_MAX = 300;
let browsingIndex = -1;      // -1 = live mode
let timelineData = [];       // { changed: bool }
let searchTerm = '';
let searchResultFrames = [];
let heatmapCanvas = null, heatmapCtx = null;
let confidenceData = [];     // All confidence values
let currentUser = null;      // Supabase user object or null

// === Auth helpers ===
function getAuthToken() { return localStorage.getItem('yolo_token'); }
function authHeaders() {
    const token = getAuthToken();
    const headers = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = 'Bearer ' + token;
    return headers;
}
async function apiFetch(url, opts = {}) {
    opts.headers = { ...authHeaders(), ...(opts.headers || {}) };
    return fetch(url, opts);
}
async function checkAuth() {
    try {
        const res = await fetch('/api/auth/config');
        const config = await res.json();
        if (!config.enabled) return; // No auth required
        const token = getAuthToken();
        if (!token) return; // Anonymous mode
        const me = await apiFetch('/api/me');
        if (me.ok) {
            currentUser = await me.json();
            showUserMenu();
        } else {
            localStorage.removeItem('yolo_token');
        }
    } catch {}
}
function showUserMenu() {
    if (!currentUser) return;
    const toolbar = document.querySelector('.toolbar-left');
    const userEl = document.createElement('span');
    userEl.id = 'userMenu';
    userEl.style.cssText = 'font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:6px;';
    userEl.innerHTML = `${currentUser.email} <a href="#" id="logoutLink" style="color:var(--danger);text-decoration:none;font-size:11px;">Logout</a>`;
    toolbar.appendChild(userEl);
    document.getElementById('logoutLink').addEventListener('click', (e) => {
        e.preventDefault();
        localStorage.removeItem('yolo_token');
        currentUser = null;
        window.location.reload();
    });
}
let annotateMode = false;
let drawStart = null;        // {x, y} for annotation drawing

// === Colors ===
const BOX_COLORS = ['#58a6ff','#3fb950','#f0883e','#f85149','#d29922','#bc8cff','#79c0ff','#56d364','#ff9bce','#ffd33d'];
const labelColorMap = {};
let colorIdx = 0;
function getColor(label) {
    if (!labelColorMap[label]) { labelColorMap[label] = BOX_COLORS[colorIdx++ % BOX_COLORS.length]; }
    return labelColorMap[label];
}

// ==============================
// WebSocket
// ==============================
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}`);
    ws.onopen = () => { connectionStatus.textContent = 'Connected'; connectionStatus.className = 'connected'; };
    ws.onclose = () => { connectionStatus.textContent = 'Disconnected'; connectionStatus.className = 'disconnected'; setTimeout(connectWS, 2000); };
    ws.onerror = () => { connectionStatus.textContent = 'Error'; connectionStatus.className = 'disconnected'; };
    ws.onmessage = e => handleMessage(JSON.parse(e.data));
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'init':
            estimatedFrames = msg.estimatedFrames;
            currentSessionId = msg.sessionId;
            statusText.textContent = `Processing: ${msg.videoPath.split('/').pop()} @ ${msg.fps} fps`;
            break;
        case 'frame':
            // Always store in history
            frameHistory.push(msg);
            if (frameHistory.length > FRAME_HISTORY_MAX) frameHistory.shift();
            // Collect confidence values
            for (const d of msg.detections) confidenceData.push(d.confidence);
            // Accumulate heatmap
            if (heatmapToggle.checked) accumulateHeatmap(msg.detections);
            // Add to timeline
            timelineData.push({ changed: msg.sceneChanged });
            // If paused or browsing, don't update canvas
            if (!isPaused && browsingIndex === -1) {
                renderFrame(msg);
                updateFrameInfo(msg);
            }
            addLogEntry(msg);
            updateTimeline();
            enableExportBtns();
            break;
        case 'stats':
            updateStats(msg);
            break;
        case 'complete':
            onComplete(msg);
            break;
        case 'paused':
            isPaused = true;
            pauseBtn.textContent = 'Resume';
            showFrameBrowser();
            break;
        case 'resumed':
            isPaused = false;
            pauseBtn.textContent = 'Pause';
            hideFrameBrowser();
            browsingIndex = -1;
            if (frameHistory.length > 0) renderFrame(frameHistory[frameHistory.length - 1]);
            break;
        case 'error':
            statusText.textContent = `Error: ${msg.message}`;
            setProcessing(false);
            break;
        case 'train_log':
            console.log('[Train]', msg.line);
            break;
        case 'train_complete':
            statusText.textContent = `Training complete: ${msg.name}`;
            break;
        case 'train_error':
            statusText.textContent = `Training error: ${msg.message}`;
            break;
    }
}

// ==============================
// Frame Rendering
// ==============================
const frameImg = new Image();
function renderFrame(msg, highlightLabel = null) {
    frameImg.onload = () => {
        canvas.width = frameImg.naturalWidth;
        canvas.height = frameImg.naturalHeight;
        canvas.style.display = 'block';
        dropZone.classList.add('hidden');
        ctx.drawImage(frameImg, 0, 0);
        // Heatmap overlay
        if (heatmapToggle.checked && heatmapCanvas) {
            ctx.globalAlpha = 0.6;
            ctx.drawImage(heatmapCanvas, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }
        drawDetections(msg.detections, canvas.width, canvas.height, highlightLabel || searchTerm);
        // Draw annotations if any
        if (msg.annotations) drawAnnotations(msg.annotations, canvas.width, canvas.height);
    };
    frameImg.src = 'data:image/png;base64,' + msg.imageBase64;
}

function drawDetections(detections, w, h, highlight = '') {
    for (const det of detections) {
        const isMatch = highlight && det.label.toLowerCase().includes(highlight.toLowerCase());
        const color = isMatch ? '#ffd33d' : getColor(det.label);
        const x = det.bbox.x * w, y = det.bbox.y * h;
        const bw = det.bbox.w * w, bh = det.bbox.h * h;

        ctx.save();
        if (highlight && !isMatch) ctx.globalAlpha = 0.3;
        if (isMatch) { ctx.shadowBlur = 12; ctx.shadowColor = '#ffd33d'; }

        ctx.strokeStyle = color;
        ctx.lineWidth = isMatch ? 3 : 2;
        ctx.strokeRect(x, y, bw, bh);

        const text = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 12px -apple-system, sans-serif';
        const tw = ctx.measureText(text).width + 8;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 16, tw, 16);
        ctx.fillStyle = '#000';
        ctx.fillText(text, x + 4, y - 4);
        ctx.restore();
    }
}

function drawAnnotations(annotations, w, h) {
    // Store clickable regions for annotation actions
    window._annotationRegions = [];
    for (const ann of annotations) {
        const x = ann.bbox_x * w, y = ann.bbox_y * h;
        const bw = ann.bbox_w * w, bh = ann.bbox_h * h;
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#f0883e';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, bw, bh);
        ctx.setLineDash([]);

        // Label + user
        const userTag = ann.user_id ? ` (${(ann.user_id || '').slice(0, 6)})` : '';
        const labelText = ann.label + userTag;
        ctx.font = 'bold 11px monospace';
        ctx.fillStyle = '#f0883e';
        ctx.fillText(labelText, x + 2, y + bh - 4);

        // Small action icons top-right of bbox (only when paused/browsing)
        if (ann.id && (isPaused || browsingIndex >= 0)) {
            // Delete X
            const iconY = y + 2, iconSize = 14;
            ctx.fillStyle = 'rgba(248, 81, 73, 0.8)';
            ctx.fillRect(x + bw - iconSize - 2, iconY, iconSize, iconSize);
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 10px sans-serif';
            ctx.fillText('X', x + bw - iconSize + 1, iconY + 11);

            // Reclassify pencil
            ctx.fillStyle = 'rgba(88, 166, 255, 0.8)';
            ctx.fillRect(x + bw - iconSize * 2 - 4, iconY, iconSize, iconSize);
            ctx.fillStyle = '#fff';
            ctx.fillText('R', x + bw - iconSize * 2 - 1, iconY + 11);

            // Store clickable regions
            window._annotationRegions.push(
                { x: x + bw - iconSize - 2, y: iconY, w: iconSize, h: iconSize, action: 'delete', id: ann.id },
                { x: x + bw - iconSize * 2 - 4, y: iconY, w: iconSize, h: iconSize, action: 'reclassify', id: ann.id }
            );
        }

        ctx.restore();
    }
}

// Handle clicks on annotation action icons
canvas.addEventListener('click', (e) => {
    if (!window._annotationRegions || window._annotationRegions.length === 0) return;
    if (annotateMode && drawStart) return; // Don't interfere with drawing

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
    const cx = (e.clientX - rect.left) * scaleX, cy = (e.clientY - rect.top) * scaleY;

    for (const region of window._annotationRegions) {
        if (cx >= region.x && cx <= region.x + region.w && cy >= region.y && cy <= region.y + region.h) {
            if (region.action === 'delete') deleteAnnotation(region.id);
            else if (region.action === 'reclassify') reclassifyAnnotation(region.id);
            return;
        }
    }
});

// ==============================
// UI Updates
// ==============================
function updateFrameInfo(msg) {
    frameCounter.textContent = `Frame: ${msg.frameNum}`;
    frameTimestamp.textContent = `Time: ${msg.timestampStr}`;
    sceneStatus.textContent = `Scene: ${msg.sceneChanged ? 'CHANGED' : 'same'} (${msg.reason})`;
    sceneStatus.style.color = msg.sceneChanged ? 'var(--accent)' : 'var(--text-muted)';
    detectionCount.textContent = `Objects: ${msg.detections.length}`;
    if (estimatedFrames > 0) progressBar.style.width = Math.min(100, (msg.frameNum / estimatedFrames) * 100) + '%';
}

function updateStats(msg) {
    statInference.textContent = msg.performance.mean + 'ms';
    statP95.textContent = msg.performance.p95 + 'ms';
    statThroughput.textContent = msg.throughput + ' fps';
    statMemory.textContent = msg.memory.rss_mb + ' MB';
    statChanges.textContent = msg.sceneAnalysis.analyzedFrames;
    statSkipped.textContent = msg.sceneAnalysis.skippedFrames;
    statSkipRate.textContent = msg.sceneAnalysis.skipRate;
    statFrames.textContent = msg.frameNum;
    updateObjectChart(msg.objectFrequency);
    renderConfidenceHist();
    if (!presenceCanvas.classList.contains('hidden')) renderPresenceTimeline();
}

function updateTimeline() {
    const recent = timelineData.slice(-200);
    sceneTimeline.innerHTML = '';
    for (const e of recent) {
        const b = document.createElement('div');
        b.className = 'timeline-block ' + (e.changed ? 'changed' : 'skipped');
        sceneTimeline.appendChild(b);
    }
}

function updateObjectChart(freq) {
    const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 15);
    if (!sorted.length) return;
    const max = sorted[0][1];
    objectChart.innerHTML = '';
    for (const [label, count] of sorted) {
        const pct = (count / max) * 100;
        const row = document.createElement('div');
        row.className = 'obj-bar-row';
        row.innerHTML = `<span class="obj-bar-label">${label}</span><div class="obj-bar-track"><div class="obj-bar-fill" style="width:${pct}%;background:${getColor(label)}"></div></div><span class="obj-bar-count">${count}</span>`;
        objectChart.appendChild(row);
    }
}

function addLogEntry(msg) {
    const entry = document.createElement('div');
    entry.className = 'log-entry' + (msg.sceneChanged ? ' scene-change' : '');
    const labels = msg.detections.map(d => d.label).join(', ') || '(none)';
    entry.innerHTML = `<span class="log-frame">[${String(msg.frameNum).padStart(5)}]</span> ${msg.timestampStr} <span class="log-objects">${labels}</span> ` + (msg.sceneChanged ? `<em>${msg.reason}</em>` : '');
    entry.dataset.labels = labels.toLowerCase();
    if (searchTerm && !labels.toLowerCase().includes(searchTerm)) entry.classList.add('log-filtered-out');
    detectionLog.appendChild(entry);
    if (detectionLog.children.length > 500) detectionLog.removeChild(detectionLog.firstChild);
    detectionLog.scrollTop = detectionLog.scrollHeight;
}

function onComplete(msg) {
    setProcessing(false);
    const s = msg.aborted ? 'Stopped' : 'Complete';
    statusText.textContent = `${s}: ${msg.totalFrames} frames in ${msg.totalTime}s | Mean: ${msg.performance.mean}ms | Skip rate: ${msg.sceneAnalysis.skipRate}`;
    statInference.textContent = msg.performance.mean + 'ms';
    statP95.textContent = msg.performance.p95 + 'ms';
    statChanges.textContent = msg.sceneAnalysis.analyzedFrames;
    statSkipped.textContent = msg.sceneAnalysis.skippedFrames;
    statSkipRate.textContent = msg.sceneAnalysis.skipRate;
    statFrames.textContent = msg.totalFrames;
    updateObjectChart(msg.objectFrequency);
    progressBar.style.width = '100%';
    renderConfidenceHist();
}

// ==============================
// Pause / Resume / Frame Browser
// ==============================
pauseBtn.addEventListener('click', () => {
    if (!ws || ws.readyState !== 1) return;
    if (isPaused) {
        ws.send(JSON.stringify({ type: 'resume' }));
    } else {
        ws.send(JSON.stringify({ type: 'pause' }));
    }
});

function showFrameBrowser() {
    if (frameHistory.length === 0) return;
    browsingIndex = frameHistory.length - 1;
    frameScrubber.min = 0;
    frameScrubber.max = frameHistory.length - 1;
    frameScrubber.value = browsingIndex;
    frameScrubberDiv.classList.remove('hidden');
    filmstripDiv.classList.remove('hidden');
    generateFilmstrip();
    updateScrubberInfo();
}

function hideFrameBrowser() {
    frameScrubberDiv.classList.add('hidden');
    filmstripDiv.classList.add('hidden');
}

frameScrubber.addEventListener('input', () => {
    browsingIndex = +frameScrubber.value;
    const msg = frameHistory[browsingIndex];
    if (msg) {
        renderFrame(msg);
        updateFrameInfo(msg);
        updateScrubberInfo();
        highlightFilmstripThumb(browsingIndex);
    }
});

function updateScrubberInfo() {
    const msg = frameHistory[browsingIndex];
    if (msg) scrubberInfo.textContent = `Frame ${msg.frameNum} | ${msg.timestampStr}`;
}

function generateFilmstrip() {
    filmstripDiv.innerHTML = '';
    const step = Math.max(1, Math.floor(frameHistory.length / 80));
    for (let i = 0; i < frameHistory.length; i += step) {
        const msg = frameHistory[i];
        const thumb = document.createElement('canvas');
        thumb.width = 80; thumb.height = 45;
        thumb.dataset.idx = i;
        thumb.addEventListener('click', () => {
            browsingIndex = i;
            frameScrubber.value = i;
            renderFrame(frameHistory[i]);
            updateFrameInfo(frameHistory[i]);
            updateScrubberInfo();
            highlightFilmstripThumb(i);
        });
        filmstripDiv.appendChild(thumb);
        // Draw thumbnail async
        const img = new Image();
        const tctx = thumb.getContext('2d');
        img.onload = () => tctx.drawImage(img, 0, 0, 80, 45);
        img.src = 'data:image/png;base64,' + msg.imageBase64;
    }
    highlightFilmstripThumb(browsingIndex);
}

function highlightFilmstripThumb(idx) {
    const thumbs = filmstripDiv.querySelectorAll('canvas');
    thumbs.forEach(t => t.classList.toggle('active', +t.dataset.idx === idx));
}

// ==============================
// Object Search
// ==============================
searchBtn.addEventListener('click', () => performSearch(searchInput.value));
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') performSearch(searchInput.value); });
clearSearchBtn.addEventListener('click', clearSearch);

function performSearch(term) {
    searchTerm = term.toLowerCase().trim();
    if (!searchTerm) { clearSearch(); return; }
    searchResultFrames = [];
    frameHistory.forEach((msg, idx) => {
        if (msg.detections.some(d => d.label.toLowerCase().includes(searchTerm))) {
            searchResultFrames.push(idx);
        }
    });
    clearSearchBtn.classList.remove('hidden');
    renderSearchResults();
    filterDetectionLog();
    // Re-render current frame with highlight
    if (browsingIndex >= 0 && frameHistory[browsingIndex]) {
        renderFrame(frameHistory[browsingIndex], searchTerm);
    } else if (frameHistory.length > 0) {
        renderFrame(frameHistory[frameHistory.length - 1], searchTerm);
    }
}

function renderSearchResults() {
    searchResults.innerHTML = `<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">${searchResultFrames.length} frames match "${searchTerm}"</div>`;
    const limit = Math.min(searchResultFrames.length, 100);
    for (let i = 0; i < limit; i++) {
        const idx = searchResultFrames[i];
        const msg = frameHistory[idx];
        const labels = msg.detections.filter(d => d.label.toLowerCase().includes(searchTerm)).map(d => d.label).join(', ');
        const row = document.createElement('div');
        row.className = 'search-result';
        row.innerHTML = `<span class="sr-frame">F${msg.frameNum} ${msg.timestampStr}</span> <span class="sr-labels">${labels}</span>`;
        row.addEventListener('click', () => {
            browsingIndex = idx;
            if (!frameScrubberDiv.classList.contains('hidden')) {
                frameScrubber.value = idx;
                updateScrubberInfo();
                highlightFilmstripThumb(idx);
            }
            renderFrame(msg, searchTerm);
            updateFrameInfo(msg);
        });
        searchResults.appendChild(row);
    }
}

function clearSearch() {
    searchTerm = '';
    searchResultFrames = [];
    searchResults.innerHTML = '';
    searchInput.value = '';
    clearSearchBtn.classList.add('hidden');
    document.querySelectorAll('.log-filtered-out').forEach(el => el.classList.remove('log-filtered-out'));
    if (browsingIndex >= 0 && frameHistory[browsingIndex]) renderFrame(frameHistory[browsingIndex]);
}

function filterDetectionLog() {
    detectionLog.querySelectorAll('.log-entry').forEach(el => {
        const has = el.dataset.labels && el.dataset.labels.includes(searchTerm);
        el.classList.toggle('log-filtered-out', !has);
    });
}

// ==============================
// Heatmap
// ==============================
function accumulateHeatmap(detections) {
    if (!heatmapCanvas) {
        heatmapCanvas = document.createElement('canvas');
        heatmapCanvas.width = 640; heatmapCanvas.height = 480;
        heatmapCtx = heatmapCanvas.getContext('2d');
    }
    for (const det of detections) {
        const cx = det.bbox.x * 640 + det.bbox.w * 640 / 2;
        const cy = det.bbox.y * 480 + det.bbox.h * 480 / 2;
        const r = Math.max(det.bbox.w, det.bbox.h) * 320;
        const grad = heatmapCtx.createRadialGradient(cx, cy, 0, cx, cy, r);
        grad.addColorStop(0, 'rgba(255, 40, 40, 0.04)');
        grad.addColorStop(1, 'transparent');
        heatmapCtx.fillStyle = grad;
        heatmapCtx.fillRect(cx - r, cy - r, r * 2, r * 2);
    }
}

heatmapToggle.addEventListener('change', () => {
    if (browsingIndex >= 0 && frameHistory[browsingIndex]) renderFrame(frameHistory[browsingIndex]);
});

// ==============================
// Confidence Histogram
// ==============================
function renderConfidenceHist() {
    if (confidenceData.length === 0) return;
    const cv = confidenceHistCanvas;
    const c = cv.getContext('2d');
    cv.width = cv.clientWidth || 300;
    const w = cv.width, h = cv.height;
    c.clearRect(0, 0, w, h);

    const bins = new Array(10).fill(0);
    for (const v of confidenceData) bins[Math.min(9, Math.floor(v * 10))]++;
    const max = Math.max(...bins, 1);

    const barW = (w - 20) / 10;
    c.fillStyle = '#58a6ff';
    for (let i = 0; i < 10; i++) {
        const barH = (bins[i] / max) * (h - 20);
        c.fillRect(10 + i * barW + 2, h - 15 - barH, barW - 4, barH);
    }
    c.fillStyle = '#484f58';
    c.font = '9px sans-serif';
    for (let i = 0; i <= 10; i++) {
        c.fillText((i / 10).toFixed(1), 8 + i * barW, h - 2);
    }
}

// ==============================
// Object Presence Timeline
// ==============================
$('togglePresence').addEventListener('click', () => {
    presenceCanvas.classList.toggle('hidden');
    if (!presenceCanvas.classList.contains('hidden')) renderPresenceTimeline();
});

function renderPresenceTimeline() {
    if (frameHistory.length === 0) return;
    const cv = presenceCanvas;
    const c = cv.getContext('2d');
    cv.width = cv.clientWidth || 300;

    // Build label → frame set
    const labelFrames = {};
    frameHistory.forEach((msg, idx) => {
        for (const d of msg.detections) {
            if (!labelFrames[d.label]) labelFrames[d.label] = new Set();
            labelFrames[d.label].add(idx);
        }
    });

    const labels = Object.entries(labelFrames).sort((a, b) => b[1].size - a[1].size).slice(0, 20);
    const rowH = 14, labelW = 90, pad = 4;
    cv.height = Math.max(50, labels.length * rowH + 10);
    const trackW = cv.width - labelW - pad;

    c.clearRect(0, 0, cv.width, cv.height);
    c.font = '10px sans-serif';

    labels.forEach(([label, frames], i) => {
        const y = i * rowH + 4;
        c.fillStyle = '#8b949e';
        c.fillText(label, 2, y + 10);

        c.fillStyle = getColor(label);
        for (const idx of frames) {
            const x = labelW + (idx / frameHistory.length) * trackW;
            c.fillRect(x, y, Math.max(1, trackW / frameHistory.length), rowH - 2);
        }
    });
}

// ==============================
// Annotation Tool
// ==============================
annotateToggle.addEventListener('change', () => {
    annotateMode = annotateToggle.checked;
    canvas.classList.toggle('annotate-mode', annotateMode);
    annotationBar.classList.toggle('hidden', !annotateMode);
    if (annotateMode) loadClasses();
});

async function loadClasses() {
    const res = await fetch('/api/classes');
    const classes = await res.json();
    annotationLabel.innerHTML = '';
    for (const c of classes) {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c;
        annotationLabel.appendChild(opt);
    }
}

$('addLabelBtn').addEventListener('click', async () => {
    const val = $('newLabel').value.trim();
    if (!val) return;
    const res = await fetch('/api/classes');
    const classes = await res.json();
    if (!classes.includes(val)) {
        classes.push(val);
        await fetch('/api/classes', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ classes }) });
    }
    $('newLabel').value = '';
    loadClasses();
});

// Canvas annotation drawing
canvas.addEventListener('mousedown', e => {
    if (!annotateMode || browsingIndex < 0) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
    drawStart = { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
});

canvas.addEventListener('mouseup', async e => {
    if (!annotateMode || !drawStart || browsingIndex < 0) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
    const endX = (e.clientX - rect.left) * scaleX, endY = (e.clientY - rect.top) * scaleY;

    const x = Math.min(drawStart.x, endX) / canvas.width;
    const y = Math.min(drawStart.y, endY) / canvas.height;
    const w = Math.abs(endX - drawStart.x) / canvas.width;
    const h = Math.abs(endY - drawStart.y) / canvas.height;

    drawStart = null;
    if (w < 0.01 || h < 0.01) return; // Too small

    const msg = frameHistory[browsingIndex];
    if (!msg || !msg.frameId) return;

    const label = annotationLabel.value;
    const res = await apiFetch('/api/annotations', {
        method: 'POST',
        body: JSON.stringify({ frameId: msg.frameId, label, bbox: { x, y, w, h }, source: 'manual' })
    });
    const data = await res.json();
    $('annotationStatus').textContent = `Saved: ${label} (id: ${data.id})`;

    // Re-render with annotation
    if (!msg.annotations) msg.annotations = [];
    msg.annotations.push({ id: data.id, label, bbox_x: x, bbox_y: y, bbox_w: w, bbox_h: h, user_id: currentUser?.id });
    renderFrame(msg);
});

// --- Annotation Delete & Reclassify ---
async function deleteAnnotation(annotationId) {
    if (!confirm('Delete this annotation?')) return;
    const res = await apiFetch(`/api/annotations/${annotationId}`, { method: 'DELETE' });
    if (res.ok) {
        // Remove from current frame's annotations in memory
        if (browsingIndex >= 0 && frameHistory[browsingIndex]?.annotations) {
            frameHistory[browsingIndex].annotations = frameHistory[browsingIndex].annotations.filter(a => a.id !== annotationId);
            renderFrame(frameHistory[browsingIndex]);
        }
        $('annotationStatus').textContent = `Deleted annotation ${annotationId}`;
    }
}

async function reclassifyAnnotation(annotationId) {
    const classes = await (await fetch('/api/classes')).json();
    const newLabel = prompt('Reclassify to which label?\n\nAvailable: ' + classes.join(', '));
    if (!newLabel) return;
    const res = await apiFetch(`/api/annotations/${annotationId}`, {
        method: 'PATCH',
        body: JSON.stringify({ label: newLabel })
    });
    if (res.ok) {
        const updated = await res.json();
        if (browsingIndex >= 0 && frameHistory[browsingIndex]?.annotations) {
            const ann = frameHistory[browsingIndex].annotations.find(a => a.id === annotationId);
            if (ann) ann.label = updated.label;
            renderFrame(frameHistory[browsingIndex]);
        }
        $('annotationStatus').textContent = `Reclassified to: ${newLabel}`;
    }
}

canvas.addEventListener('mousemove', e => {
    if (!annotateMode || !drawStart || browsingIndex < 0) return;
    const msg = frameHistory[browsingIndex];
    if (!msg) return;
    // Re-render base frame then draw rubber-band rect
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
    const curX = (e.clientX - rect.left) * scaleX, curY = (e.clientY - rect.top) * scaleY;

    renderFrame(msg);
    ctx.save();
    ctx.strokeStyle = '#f0883e';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(drawStart.x, drawStart.y, curX - drawStart.x, curY - drawStart.y);
    ctx.restore();
});

// ==============================
// Export
// ==============================
exportCsvBtn.addEventListener('click', () => {
    if (frameHistory.length === 0) return;
    const rows = ['frameNum,timestamp,label,confidence,bbox_x,bbox_y,bbox_w,bbox_h,sceneChanged,reason'];
    for (const msg of frameHistory) {
        if (msg.detections.length === 0) {
            rows.push(`${msg.frameNum},${msg.timestamp},,,,,,${msg.sceneChanged},${msg.reason}`);
        } else {
            for (const d of msg.detections) {
                rows.push(`${msg.frameNum},${msg.timestamp},${d.label},${d.confidence},${d.bbox.x.toFixed(4)},${d.bbox.y.toFixed(4)},${d.bbox.w.toFixed(4)},${d.bbox.h.toFixed(4)},${msg.sceneChanged},${msg.reason}`);
            }
        }
    }
    downloadText(rows.join('\n'), 'detections.csv', 'text/csv');
});

exportFrameBtn.addEventListener('click', () => {
    canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `frame_annotated.png`;
        a.click();
        URL.revokeObjectURL(url);
    }, 'image/png');
});

function downloadText(text, filename, mime) {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
}

function enableExportBtns() {
    exportCsvBtn.disabled = false;
    exportFrameBtn.disabled = false;
}

// ==============================
// Controls & Lifecycle
// ==============================
function setProcessing(processing) {
    isProcessing = processing;
    startBtn.disabled = processing || !selectedVideoPath;
    stopBtn.disabled = !processing;
    pauseBtn.disabled = !processing;
    videoSelect.disabled = processing;
    fpsSlider.disabled = processing;
    confSlider.disabled = processing;
    if (!processing) { isPaused = false; pauseBtn.textContent = 'Pause'; hideFrameBrowser(); }
}

startBtn.addEventListener('click', () => {
    if (!ws || ws.readyState !== 1 || !selectedVideoPath) return;
    // Reset
    timelineData = []; sceneTimeline.innerHTML = '';
    objectChart.innerHTML = ''; detectionLog.innerHTML = '';
    progressBar.style.width = '0%';
    frameHistory = []; browsingIndex = -1; confidenceData = [];
    heatmapCanvas = null; heatmapCtx = null;
    colorIdx = 0; Object.keys(labelColorMap).forEach(k => delete labelColorMap[k]);
    searchTerm = ''; searchResults.innerHTML = ''; searchInput.value = '';
    clearSearchBtn.classList.add('hidden');

    ws.send(JSON.stringify({
        type: 'start',
        videoPath: selectedVideoPath,
        fps: parseFloat(fpsSlider.value),
        confidence: parseFloat(confSlider.value),
        maxFrames: parseInt(maxFramesInput.value, 10) || 0
    }));
    setProcessing(true);
    statusText.textContent = 'Starting...';
});

stopBtn.addEventListener('click', () => {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'stop' }));
    statusText.textContent = 'Stopping...';
});

fpsSlider.addEventListener('input', () => fpsValue.textContent = fpsSlider.value);
confSlider.addEventListener('input', () => confValue.textContent = parseFloat(confSlider.value).toFixed(2));
videoSelect.addEventListener('change', () => { selectedVideoPath = videoSelect.value; startBtn.disabled = !selectedVideoPath || isProcessing; });

// ==============================
// Video List & Upload
// ==============================
async function loadVideoList() {
    try {
        const res = await fetch('/api/videos');
        const videos = await res.json();
        while (videoSelect.options.length > 1) videoSelect.remove(1);
        const sources = {};
        for (const v of videos) { if (!sources[v.source]) sources[v.source] = []; sources[v.source].push(v); }
        for (const [source, list] of Object.entries(sources)) {
            const group = document.createElement('optgroup');
            group.label = source === 'library' ? 'Video Library' : 'Uploaded';
            for (const v of list) {
                const opt = document.createElement('option');
                opt.value = v.path;
                opt.textContent = `${v.name} (${(v.size / 1024 / 1024).toFixed(0)} MB)`;
                group.appendChild(opt);
            }
            videoSelect.appendChild(group);
        }
    } catch (err) { console.error('Failed to load video list:', err); }
}

fileInput.addEventListener('change', async () => { const f = fileInput.files[0]; if (f) { await uploadVideo(f); fileInput.value = ''; } });
async function uploadVideo(file) {
    statusText.textContent = `Uploading: ${file.name}...`;
    const fd = new FormData(); fd.append('video', file);
    try {
        const res = await fetch('/api/upload', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error) { statusText.textContent = `Upload failed: ${data.error}`; return; }
        statusText.textContent = `Uploaded: ${file.name}`;
        await loadVideoList();
        selectedVideoPath = data.path; videoSelect.value = data.path; startBtn.disabled = false;
    } catch (err) { statusText.textContent = `Upload error: ${err.message}`; }
}

// Drag & Drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', async e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && /\.(mp4|mkv|avi|mov|webm)$/i.test(file.name)) await uploadVideo(file);
    else statusText.textContent = 'Unsupported file format.';
});
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => e.preventDefault());

// ==============================
// Model Tips Modal
// ==============================
$('tipsBtn').addEventListener('click', () => $('tips-modal').classList.remove('hidden'));
$('closeTipsBtn').addEventListener('click', () => $('tips-modal').classList.add('hidden'));
document.querySelector('.modal-backdrop')?.addEventListener('click', () => $('tips-modal').classList.add('hidden'));

// ==============================
// Init
// ==============================
checkAuth();
connectWS();
loadVideoList();
