/**
 * YOLO Video Game Observer — Web Server
 *
 * Express + WebSocket server with:
 * - Real-time YOLO detection streaming
 * - SQLite persistence (sessions, frames, detections, annotations)
 * - Pause/resume video processing
 * - Annotation management + multi-format export
 * - Model training pipeline
 */

const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const sharp = require('sharp');

// Load .env if present
try { require('dotenv').config(); } catch {}

const YOLODetectionService = require('./src/services/yolo/YOLODetectionService');
const YOLOSceneAnalyzer = require('./src/services/yolo/YOLOSceneAnalyzer');
const SessionStore = require('./src/services/db/SessionStore');
const AnnotationManager = require('./src/services/annotation/AnnotationManager');
const TrainingPipeline = require('./src/services/training/TrainingPipeline');
const SupabaseAuth = require('./src/services/auth/SupabaseAuth');
const { getVideoInfo, createFrameStream, extractPNGFrames, formatTime } = require('./src/services/video/VideoFrameExtractor');

const PORT = process.env.PORT || 6600;
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const FRAMES_DIR = path.join(__dirname, 'data/frames');

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(FRAMES_DIR)) fs.mkdirSync(FRAMES_DIR, { recursive: true });

// --- Express setup ---

const app = express();

// --- Simple password gate (when AUTH_PASSWORD is set) ---
const AUTH_PASSWORD = process.env.AUTH_PASSWORD;
if (AUTH_PASSWORD) {
    console.log('[Auth] Password protection enabled');
    app.use((req, res, next) => {
        // Allow WebSocket upgrades (handled separately)
        if (req.headers.upgrade === 'websocket') return next();
        // Check for session cookie
        const cookies = req.headers.cookie || '';
        const authed = cookies.split(';').some(c => c.trim() === 'miranda_auth=1');
        if (authed) return next();
        // Login page and login endpoint are always accessible
        if (req.path === '/login' || req.path === '/api/login') return next();
        // Redirect to login
        res.redirect('/login');
    });
    app.get('/login', (req, res) => {
        res.send(`<!DOCTYPE html><html><head><title>MIRANDA — Login</title>
        <style>body{font-family:system-ui;background:#0d1117;color:#c9d1d9;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}
        .card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:32px;width:340px}
        h1{font-size:20px;color:#58a6ff;margin:0 0 4px}p{font-size:13px;color:#8b949e;margin:0 0 20px}
        input{width:100%;padding:10px;font-size:14px;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;box-sizing:border-box;margin-bottom:12px}
        button{width:100%;padding:10px;font-size:14px;font-weight:600;background:#58a6ff;color:#000;border:none;border-radius:6px;cursor:pointer}
        .err{color:#f85149;font-size:13px;display:none}</style></head>
        <body><div class="card"><h1>🎯 MIRANDA YOLO</h1><p>Enter password to continue</p>
        <form onsubmit="login(event)"><input type="password" id="pw" placeholder="Password" autofocus>
        <div class="err" id="err">Incorrect password</div>
        <button type="submit">Sign In</button></form></div>
        <script>async function login(e){e.preventDefault();const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:document.getElementById('pw').value})});if(r.ok)location.href='/';else document.getElementById('err').style.display='block'}</script>
        </body></html>`);
    });
    app.post('/api/login', express.json(), (req, res) => {
        if (req.body && req.body.password === AUTH_PASSWORD) {
            res.setHeader('Set-Cookie', 'miranda_auth=1; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800');
            res.json({ ok: true });
        } else {
            res.status(401).json({ error: 'Wrong password' });
        }
    });
}

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
// Serve saved frame images
app.use('/data/frames', express.static(FRAMES_DIR));

const upload = multer({
    dest: UPLOADS_DIR,
    limits: { fileSize: 100 * 1024 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        if (['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'].includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error('Unsupported video format'));
        }
    }
});

// --- Services ---

const store = new SessionStore();
const auth = new SupabaseAuth();
const detector = new YOLODetectionService({ confidenceThreshold: 0.3 });
const annotationMgr = new AnnotationManager(store);
const trainingPipeline = new TrainingPipeline(store, annotationMgr, detector);
let detectorReady = false;

// Auth middleware shortcuts
const requireAuth = auth.requireAuth();
const optionalAuth = auth.optionalAuth();

(async () => {
    detectorReady = await detector.initialize();
    if (detectorReady) {
        console.log('[Server] YOLO model loaded');
    } else {
        console.error('[Server] YOLO model failed to load:', detector.initError);
    }
})();

// --- State ---

let currentJob = null; // { id, sessionId, ffmpegProc, aborted, paused }

// =====================================================
// REST API Routes
// =====================================================

// --- Video Management ---

app.post('/api/upload', upload.single('video'), (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No video file provided' });
    const ext = path.extname(req.file.originalname);
    const newPath = path.join(UPLOADS_DIR, req.file.filename + ext);
    fs.renameSync(req.file.path, newPath);
    res.json({ id: req.file.filename, name: req.file.originalname, path: newPath, size: req.file.size });
});

app.get('/api/videos', (req, res) => {
    const videos = [];
    const uploadedFiles = fs.readdirSync(UPLOADS_DIR).filter(f => /\.(mp4|mkv|avi|mov|webm|flv|wmv)$/i.test(f));
    for (const f of uploadedFiles) {
        const fullPath = path.join(UPLOADS_DIR, f);
        const stat = fs.statSync(fullPath);
        videos.push({ name: f, path: fullPath, size: stat.size, source: 'upload' });
    }
    const movieDir = process.env.MOVIE_DIR || path.join(process.env.HOME, 'Documents/src/movies');
    if (fs.existsSync(movieDir)) {
        const movieFiles = fs.readdirSync(movieDir).filter(f => /\.(mp4|mkv|avi|mov|webm)$/i.test(f));
        for (const f of movieFiles) {
            const fullPath = path.join(movieDir, f);
            const stat = fs.statSync(fullPath);
            videos.push({ name: f, path: fullPath, size: stat.size, source: 'library' });
        }
    }
    res.json(videos);
});

// --- Auth Config (tells frontend whether Supabase is available) ---

app.get('/api/auth/config', (req, res) => {
    res.json(auth.getConfig());
});

app.get('/api/me', requireAuth, (req, res) => {
    res.json(req.user ? { id: req.user.id, email: req.user.email, metadata: req.user.user_metadata } : null);
});

// --- Session Management ---

app.get('/api/sessions', optionalAuth, (req, res) => {
    const userId = req.user?.id || null;
    res.json(store.listSessions(userId));
});

app.get('/api/sessions/:id', optionalAuth, (req, res) => {
    const session = store.getSession(+req.params.id);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    res.json(session);
});

app.get('/api/sessions/:id/search', (req, res) => {
    const label = req.query.label;
    if (!label) return res.status(400).json({ error: 'label query param required' });
    res.json(store.searchByLabel(+req.params.id, label));
});

app.get('/api/sessions/:id/frames/:frameNum', (req, res) => {
    const frame = store.getFrame(+req.params.id, +req.params.frameNum);
    if (!frame) return res.status(404).json({ error: 'Frame not found' });
    res.json(frame);
});

app.get('/api/sessions/:id/export', (req, res) => {
    const session = store.getSession(+req.params.id);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    const frequency = store.getObjectFrequency(+req.params.id);
    res.json({ session, objectFrequency: frequency });
});

// --- Session Sharing ---

app.post('/api/sessions/:id/share', requireAuth, (req, res) => {
    const session = store.getSession(+req.params.id);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    if (session.user_id !== req.user?.id) return res.status(403).json({ error: 'Only the owner can share' });
    const { userId, permission } = req.body;
    if (!userId) return res.status(400).json({ error: 'userId required' });
    store.shareSession(+req.params.id, userId, permission || 'annotate');
    res.json({ shared: true });
});

app.delete('/api/sessions/:id/share/:userId', requireAuth, (req, res) => {
    const session = store.getSession(+req.params.id);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    if (session.user_id !== req.user?.id) return res.status(403).json({ error: 'Only the owner can revoke' });
    store.unshareSession(+req.params.id, req.params.userId);
    res.json({ revoked: true });
});

app.patch('/api/sessions/:id/visibility', requireAuth, (req, res) => {
    const session = store.getSession(+req.params.id);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    if (session.user_id !== req.user?.id) return res.status(403).json({ error: 'Only the owner can change visibility' });
    const { visibility } = req.body;
    if (!['public', 'private'].includes(visibility)) return res.status(400).json({ error: 'visibility must be public or private' });
    store.setSessionVisibility(+req.params.id, visibility);
    res.json({ visibility });
});

app.get('/api/sessions/:id/shares', optionalAuth, (req, res) => {
    res.json(store.getSessionShares(+req.params.id));
});

// --- Annotations (full CRUD) ---

app.post('/api/annotations', optionalAuth, (req, res) => {
    const { frameId, label, bbox, source } = req.body;
    if (!frameId || !label || !bbox) return res.status(400).json({ error: 'frameId, label, bbox required' });
    const userId = req.user?.id || null;
    const id = store.saveAnnotation(frameId, label, bbox, source || 'manual', userId);
    res.json({ id });
});

app.patch('/api/annotations/:id', optionalAuth, (req, res) => {
    const { label } = req.body;
    if (!label) return res.status(400).json({ error: 'label required' });
    const userId = req.user?.id || null;
    const result = store.reclassifyAnnotation(+req.params.id, label, userId);
    if (!result) return res.status(404).json({ error: 'Annotation not found or already deleted' });
    res.json(result);
});

app.delete('/api/annotations/:id', optionalAuth, (req, res) => {
    const userId = req.user?.id || null;
    const result = store.deleteAnnotation(+req.params.id, userId);
    if (!result) return res.status(404).json({ error: 'Annotation not found or already deleted' });
    res.json(result);
});

app.get('/api/annotations/:id/history', (req, res) => {
    const result = store.getAnnotationHistory(+req.params.id);
    if (!result) return res.status(404).json({ error: 'Annotation not found' });
    res.json(result);
});

app.post('/api/annotations/promote/:detectionId', optionalAuth, (req, res) => {
    const userId = req.user?.id || null;
    const id = store.promoteDetection(+req.params.detectionId, userId);
    if (!id) return res.status(404).json({ error: 'Detection not found' });
    res.json({ annotationId: id });
});

app.get('/api/annotations/stats', (req, res) => {
    res.json(store.getAnnotationStats());
});

app.get('/api/classes', (req, res) => {
    res.json(annotationMgr.getClasses());
});

app.put('/api/classes', (req, res) => {
    annotationMgr.saveClasses(req.body.classes);
    res.json({ saved: true });
});

// --- Export ---

app.post('/api/export/yolo', async (req, res) => {
    try {
        const { name } = req.body;
        if (!name) return res.status(400).json({ error: 'name required' });
        const result = await annotationMgr.exportYOLO(name, req.body);
        res.json(result);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

app.post('/api/export/coco', (req, res) => {
    try {
        const outputPath = path.join(__dirname, 'data/exports', `coco_${Date.now()}.json`);
        fs.mkdirSync(path.dirname(outputPath), { recursive: true });
        const result = annotationMgr.exportCOCO(outputPath);
        res.json(result);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

app.post('/api/export/csv', (req, res) => {
    try {
        const outputPath = path.join(__dirname, 'data/exports', `annotations_${Date.now()}.csv`);
        fs.mkdirSync(path.dirname(outputPath), { recursive: true });
        const result = annotationMgr.exportCSV(outputPath);
        res.json(result);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

app.post('/api/export/roboflow', async (req, res) => {
    try {
        const { apiKey, projectId } = req.body;
        const result = await annotationMgr.uploadToRoboflow(apiKey, projectId);
        res.json(result);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// --- Models ---

app.get('/api/models', (req, res) => {
    res.json(store.listModels());
});

app.post('/api/models/:id/activate', async (req, res) => {
    try {
        const result = await trainingPipeline.activateModel(+req.params.id);
        res.json(result);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

app.post('/api/stop', (req, res) => {
    if (currentJob) {
        currentJob.aborted = true;
        if (currentJob.ffmpegProc) currentJob.ffmpegProc.kill('SIGTERM');
        res.json({ stopped: true });
    } else {
        res.json({ stopped: false, message: 'No active job' });
    }
});

// =====================================================
// WebSocket
// =====================================================

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

wss.on('connection', (ws) => {
    console.log('[WS] Client connected');

    ws.on('message', async (raw) => {
        let msg;
        try { msg = JSON.parse(raw); } catch { return; }

        switch (msg.type) {
            case 'start':
                await handleStartProcessing(ws, msg);
                break;

            case 'stop':
                if (currentJob) {
                    currentJob.aborted = true;
                    if (currentJob.ffmpegProc) currentJob.ffmpegProc.kill('SIGTERM');
                }
                break;

            case 'pause':
                if (currentJob && !currentJob.paused && currentJob.ffmpegProc) {
                    currentJob.paused = true;
                    currentJob.ffmpegProc.kill('SIGSTOP');
                    ws.send(JSON.stringify({ type: 'paused' }));
                }
                break;

            case 'resume':
                if (currentJob && currentJob.paused && currentJob.ffmpegProc) {
                    currentJob.paused = false;
                    currentJob.ffmpegProc.kill('SIGCONT');
                    ws.send(JSON.stringify({ type: 'resumed' }));
                }
                break;

            case 'train':
                await handleTraining(ws, msg);
                break;
        }
    });

    ws.on('close', () => {
        console.log('[WS] Client disconnected');
        // Resume ffmpeg if client disconnects while paused
        if (currentJob && currentJob.paused && currentJob.ffmpegProc) {
            try { currentJob.ffmpegProc.kill('SIGCONT'); } catch {}
        }
    });
});

// =====================================================
// Processing Logic
// =====================================================

async function handleStartProcessing(ws, msg) {
    const { videoPath, fps = 1, confidence = 0.3, maxFrames = 0 } = msg;

    if (!detectorReady) {
        ws.send(JSON.stringify({ type: 'error', message: 'YOLO model not loaded' }));
        return;
    }

    if (currentJob && !currentJob.aborted) {
        ws.send(JSON.stringify({ type: 'error', message: 'A job is already running. Stop it first.' }));
        return;
    }

    if (!fs.existsSync(videoPath)) {
        ws.send(JSON.stringify({ type: 'error', message: `Video not found: ${videoPath}` }));
        return;
    }

    detector.setConfidenceThreshold(confidence);
    detector.resetPerformanceStats();

    const analyzer = new YOLOSceneAnalyzer({ changeThreshold: 0.3, forceAnalyzeEveryN: 10 });

    let videoInfo = { duration: 0, width: 0, height: 0 };
    try { videoInfo = await getVideoInfo(videoPath); } catch {}

    const estimatedFrames = maxFrames > 0 ? maxFrames : Math.floor(videoInfo.duration * fps) || 0;

    // Create DB session
    const sessionId = store.createSession(videoPath, { fps, confidence });
    const sessionFramesDir = path.join(FRAMES_DIR, String(sessionId));
    if (!fs.existsSync(sessionFramesDir)) fs.mkdirSync(sessionFramesDir, { recursive: true });

    ws.send(JSON.stringify({
        type: 'init',
        videoPath,
        sessionId,
        duration: videoInfo.duration,
        estimatedFrames,
        fps,
        confidence
    }));

    const ffmpegProc = createFrameStream(videoPath, fps, detector.inputSize, detector.inputSize);
    const jobId = Date.now().toString(36);
    currentJob = { id: jobId, sessionId, ffmpegProc, aborted: false, paused: false };

    const objectFrequency = {};
    let frameNum = 0;
    const startTime = performance.now();

    try {
        for await (const pngBuffer of extractPNGFrames(ffmpegProc.stdout)) {
            if (currentJob.aborted || ws.readyState !== 1) break;

            frameNum++;
            if (maxFrames > 0 && frameNum > maxFrames) break;

            const timestamp = (frameNum - 1) / fps;
            const detections = await detector.detect(pngBuffer);
            const scene = analyzer.analyzeFrame(detections);

            // Save frame image to disk as JPEG
            const frameName = `frame_${String(frameNum).padStart(5, '0')}.jpg`;
            const framePath = path.join(sessionFramesDir, frameName);
            try {
                await sharp(pngBuffer).jpeg({ quality: 80 }).toFile(framePath);
            } catch {}

            // Save to SQLite
            const detMapped = detections.map(d => ({
                label: d.label,
                confidence: +d.confidence.toFixed(3),
                bbox: d.bbox
            }));
            const frameId = store.saveFrame(sessionId, {
                frameNum,
                timestamp: +timestamp.toFixed(2),
                imagePath: framePath,
                sceneChanged: scene.changed,
                reason: scene.reason,
                jaccard: +scene.jaccard.toFixed(3)
            }, detMapped);

            for (const d of detections) {
                objectFrequency[d.label] = (objectFrequency[d.label] || 0) + 1;
            }

            // Send frame data to client
            const frameMsg = {
                type: 'frame',
                frameNum,
                frameId,
                sessionId,
                timestamp: +timestamp.toFixed(2),
                timestampStr: formatTime(timestamp),
                imageBase64: pngBuffer.toString('base64'),
                detections: detMapped,
                sceneChanged: scene.changed,
                reason: scene.reason,
                jaccard: +scene.jaccard.toFixed(3)
            };

            if (ws.readyState === 1) {
                ws.send(JSON.stringify(frameMsg));
            }

            // Stats every 10 frames
            if (frameNum % 10 === 0) {
                const perfStats = detector.getPerformanceStats();
                const sceneStats = analyzer.getStats();
                const mem = process.memoryUsage();
                const elapsed = (performance.now() - startTime) / 1000;

                if (ws.readyState === 1) {
                    ws.send(JSON.stringify({
                        type: 'stats',
                        frameNum,
                        sessionId,
                        elapsed: +elapsed.toFixed(1),
                        throughput: +(frameNum / elapsed).toFixed(1),
                        estimatedFrames,
                        performance: perfStats,
                        sceneAnalysis: sceneStats,
                        objectFrequency,
                        memory: { rss_mb: +(mem.rss / 1024 / 1024).toFixed(1) }
                    }));
                }
            }
        }
    } catch (err) {
        if (!currentJob.aborted) {
            console.error('[Process] Error:', err.message);
            if (ws.readyState === 1) {
                ws.send(JSON.stringify({ type: 'error', message: err.message }));
            }
        }
    }

    try { ffmpegProc.kill('SIGTERM'); } catch {}

    const totalTime = (performance.now() - startTime) / 1000;
    const perfStats = detector.getPerformanceStats();
    const sceneStats = analyzer.getStats();

    // Complete session in DB
    store.completeSession(sessionId, {
        aborted: currentJob.aborted,
        totalFrames: frameNum,
        meanInference: perfStats.mean,
        skipRate: sceneStats.skipRate
    });

    // Auto-export JSON report
    const resultsDir = path.join(__dirname, 'results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    const videoName = path.basename(videoPath, path.extname(videoPath));
    const reportPath = path.join(resultsDir, `${videoName}_${sessionId}.json`);
    try {
        fs.writeFileSync(reportPath, JSON.stringify({
            session: store.getSession(sessionId),
            performance: perfStats,
            sceneAnalysis: sceneStats,
            objectFrequency
        }, null, 2));
    } catch {}

    if (ws.readyState === 1) {
        ws.send(JSON.stringify({
            type: 'complete',
            sessionId,
            aborted: currentJob.aborted,
            totalFrames: frameNum,
            totalTime: +totalTime.toFixed(1),
            performance: perfStats,
            sceneAnalysis: sceneStats,
            objectFrequency
        }));
    }

    currentJob = null;
    console.log(`[Process] Session ${sessionId}: ${frameNum} frames in ${totalTime.toFixed(1)}s`);
}

// =====================================================
// Training
// =====================================================

async function handleTraining(ws, msg) {
    const { name, epochs = 100, imgSize = 640, baseModel = 'yolov8n.pt' } = msg;

    if (!name) {
        ws.send(JSON.stringify({ type: 'train_error', message: 'Model name required' }));
        return;
    }

    ws.send(JSON.stringify({ type: 'train_started', name }));

    try {
        const result = await trainingPipeline.train(name, { epochs, imgSize, baseModel }, (line) => {
            if (ws.readyState === 1) {
                ws.send(JSON.stringify({ type: 'train_log', line }));
            }
        });

        if (ws.readyState === 1) {
            ws.send(JSON.stringify({ type: 'train_complete', ...result }));
        }
    } catch (err) {
        if (ws.readyState === 1) {
            ws.send(JSON.stringify({ type: 'train_error', message: err.message }));
        }
    }
}

// =====================================================
// Start Server
// =====================================================

server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        const fallback = PORT + 1;
        console.log(`  Port ${PORT} is in use, trying ${fallback}...`);
        server.listen(fallback, '0.0.0.0', () => {
            console.log(`\n  YOLO Video Game Observer`);
            console.log(`  http://localhost:${fallback}\n`);
        });
    } else {
        throw err;
    }
});

server.listen(PORT, '0.0.0.0', () => {
    console.log(`\n  YOLO Video Game Observer`);
    console.log(`  http://localhost:${PORT}\n`);
});
