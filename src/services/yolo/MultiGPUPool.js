/**
 * MultiGPUPool — Distributes YOLO inference across multiple GPUs
 *
 * Creates one ONNX session per GPU and round-robins frames across them.
 * Falls back to a single CPU session if no GPUs are available.
 */

const path = require('path');
const fs = require('fs');
const sharp = require('sharp');
const { execSync } = require('child_process');

let ort = null;
try {
    ort = require('onnxruntime-node');
} catch (err) {
    console.warn('[MultiGPU] onnxruntime-node not available:', err.message);
}

const COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
];

class MultiGPUPool {
    constructor(options = {}) {
        this.modelPath = options.modelPath || this._resolveModelPath();
        this.confidenceThreshold = options.confidenceThreshold || 0.5;
        this.iouThreshold = options.iouThreshold || 0.45;
        this.inputSize = options.inputSize || 640;
        this.workers = [];        // Array of { session, gpuId, busy }
        this.initialized = false;
        this.initError = null;
        this._inferenceTimes = [];
        this._roundRobinIdx = 0;
    }

    _resolveModelPath() {
        // Prefer larger models on multi-GPU systems
        const modelsDir = path.join(__dirname, '../../../models');
        const preferred = [
            'yolov8x_1280.onnx',
            'yolov8l_1280.onnx',
            'yolov8x_640.onnx',
            'yolov8l_640.onnx',
            'yolov8m_640.onnx',
            'yolov8n_640.onnx'
        ];
        for (const name of preferred) {
            const p = path.join(modelsDir, name);
            if (fs.existsSync(p)) return p;
        }
        return path.join(modelsDir, 'yolov8n_640.onnx');
    }

    /**
     * Detect number of NVIDIA GPUs
     */
    _detectGPUCount() {
        try {
            const output = execSync('nvidia-smi --query-gpu=index --format=csv,noheader', {
                encoding: 'utf8', timeout: 5000
            });
            return output.trim().split('\n').length;
        } catch {
            return 0;
        }
    }

    /**
     * Initialize ONNX sessions — one per GPU
     */
    async initialize() {
        if (this.initialized) return true;

        if (!ort) {
            this.initError = 'onnxruntime-node is not installed';
            console.error(`[MultiGPU] ${this.initError}`);
            return false;
        }

        if (!fs.existsSync(this.modelPath)) {
            this.initError = `Model not found at ${this.modelPath}. Run: npm run download-model`;
            console.error(`[MultiGPU] ${this.initError}`);
            return false;
        }

        // Parse input size from model filename (e.g. yolov8x_1280.onnx)
        const match = this.modelPath.match(/_(\d+)\.onnx$/);
        if (match) this.inputSize = parseInt(match[1]);

        const gpuCount = this._detectGPUCount();
        console.log(`[MultiGPU] Detected ${gpuCount} GPU(s), model: ${path.basename(this.modelPath)}, input: ${this.inputSize}px`);

        const sessionOpts = {
            graphOptimizationLevel: 'all',
            intraOpNumThreads: 0,
            interOpNumThreads: 0
        };

        if (gpuCount > 0) {
            // Create one session per GPU
            for (let gpuId = 0; gpuId < gpuCount; gpuId++) {
                try {
                    const session = await ort.InferenceSession.create(this.modelPath, {
                        ...sessionOpts,
                        executionProviders: [{
                            name: 'cuda',
                            deviceId: gpuId
                        }]
                    });
                    this.workers.push({ session, gpuId, busy: false });
                    console.log(`[MultiGPU] GPU ${gpuId}: session ready`);
                } catch (err) {
                    console.warn(`[MultiGPU] GPU ${gpuId} failed: ${err.message}`);
                }
            }
        }

        // Fallback to CPU if no GPU sessions worked
        if (this.workers.length === 0) {
            try {
                const session = await ort.InferenceSession.create(this.modelPath, {
                    ...sessionOpts,
                    executionProviders: ['cpu']
                });
                this.workers.push({ session, gpuId: 'cpu', busy: false });
                console.log('[MultiGPU] CPU fallback session ready');
            } catch (err) {
                this.initError = err.message;
                console.error('[MultiGPU] Failed to initialize:', err.message);
                return false;
            }
        }

        this.initialized = true;
        this.initError = null;
        console.log(`[MultiGPU] Ready with ${this.workers.length} worker(s)`);
        return true;
    }

    /**
     * Get an available worker (round-robin with busy tracking)
     */
    _getWorker() {
        // Try round-robin first
        for (let i = 0; i < this.workers.length; i++) {
            const idx = (this._roundRobinIdx + i) % this.workers.length;
            if (!this.workers[idx].busy) {
                this._roundRobinIdx = (idx + 1) % this.workers.length;
                return this.workers[idx];
            }
        }
        // All busy — return next in round-robin anyway (will queue in ONNX runtime)
        const idx = this._roundRobinIdx;
        this._roundRobinIdx = (idx + 1) % this.workers.length;
        return this.workers[idx];
    }

    /**
     * Run detection on a single frame using an available GPU worker
     */
    async detect(imageBuffer) {
        if (!this.initialized) return [];

        const worker = this._getWorker();
        worker.busy = true;
        const startTime = performance.now();

        try {
            const { data: rawPixels } = await sharp(imageBuffer)
                .resize(this.inputSize, this.inputSize, { fit: 'fill' })
                .removeAlpha()
                .raw()
                .toBuffer({ resolveWithObject: true });

            const inputTensor = this._preprocessToTensor(rawPixels);

            const feeds = {};
            feeds[worker.session.inputNames[0]] = inputTensor;
            const results = await worker.session.run(feeds);
            const output = results[worker.session.outputNames[0]];

            const detections = this._postprocess(output);

            const elapsed = performance.now() - startTime;
            this._inferenceTimes.push(elapsed);

            return detections;
        } catch (err) {
            console.error(`[MultiGPU] Detection error on GPU ${worker.gpuId}:`, err.message);
            return [];
        } finally {
            worker.busy = false;
        }
    }

    /**
     * Process multiple frames in parallel across all GPUs
     * Returns array of { frameIdx, detections } in order
     */
    async detectBatch(imageBuffers) {
        const promises = imageBuffers.map((buf, idx) =>
            this.detect(buf).then(detections => ({ frameIdx: idx, detections }))
        );
        return Promise.all(promises);
    }

    _preprocessToTensor(rawPixels) {
        const pixels = this.inputSize * this.inputSize;
        const float32Data = new Float32Array(3 * pixels);

        for (let i = 0; i < pixels; i++) {
            float32Data[i] = rawPixels[i * 3] / 255.0;
            float32Data[pixels + i] = rawPixels[i * 3 + 1] / 255.0;
            float32Data[2 * pixels + i] = rawPixels[i * 3 + 2] / 255.0;
        }

        return new ort.Tensor('float32', float32Data, [1, 3, this.inputSize, this.inputSize]);
    }

    _postprocess(output) {
        const data = output.data;
        const [batch, channels, numAnchors] = output.dims;
        const numClasses = channels - 4;

        const candidates = [];

        for (let i = 0; i < numAnchors; i++) {
            const cx = data[0 * numAnchors + i];
            const cy = data[1 * numAnchors + i];
            const w = data[2 * numAnchors + i];
            const h = data[3 * numAnchors + i];

            let maxScore = 0;
            let maxClassIdx = 0;
            for (let c = 0; c < numClasses; c++) {
                const score = data[(4 + c) * numAnchors + i];
                if (score > maxScore) {
                    maxScore = score;
                    maxClassIdx = c;
                }
            }

            if (maxScore >= this.confidenceThreshold) {
                candidates.push({
                    label: COCO_LABELS[maxClassIdx] || `class_${maxClassIdx}`,
                    confidence: maxScore,
                    bbox: {
                        x: (cx - w / 2) / this.inputSize,
                        y: (cy - h / 2) / this.inputSize,
                        w: w / this.inputSize,
                        h: h / this.inputSize
                    },
                    _classIdx: maxClassIdx
                });
            }
        }

        const nmsResults = this._nms(candidates);
        return nmsResults.map(({ _classIdx, ...rest }) => rest);
    }

    _nms(detections) {
        if (detections.length === 0) return [];
        const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
        const kept = [];

        while (sorted.length > 0) {
            const best = sorted.shift();
            kept.push(best);
            for (let i = sorted.length - 1; i >= 0; i--) {
                if (sorted[i]._classIdx === best._classIdx &&
                    this._iou(best.bbox, sorted[i].bbox) > this.iouThreshold) {
                    sorted.splice(i, 1);
                }
            }
        }
        return kept;
    }

    _iou(a, b) {
        const x1 = Math.max(a.x, b.x);
        const y1 = Math.max(a.y, b.y);
        const x2 = Math.min(a.x + a.w, b.x + b.w);
        const y2 = Math.min(a.y + a.h, b.y + b.h);
        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const union = a.w * a.h + b.w * b.h - intersection;
        return union > 0 ? intersection / union : 0;
    }

    setConfidenceThreshold(threshold) {
        this.confidenceThreshold = Math.max(0.05, Math.min(0.95, threshold));
    }

    getPerformanceStats() {
        if (this._inferenceTimes.length === 0) {
            return { count: 0, mean: 0, p95: 0, max: 0, min: 0, gpuCount: this.workers.length };
        }
        const sorted = [...this._inferenceTimes].sort((a, b) => a - b);
        const sum = sorted.reduce((a, b) => a + b, 0);
        const p95Idx = Math.floor(sorted.length * 0.95);
        return {
            count: sorted.length,
            mean: Math.round(sum / sorted.length),
            p95: Math.round(sorted[p95Idx] || sorted[sorted.length - 1]),
            max: Math.round(sorted[sorted.length - 1]),
            min: Math.round(sorted[0]),
            gpuCount: this.workers.length
        };
    }

    resetPerformanceStats() {
        this._inferenceTimes = [];
    }

    async dispose() {
        for (const worker of this.workers) {
            try { await worker.session.release(); } catch {}
        }
        this.workers = [];
        this.initialized = false;
    }
}

module.exports = MultiGPUPool;
