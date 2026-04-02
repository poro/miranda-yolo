/**
 * YOLODetectionService - Local object detection using YOLOv8 via ONNX Runtime
 *
 * Provides real-time object detection on gameplay screenshots for:
 * - Smart scene change detection (reducing unnecessary LLM API calls)
 * - Standalone behavioral analytics (object counts, scene classification)
 *
 * Adapted from MIRANDA's detection service for standalone prototype use.
 */

const path = require('path');
const fs = require('fs');
const sharp = require('sharp');

// Lazy-load onnxruntime-node to allow graceful fallback
let ort = null;
try {
    ort = require('onnxruntime-node');
} catch (err) {
    console.warn('[YOLO] onnxruntime-node not available:', err.message);
}

// COCO dataset class labels (80 classes)
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

class YOLODetectionService {
    constructor(options = {}) {
        this.modelPath = options.modelPath || this._resolveModelPath();
        this.confidenceThreshold = options.confidenceThreshold || 0.5;
        this.iouThreshold = options.iouThreshold || 0.45;
        this.inputSize = 640;
        this.session = null;
        this.initialized = false;
        this.initError = null;

        // Performance tracking
        this._inferenceTimes = [];
    }

    /**
     * Resolve model path - checks project models/ directory and cwd
     */
    _resolveModelPath() {
        const candidates = [
            path.join(__dirname, '../../../models/yolov8n.onnx'),
            path.join(process.cwd(), 'models/yolov8n.onnx')
        ];
        for (const p of candidates) {
            if (fs.existsSync(p)) return p;
        }
        return candidates[0];
    }

    /**
     * Initialize the ONNX inference session
     * @returns {Promise<boolean>} true if initialization succeeded
     */
    async initialize() {
        if (this.initialized) return true;

        if (!ort) {
            this.initError = 'onnxruntime-node is not installed';
            console.error(`[YOLO] ${this.initError}`);
            return false;
        }

        if (!fs.existsSync(this.modelPath)) {
            this.initError = `Model not found at ${this.modelPath}. Run: npm run download-model`;
            console.error(`[YOLO] ${this.initError}`);
            return false;
        }

        try {
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['cpu'],
                graphOptimizationLevel: 'all',
                intraOpNumThreads: 0,  // 0 = use all CPU cores
                interOpNumThreads: 0
            });
            this.initialized = true;
            this.initError = null;
            console.log('[YOLO] Model loaded successfully from', this.modelPath);
            return true;
        } catch (err) {
            this.initError = err.message;
            console.error('[YOLO] Failed to initialize:', err.message);
            return false;
        }
    }

    /**
     * Run object detection on an image buffer
     * @param {Buffer} imageBuffer - Raw image buffer (PNG/JPEG)
     * @returns {Promise<Array>} Array of {label, confidence, bbox: {x, y, w, h}}
     */
    async detect(imageBuffer) {
        if (!this.initialized) {
            return [];
        }

        const startTime = performance.now();

        try {
            // Preprocess: resize to 640x640 if needed, decode to raw RGB
            const { data: rawPixels } = await sharp(imageBuffer)
                .resize(this.inputSize, this.inputSize, { fit: 'fill' })
                .removeAlpha()
                .raw()
                .toBuffer({ resolveWithObject: true });

            const inputTensor = this._preprocessToTensor(rawPixels);

            // Run inference
            const feeds = {};
            const inputName = this.session.inputNames[0];
            feeds[inputName] = inputTensor;

            const results = await this.session.run(feeds);
            const outputName = this.session.outputNames[0];
            const output = results[outputName];

            // Post-process
            const detections = this._postprocess(output);

            const elapsed = performance.now() - startTime;
            this._inferenceTimes.push(elapsed);

            return detections;
        } catch (err) {
            console.error('[YOLO] Detection error:', err.message);
            return [];
        }
    }

    /**
     * Convenience: detect objects from an image file path
     * @param {string} filePath - Path to image file
     * @returns {Promise<Array>} Array of detections
     */
    async detectFromFile(filePath) {
        const imageBuffer = fs.readFileSync(filePath);
        return this.detect(imageBuffer);
    }

    /**
     * Convert raw RGB pixel buffer to float32 CHW tensor
     */
    _preprocessToTensor(rawPixels) {
        const pixels = this.inputSize * this.inputSize;
        const float32Data = new Float32Array(3 * pixels);

        // Convert HWC uint8 [0-255] to CHW float32 [0-1]
        for (let i = 0; i < pixels; i++) {
            float32Data[i] = rawPixels[i * 3] / 255.0;               // R channel
            float32Data[pixels + i] = rawPixels[i * 3 + 1] / 255.0;  // G channel
            float32Data[2 * pixels + i] = rawPixels[i * 3 + 2] / 255.0; // B channel
        }

        return new ort.Tensor('float32', float32Data, [1, 3, this.inputSize, this.inputSize]);
    }

    /**
     * Post-process YOLOv8 output: extract detections and apply NMS
     * YOLOv8 output shape: [1, 84, 8400] (4 bbox + 80 classes, 8400 anchors)
     */
    _postprocess(output) {
        const data = output.data;
        const [batch, channels, numAnchors] = output.dims;
        const numClasses = channels - 4;

        const candidates = [];

        for (let i = 0; i < numAnchors; i++) {
            // Extract bbox: center_x, center_y, width, height
            const cx = data[0 * numAnchors + i];
            const cy = data[1 * numAnchors + i];
            const w = data[2 * numAnchors + i];
            const h = data[3 * numAnchors + i];

            // Find best class
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

        // Apply Non-Maximum Suppression
        const nmsResults = this._nms(candidates);

        // Remove internal field
        return nmsResults.map(({ _classIdx, ...rest }) => rest);
    }

    /**
     * Non-Maximum Suppression: remove overlapping detections
     */
    _nms(detections) {
        if (detections.length === 0) return [];

        // Sort by confidence descending
        const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
        const kept = [];

        while (sorted.length > 0) {
            const best = sorted.shift();
            kept.push(best);

            // Remove detections with high IoU overlap (same class only)
            for (let i = sorted.length - 1; i >= 0; i--) {
                if (sorted[i]._classIdx === best._classIdx &&
                    this._iou(best.bbox, sorted[i].bbox) > this.iouThreshold) {
                    sorted.splice(i, 1);
                }
            }
        }

        return kept;
    }

    /**
     * Calculate Intersection over Union between two bounding boxes
     */
    _iou(a, b) {
        const x1 = Math.max(a.x, b.x);
        const y1 = Math.max(a.y, b.y);
        const x2 = Math.min(a.x + a.w, b.x + b.w);
        const y2 = Math.min(a.y + a.h, b.y + b.h);

        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = a.w * a.h;
        const areaB = b.w * b.h;
        const union = areaA + areaB - intersection;

        return union > 0 ? intersection / union : 0;
    }

    /**
     * Get performance statistics from tracked inference times
     */
    getPerformanceStats() {
        if (this._inferenceTimes.length === 0) {
            return { count: 0, mean: 0, p95: 0, max: 0, min: 0 };
        }

        const sorted = [...this._inferenceTimes].sort((a, b) => a - b);
        const sum = sorted.reduce((a, b) => a + b, 0);
        const p95Idx = Math.floor(sorted.length * 0.95);

        return {
            count: sorted.length,
            mean: Math.round(sum / sorted.length),
            p95: Math.round(sorted[p95Idx] || sorted[sorted.length - 1]),
            max: Math.round(sorted[sorted.length - 1]),
            min: Math.round(sorted[0])
        };
    }

    /**
     * Reset performance tracking
     */
    resetPerformanceStats() {
        this._inferenceTimes = [];
    }

    /**
     * Update confidence threshold at runtime
     */
    setConfidenceThreshold(threshold) {
        this.confidenceThreshold = Math.max(0.05, Math.min(0.95, threshold));
    }

    /**
     * Release ONNX session resources
     */
    async dispose() {
        if (this.session) {
            try {
                await this.session.release();
            } catch (err) {
                console.warn('[YOLO] Error releasing session:', err.message);
            }
            this.session = null;
            this.initialized = false;
        }
    }
}

module.exports = YOLODetectionService;
