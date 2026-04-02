/**
 * TrainingPipeline — Orchestrates YOLO model fine-tuning
 *
 * 1. Exports annotations to YOLO dataset format
 * 2. Spawns Python subprocess for Ultralytics training
 * 3. Exports trained model to ONNX
 * 4. Registers in model registry (SQLite)
 * 5. Hot-swaps the active ONNX model
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const MODELS_DIR = path.join(__dirname, '../../../models');

class TrainingPipeline {
    constructor(sessionStore, annotationManager, detectionService) {
        this.store = sessionStore;
        this.annotations = annotationManager;
        this.detector = detectionService;
        this.activeTraining = null;
    }

    /**
     * Start a training run
     * @param {string} name - Model name
     * @param {object} config - { epochs, imgSize, trainSplit, baseModel }
     * @param {function} onLog - Callback for training log lines
     * @returns {Promise<object>} Training results
     */
    async train(name, config = {}, onLog = () => {}) {
        const {
            epochs = 100,
            imgSize = 640,
            trainSplit = 0.8,
            baseModel = 'yolov8n.pt'
        } = config;

        if (this.activeTraining) {
            throw new Error('A training run is already active');
        }

        // Step 1: Export dataset
        onLog('[1/4] Exporting annotations to YOLO format...');
        const dataset = await this.annotations.exportYOLO(name, { trainSplit });
        onLog(`  Dataset: ${dataset.trainCount} train, ${dataset.valCount} val, ${dataset.classes} classes`);

        if (dataset.trainCount === 0) {
            throw new Error('No training images. Annotate more frames first.');
        }

        // Step 2: Register model in DB
        const modelId = this.store.registerModel(name, {
            baseModel,
            datasetPath: dataset.datasetDir,
            epochs
        });
        this.store.updateModel(modelId, { status: 'training' });

        onLog(`[2/4] Starting training (${epochs} epochs, ${imgSize}px)...`);

        // Step 3: Train via Python
        const onnxOutputPath = path.join(dataset.datasetDir, 'runs', 'detect', 'train', 'weights', 'best.onnx');

        try {
            const trainScript = `
import sys, json
from ultralytics import YOLO

model = YOLO('${baseModel}')
results = model.train(
    data='${dataset.yamlPath}',
    epochs=${epochs},
    imgsz=${imgSize},
    pretrained=True,
    verbose=True
)

# Export to ONNX
model.export(format='onnx', imgsz=${imgSize}, simplify=True)

# Output metrics
metrics = {
    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
    'mAP50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
    'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
    'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
}
print('METRICS:' + json.dumps(metrics))
`;

            const metrics = await this._runPython(trainScript, onLog);
            onLog('[3/4] Training complete. Exporting ONNX...');

            // Find the best.onnx — Ultralytics puts it in runs/detect/train/weights/
            const runsDir = path.join(dataset.datasetDir, 'runs', 'detect');
            let onnxPath = this._findFile(runsDir, 'best.onnx');

            if (!onnxPath) {
                // Try looking for any .onnx file
                onnxPath = this._findFile(runsDir, '.onnx');
            }

            if (!onnxPath) {
                throw new Error('ONNX export not found after training');
            }

            // Step 4: Copy to models directory and register
            const destOnnxPath = path.join(MODELS_DIR, `${name}.onnx`);
            fs.copyFileSync(onnxPath, destOnnxPath);

            this.store.updateModel(modelId, {
                status: 'ready',
                onnx_path: destOnnxPath,
                metrics: JSON.stringify(metrics)
            });

            onLog(`[4/4] Model saved: ${destOnnxPath}`);
            onLog(`  Metrics: ${JSON.stringify(metrics, null, 2)}`);

            this.activeTraining = null;

            return {
                modelId,
                name,
                onnxPath: destOnnxPath,
                metrics,
                dataset
            };

        } catch (err) {
            this.store.updateModel(modelId, { status: 'failed' });
            this.activeTraining = null;
            throw err;
        }
    }

    /**
     * Activate a trained model — hot-swap the ONNX session
     */
    async activateModel(modelId) {
        const model = this.store.db.prepare('SELECT * FROM models WHERE id = ?').get(modelId);
        if (!model) throw new Error('Model not found');
        if (!model.onnx_path || !fs.existsSync(model.onnx_path)) {
            throw new Error('ONNX file not found: ' + model.onnx_path);
        }

        // Dispose current session and reinitialize with new model
        await this.detector.dispose();
        this.detector.modelPath = model.onnx_path;
        const ok = await this.detector.initialize();

        if (ok) {
            this.store.activateModel(modelId);
            return { success: true, model: model.name, path: model.onnx_path };
        } else {
            throw new Error('Failed to load model: ' + this.detector.initError);
        }
    }

    /**
     * Stop active training
     */
    stopTraining() {
        if (this.activeTraining && this.activeTraining.process) {
            this.activeTraining.process.kill('SIGTERM');
            this.activeTraining = null;
        }
    }

    // --- Internal ---

    _runPython(script, onLog) {
        return new Promise((resolve, reject) => {
            const proc = spawn('python3', ['-u', '-c', script], {
                cwd: path.join(__dirname, '../../..'),
                stdio: ['ignore', 'pipe', 'pipe']
            });

            this.activeTraining = { process: proc };

            let metrics = {};

            proc.stdout.on('data', (data) => {
                const lines = data.toString().split('\n');
                for (const line of lines) {
                    if (line.trim()) {
                        if (line.startsWith('METRICS:')) {
                            try {
                                metrics = JSON.parse(line.slice(8));
                            } catch {}
                        } else {
                            onLog(line);
                        }
                    }
                }
            });

            proc.stderr.on('data', (data) => {
                const lines = data.toString().split('\n');
                for (const line of lines) {
                    if (line.trim()) onLog(`[stderr] ${line}`);
                }
            });

            proc.on('close', (code) => {
                if (code === 0) {
                    resolve(metrics);
                } else {
                    reject(new Error(`Training process exited with code ${code}`));
                }
            });

            proc.on('error', (err) => {
                reject(new Error(`Failed to start Python: ${err.message}`));
            });
        });
    }

    _findFile(dir, pattern) {
        if (!fs.existsSync(dir)) return null;
        const entries = fs.readdirSync(dir, { withFileTypes: true, recursive: true });
        for (const entry of entries) {
            if (entry.isFile() && entry.name.includes(pattern)) {
                return path.join(entry.parentPath || entry.path, entry.name);
            }
        }
        return null;
    }
}

module.exports = TrainingPipeline;
