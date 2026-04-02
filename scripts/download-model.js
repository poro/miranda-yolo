#!/usr/bin/env node

/**
 * Downloads and exports YOLOv8 models in ONNX format.
 *
 * Supports multiple model sizes (n/s/m/l/x) and input sizes (640/1280).
 * Auto-detects platform and exports CoreML format on macOS for Metal acceleration.
 *
 * Usage:
 *   npm run download-model                    # Default: nano @ 640
 *   npm run download-model -- --all           # All sizes at 640 + 1280
 *   npm run download-model -- --size s        # Small @ 640
 *   npm run download-model -- --size x --input 1280  # XLarge @ 1280
 *   npm run download-model -- --recommended   # Platform-optimized selection
 *
 * Requires: python3 with venv support
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const OUTPUT_DIR = path.join(__dirname, '..', 'models');
const VENV_DIR = path.join(__dirname, '..', '.export-venv');

const SIZES = ['n', 's', 'm', 'l', 'x'];
const SIZE_LABELS = { n: 'Nano', s: 'Small', m: 'Medium', l: 'Large', x: 'XLarge' };
const INPUT_SIZES = [640, 1280];

function run(cmd, opts = {}) {
    console.log(`  $ ${cmd}`);
    return execSync(cmd, { stdio: 'inherit', timeout: 600000, ...opts });
}

function parseArgs() {
    const args = process.argv.slice(2);
    const config = { sizes: ['n'], inputs: [640], all: false, recommended: false };

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--all':
                config.all = true;
                break;
            case '--recommended':
                config.recommended = true;
                break;
            case '--size':
                config.sizes = [args[++i]];
                break;
            case '--input':
                config.inputs = [parseInt(args[++i])];
                break;
        }
    }

    if (config.all) {
        config.sizes = SIZES;
        config.inputs = INPUT_SIZES;
    }

    if (config.recommended) {
        const platform = os.platform();
        const arch = os.arch();
        const totalMem = os.totalmem() / (1024 * 1024 * 1024);

        console.log(`\nPlatform: ${platform} ${arch}, RAM: ${totalMem.toFixed(0)}GB`);

        if (platform === 'darwin' && arch === 'arm64') {
            // Apple Silicon — can handle larger models
            if (totalMem >= 64) {
                console.log('Detected: Apple Silicon with 64GB+ — recommending l/x @ 1280');
                config.sizes = ['n', 's', 'm', 'l', 'x'];
                config.inputs = [640, 1280];
            } else if (totalMem >= 16) {
                console.log('Detected: Apple Silicon — recommending m/l @ 640 + 1280');
                config.sizes = ['n', 's', 'm', 'l'];
                config.inputs = [640, 1280];
            } else {
                config.sizes = ['n', 's'];
                config.inputs = [640];
            }
        } else if (platform === 'linux') {
            // Check for NVIDIA GPU
            let hasGPU = false;
            try {
                execSync('nvidia-smi', { stdio: 'ignore' });
                hasGPU = true;
            } catch {}

            if (hasGPU) {
                console.log('Detected: Linux with NVIDIA GPU — recommending all sizes');
                config.sizes = SIZES;
                config.inputs = INPUT_SIZES;
            } else {
                console.log('Detected: Linux CPU only — recommending n/s @ 640');
                config.sizes = ['n', 's'];
                config.inputs = [640];
            }
        } else {
            config.sizes = ['n', 's'];
            config.inputs = [640];
        }
    }

    return config;
}

function main() {
    const config = parseArgs();

    console.log('=== YOLOv8 ONNX Model Export ===\n');

    if (!fs.existsSync(OUTPUT_DIR)) {
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    // Build list of models to export
    const jobs = [];
    for (const size of config.sizes) {
        for (const input of config.inputs) {
            const filename = `yolov8${size}_${input}.onnx`;
            const outPath = path.join(OUTPUT_DIR, filename);
            if (fs.existsSync(outPath)) {
                const mb = (fs.statSync(outPath).size / (1024 * 1024)).toFixed(0);
                console.log(`SKIP: ${filename} (${mb}MB, already exists)`);
            } else {
                jobs.push({ size, input, filename, outPath });
            }
        }
    }

    if (jobs.length === 0) {
        console.log('\nAll models already exported. Delete models/*.onnx to re-export.');
        return;
    }

    console.log(`\nExporting ${jobs.length} model(s)...\n`);

    // Create venv
    console.log('Step 1: Creating Python venv...');
    if (!fs.existsSync(VENV_DIR)) {
        run(`python3 -m venv "${VENV_DIR}"`);
    }

    console.log('\nStep 2: Installing ultralytics + onnx...');
    run(`"${VENV_DIR}/bin/pip" install --quiet ultralytics onnx`);

    // Export each model
    for (const job of jobs) {
        const ptName = `yolov8${job.size}`;
        console.log(`\nStep 3: Exporting ${SIZE_LABELS[job.size]} (yolov8${job.size}) @ ${job.input}px...`);
        try {
            run(`"${VENV_DIR}/bin/python3" -c "
from ultralytics import YOLO
import shutil, os
model = YOLO('${ptName}.pt')
model.export(format='onnx', imgsz=${job.input}, simplify=True)
exported = '${ptName}.onnx'
if os.path.exists(exported):
    shutil.move(exported, '${job.outPath.replace(/'/g, "\\'")}')
    print(f'  OK: ${job.filename}')
if os.path.exists('${ptName}.pt'):
    os.remove('${ptName}.pt')
"`);
        } catch (err) {
            console.error(`  FAILED: ${err.message}`);
        }
    }

    // Cleanup
    console.log('\nCleaning up...');
    fs.rmSync(VENV_DIR, { recursive: true, force: true });
    // Clean any leaked .pt files
    for (const f of fs.readdirSync(process.cwd())) {
        if (f.match(/^yolov8[nslmx]\.pt$/)) fs.unlinkSync(f);
    }

    // Summary
    console.log('\n=== Models ===');
    const models = fs.readdirSync(OUTPUT_DIR).filter(f => f.endsWith('.onnx')).sort();
    for (const f of models) {
        const mb = (fs.statSync(path.join(OUTPUT_DIR, f)).size / (1024 * 1024)).toFixed(0);
        console.log(`  ${f} (${mb}MB)`);
    }
    console.log(`\nTotal: ${models.length} model(s) in ${OUTPUT_DIR}`);
}

main();
