#!/usr/bin/env node

/**
 * Downloads YOLOv8n .pt from Ultralytics and exports to ONNX using Python.
 *
 * Requires: python3 with venv support
 * Output: models/yolov8n.onnx (~12MB, 80 COCO object classes)
 *
 * Alternative: manually export with `yolo export model=yolov8n.pt format=onnx`
 * and place the result in models/yolov8n.onnx
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const OUTPUT_DIR = path.join(__dirname, '..', 'models');
const OUTPUT_PATH = path.join(OUTPUT_DIR, 'yolov8n.onnx');
const VENV_DIR = path.join(__dirname, '..', '.export-venv');

function run(cmd, opts = {}) {
    console.log(`  $ ${cmd}`);
    return execSync(cmd, { stdio: 'inherit', timeout: 300000, ...opts });
}

async function main() {
    console.log('=== YOLOv8n ONNX Model Setup ===\n');

    if (fs.existsSync(OUTPUT_PATH)) {
        const stat = fs.statSync(OUTPUT_PATH);
        console.log(`Model already exists at ${OUTPUT_PATH} (${(stat.size / 1024 / 1024).toFixed(1)}MB)`);
        console.log('Delete it manually and re-run to re-export.');
        return;
    }

    if (!fs.existsSync(OUTPUT_DIR)) {
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    console.log('Step 1: Creating temporary Python venv...');
    run(`python3 -m venv "${VENV_DIR}"`);

    console.log('\nStep 2: Installing ultralytics + onnx...');
    run(`"${VENV_DIR}/bin/pip" install ultralytics onnx`);

    console.log('\nStep 3: Exporting YOLOv8n to ONNX...');
    run(`"${VENV_DIR}/bin/python3" -c "
from ultralytics import YOLO
import shutil, os
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, simplify=True)
shutil.move('yolov8n.onnx', '${OUTPUT_PATH.replace(/'/g, "\\'")}')
if os.path.exists('yolov8n.pt'):
    os.remove('yolov8n.pt')
"`);

    console.log('\nStep 4: Cleaning up venv...');
    fs.rmSync(VENV_DIR, { recursive: true, force: true });

    // Clean up .pt if it leaked to cwd
    const ptPath = path.join(process.cwd(), 'yolov8n.pt');
    if (fs.existsSync(ptPath)) fs.unlinkSync(ptPath);

    if (fs.existsSync(OUTPUT_PATH)) {
        const stat = fs.statSync(OUTPUT_PATH);
        console.log(`\nModel exported successfully: ${OUTPUT_PATH} (${(stat.size / 1024 / 1024).toFixed(1)}MB)`);
    } else {
        console.error('\nExport failed — model file not found.');
        console.error('Manual alternative:');
        console.error('  pip install ultralytics');
        console.error('  yolo export model=yolov8n.pt format=onnx imgsz=640');
        console.error(`  mv yolov8n.onnx ${OUTPUT_PATH}`);
        process.exit(1);
    }
}

main();
