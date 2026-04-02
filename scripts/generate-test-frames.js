#!/usr/bin/env node

/**
 * Generates synthetic test frames for YOLO testing.
 * Creates a mix of: solid colors, gradients, noise patterns.
 * These won't produce meaningful YOLO detections but validate the pipeline works.
 *
 * For real testing, add actual gameplay screenshots to test-frames/.
 */

const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.join(__dirname, '..', 'test-frames');

async function createSolidFrame(name, r, g, b) {
    const buf = Buffer.alloc(640 * 480 * 3);
    for (let i = 0; i < 640 * 480; i++) {
        buf[i * 3] = r;
        buf[i * 3 + 1] = g;
        buf[i * 3 + 2] = b;
    }
    await sharp(buf, { raw: { width: 640, height: 480, channels: 3 } })
        .png()
        .toFile(path.join(OUTPUT_DIR, name));
}

async function createNoiseFrame(name) {
    const buf = Buffer.alloc(640 * 480 * 3);
    for (let i = 0; i < buf.length; i++) {
        buf[i] = Math.floor(Math.random() * 256);
    }
    await sharp(buf, { raw: { width: 640, height: 480, channels: 3 } })
        .png()
        .toFile(path.join(OUTPUT_DIR, name));
}

async function createGradientFrame(name) {
    const buf = Buffer.alloc(640 * 480 * 3);
    for (let y = 0; y < 480; y++) {
        for (let x = 0; x < 640; x++) {
            const idx = (y * 640 + x) * 3;
            buf[idx] = Math.floor((x / 640) * 255);
            buf[idx + 1] = Math.floor((y / 480) * 255);
            buf[idx + 2] = 128;
        }
    }
    await sharp(buf, { raw: { width: 640, height: 480, channels: 3 } })
        .png()
        .toFile(path.join(OUTPUT_DIR, name));
}

async function main() {
    if (!fs.existsSync(OUTPUT_DIR)) {
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    console.log('Generating synthetic test frames...\n');

    // Black screen (loading/dark screen test)
    await createSolidFrame('frame_001_black.png', 0, 0, 0);
    console.log('  frame_001_black.png (black/loading screen)');

    // Duplicate black (should be skipped by scene analyzer)
    await createSolidFrame('frame_002_black.png', 0, 0, 0);
    console.log('  frame_002_black.png (duplicate black)');

    // Gradient (scene change from black)
    await createGradientFrame('frame_003_gradient.png');
    console.log('  frame_003_gradient.png (gradient)');

    // Another gradient (should be skipped - same content)
    await createGradientFrame('frame_004_gradient.png');
    console.log('  frame_004_gradient.png (duplicate gradient)');

    // Noise frames (simulate visual change)
    for (let i = 5; i <= 10; i++) {
        const name = `frame_${String(i).padStart(3, '0')}_noise.png`;
        await createNoiseFrame(name);
        console.log(`  ${name} (random noise)`);
    }

    // White screen
    await createSolidFrame('frame_011_white.png', 255, 255, 255);
    console.log('  frame_011_white.png (white screen)');

    // Blue screen (scene change)
    await createSolidFrame('frame_012_blue.png', 30, 60, 180);
    console.log('  frame_012_blue.png (blue screen)');

    console.log(`\nGenerated ${12} test frames in ${OUTPUT_DIR}`);
    console.log('For real testing, add actual gameplay screenshots to that folder.');
}

main().catch(console.error);
