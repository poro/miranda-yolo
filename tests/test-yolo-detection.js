#!/usr/bin/env node

/**
 * Test script: processes a folder of screenshots and outputs detection results
 * + scene change decisions.
 *
 * Usage:
 *   node tests/test-yolo-detection.js [path/to/frames]
 *
 * Defaults to ./test-frames/ if no path is given.
 * Place PNG/JPEG screenshots in that folder before running.
 */

const path = require('path');
const fs = require('fs');
const YOLODetectionService = require('../src/services/yolo/YOLODetectionService');
const YOLOSceneAnalyzer = require('../src/services/yolo/YOLOSceneAnalyzer');

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.bmp', '.webp']);

async function main() {
    const framesDir = path.resolve(process.argv[2] || './test-frames');

    console.log('=== YOLO Detection Test ===');
    console.log(`Frames directory: ${framesDir}\n`);

    // Check frames directory
    if (!fs.existsSync(framesDir)) {
        console.error(`Directory not found: ${framesDir}`);
        console.error('Create it and add PNG/JPEG gameplay screenshots, then re-run.');
        process.exit(1);
    }

    const files = fs.readdirSync(framesDir)
        .filter(f => IMAGE_EXTENSIONS.has(path.extname(f).toLowerCase()))
        .sort();

    if (files.length === 0) {
        console.error('No image files found in', framesDir);
        console.error('Add PNG/JPEG screenshots and re-run.');
        process.exit(1);
    }

    console.log(`Found ${files.length} image(s)\n`);

    // Initialize services
    const detector = new YOLODetectionService({ confidenceThreshold: 0.5 });
    const analyzer = new YOLOSceneAnalyzer({ changeThreshold: 0.3, forceAnalyzeEveryN: 10 });

    const ok = await detector.initialize();
    if (!ok) {
        console.error('Failed to initialize YOLO:', detector.initError);
        process.exit(1);
    }

    // Process each frame
    const results = [];

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const filePath = path.join(framesDir, file);

        try {
            const detections = await detector.detectFromFile(filePath);
            const scene = analyzer.analyzeFrame(detections);

            const labels = detections.map(d => `${d.label}(${d.confidence.toFixed(2)})`);

            console.log(`[${String(i + 1).padStart(3)}] ${file}`);
            console.log(`      Objects (${detections.length}): ${labels.join(', ') || '(none)'}`);
            console.log(`      Scene changed: ${scene.changed} (${scene.reason}, jaccard=${scene.jaccard.toFixed(3)})`);
            console.log();

            results.push({
                file,
                detectionCount: detections.length,
                labels: detections.map(d => d.label),
                sceneChanged: scene.changed,
                reason: scene.reason,
                jaccard: scene.jaccard
            });
        } catch (err) {
            console.log(`[${String(i + 1).padStart(3)}] ${file} — ERROR: ${err.message}\n`);
            results.push({ file, error: err.message });
        }
    }

    // Summary
    const perfStats = detector.getPerformanceStats();
    const sceneStats = analyzer.getStats();
    const mem = process.memoryUsage();

    console.log('=== Summary ===');
    console.log(`Frames processed:    ${results.length}`);
    console.log(`Detection errors:    ${results.filter(r => r.error).length}`);
    console.log();
    console.log('--- Performance ---');
    console.log(`Mean inference:      ${perfStats.mean}ms`);
    console.log(`P95 inference:       ${perfStats.p95}ms`);
    console.log(`Max inference:       ${perfStats.max}ms`);
    console.log(`Min inference:       ${perfStats.min}ms`);
    console.log();
    console.log('--- Scene Analysis ---');
    console.log(`Analyzed frames:     ${sceneStats.analyzedFrames}`);
    console.log(`Skipped frames:      ${sceneStats.skippedFrames}`);
    console.log(`Skip rate:           ${sceneStats.skipRate}`);
    console.log();
    console.log('--- Memory ---');
    console.log(`RSS:                 ${(mem.rss / 1024 / 1024).toFixed(1)}MB`);
    console.log(`Heap used:           ${(mem.heapUsed / 1024 / 1024).toFixed(1)}MB`);
    console.log(`Heap total:          ${(mem.heapTotal / 1024 / 1024).toFixed(1)}MB`);

    // Write results to JSON
    const outputPath = path.join(framesDir, '../test-results.json');
    fs.writeFileSync(outputPath, JSON.stringify({
        timestamp: new Date().toISOString(),
        framesDir,
        performance: perfStats,
        sceneAnalysis: sceneStats,
        memory: {
            rss_mb: +(mem.rss / 1024 / 1024).toFixed(1),
            heapUsed_mb: +(mem.heapUsed / 1024 / 1024).toFixed(1)
        },
        frames: results,
        transitions: analyzer.getTransitions()
    }, null, 2));
    console.log(`\nResults written to: ${outputPath}`);

    await detector.dispose();
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
