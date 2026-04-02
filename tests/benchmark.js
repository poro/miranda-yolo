#!/usr/bin/env node

/**
 * Performance benchmark: processes frames repeatedly to measure
 * inference latency and peak memory usage.
 *
 * Usage:
 *   node tests/benchmark.js [path/to/frames] [iterations]
 *
 * Defaults to ./test-frames/ and 500 iterations (cycling through available images).
 */

const path = require('path');
const fs = require('fs');
const YOLODetectionService = require('../src/services/yolo/YOLODetectionService');
const YOLOSceneAnalyzer = require('../src/services/yolo/YOLOSceneAnalyzer');

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.bmp', '.webp']);

async function main() {
    const framesDir = path.resolve(process.argv[2] || './test-frames');
    const targetIterations = parseInt(process.argv[3], 10) || 500;

    console.log('=== YOLO Performance Benchmark ===');
    console.log(`Frames directory: ${framesDir}`);
    console.log(`Target iterations: ${targetIterations}\n`);

    if (!fs.existsSync(framesDir)) {
        console.error(`Directory not found: ${framesDir}`);
        process.exit(1);
    }

    const files = fs.readdirSync(framesDir)
        .filter(f => IMAGE_EXTENSIONS.has(path.extname(f).toLowerCase()))
        .sort();

    if (files.length === 0) {
        console.error('No image files found. Add screenshots to', framesDir);
        process.exit(1);
    }

    // Pre-load all image buffers into memory to isolate inference time from I/O
    console.log(`Loading ${files.length} image(s) into memory...`);
    const buffers = files.map(f => fs.readFileSync(path.join(framesDir, f)));

    const detector = new YOLODetectionService({ confidenceThreshold: 0.5 });
    const analyzer = new YOLOSceneAnalyzer({ changeThreshold: 0.3 });

    const ok = await detector.initialize();
    if (!ok) {
        console.error('Failed to initialize YOLO:', detector.initError);
        process.exit(1);
    }

    // Warm-up: run 3 frames to stabilize ONNX runtime
    console.log('Warming up (3 frames)...');
    for (let i = 0; i < Math.min(3, buffers.length); i++) {
        await detector.detect(buffers[i]);
    }
    detector.resetPerformanceStats();
    analyzer.reset();

    // Benchmark
    console.log(`\nRunning ${targetIterations} iterations...\n`);
    let peakRSS = 0;

    const benchmarkStart = performance.now();

    for (let i = 0; i < targetIterations; i++) {
        const buffer = buffers[i % buffers.length];
        const detections = await detector.detect(buffer);
        analyzer.analyzeFrame(detections);

        const currentRSS = process.memoryUsage().rss;
        if (currentRSS > peakRSS) peakRSS = currentRSS;

        // Progress indicator every 100 frames
        if ((i + 1) % 100 === 0) {
            const stats = detector.getPerformanceStats();
            console.log(`  [${i + 1}/${targetIterations}] mean=${stats.mean}ms, p95=${stats.p95}ms, RSS=${(process.memoryUsage().rss / 1024 / 1024).toFixed(0)}MB`);
        }
    }

    const totalTime = performance.now() - benchmarkStart;

    // Final report
    const perfStats = detector.getPerformanceStats();
    const sceneStats = analyzer.getStats();
    const mem = process.memoryUsage();

    console.log('\n=== Benchmark Results ===');
    console.log();
    console.log('--- Inference Latency ---');
    console.log(`Frames processed:    ${perfStats.count}`);
    console.log(`Mean:                ${perfStats.mean}ms`);
    console.log(`P95:                 ${perfStats.p95}ms`);
    console.log(`Max:                 ${perfStats.max}ms`);
    console.log(`Min:                 ${perfStats.min}ms`);
    console.log(`Total wall time:     ${(totalTime / 1000).toFixed(1)}s`);
    console.log(`Throughput:          ${(perfStats.count / (totalTime / 1000)).toFixed(1)} fps`);
    console.log();
    console.log('--- Memory ---');
    console.log(`Peak RSS:            ${(peakRSS / 1024 / 1024).toFixed(1)}MB`);
    console.log(`Final RSS:           ${(mem.rss / 1024 / 1024).toFixed(1)}MB`);
    console.log(`Heap used:           ${(mem.heapUsed / 1024 / 1024).toFixed(1)}MB`);
    console.log();
    console.log('--- Scene Analysis ---');
    console.log(`Skip rate:           ${sceneStats.skipRate}`);
    console.log(`Analyzed frames:     ${sceneStats.analyzedFrames}`);
    console.log(`Skipped frames:      ${sceneStats.skippedFrames}`);

    // PRD success criteria check
    console.log('\n=== PRD Success Criteria ===');
    const checks = [
        { name: 'Inference < 200ms (mean)', pass: perfStats.mean < 200, value: `${perfStats.mean}ms` },
        { name: 'Memory < 500MB (peak RSS)', pass: peakRSS / 1024 / 1024 < 500, value: `${(peakRSS / 1024 / 1024).toFixed(1)}MB` },
    ];

    for (const c of checks) {
        console.log(`${c.pass ? 'PASS' : 'FAIL'} ${c.name}: ${c.value}`);
    }

    await detector.dispose();
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
