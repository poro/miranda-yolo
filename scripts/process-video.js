#!/usr/bin/env node

/**
 * Process a gameplay video file through YOLO detection pipeline.
 *
 * Extracts frames at a target FPS using ffmpeg, runs YOLO detection on each,
 * and performs scene change analysis. Results are written to a JSON report.
 *
 * Usage:
 *   node scripts/process-video.js <video-path> [options]
 *
 * Options:
 *   --fps <n>          Frame extraction rate (default: 1)
 *   --max-frames <n>   Stop after N frames (default: unlimited)
 *   --confidence <n>   Detection confidence threshold (default: 0.3)
 *   --output <path>    Output JSON report path (default: ./results/<video-name>.json)
 *   --save-frames      Also save extracted frames to disk
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const sharp = require('sharp');
const YOLODetectionService = require('../src/services/yolo/YOLODetectionService');
const YOLOSceneAnalyzer = require('../src/services/yolo/YOLOSceneAnalyzer');

// --- Argument parsing ---

function parseArgs() {
    const args = process.argv.slice(2);
    const opts = {
        videoPath: null,
        fps: 1,
        maxFrames: Infinity,
        confidence: 0.3,
        output: null,
        saveFrames: false,
    };

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--fps':        opts.fps = parseFloat(args[++i]); break;
            case '--max-frames': opts.maxFrames = parseInt(args[++i], 10); break;
            case '--confidence': opts.confidence = parseFloat(args[++i]); break;
            case '--output':     opts.output = args[++i]; break;
            case '--save-frames': opts.saveFrames = true; break;
            default:
                if (!opts.videoPath && !args[i].startsWith('--')) {
                    opts.videoPath = args[i];
                }
        }
    }

    if (!opts.videoPath) {
        console.error('Usage: node scripts/process-video.js <video-path> [options]');
        console.error('');
        console.error('Options:');
        console.error('  --fps <n>          Frame extraction rate (default: 1)');
        console.error('  --max-frames <n>   Stop after N frames (default: unlimited)');
        console.error('  --confidence <n>   Detection threshold (default: 0.3)');
        console.error('  --output <path>    Output JSON path');
        console.error('  --save-frames      Save extracted frames to disk');
        process.exit(1);
    }

    return opts;
}

// --- Get video duration via ffprobe ---

function getVideoDuration(videoPath) {
    return new Promise((resolve, reject) => {
        const proc = spawn('ffprobe', [
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            videoPath
        ]);
        let stdout = '';
        proc.stdout.on('data', d => stdout += d);
        proc.on('close', code => {
            if (code !== 0) return reject(new Error(`ffprobe exited with code ${code}`));
            try {
                const info = JSON.parse(stdout);
                resolve(parseFloat(info.format.duration));
            } catch (e) {
                reject(e);
            }
        });
    });
}

// --- Stream frames from ffmpeg ---

function createFrameStream(videoPath, fps) {
    const proc = spawn('ffmpeg', [
        '-i', videoPath,
        '-vf', `fps=${fps}`,
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-'
    ], { stdio: ['ignore', 'pipe', 'ignore'] });

    return proc;
}

/**
 * Read complete PNG images from a raw byte stream.
 * PNG files start with an 8-byte signature and end with IEND chunk.
 */
async function* extractPNGFrames(stream) {
    const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    const IEND_MARKER = Buffer.from([0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]);

    let buffer = Buffer.alloc(0);

    for await (const chunk of stream) {
        buffer = Buffer.concat([buffer, chunk]);

        while (true) {
            // Find start of next PNG
            const sigIdx = buffer.indexOf(PNG_SIGNATURE);
            if (sigIdx === -1) break;

            // Find end of this PNG (IEND chunk)
            const endIdx = buffer.indexOf(IEND_MARKER, sigIdx + 8);
            if (endIdx === -1) break; // Need more data

            const frameEnd = endIdx + IEND_MARKER.length;
            const frame = buffer.subarray(sigIdx, frameEnd);

            yield Buffer.from(frame); // Copy before we slice the buffer

            buffer = buffer.subarray(frameEnd);
        }
    }
}

// --- Main ---

async function main() {
    const opts = parseArgs();
    const videoPath = path.resolve(opts.videoPath);
    const videoName = path.basename(videoPath, path.extname(videoPath));

    console.log('=== YOLO Video Processor ===');
    console.log(`Video:       ${videoPath}`);
    console.log(`FPS:         ${opts.fps}`);
    console.log(`Max frames:  ${opts.maxFrames === Infinity ? 'unlimited' : opts.maxFrames}`);
    console.log(`Confidence:  ${opts.confidence}`);
    console.log();

    if (!fs.existsSync(videoPath)) {
        console.error(`Video file not found: ${videoPath}`);
        process.exit(1);
    }

    // Get video duration
    let duration;
    try {
        duration = await getVideoDuration(videoPath);
        const expectedFrames = Math.floor(duration * opts.fps);
        console.log(`Duration:    ${(duration / 60).toFixed(1)} minutes`);
        console.log(`Est. frames: ${expectedFrames}`);
        console.log();
    } catch {
        console.log('Duration:    unknown (ffprobe failed)\n');
    }

    // Setup output
    const resultsDir = path.join(process.cwd(), 'results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

    const outputPath = opts.output || path.join(resultsDir, `${videoName}.json`);

    let framesDir = null;
    if (opts.saveFrames) {
        framesDir = path.join(resultsDir, `${videoName}-frames`);
        if (!fs.existsSync(framesDir)) fs.mkdirSync(framesDir, { recursive: true });
    }

    // Initialize YOLO
    const detector = new YOLODetectionService({ confidenceThreshold: opts.confidence });
    const analyzer = new YOLOSceneAnalyzer({ changeThreshold: 0.3, forceAnalyzeEveryN: 10 });

    const ok = await detector.initialize();
    if (!ok) {
        console.error('Failed to initialize YOLO:', detector.initError);
        process.exit(1);
    }

    // Start ffmpeg frame extraction
    console.log('Extracting and analyzing frames...\n');
    const ffmpeg = createFrameStream(videoPath, opts.fps);

    const frames = [];
    let frameNum = 0;
    const startTime = performance.now();

    for await (const pngBuffer of extractPNGFrames(ffmpeg.stdout)) {
        frameNum++;
        if (frameNum > opts.maxFrames) break;

        const timestamp = (frameNum - 1) / opts.fps;
        const detections = await detector.detect(pngBuffer);
        const scene = analyzer.analyzeFrame(detections);

        // Save frame image if requested
        if (framesDir) {
            const framePath = path.join(framesDir, `frame_${String(frameNum).padStart(5, '0')}.png`);
            fs.writeFileSync(framePath, pngBuffer);
        }

        const labels = [...new Set(detections.map(d => d.label))];
        const frameResult = {
            frame: frameNum,
            timestamp: +timestamp.toFixed(2),
            timestampStr: formatTime(timestamp),
            detectionCount: detections.length,
            uniqueLabels: labels,
            detections: detections.map(d => ({
                label: d.label,
                confidence: +d.confidence.toFixed(3),
                bbox: {
                    x: +d.bbox.x.toFixed(3),
                    y: +d.bbox.y.toFixed(3),
                    w: +d.bbox.w.toFixed(3),
                    h: +d.bbox.h.toFixed(3)
                }
            })),
            sceneChanged: scene.changed,
            sceneReason: scene.reason,
            jaccard: +scene.jaccard.toFixed(3)
        };

        frames.push(frameResult);

        // Progress output
        const elapsed = (performance.now() - startTime) / 1000;
        const fps = frameNum / elapsed;
        const statusIcon = scene.changed ? '*' : ' ';
        const objectStr = labels.length > 0 ? labels.join(', ') : '(none)';

        if (frameNum <= 5 || frameNum % 10 === 0 || scene.changed) {
            console.log(
                `${statusIcon} [${String(frameNum).padStart(5)}] ${formatTime(timestamp)}  ` +
                `objects=${String(detections.length).padStart(2)}  ` +
                `scene=${scene.changed ? 'CHANGED' : 'same   '} ` +
                `(${scene.reason})  ` +
                `${objectStr}`
            );
        }

        // Periodic summary
        if (frameNum % 100 === 0) {
            const perf = detector.getPerformanceStats();
            const mem = process.memoryUsage();
            console.log(
                `  --- ${frameNum} frames, ${perf.mean}ms avg, ` +
                `${(mem.rss / 1024 / 1024).toFixed(0)}MB RSS, ` +
                `${fps.toFixed(1)} fps throughput ---`
            );
        }
    }

    // Kill ffmpeg if we stopped early
    ffmpeg.kill('SIGTERM');

    const totalTime = (performance.now() - startTime) / 1000;

    // Collect final stats
    const perfStats = detector.getPerformanceStats();
    const sceneStats = analyzer.getStats();
    const mem = process.memoryUsage();

    // Aggregate object frequency across all frames
    const objectFrequency = {};
    for (const f of frames) {
        for (const d of f.detections) {
            objectFrequency[d.label] = (objectFrequency[d.label] || 0) + 1;
        }
    }
    const sortedObjects = Object.entries(objectFrequency)
        .sort((a, b) => b[1] - a[1]);

    // Summary
    console.log('\n=== Processing Complete ===\n');
    console.log(`Frames analyzed:     ${frames.length}`);
    console.log(`Total time:          ${totalTime.toFixed(1)}s`);
    console.log(`Throughput:          ${(frames.length / totalTime).toFixed(1)} fps`);
    console.log();
    console.log('--- Inference ---');
    console.log(`Mean:                ${perfStats.mean}ms`);
    console.log(`P95:                 ${perfStats.p95}ms`);
    console.log(`Max:                 ${perfStats.max}ms`);
    console.log();
    console.log('--- Scene Analysis ---');
    console.log(`Scene changes:       ${sceneStats.analyzedFrames}`);
    console.log(`Skipped (static):    ${sceneStats.skippedFrames}`);
    console.log(`Skip rate:           ${sceneStats.skipRate}`);
    console.log();
    console.log('--- Top Detected Objects ---');
    for (const [label, count] of sortedObjects.slice(0, 15)) {
        const pct = ((count / frames.length) * 100).toFixed(1);
        console.log(`  ${label.padEnd(20)} ${String(count).padStart(5)} frames (${pct}%)`);
    }
    console.log();
    console.log('--- Memory ---');
    console.log(`RSS:                 ${(mem.rss / 1024 / 1024).toFixed(1)}MB`);

    // Write report
    const report = {
        video: {
            path: videoPath,
            name: videoName,
            duration: duration || null,
            durationStr: duration ? formatTime(duration) : null
        },
        config: {
            fps: opts.fps,
            confidenceThreshold: opts.confidence,
            changeThreshold: 0.3,
            forceAnalyzeEveryN: 10
        },
        summary: {
            totalFrames: frames.length,
            totalTimeSeconds: +totalTime.toFixed(1),
            throughputFps: +(frames.length / totalTime).toFixed(1),
            inference: perfStats,
            sceneAnalysis: sceneStats,
            objectFrequency: Object.fromEntries(sortedObjects),
            memory: {
                rss_mb: +(mem.rss / 1024 / 1024).toFixed(1),
                heapUsed_mb: +(mem.heapUsed / 1024 / 1024).toFixed(1)
            }
        },
        transitions: analyzer.getTransitions(),
        frames
    };

    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
    console.log(`\nReport written to: ${outputPath}`);

    await detector.dispose();
}

function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${m}:${String(s).padStart(2, '0')}`;
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
