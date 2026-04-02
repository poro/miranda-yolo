/**
 * VideoFrameExtractor - Shared video frame extraction via ffmpeg
 *
 * Provides streaming frame extraction from video files for both
 * the CLI processor and the web GUI server.
 */

const { spawn } = require('child_process');

/**
 * Get video duration and metadata via ffprobe
 * @param {string} videoPath
 * @returns {Promise<{duration: number, width: number, height: number}>}
 */
function getVideoInfo(videoPath) {
    return new Promise((resolve, reject) => {
        const proc = spawn('ffprobe', [
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            videoPath
        ]);
        let stdout = '';
        proc.stdout.on('data', d => stdout += d);
        proc.on('close', code => {
            if (code !== 0) return reject(new Error(`ffprobe exited with code ${code}`));
            try {
                const info = JSON.parse(stdout);
                const duration = parseFloat(info.format.duration) || 0;
                const videoStream = (info.streams || []).find(s => s.codec_type === 'video');
                resolve({
                    duration,
                    width: videoStream ? videoStream.width : 0,
                    height: videoStream ? videoStream.height : 0
                });
            } catch (e) {
                reject(e);
            }
        });
    });
}

/**
 * Spawn ffmpeg to extract frames as a PNG stream
 * @param {string} videoPath
 * @param {number} fps - Frames per second to extract
 * @returns {import('child_process').ChildProcess}
 */
function createFrameStream(videoPath, fps, targetWidth = 640, targetHeight = 640) {
    return spawn('ffmpeg', [
        '-i', videoPath,
        '-vf', `fps=${fps},scale=${targetWidth}:${targetHeight}`,
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-'
    ], { stdio: ['ignore', 'pipe', 'ignore'] });
}

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
const IEND_MARKER = Buffer.from([0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]);

/**
 * Async generator that yields complete PNG buffers from an ffmpeg stdout stream
 * @param {import('stream').Readable} stream
 * @yields {Buffer} Complete PNG image buffer
 */
async function* extractPNGFrames(stream) {
    let buffer = Buffer.alloc(0);

    for await (const chunk of stream) {
        buffer = Buffer.concat([buffer, chunk]);

        while (true) {
            const sigIdx = buffer.indexOf(PNG_SIGNATURE);
            if (sigIdx === -1) break;

            const endIdx = buffer.indexOf(IEND_MARKER, sigIdx + 8);
            if (endIdx === -1) break;

            const frameEnd = endIdx + IEND_MARKER.length;
            const frame = buffer.subarray(sigIdx, frameEnd);

            yield Buffer.from(frame);

            buffer = buffer.subarray(frameEnd);
        }
    }
}

/**
 * Format seconds to H:MM:SS or M:SS string
 */
function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${m}:${String(s).padStart(2, '0')}`;
}

module.exports = {
    getVideoInfo,
    createFrameStream,
    extractPNGFrames,
    formatTime
};
