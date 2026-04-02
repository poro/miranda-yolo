/**
 * YOLOSceneAnalyzer - Scene change detection from YOLO detections
 *
 * Uses Jaccard distance on detected object sets to determine whether the gameplay
 * scene has meaningfully changed between frames, enabling smart frame skipping.
 *
 * Adapted from MIRANDA's scene analyzer for standalone prototype use.
 */

class YOLOSceneAnalyzer {
    constructor(options = {}) {
        this.changeThreshold = options.changeThreshold || 0.3;
        this.forceAnalyzeEveryN = options.forceAnalyzeEveryN || 10;
        this.lastDetections = null;
        this.frameCount = 0;
        this.skippedFrames = 0;
        this.analyzedFrames = 0;

        // Detailed transition log
        this._transitions = [];
    }

    /**
     * Determine if the scene has changed enough to warrant analysis
     * @param {Array} currentDetections - Array of {label, confidence, bbox}
     * @returns {{ changed: boolean, reason: string, jaccard: number }}
     */
    analyzeFrame(currentDetections) {
        this.frameCount++;

        // Always analyze the first frame
        if (!this.lastDetections) {
            this.lastDetections = currentDetections;
            this.analyzedFrames++;
            const result = { changed: true, reason: 'first_frame', jaccard: 1.0 };
            this._logTransition(result, currentDetections);
            return result;
        }

        // Safety net: force analysis every N frames even if no change detected
        if (this.frameCount % this.forceAnalyzeEveryN === 0) {
            const jaccard = this._computeJaccard(this.lastDetections, currentDetections);
            this.lastDetections = currentDetections;
            this.analyzedFrames++;
            const result = { changed: true, reason: 'forced_reanalysis', jaccard };
            this._logTransition(result, currentDetections);
            return result;
        }

        // Compare object label sets using Jaccard distance
        const jaccard = this._computeJaccard(this.lastDetections, currentDetections);
        const countChanged = this._hasCountChanged(this.lastDetections, currentDetections);

        let changed = false;
        let reason = 'no_change';

        if (jaccard > this.changeThreshold) {
            changed = true;
            reason = 'jaccard_threshold';
        } else if (countChanged) {
            changed = true;
            reason = 'count_change';
        }

        if (changed) {
            this.lastDetections = currentDetections;
            this.analyzedFrames++;
        } else {
            this.skippedFrames++;
        }

        const result = { changed, reason, jaccard };
        if (changed) this._logTransition(result, currentDetections);
        return result;
    }

    /**
     * Backward-compatible: returns boolean only
     */
    hasSceneChanged(currentDetections) {
        return this.analyzeFrame(currentDetections).changed;
    }

    /**
     * Compute Jaccard distance between two detection sets
     */
    _computeJaccard(prev, curr) {
        const prevLabels = new Set(prev.map(d => d.label));
        const currLabels = new Set(curr.map(d => d.label));

        const union = new Set([...prevLabels, ...currLabels]);
        const intersection = new Set([...prevLabels].filter(x => currLabels.has(x)));

        return union.size === 0 ? 0 : 1 - (intersection.size / union.size);
    }

    /**
     * Check if object counts have changed significantly (even if same labels)
     */
    _hasCountChanged(prev, curr) {
        const prevCounts = this._countByLabel(prev);
        const currCounts = this._countByLabel(curr);

        for (const label of Object.keys(currCounts)) {
            const prevCount = prevCounts[label] || 0;
            const currCount = currCounts[label];
            if (Math.abs(currCount - prevCount) > 1) {
                return true;
            }
        }
        return false;
    }

    /**
     * Count detections by label
     */
    _countByLabel(detections) {
        const counts = {};
        for (const d of detections) {
            counts[d.label] = (counts[d.label] || 0) + 1;
        }
        return counts;
    }

    /**
     * Log a scene transition for detailed reporting
     */
    _logTransition(result, detections) {
        this._transitions.push({
            frame: this.frameCount,
            reason: result.reason,
            jaccard: result.jaccard,
            objects: detections.map(d => d.label)
        });
    }

    /**
     * Get summary statistics
     */
    getStats() {
        const total = this.analyzedFrames + this.skippedFrames;
        return {
            totalFrames: this.frameCount,
            analyzedFrames: this.analyzedFrames,
            skippedFrames: this.skippedFrames,
            skipRate: total > 0 ? (this.skippedFrames / total * 100).toFixed(1) + '%' : '0%'
        };
    }

    /**
     * Get detailed transition log
     */
    getTransitions() {
        return this._transitions;
    }

    /**
     * Reset state for a new session
     */
    reset() {
        this.lastDetections = null;
        this.frameCount = 0;
        this.skippedFrames = 0;
        this.analyzedFrames = 0;
        this._transitions = [];
    }
}

module.exports = YOLOSceneAnalyzer;
