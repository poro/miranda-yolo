/**
 * SessionStore — SQLite persistence layer for MIRANDA SENSE
 *
 * Stores sessions, frames, detections, annotations (with audit trail),
 * models, users, session sharing, and model contributors.
 * Uses better-sqlite3 for synchronous, fast local storage.
 */

const path = require('path');
const fs = require('fs');
const Database = require('better-sqlite3');

const DB_PATH = path.join(__dirname, '../../../data/yolo-observer.db');
const FRAMES_DIR = path.join(__dirname, '../../../data/frames');

class SessionStore {
    constructor(dbPath = DB_PATH) {
        const dir = path.dirname(dbPath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

        this.db = new Database(dbPath);
        this.db.pragma('journal_mode = WAL');
        this.db.pragma('foreign_keys = ON');
        this._createTables();
        this._migrate();
    }

    _createTables() {
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT NOT NULL,
                video_name TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT,
                fps REAL NOT NULL DEFAULT 1,
                confidence REAL NOT NULL DEFAULT 0.3,
                status TEXT NOT NULL DEFAULT 'running',
                total_frames INTEGER DEFAULT 0,
                mean_inference_ms REAL,
                skip_rate TEXT,
                user_id TEXT,
                visibility TEXT NOT NULL DEFAULT 'private'
            );

            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                frame_num INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                image_path TEXT,
                scene_changed INTEGER NOT NULL DEFAULT 0,
                reason TEXT,
                jaccard REAL,
                detection_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id, frame_num);

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL REFERENCES frames(id),
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_w REAL NOT NULL,
                bbox_h REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id);
            CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label);

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL REFERENCES frames(id),
                label TEXT NOT NULL,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_w REAL NOT NULL,
                bbox_h REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                source TEXT NOT NULL DEFAULT 'manual',
                user_id TEXT,
                deleted_at TEXT,
                original_label TEXT,
                updated_at TEXT,
                updated_by TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_annotations_frame ON annotations(frame_id);
            CREATE INDEX IF NOT EXISTS idx_annotations_user ON annotations(user_id);

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                base_model TEXT NOT NULL DEFAULT 'yolov8n.pt',
                dataset_path TEXT,
                epochs INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                onnx_path TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                metrics TEXT,
                active INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS session_shares (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                user_id TEXT NOT NULL,
                permission TEXT NOT NULL DEFAULT 'annotate',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(session_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS model_contributors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL REFERENCES models(id),
                user_id TEXT NOT NULL,
                annotation_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        `);
    }

    /** Add columns to existing tables if they don't exist yet (safe migration) */
    _migrate() {
        const cols = (table) => {
            const info = this.db.prepare(`PRAGMA table_info(${table})`).all();
            return new Set(info.map(c => c.name));
        };

        // Annotations migration
        const annCols = cols('annotations');
        if (!annCols.has('user_id')) this.db.exec('ALTER TABLE annotations ADD COLUMN user_id TEXT');
        if (!annCols.has('deleted_at')) this.db.exec('ALTER TABLE annotations ADD COLUMN deleted_at TEXT');
        if (!annCols.has('original_label')) this.db.exec('ALTER TABLE annotations ADD COLUMN original_label TEXT');
        if (!annCols.has('updated_at')) this.db.exec('ALTER TABLE annotations ADD COLUMN updated_at TEXT');
        if (!annCols.has('updated_by')) this.db.exec('ALTER TABLE annotations ADD COLUMN updated_by TEXT');

        // Detections migration
        const detCols = cols('detections');
        if (!detCols.has('deleted_at')) this.db.exec('ALTER TABLE detections ADD COLUMN deleted_at TEXT');
        if (!detCols.has('original_label')) this.db.exec('ALTER TABLE detections ADD COLUMN original_label TEXT');

        // Sessions migration
        const sesCols = cols('sessions');
        if (!sesCols.has('user_id')) this.db.exec('ALTER TABLE sessions ADD COLUMN user_id TEXT');
        if (!sesCols.has('visibility')) this.db.exec("ALTER TABLE sessions ADD COLUMN visibility TEXT NOT NULL DEFAULT 'private'");
    }

    // =====================================================
    // Sessions
    // =====================================================

    createSession(videoPath, config = {}) {
        const videoName = path.basename(videoPath, path.extname(videoPath));
        const result = this.db.prepare(`
            INSERT INTO sessions (video_path, video_name, fps, confidence, status, user_id, visibility)
            VALUES (?, ?, ?, ?, 'running', ?, ?)
        `).run(videoPath, videoName, config.fps || 1, config.confidence || 0.3, config.userId || null, config.visibility || 'private');
        const sessionId = result.lastInsertRowid;

        const sessionFramesDir = path.join(FRAMES_DIR, String(sessionId));
        if (!fs.existsSync(sessionFramesDir)) fs.mkdirSync(sessionFramesDir, { recursive: true });

        return sessionId;
    }

    completeSession(sessionId, stats = {}) {
        this.db.prepare(`
            UPDATE sessions SET completed_at = datetime('now'), status = ?,
                total_frames = ?, mean_inference_ms = ?, skip_rate = ?
            WHERE id = ?
        `).run(stats.aborted ? 'stopped' : 'complete', stats.totalFrames || 0,
            stats.meanInference || null, stats.skipRate || null, sessionId);
    }

    listSessions(userId = null) {
        if (!userId) {
            // No auth — return all
            return this.db.prepare(`
                SELECT s.*,
                    (SELECT COUNT(*) FROM frames WHERE session_id = s.id) as frame_count,
                    (SELECT COUNT(*) FROM annotations a JOIN frames f ON a.frame_id = f.id
                     WHERE f.session_id = s.id AND a.deleted_at IS NULL) as annotation_count
                FROM sessions s ORDER BY s.started_at DESC
            `).all();
        }
        // With auth — own + shared + public
        return this.db.prepare(`
            SELECT s.*,
                (SELECT COUNT(*) FROM frames WHERE session_id = s.id) as frame_count,
                (SELECT COUNT(*) FROM annotations a JOIN frames f ON a.frame_id = f.id
                 WHERE f.session_id = s.id AND a.deleted_at IS NULL) as annotation_count,
                CASE
                    WHEN s.user_id = ? THEN 'owner'
                    WHEN s.visibility = 'public' THEN 'public'
                    WHEN EXISTS(SELECT 1 FROM session_shares ss WHERE ss.session_id = s.id AND ss.user_id = ?) THEN 'shared'
                    ELSE NULL
                END as access_type
            FROM sessions s
            WHERE s.user_id = ? OR s.visibility = 'public'
                OR EXISTS(SELECT 1 FROM session_shares ss WHERE ss.session_id = s.id AND ss.user_id = ?)
            ORDER BY s.started_at DESC
        `).all(userId, userId, userId, userId);
    }

    getSession(sessionId) {
        return this.db.prepare('SELECT * FROM sessions WHERE id = ?').get(sessionId);
    }

    canUserAccessSession(sessionId, userId) {
        if (!userId) return true; // No auth mode
        const session = this.getSession(sessionId);
        if (!session) return false;
        if (session.user_id === userId) return true;
        if (session.visibility === 'public') return true;
        const share = this.db.prepare(
            'SELECT 1 FROM session_shares WHERE session_id = ? AND user_id = ?'
        ).get(sessionId, userId);
        return !!share;
    }

    canUserAnnotateSession(sessionId, userId) {
        if (!userId) return true;
        const session = this.getSession(sessionId);
        if (!session) return false;
        if (session.user_id === userId) return true;
        if (session.visibility === 'public') return true;
        const share = this.db.prepare(
            "SELECT 1 FROM session_shares WHERE session_id = ? AND user_id = ? AND permission = 'annotate'"
        ).get(sessionId, userId);
        return !!share;
    }

    // --- Sharing ---

    shareSession(sessionId, targetUserId, permission = 'annotate') {
        return this.db.prepare(`
            INSERT OR REPLACE INTO session_shares (session_id, user_id, permission)
            VALUES (?, ?, ?)
        `).run(sessionId, targetUserId, permission);
    }

    unshareSession(sessionId, targetUserId) {
        return this.db.prepare(
            'DELETE FROM session_shares WHERE session_id = ? AND user_id = ?'
        ).run(sessionId, targetUserId);
    }

    setSessionVisibility(sessionId, visibility) {
        return this.db.prepare(
            'UPDATE sessions SET visibility = ? WHERE id = ?'
        ).run(visibility, sessionId);
    }

    getSessionShares(sessionId) {
        return this.db.prepare(
            'SELECT * FROM session_shares WHERE session_id = ?'
        ).all(sessionId);
    }

    // =====================================================
    // Frames
    // =====================================================

    saveFrame(sessionId, frameData, detections = []) {
        const result = this.db.prepare(`
            INSERT INTO frames (session_id, frame_num, timestamp, image_path, scene_changed, reason, jaccard, detection_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `).run(sessionId, frameData.frameNum, frameData.timestamp, frameData.imagePath || null,
            frameData.sceneChanged ? 1 : 0, frameData.reason || null, frameData.jaccard || null, detections.length);
        const frameId = result.lastInsertRowid;

        const detectionIds = [];
        if (detections.length > 0) {
            const detStmt = this.db.prepare(`
                INSERT INTO detections (frame_id, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            `);
            const insertMany = this.db.transaction((dets) => {
                for (const d of dets) {
                    const r = detStmt.run(frameId, d.label, d.confidence, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h);
                    detectionIds.push(Number(r.lastInsertRowid));
                }
            });
            insertMany(detections);
        }

        return { frameId, detectionIds };
    }

    getFrame(sessionId, frameNum) {
        const frame = this.db.prepare(
            'SELECT * FROM frames WHERE session_id = ? AND frame_num = ?'
        ).get(sessionId, frameNum);
        if (!frame) return null;
        frame.detections = this.db.prepare('SELECT * FROM detections WHERE frame_id = ? AND deleted_at IS NULL').all(frame.id);
        frame.annotations = this.db.prepare('SELECT * FROM annotations WHERE frame_id = ? AND deleted_at IS NULL').all(frame.id);
        return frame;
    }

    getFrameById(frameId) {
        const frame = this.db.prepare('SELECT * FROM frames WHERE id = ?').get(frameId);
        if (!frame) return null;
        frame.detections = this.db.prepare('SELECT * FROM detections WHERE frame_id = ? AND deleted_at IS NULL').all(frame.id);
        frame.annotations = this.db.prepare('SELECT * FROM annotations WHERE frame_id = ? AND deleted_at IS NULL').all(frame.id);
        return frame;
    }

    // --- Detection CRUD ---

    deleteDetection(detectionId, userId = null) {
        const det = this.db.prepare('SELECT * FROM detections WHERE id = ? AND deleted_at IS NULL').get(detectionId);
        if (!det) return null;
        this.db.prepare('UPDATE detections SET deleted_at = datetime(?) WHERE id = ?')
            .run('now', detectionId);
        return { deleted: true, id: detectionId };
    }

    reclassifyDetection(detectionId, newLabel, userId = null) {
        const det = this.db.prepare('SELECT * FROM detections WHERE id = ? AND deleted_at IS NULL').get(detectionId);
        if (!det) return null;
        const originalLabel = det.original_label || det.label;
        this.db.prepare('UPDATE detections SET label = ?, original_label = ? WHERE id = ?')
            .run(newLabel, originalLabel, detectionId);
        return this.db.prepare('SELECT * FROM detections WHERE id = ?').get(detectionId);
    }

    // =====================================================
    // Search
    // =====================================================

    searchByLabel(sessionId, label) {
        const query = sessionId
            ? `SELECT DISTINCT f.*, d.label, d.confidence
               FROM frames f JOIN detections d ON d.frame_id = f.id
               WHERE f.session_id = ? AND d.label LIKE ? ORDER BY f.frame_num`
            : `SELECT DISTINCT f.*, d.label, d.confidence, s.video_name
               FROM frames f JOIN detections d ON d.frame_id = f.id JOIN sessions s ON f.session_id = s.id
               WHERE d.label LIKE ? ORDER BY s.started_at DESC, f.frame_num`;
        const params = sessionId ? [sessionId, `%${label}%`] : [`%${label}%`];
        return this.db.prepare(query).all(...params);
    }

    getObjectFrequency(sessionId) {
        return this.db.prepare(`
            SELECT d.label, COUNT(*) as count FROM detections d JOIN frames f ON d.frame_id = f.id
            WHERE f.session_id = ? GROUP BY d.label ORDER BY count DESC
        `).all(sessionId);
    }

    // =====================================================
    // Annotations (full CRUD + audit trail)
    // =====================================================

    saveAnnotation(frameId, label, bbox, source = 'manual', userId = null) {
        const result = this.db.prepare(`
            INSERT INTO annotations (frame_id, label, bbox_x, bbox_y, bbox_w, bbox_h, source, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `).run(frameId, label, bbox.x, bbox.y, bbox.w, bbox.h, source, userId);
        return result.lastInsertRowid;
    }

    reclassifyAnnotation(annotationId, newLabel, userId = null) {
        const ann = this.db.prepare('SELECT * FROM annotations WHERE id = ? AND deleted_at IS NULL').get(annotationId);
        if (!ann) return null;

        // Preserve original label on first reclassification
        const originalLabel = ann.original_label || ann.label;

        this.db.prepare(`
            UPDATE annotations SET label = ?, original_label = ?, updated_at = datetime('now'), updated_by = ?
            WHERE id = ?
        `).run(newLabel, originalLabel, userId, annotationId);

        return this.db.prepare('SELECT * FROM annotations WHERE id = ?').get(annotationId);
    }

    deleteAnnotation(annotationId, userId = null) {
        const ann = this.db.prepare('SELECT * FROM annotations WHERE id = ? AND deleted_at IS NULL').get(annotationId);
        if (!ann) return null;

        this.db.prepare(`
            UPDATE annotations SET deleted_at = datetime('now'), updated_by = ?
            WHERE id = ?
        `).run(userId, annotationId);

        return { deleted: true, id: annotationId };
    }

    getAnnotation(annotationId) {
        return this.db.prepare('SELECT * FROM annotations WHERE id = ?').get(annotationId);
    }

    getAnnotationHistory(annotationId) {
        const ann = this.db.prepare('SELECT * FROM annotations WHERE id = ?').get(annotationId);
        if (!ann) return null;
        return {
            ...ann,
            was_reclassified: !!ann.original_label,
            was_deleted: !!ann.deleted_at
        };
    }

    getAnnotationsByUser(userId, sessionId = null) {
        if (sessionId) {
            return this.db.prepare(`
                SELECT a.* FROM annotations a JOIN frames f ON a.frame_id = f.id
                WHERE a.user_id = ? AND f.session_id = ? AND a.deleted_at IS NULL
                ORDER BY a.created_at DESC
            `).all(userId, sessionId);
        }
        return this.db.prepare(`
            SELECT a.* FROM annotations a WHERE a.user_id = ? AND a.deleted_at IS NULL
            ORDER BY a.created_at DESC
        `).all(userId);
    }

    promoteDetection(detectionId, userId = null) {
        const det = this.db.prepare('SELECT * FROM detections WHERE id = ?').get(detectionId);
        if (!det) return null;
        return this.saveAnnotation(det.frame_id, det.label, {
            x: det.bbox_x, y: det.bbox_y, w: det.bbox_w, h: det.bbox_h
        }, 'auto', userId);
    }

    getAnnotationStats(excludeDeleted = true) {
        const where = excludeDeleted ? 'WHERE a.deleted_at IS NULL' : '';
        return this.db.prepare(`
            SELECT a.label, a.source, a.user_id, COUNT(*) as count
            FROM annotations a ${where}
            GROUP BY a.label, a.source, a.user_id
            ORDER BY count DESC
        `).all();
    }

    getAllAnnotationsForExport(sessionIds = null) {
        let query = `
            SELECT a.*, f.image_path, f.frame_num, f.session_id, s.video_name
            FROM annotations a
            JOIN frames f ON a.frame_id = f.id
            JOIN sessions s ON f.session_id = s.id
            WHERE a.deleted_at IS NULL
        `;
        if (sessionIds && sessionIds.length > 0) {
            const placeholders = sessionIds.map(() => '?').join(',');
            query += ` AND f.session_id IN (${placeholders})`;
            query += ' ORDER BY s.id, f.frame_num';
            return this.db.prepare(query).all(...sessionIds);
        }
        query += ' ORDER BY s.id, f.frame_num';
        return this.db.prepare(query).all();
    }

    // =====================================================
    // Models
    // =====================================================

    registerModel(name, config = {}) {
        const result = this.db.prepare(`
            INSERT INTO models (name, base_model, dataset_path, epochs, status)
            VALUES (?, ?, ?, ?, 'pending')
        `).run(name, config.baseModel || 'yolov8n.pt', config.datasetPath || null, config.epochs || 100);
        return result.lastInsertRowid;
    }

    updateModel(modelId, updates) {
        const sets = [], vals = [];
        for (const [key, val] of Object.entries(updates)) {
            sets.push(`${key} = ?`);
            vals.push(typeof val === 'object' ? JSON.stringify(val) : val);
        }
        vals.push(modelId);
        this.db.prepare(`UPDATE models SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
    }

    activateModel(modelId) {
        this.db.prepare('UPDATE models SET active = 0').run();
        this.db.prepare('UPDATE models SET active = 1 WHERE id = ?').run(modelId);
    }

    getActiveModel() { return this.db.prepare('SELECT * FROM models WHERE active = 1').get(); }
    listModels() { return this.db.prepare('SELECT * FROM models ORDER BY created_at DESC').all(); }

    addModelContributor(modelId, userId, annotationCount) {
        this.db.prepare(`
            INSERT INTO model_contributors (model_id, user_id, annotation_count) VALUES (?, ?, ?)
        `).run(modelId, userId, annotationCount);
    }

    getModelContributors(modelId) {
        return this.db.prepare('SELECT * FROM model_contributors WHERE model_id = ?').all(modelId);
    }

    // =====================================================
    // Cleanup
    // =====================================================

    close() { this.db.close(); }
}

module.exports = SessionStore;
