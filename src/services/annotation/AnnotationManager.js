/**
 * AnnotationManager — Multi-format annotation export
 *
 * Exports annotations to:
 * 1. YOLO format (local training)
 * 2. COCO JSON (universal interchange)
 * 3. CSV (spreadsheet analysis)
 * 4. Roboflow upload (cloud training)
 */

const path = require('path');
const fs = require('fs');
const sharp = require('sharp');

const DATASETS_DIR = path.join(__dirname, '../../../data/datasets');
const CLASSES_PATH = path.join(__dirname, '../../../data/classes.json');

class AnnotationManager {
    constructor(sessionStore) {
        this.store = sessionStore;
    }

    // --- Class Management ---

    getClasses() {
        if (fs.existsSync(CLASSES_PATH)) {
            return JSON.parse(fs.readFileSync(CLASSES_PATH, 'utf-8'));
        }
        // Default game-oriented classes
        const defaults = [
            'player', 'enemy', 'npc', 'health_bar', 'minimap',
            'inventory', 'quest_marker', 'hud_element', 'weapon',
            'loot_item', 'damage_number', 'crosshair', 'ability_cooldown',
            'ammo_counter', 'score_display', 'menu_button', 'dialog_box'
        ];
        this.saveClasses(defaults);
        return defaults;
    }

    saveClasses(classes) {
        const dir = path.dirname(CLASSES_PATH);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(CLASSES_PATH, JSON.stringify(classes, null, 2));
    }

    // --- YOLO Format Export ---

    async exportYOLO(datasetName, options = {}) {
        const { trainSplit = 0.8 } = options;
        const classes = this.getClasses();
        const annotations = this.store.getAllAnnotationsForExport();

        if (annotations.length === 0) {
            throw new Error('No annotations to export');
        }

        const datasetDir = path.join(DATASETS_DIR, datasetName);
        const dirs = ['images/train', 'images/val', 'labels/train', 'labels/val'];
        for (const d of dirs) {
            fs.mkdirSync(path.join(datasetDir, d), { recursive: true });
        }

        // Group annotations by frame
        const frameGroups = {};
        for (const ann of annotations) {
            const key = ann.frame_id;
            if (!frameGroups[key]) {
                frameGroups[key] = { imagePath: ann.image_path, annotations: [] };
            }
            frameGroups[key].annotations.push(ann);
        }

        const frameIds = Object.keys(frameGroups);
        const splitIdx = Math.floor(frameIds.length * trainSplit);

        let trainCount = 0, valCount = 0;

        for (let i = 0; i < frameIds.length; i++) {
            const frameId = frameIds[i];
            const group = frameGroups[frameId];
            const split = i < splitIdx ? 'train' : 'val';

            if (!group.imagePath || !fs.existsSync(group.imagePath)) continue;

            // Copy image
            const imgName = `frame_${frameId}.jpg`;
            const imgDest = path.join(datasetDir, 'images', split, imgName);
            fs.copyFileSync(group.imagePath, imgDest);

            // Write label file
            const labelLines = [];
            for (const ann of group.annotations) {
                const classIdx = classes.indexOf(ann.label);
                if (classIdx === -1) continue;

                // YOLO format: class_id center_x center_y width height (all normalized)
                const cx = ann.bbox_x + ann.bbox_w / 2;
                const cy = ann.bbox_y + ann.bbox_h / 2;
                labelLines.push(`${classIdx} ${cx.toFixed(6)} ${cy.toFixed(6)} ${ann.bbox_w.toFixed(6)} ${ann.bbox_h.toFixed(6)}`);
            }

            const labelDest = path.join(datasetDir, 'labels', split, `frame_${frameId}.txt`);
            fs.writeFileSync(labelDest, labelLines.join('\n'));

            if (split === 'train') trainCount++;
            else valCount++;
        }

        // Write dataset.yaml
        const yamlContent = [
            `path: ${datasetDir}`,
            'train: images/train',
            'val: images/val',
            '',
            'names:'
        ];
        classes.forEach((name, idx) => {
            yamlContent.push(`  ${idx}: ${name}`);
        });

        const yamlPath = path.join(datasetDir, 'dataset.yaml');
        fs.writeFileSync(yamlPath, yamlContent.join('\n'));

        return {
            datasetDir,
            yamlPath,
            trainCount,
            valCount,
            classes: classes.length,
            totalAnnotations: annotations.length
        };
    }

    // --- COCO JSON Export ---

    exportCOCO(outputPath) {
        const classes = this.getClasses();
        const annotations = this.store.getAllAnnotationsForExport();

        const coco = {
            info: {
                description: 'YOLO Game Observer Annotations',
                date_created: new Date().toISOString(),
                version: '1.0'
            },
            images: [],
            annotations: [],
            categories: classes.map((name, idx) => ({
                id: idx,
                name,
                supercategory: 'game_element'
            }))
        };

        const imageMap = {};
        let annId = 1;

        for (const ann of annotations) {
            if (!ann.image_path) continue;

            // Register image if new
            if (!imageMap[ann.frame_id]) {
                const imageId = Object.keys(imageMap).length + 1;
                imageMap[ann.frame_id] = imageId;
                coco.images.push({
                    id: imageId,
                    file_name: path.basename(ann.image_path),
                    width: 640,  // Will be actual size from saved frame
                    height: 480
                });
            }

            const classIdx = classes.indexOf(ann.label);
            if (classIdx === -1) continue;

            // COCO uses absolute pixel coords — we store normalized, convert assuming 640x480
            // The actual image dimensions should be read, but for export we note this limitation
            coco.annotations.push({
                id: annId++,
                image_id: imageMap[ann.frame_id],
                category_id: classIdx,
                bbox: [
                    ann.bbox_x * 640,
                    ann.bbox_y * 480,
                    ann.bbox_w * 640,
                    ann.bbox_h * 480
                ],
                area: ann.bbox_w * 640 * ann.bbox_h * 480,
                iscrowd: 0
            });
        }

        fs.writeFileSync(outputPath, JSON.stringify(coco, null, 2));
        return { outputPath, images: coco.images.length, annotations: coco.annotations.length };
    }

    // --- CSV Export ---

    exportCSV(outputPath) {
        const annotations = this.store.getAllAnnotationsForExport();

        const header = 'image_path,video_name,frame_num,label,confidence,bbox_x,bbox_y,bbox_w,bbox_h,source,created_at';
        const rows = annotations.map(a =>
            `${a.image_path || ''},${a.video_name},${a.frame_num},${a.label},,${a.bbox_x},${a.bbox_y},${a.bbox_w},${a.bbox_h},${a.source},${a.created_at}`
        );

        fs.writeFileSync(outputPath, [header, ...rows].join('\n'));
        return { outputPath, count: annotations.length };
    }

    // --- Roboflow Upload ---

    async uploadToRoboflow(apiKey, projectId) {
        const annotations = this.store.getAllAnnotationsForExport();

        if (!apiKey || !projectId) {
            throw new Error('Roboflow API key and project ID required');
        }

        const results = { uploaded: 0, failed: 0, errors: [] };

        // Group by frame
        const frameGroups = {};
        for (const ann of annotations) {
            if (!frameGroups[ann.frame_id]) {
                frameGroups[ann.frame_id] = { imagePath: ann.image_path, annotations: [] };
            }
            frameGroups[ann.frame_id].annotations.push(ann);
        }

        for (const [frameId, group] of Object.entries(frameGroups)) {
            if (!group.imagePath || !fs.existsSync(group.imagePath)) continue;

            try {
                const imageBuffer = fs.readFileSync(group.imagePath);
                const base64Image = imageBuffer.toString('base64');

                // Roboflow Upload API
                const url = `https://api.roboflow.com/dataset/${projectId}/upload?api_key=${apiKey}&name=frame_${frameId}.jpg`;

                const response = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: base64Image
                });

                if (response.ok) {
                    results.uploaded++;
                    // TODO: Upload annotations via Roboflow annotation API
                } else {
                    results.failed++;
                    results.errors.push(`Frame ${frameId}: HTTP ${response.status}`);
                }
            } catch (err) {
                results.failed++;
                results.errors.push(`Frame ${frameId}: ${err.message}`);
            }
        }

        return results;
    }
}

module.exports = AnnotationManager;
