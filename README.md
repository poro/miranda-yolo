# YOLO Video Game Observer

An open-source, self-contained system for real-time object detection in gameplay video using YOLOv8. Includes a web GUI for live analysis, frame-by-frame inspection, annotation, multi-format export, and model fine-tuning — all aimed at building advanced YOLO models purpose-built for video games.

## Why This Exists

Standard YOLO models (trained on COCO) recognize 80 generic object classes: person, car, dog, chair. They have no understanding of game-specific elements like health bars, minimaps, quest markers, enemy indicators, or HUD elements. The model sees "person with sword near door" — not "player choosing between two paths."

This project provides the complete pipeline to go from generic COCO detection to a game-specific model:

1. **Observe** — Process gameplay video through YOLOv8, see what the model detects
2. **Annotate** — Draw bounding boxes around game-specific objects the model misses
3. **Train** — Fine-tune YOLOv8 on your annotations
4. **Deploy** — Hot-swap the improved model and see results immediately

## Features

### Real-Time Detection
- Stream gameplay video through YOLOv8n at ~50ms/frame (CPU)
- Live bounding box overlays on each frame
- Scene change detection using Jaccard distance on object sets
- Smart frame skipping (typically 40-80% of static frames)

### Interactive Analysis
- **Pause/Resume** — Freeze processing to inspect individual frames
- **Frame Browser** — Scrubber + filmstrip to browse frame history
- **Object Search** — Find frames containing specific objects, jump to them
- **Highlighted Results** — Searched objects glow yellow, others dim

### Research Visualizations
- **Object Heatmap** — Aggregate detection positions across frames, see where objects cluster
- **Confidence Histogram** — Distribution of detection confidence scores
- **Object Presence Timeline** — Gantt-style chart showing when each object appears
- **Scene Change Timeline** — Visual strip of changed vs. skipped frames

### Data Persistence
- **SQLite Database** — All sessions, frames, detections, and annotations stored locally
- **Frame Images** — Saved as JPEG to `data/frames/{sessionId}/`
- **JSON Reports** — Auto-exported on session completion
- **Full Audit Trail** — Every annotation tracked with timestamp and source

### Annotation & Training Pipeline
- **In-App Annotation** — Draw bounding boxes directly on paused frames
- **Custom Class Management** — Define game-specific classes (health_bar, minimap, enemy, etc.)
- **Auto-Promote** — Convert YOLO auto-detections to training annotations with one click
- **Multi-Format Export:**
  - YOLO format (local training with Ultralytics)
  - COCO JSON (universal interchange — CVAT, Label Studio, FiftyOne)
  - CSV (spreadsheet analysis)
  - Roboflow upload (cloud training + hosted inference)
- **Integrated Training** — Fine-tune YOLOv8 from the GUI, live training logs
- **Model Registry** — Track trained models with metrics (mAP, precision, recall)
- **Hot-Swap** — Activate a new model without restarting the server

### Annotation CRUD
- **Delete Annotations** — Soft delete with full audit trail (data preserved, excluded from exports)
- **Reclassify Annotations** — Change label with original preserved for audit (`original_label` field)
- **Works on both auto-detections and manual annotations**
- **Visual action icons** — Click X (delete) or R (reclassify) on annotation boxes when paused

### Multi-User Authentication (Supabase)
- **Email/Password** sign-up and sign-in
- **OAuth** — Google and GitHub SSO (via Supabase)
- **Graceful fallback** — Works in single-user mode without Supabase configured
- **User attribution** — Every annotation tagged with `user_id`
- **User menu** — Shows email + logout in toolbar when authenticated

### Session Sharing
- **Public sessions** — Any authenticated user can view and annotate
- **Private sessions** — Only owner can see; share with specific users
- **Permission levels** — `view` (read-only) or `annotate` (can add/edit annotations)
- **Session list** shows: My Sessions, Shared with Me, Public

### Collaborative Training
- **Multi-session training** — Merge annotations from multiple sessions into one dataset
- **Contributor tracking** — `model_contributors` table records who annotated what
- **Cross-user annotations** — Multiple users annotate the same video, all annotations included

### Export
- **CSV Export** — Download all detection data as spreadsheet
- **Annotated Frame Export** — Save current frame with bounding box overlays as PNG

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.8+ (for model training only)
- ffmpeg (for video processing)

### Install

```bash
git clone <repo-url>
cd yolo
npm install

# Download models (choose one):
npm run download-model              # Default: Nano @ 640 (~12MB, fastest)
npm run download-recommended         # Auto-detect platform, pick best models
npm run download-all                 # All 10 variants: n/s/m/l/x @ 640+1280

# Or specific models:
npm run download-model -- --size x --input 1280   # XLarge @ 1280
```

### Platform Recommendations

| Platform | Recommended Model | Expected FPS |
|---|---|---|
| **M4 Max MacBook (128GB)** | Large/XLarge @ 1280 | 20-30+ (CoreML) |
| **Linux + NVIDIA H100** | XLarge @ 1280 | 60+ (CUDA) |
| **Linux CPU (i3/i5)** | Nano @ 640 | 15-20 |
| **Raspberry Pi** | Nano @ 640 | 2-5 |

The server auto-detects CoreML (macOS) and CUDA (Linux + NVIDIA) execution providers.

### Run

```bash
npm start
# Open http://localhost:6600
```

### Usage

1. Select a video from the dropdown (auto-scans `~/Documents/src/movies/`)
2. Adjust FPS (1 = one frame per second), confidence threshold, max frames
3. Click **Start** — watch detections stream in real-time
4. Click **Pause** to inspect frames, use the scrubber to browse
5. Use **Search** to find specific objects across all frames
6. Toggle **Heatmap** to see object clustering patterns
7. Toggle **Annotate** to draw bounding boxes for training data
8. Click **Model Tips** for guidance on fine-tuning

## Multi-User Setup (Optional)

By default, the app runs in single-user mode with no authentication required. To enable multi-user collaboration:

### 1. Create a Supabase Project
- Go to [supabase.com](https://supabase.com) and create a free project
- Enable Email/Password auth (enabled by default)
- Optionally enable Google and/or GitHub OAuth providers in Authentication > Providers

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Supabase credentials:
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_ANON_KEY=eyJ...your-anon-key
```

### 3. Restart
```bash
npm start
```

Users will now see a login page at `/auth.html`. They can sign up, sign in, or continue in anonymous mode.

### Sharing Workflow
1. User A processes a video → session created with `user_id = A`, `visibility = 'private'`
2. User A shares via API: `POST /api/sessions/:id/share { userId: "B", permission: "annotate" }`
3. Or sets public: `PATCH /api/sessions/:id/visibility { visibility: "public" }`
4. User B sees the session in their list, annotates frames
5. Both users' annotations are included when training a model

## Architecture

```
Browser (HTML/JS/CSS)  <-- WebSocket -->  Express Server (Node.js)
                                               |
                                    +----------+-----------+
                                    |          |           |
                                  FFmpeg   YOLO/ONNX    SQLite
                                  (frames) (inference)  (persistence)
                                               |
                                          Python/Ultralytics
                                          (training, export)
```

### Project Structure

```
yolo/
├── server.js                          # Express + WebSocket server
├── public/
│   ├── index.html                     # Web GUI
│   ├── app.js                         # Frontend logic
│   └── style.css                      # Dark theme styles
├── src/services/
│   ├── yolo/
│   │   ├── YOLODetectionService.js    # ONNX inference engine
│   │   └── YOLOSceneAnalyzer.js       # Jaccard-based scene change detection
│   ├── video/
│   │   └── VideoFrameExtractor.js     # FFmpeg frame streaming
│   ├── auth/
│   │   └── SupabaseAuth.js            # Auth middleware (Supabase + fallback)
│   ├── db/
│   │   └── SessionStore.js            # SQLite persistence (better-sqlite3)
│   ├── annotation/
│   │   └── AnnotationManager.js       # Multi-format annotation export
│   └── training/
│       └── TrainingPipeline.js        # Fine-tuning orchestrator
├── models/
│   └── yolov8n.onnx                   # Base model (80 COCO classes)
├── data/
│   ├── yolo-observer.db               # SQLite database
│   ├── frames/                        # Saved frame images by session
│   ├── datasets/                      # Exported training datasets
│   ├── classes.json                   # Custom class definitions
│   └── exports/                       # COCO JSON / CSV exports
├── results/                           # Auto-generated JSON reports
├── scripts/
│   ├── process-video.js               # CLI video processor
│   ├── download-model.js              # Model download/export script
│   └── generate-test-frames.js        # Synthetic test frame generator
└── tests/
    ├── test-yolo-detection.js         # Detection test suite
    └── benchmark.js                   # Performance benchmarks
```

### Database Schema

```sql
sessions         — Video processing sessions with config, stats, owner, visibility
frames           — Individual frames with timestamps and scene change data
detections       — YOLO auto-detections per frame (label, confidence, bbox)
annotations      — Manual + promoted annotations with audit trail
                   (user_id, deleted_at, original_label, updated_at, updated_by)
models           — Trained model registry with metrics and ONNX paths
session_shares   — Per-user sharing permissions (view/annotate)
model_contributors — Tracks which users contributed annotations to each model
```

## API Reference

### REST Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/auth/config` | No | Check if Supabase auth is enabled |
| GET | `/api/me` | Yes | Get current user info |
| GET | `/api/videos` | No | List available videos (uploads + library) |
| POST | `/api/upload` | No | Upload a video file |
| GET | `/api/sessions` | Opt | List sessions (filtered by access if auth enabled) |
| GET | `/api/sessions/:id` | Opt | Session details |
| GET | `/api/sessions/:id/search?label=X` | No | Search frames by object label |
| GET | `/api/sessions/:id/frames/:num` | No | Get frame + detections + annotations |
| POST | `/api/sessions/:id/share` | Yes | Share session with a user |
| DELETE | `/api/sessions/:id/share/:uid` | Yes | Revoke share access |
| PATCH | `/api/sessions/:id/visibility` | Yes | Set public/private |
| GET | `/api/sessions/:id/shares` | Opt | List shares for a session |
| POST | `/api/annotations` | Opt | Create annotation (user_id auto-set) |
| PATCH | `/api/annotations/:id` | Opt | Reclassify (preserves original_label) |
| DELETE | `/api/annotations/:id` | Opt | Soft delete (sets deleted_at) |
| GET | `/api/annotations/:id/history` | No | Audit trail for annotation |
| POST | `/api/annotations/promote/:detId` | Opt | Promote auto-detection to annotation |
| GET | `/api/annotations/stats` | No | Annotation counts by label/source/user |
| GET | `/api/classes` | No | Get custom class list |
| PUT | `/api/classes` | No | Update custom class list |
| POST | `/api/export/yolo` | No | Export annotations as YOLO dataset |
| POST | `/api/export/coco` | No | Export as COCO JSON |
| POST | `/api/export/csv` | No | Export as CSV |
| POST | `/api/export/roboflow` | No | Upload to Roboflow |
| GET | `/api/models` | No | List trained models |
| POST | `/api/models/:id/activate` | No | Hot-swap active model |

*Auth column: Yes = required, Opt = uses user if present, No = public*

### WebSocket Messages

| Direction | Type | Description |
|-----------|------|-------------|
| Client -> | `start` | Begin processing (videoPath, fps, confidence, maxFrames) |
| Client -> | `pause` | Freeze ffmpeg (SIGSTOP) |
| Client -> | `resume` | Unfreeze ffmpeg (SIGCONT) |
| Client -> | `stop` | Abort processing |
| Client -> | `train` | Start model training (name, epochs, imgSize) |
| Server -> | `init` | Session started (sessionId, estimatedFrames, duration) |
| Server -> | `frame` | Frame data (imageBase64, detections, sceneChanged) |
| Server -> | `stats` | Periodic stats (every 10 frames) |
| Server -> | `paused` / `resumed` | Pause state confirmation |
| Server -> | `complete` | Processing finished with final stats |
| Server -> | `train_log` | Training progress line |
| Server -> | `train_complete` | Training finished with metrics |

## Road to a Game-Specific Model

### Phase 1: Observe (Current)
Process gameplay videos with the COCO-trained model. Note what it detects (person, chair, tv) and what it misses (health bars, enemies, HUD).

### Phase 2: Annotate
Pause on frames and use the annotation tool to draw bounding boxes around game elements. Target 500-2000 annotated frames across diverse gameplay situations.

### Phase 3: Train
Export annotations as a YOLO dataset, then train via the GUI or CLI:

```bash
# Via GUI: click Train Model, configure epochs, start
# Via CLI:
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data/datasets/my-game/dataset.yaml', epochs=100, imgsz=640)
model.export(format='onnx', imgsz=640)
"
# Copy the .onnx to models/ and activate via API
```

### Phase 4: Iterate
Activate the new model, process more video, annotate the remaining mistakes, retrain. Each cycle improves accuracy.

### Export Channels

| Channel | When to Use |
|---------|-------------|
| **Local YOLO** | Full control, privacy, offline. Free, fast iteration. |
| **Roboflow** | Team collaboration, production deployment. Auto-augmentation, versioning, hosted inference API. |
| **COCO JSON** | Interop with CVAT, Label Studio, FiftyOne, or any ML tool. |
| **CSV** | Manual review in Excel/Sheets. |

## Performance

Benchmarked on Apple Silicon (M-series, CPU only):

| Metric | Value |
|--------|-------|
| Inference (mean) | ~50ms/frame |
| Inference (P95) | ~85ms |
| Throughput | ~6 fps |
| Memory (peak RSS) | ~500MB |
| Scene skip rate | 40-80% |

## CLI Tools

```bash
# Process a video from command line (no GUI)
npm run process-video -- "/path/to/video.mp4" --fps 1 --confidence 0.3

# Run detection tests on screenshot folder
npm test

# Performance benchmark
npm run benchmark
```

## Known Limitations

- COCO model recognizes generic objects, not game-specific elements (until fine-tuned)
- Single-frame detection — no temporal dynamics (deliberation, backtracking)
- No GPU acceleration (CPU only via onnxruntime-node)
- Pause/resume uses SIGSTOP/SIGCONT (macOS/Linux only)
- Frame history buffer capped at 300 frames in browser memory

## Future Work

- Fine-tuned models for specific game genres (RPG, FPS, strategy)
- Object trajectory tracking across frames
- Behavioral metric derivation (exploration coverage, action frequency)
- Integration with LLM analysis for semantic understanding
- GPU acceleration via onnxruntime-gpu
- Multi-game model hub / community model sharing

## License

MIT
