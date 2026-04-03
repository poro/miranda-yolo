# MIRANDA SENSE — Scale Architecture Roadmap

**Version**: 1.0
**Date**: April 2026
**Status**: Strategic Planning

---

## Executive Summary

MIRANDA SENSE is being architected as an open-source, self-contained platform for gaming computer vision at scale. The vision: any game studio, researcher, or community contributor can process gameplay video, annotate game-specific objects, train custom YOLO models, and deploy them — without depending on Roboflow or any external platform.

The competitive landscape is weak. Roboflow's gaming coverage is limited to a handful of FPS aimbot datasets, a 228-image HUD elements set, and some board games. No platform offers end-to-end video-to-model pipelines purpose-built for games. No one has large-scale annotated datasets for open-world RPGs, narrative games, or game UI elements.

**Our advantages:**
- 372GB video library spanning 15+ game franchises (AC Valhalla, Red Dead, Celeste, Life is Strange, etc.)
- End-to-end pipeline: video ingestion, YOLO detection, annotation, training, deployment
- MIRANDA knowledge graph upstream for behavioral/learning analysis
- Scene-level temporal understanding (not just frame-by-frame)
- Self-contained — runs fully offline, no cloud dependency

---

## Current State

### What Exists Today

| Component | Status | Tech | Scale |
|-----------|--------|------|-------|
| **MIRANDA Core** (miranda-axio-v0) | Mature | Electron, OpenAI Vision, Mermaid graphs | Single-user desktop |
| **MIRANDA SENSE** (miranda-yolo) | Beta | Node.js, ONNX Runtime, Express, SQLite | ~500KB/session |
| **Game QA Analyzer** | Prototype | Python, Streamlit, YOLOv8, FER | GPU-bound |
| **Clash-Commander** | Production | Kotlin, Roboflow API, Claude/Gemini | Android APK |
| **Video Library** | Growing | 169 files, 372GB | 15+ game franchises |

### MIRANDA SENSE Capabilities
- 11 ONNX models (n/s/m/l/x at 640+1280px)
- CoreML acceleration on Apple Silicon, CUDA on Linux
- Real-time WebSocket streaming at 15-50 fps inference
- SQLite persistence with 8 tables (sessions, frames, detections, annotations, models, shares, contributors)
- Multi-user auth (Supabase), session sharing (public/private)
- Annotation CRUD with full audit trail (soft delete, reclassify with original preserved)
- Multi-format export (YOLO, COCO JSON, CSV, Roboflow upload)
- Integrated training pipeline (Python subprocess, model registry, hot-swap)
- Pause/resume, frame browser, object search, heatmap, confidence histogram, presence timeline

---

## Phase 1: Foundation (Q2 2026)

### 1.1 Dataset Pipeline — Video Library to Labeled Frames

**Goal**: Turn 372GB of gameplay video into the largest open-source gaming CV dataset.

**Architecture**:
```
Video Library (372GB, 169 files)
    ↓
Batch Frame Extractor (ffmpeg, configurable fps)
    ↓
YOLO Auto-Detection (pre-label with COCO model)
    ↓
Active Learning Selector (pick most uncertain frames)
    ↓
Annotation Queue (prioritized by uncertainty + diversity)
    ↓
Human Review (approve/correct/reclassify auto-labels)
    ↓
Versioned Dataset (lakeFS branches, YOLO + COCO formats)
```

**Batch processing service** (`src/services/batch/BatchProcessor.js`):
- Queue-based: submit a video, get a job ID, process in background
- Worker pool: 1 worker per CPU core, each running ffmpeg + YOLO
- Progress tracking via WebSocket
- Output: frames saved to `data/frames/`, detections to SQLite, auto-annotations created

**Active learning scorer**:
- After initial YOLO pass, score each frame by:
  - **Model uncertainty**: average confidence of detections (lower = more valuable to annotate)
  - **Diversity**: Jaccard distance from already-annotated frames (higher = more novel)
  - **Event density**: frames with many detections or scene changes ranked higher
- Surface top-N frames in the annotation UI as "Suggested for Review"

**Scale targets**:
- Process entire 372GB library in batch (est. 500K-1M frames at 1fps)
- Auto-label all frames with COCO model
- Surface 10K-50K frames for human annotation via active learning

### 1.2 Annotation Quality System

**Multi-layer quality control** (adapted from CVAT/Scale AI patterns):

1. **Gold sets**: 100-200 expert-annotated frames per game, used as ground truth
2. **Interleaved monitoring**: 5-10% of annotation tasks are gold items — annotators don't know which
3. **Inter-annotator agreement (IAA)**: When 2+ users annotate the same frame, compute IoU agreement
4. **Review queue**: Annotations ordered by agreement score — low agreement = needs expert review
5. **Annotator scoring**: Per-user accuracy tracked over time, top annotators get more weight in consensus

**Schema additions**:
```sql
annotation_reviews (
    id, annotation_id, reviewer_user_id,
    verdict TEXT ('approved', 'rejected', 'corrected'),
    corrected_label TEXT, corrected_bbox TEXT,
    created_at
)

gold_frames (
    id, frame_id, is_active, created_by, created_at
)

annotator_scores (
    user_id, total_annotations, gold_accuracy,
    agreement_score, last_calculated_at
)
```

### 1.3 Game-Specific Class Taxonomy

**Default taxonomy** (replace current flat list with hierarchical):

```yaml
game_ui:
  - health_bar
  - enemy_health_bar
  - mana_bar
  - stamina_bar
  - minimap
  - minimap_icon
  - quest_marker
  - waypoint
  - compass
  - crosshair
  - ammo_counter
  - ability_cooldown
  - score_display
  - level_indicator

game_entities:
  - player_character
  - enemy
  - npc
  - companion
  - boss
  - mount

game_objects:
  - weapon
  - loot_item
  - chest
  - door
  - interactable
  - vehicle

game_screens:
  - main_menu
  - pause_menu
  - inventory
  - map_screen
  - dialog_box
  - cutscene
  - loading_screen
  - death_screen
  - title_screen

game_effects:
  - damage_number
  - hit_marker
  - particle_effect
  - status_icon
```

Each game can extend this base taxonomy with game-specific classes.

---

## Phase 2: Scale Infrastructure (Q3 2026)

### 2.1 Database Migration: SQLite to PostgreSQL

**Why**: SQLite is single-writer. Multi-user annotation at scale needs concurrent writes.

**Approach**:
- Keep SQLite as the local/offline default (single-user mode still works)
- Add PostgreSQL support via connection string in `.env`
- `SessionStore.js` gets a `DatabaseAdapter` interface — implementations for SQLite and PostgreSQL
- Migration tool: export SQLite to PostgreSQL dump

```
DATABASE_URL=postgresql://user:pass@host:5432/miranda_sense
# OR leave unset for SQLite (local mode)
```

### 2.2 Object Storage: Local to S3/R2

**Why**: Frame images (JPEG) don't belong in a database or local filesystem at scale.

**Architecture**:
```
Frames → S3-compatible storage (Cloudflare R2, AWS S3, MinIO)
    ↓
Metadata → PostgreSQL (image_url instead of image_path)
    ↓
CDN → Serve frames to annotation UI via signed URLs
```

- **Cloudflare R2**: Zero egress fees, S3-compatible API, already in your Cloudflare account
- Frame URLs become: `https://r2.miranda-sense.com/frames/{sessionId}/{frameNum}.jpg`
- Local mode: still saves to `data/frames/` with file:// paths

### 2.3 Job Queue: Background Processing

**Why**: Video processing, model training, and batch exports are long-running. They shouldn't block the web server.

**Architecture**:
```
Express API (receives job requests)
    ↓
BullMQ (Redis-backed job queue)
    ↓
Worker Pool (N workers, each with ffmpeg + YOLO)
    ↓
WebSocket (progress updates to UI)
```

**Job types**:
- `process_video` — extract frames + YOLO detection
- `train_model` — fine-tune YOLO on annotations
- `export_dataset` — generate YOLO/COCO/CSV exports
- `batch_inference` — run new model on existing frames (re-detect)
- `active_learning_score` — compute uncertainty scores for annotation queue

### 2.4 Multi-GPU Training

**YOLO26** (latest Ultralytics, Sep 2025) with Distributed Data Parallel:

```python
from ultralytics import YOLO
model = YOLO("yolo26m.pt")
model.train(
    data="game_dataset.yaml",
    epochs=100,
    device=[0, 1, 2, 3],  # Multi-GPU
    batch=64,              # Must be multiple of GPU count
    imgsz=640
)
```

**Key YOLO26 improvements relevant to gaming**:
- **NMS-free inference** — no post-processing bottleneck
- **STAL** (Small-Target-Aware Label Assignment) — critical for small UI elements (minimap icons, ammo counters)
- **Progressive Loss Balancing** — better convergence on imbalanced gaming datasets

**Cloud training option**:
- Ultralytics Cloud (managed GPU training with metrics streaming)
- Or self-hosted: RunPod, Lambda Labs, or your own GPU cluster

### 2.5 Experiment Tracking: W&B + MLflow

```
Training Run
    ↓
Weights & Biases (real-time experiment visualization)
    ├── Loss curves, mAP per epoch
    ├── Per-class AP breakdown
    ├── Confusion matrices
    └── Model checkpoints
    ↓
MLflow Model Registry (deployment pipeline)
    ├── Staging → Production → Archived
    ├── Approval workflows
    └── REST API for model serving
```

**Setup**:
```bash
pip install ultralytics wandb mlflow
yolo settings wandb=True
```

Automatically logs all training metrics, hyperparameters, and model artifacts.

---

## Phase 3: Community Platform (Q4 2026)

### 3.1 Contributor Portal

**Web application** for community annotators:

```
Landing Page → Sign Up (Supabase Auth)
    ↓
Dashboard
    ├── My Annotations (count, accuracy, leaderboard rank)
    ├── Annotation Queue (active learning suggestions)
    ├── Browse Games (filter by title, genre)
    ├── My Models (trained from my annotations)
    └── Community Models (shared, top-rated)
```

**Gamification**:
- Annotation count leaderboard
- Accuracy badges (based on gold set performance)
- Per-game contribution tracking
- "Dataset Champion" for top contributors per game title

### 3.2 Dataset Versioning & Distribution

**lakeFS** for internal dataset management:
```
main (production dataset)
    ├── branch: ac-valhalla-v2 (adding 500 new frames)
    ├── branch: experiment/yolo26-large (testing larger model)
    └── branch: contributor/user123 (pending review)
```

**Hugging Face Hub** for community distribution:
```
miranda-sense/gaming-cv-dataset
    ├── README.md (dataset card)
    ├── game_ui/ (health bars, minimaps, HUD elements)
    ├── game_entities/ (players, enemies, NPCs)
    ├── metadata.parquet (annotations + frame metadata)
    └── streaming support (no full download required)
```

### 3.3 Pre-Trained Model Zoo

Publish fine-tuned models for specific games and game genres:

```
miranda-sense/models/
    ├── game-ui-detector-v1      (generic HUD elements)
    ├── ac-valhalla-detector-v1  (AC Valhalla specific)
    ├── fps-player-detector-v1   (FPS games, player detection)
    ├── rdr2-detector-v1         (Red Dead Redemption 2)
    └── rpg-inventory-v1         (inventory/menu screens)
```

Each model published to Hugging Face Hub with:
- Model card (training data, metrics, limitations)
- ONNX export (ready to use in MIRANDA SENSE)
- YOLO .pt format (for further fine-tuning)
- Inference API endpoint

### 3.4 API & SDK

**REST API** for programmatic access:
```
POST /api/v1/detect          — Run detection on uploaded image
POST /api/v1/detect/video    — Submit video for batch processing
GET  /api/v1/datasets        — List available datasets
GET  /api/v1/models          — List available models
POST /api/v1/train           — Submit training job
GET  /api/v1/jobs/:id        — Job status + progress
```

**Python SDK**:
```python
from miranda_sense import MirandaSense

ms = MirandaSense(api_key="...")

# Detect objects in a game screenshot
results = ms.detect("screenshot.png", model="game-ui-detector-v1")

# Process a gameplay video
job = ms.process_video("gameplay.mp4", fps=1, model="ac-valhalla-v1")
job.wait()

# Download a dataset
dataset = ms.dataset("gaming-cv-v1").download(format="yolov8")
```

**Node.js SDK**:
```javascript
const { MirandaSense } = require('miranda-sense');
const ms = new MirandaSense({ apiKey: '...' });

const detections = await ms.detect('screenshot.png', { model: 'game-ui-v1' });
```

---

## Phase 4: Video Intelligence (Q1 2027)

### 4.1 Object Tracking: YOLO + ByteTrack

Move beyond frame-by-frame detection to object tracking across video:

```python
from ultralytics import YOLO
model = YOLO("game-ui-v1.pt")
results = model.track(source="gameplay.mp4", tracker="bytetrack.yaml")
```

**What this enables**:
- Track a specific NPC across frames (assign persistent ID)
- Measure how long a player looks at their health bar
- Detect "deliberation" — player stops moving, camera pans between options
- Track enemy movement patterns over time

### 4.2 VideoMAE Pre-Training

Use self-supervised pre-training on unlabeled gameplay footage:

```
372GB unlabeled gameplay video
    ↓
VideoMAE (90-95% masking ratio)
    ↓
Pre-trained video encoder (understands game visual patterns)
    ↓
Fine-tune detection head on labeled data
```

**Why this matters**: VideoMAE with 90-95% masking exploits temporal redundancy in video. You can pre-train on your entire 372GB library without any labels, then fine-tune on the much smaller annotated subset. This gives the model implicit understanding of game visual patterns (menus look different from gameplay, cutscenes have cinematic framing, etc.).

### 4.3 Behavioral Signal Extraction

Connect YOLO detection + tracking to MIRANDA's knowledge graph:

```
YOLO Detections (frame-level)
    + ByteTrack (object persistence)
    + Scene Analyzer (transition detection)
    ↓
Behavioral Signals:
    ├── Exploration coverage (minimap tracking)
    ├── Combat frequency (enemy encounter rate)
    ├── Deliberation time (pause duration at choice points)
    ├── Inventory management patterns
    ├── Menu navigation efficiency
    └── Death/retry patterns
    ↓
MIRANDA Knowledge Graph
    ↓
Learning/Behavioral Analysis
```

### 4.4 Multi-Modal Fusion

Combine game-qa-analyzer's emotion pipeline with MIRANDA SENSE's detection:

```
Gameplay Video → YOLO (game events) ─────────────┐
                                                   ├→ Temporal Alignment → Fused Analysis
Facecam Video → FER (player emotions) ────────────┘

Audio → Speech/Sound Analysis ────────────────────┘
```

**Output**: "Player encountered boss at 3:42, showed frustration (anger 0.7), died at 3:58, retried 4 times, showed relief (joy 0.8) on success at 5:12"

---

## Phase 5: Enterprise (Q2-Q3 2027)

### 5.1 Game Studio Integration

**SDK for game studios** to integrate MIRANDA SENSE into their QA pipeline:

```
Game Build Pipeline
    ↓
Automated Playtest Recording
    ↓
MIRANDA SENSE API (batch processing)
    ↓
QA Dashboard
    ├── UI element visibility scores
    ├── Player confusion hotspots
    ├── Accessibility compliance checks
    ├── HUD occlusion detection
    └── Tutorial effectiveness metrics
```

### 5.2 Federated Learning

Game studios contribute annotations without sharing footage:

```
Studio A (private footage) → Train local model → Share model gradients
Studio B (private footage) → Train local model → Share model gradients
                                ↓
                    Federated aggregation server
                                ↓
                    Improved global model (no footage shared)
```

### 5.3 Deployment Options

| Tier | Target | Infrastructure |
|------|--------|---------------|
| **Local** | Individual researcher | npm start, SQLite, local GPU |
| **Team** | Small studio | Docker Compose, PostgreSQL, shared GPU |
| **Cloud** | Large studio | Kubernetes, R2/S3, GPU cluster, managed DB |
| **SaaS** | Community | Hosted platform, API access, pay-per-inference |

---

## Technical Architecture (Target State)

```
┌─────────────────────────────────────────────────────────────┐
│                    MIRANDA SENSE Platform                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Web GUI  │  │ REST API │  │Python SDK│  │ Node SDK │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       └──────────────┼──────────────┼──────────────┘          │
│                      ▼                                        │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              API Gateway (Express)                    │     │
│  │  Auth (Supabase) │ Rate Limit │ Session Management   │     │
│  └──────────────────────┬──────────────────────────────┘     │
│                          ▼                                    │
│  ┌────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐    │
│  │ Video  │  │Detection │  │Annotation │  │ Training  │    │
│  │Process │  │ Service  │  │ Service   │  │ Pipeline  │    │
│  │Service │  │(YOLO/ORT)│  │(CRUD+QC)  │  │(YOLO+DDP) │    │
│  └───┬────┘  └────┬─────┘  └─────┬─────┘  └─────┬─────┘    │
│      │            │              │              │            │
│  ┌───┴────────────┴──────────────┴──────────────┴──────┐    │
│  │              Job Queue (BullMQ / Redis)               │    │
│  └──────────────────────┬──────────────────────────────┘     │
│                          ▼                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │PostgreSQL│  │  S3/R2   │  │  Redis   │  │  MLflow  │    │
│  │(metadata)│  │ (frames) │  │ (cache)  │  │(models)  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │           Dataset Layer (lakeFS + HuggingFace)        │     │
│  │  Versioning │ Branching │ Streaming │ Distribution    │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset Scale Projections

| Metric | Current | Phase 1 | Phase 3 | Phase 5 |
|--------|---------|---------|---------|---------|
| **Video hours** | 200+ hrs | 500 hrs | 2,000 hrs | 10,000+ hrs |
| **Extracted frames** | ~1K | 500K | 2M | 10M+ |
| **Annotated frames** | ~50 | 10K | 100K | 1M+ |
| **Object classes** | 80 (COCO) | 50 game-specific | 200+ | 500+ |
| **Trained models** | 1 (generic) | 5 (per-game) | 20+ | 100+ (community) |
| **Contributors** | 1 | 5-10 | 100+ | 1,000+ |
| **Games covered** | 15 | 30 | 100+ | 500+ |

---

## Competitive Positioning

| Capability | Roboflow | MIRANDA SENSE |
|-----------|----------|---------------|
| Gaming datasets | ~10 datasets, mostly FPS aimbots | Purpose-built for gaming, all genres |
| Annotation tool | Generic (no game awareness) | Game-aware (HUD detection, scene classification) |
| Video processing | Frame-by-frame only | Temporal: scene change, object tracking, behavioral signals |
| Training pipeline | Cloud-only (Roboflow Train) | Self-contained: local + cloud + distributed |
| Model zoo | Community-contributed, unvetted | Curated per-game models with quality scores |
| Behavioral analysis | None | MIRANDA knowledge graph integration |
| Emotion correlation | None | FER pipeline (game-qa-analyzer) |
| Open source | Partially (inference server) | Fully open source |
| Offline capable | No (cloud-dependent) | Yes (local-first architecture) |
| Price | Per-API-call | Free (self-hosted) |

---

## Key Dependencies & Decisions

| Decision | Options | Recommendation | Rationale |
|----------|---------|----------------|-----------|
| Database at scale | PostgreSQL, CockroachDB, Supabase | PostgreSQL + Supabase Auth | Proven, Supabase gives auth for free |
| Object storage | S3, R2, MinIO | Cloudflare R2 | Zero egress, S3-compatible, already in your CF account |
| Job queue | BullMQ, RabbitMQ, SQS | BullMQ (Redis) | Node.js native, simple, battle-tested |
| Dataset versioning | DVC, lakeFS, HF Datasets | lakeFS + HF Hub | lakeFS for internal, HF for distribution |
| Experiment tracking | W&B, MLflow, Trackio | W&B + MLflow | W&B for viz, MLflow for registry |
| YOLO version | YOLOv8, YOLO11, YOLO26 | YOLO26 | NMS-free, STAL for small targets |
| Video tracking | ByteTrack, BoT-SORT, DeepSORT | ByteTrack | Built into Ultralytics, no appearance model needed |
| Annotation platform | Build own, CVAT, Label Studio | Own (already built) + CVAT export | We have the core, CVAT for power users |

---

## Next Steps (Immediate)

1. **Batch processing service** — process entire video library overnight
2. **Active learning scorer** — prioritize frames for annotation
3. **Game-specific class taxonomy** — replace flat COCO list
4. **Gold set creation** — annotate 200 expert frames across 5 games
5. **Docker containerization** — package for team deployment
6. **CI/CD pipeline** — GitHub Actions for test + build + deploy
