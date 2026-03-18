# OcuTrace Phased Development Design

**Date:** 2026-03-18
**Approach:** Vertical Slice + Risk Spike (Approach A with spike from C)

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hardware | Dahili webcam (30fps) ile başla, harici webcam (60fps) desteği ekle | Erişilebilirlik — çoğu klinikte standart laptop var |
| Stimulus | PsychoPy (flip-based timing) | Klinik doğruluk için altın standart, `time.sleep()` kabul edilemez |
| Storage | SQLite önce, MariaDB sonra (repository pattern) | Hızlı başlangıç + ölçeklenebilirlik |
| Dashboard | Sonraki fazlarda (Faz 7) | Core pipeline öncelikli |
| Testing | Unit + sentetik integration + klinik validation | Klinik güvenilirlik için zorunlu |
| Glasses | Kalite kapısı yaklaşımı — tespit et, kaliteyi ölç, bilgilendir | Zorunlu çıkarma yerine veri kalitesi odaklı |

## Project Structure

```
OcuTrace/
├── config/
│   └── settings.yaml
├── src/
│   ├── __init__.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── iris_tracker.py
│   │   ├── calibration.py
│   │   └── glasses_detector.py
│   ├── experiment/
│   │   ├── __init__.py
│   │   ├── stimulus.py
│   │   ├── paradigm.py
│   │   └── session.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── signal_processing.py
│   │   ├── saccade_detector.py
│   │   └── metrics.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── repository.py
│   │   ├── sqlite_repo.py
│   │   └── models.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── reports.py
│   └── dashboard/
│       └── __init__.py
├── scripts/
│   ├── run_session.py
│   ├── calibrate.py
│   └── analyze.py
├── tests/
│   ├── conftest.py
│   ├── test_tracking.py
│   ├── test_saccade_detector.py
│   ├── test_metrics.py
│   ├── test_paradigm.py
│   └── test_storage.py
├── data/
├── docs/plans/
├── requirements.txt
└── README.md
```

## Phase Plan

### Phase 1 — Project Skeleton + Iris Tracking Spike + Glasses Detection

**Goal:** Set up project infrastructure, validate MediaPipe iris tracking on internal webcam, implement glasses detection with quality gate.

**Deliverables:**
- Directory structure, `requirements.txt`, `config/settings.yaml`
- `iris_tracker.py` — real-time iris coordinate extraction via MediaPipe FaceMesh
- `glasses_detector.py` — glasses presence detection + tracking quality scoring
- Spike: measure landmark stability at 30fps (jitter, drop rate)
- `storage/` — repository pattern + SQLite for raw gaze data
- `models.py` — Session, Trial, GazeData dataclasses
- `conftest.py` — synthetic gaze data fixtures
- Unit tests: tracker with mock, storage with real SQLite

**Success criteria:**
- Landmark 468/473 detected in >= 95% of frames
- X-coordinate jitter < 2px during fixation
- Glasses detection accuracy validated manually
- Quality gate warns when tracking degrades

### Phase 2 — Calibration System

**Goal:** Convert pixel coordinates to visual angle (degrees).

**Deliverables:**
- `calibration.py` — 5 or 9-point calibration routine (PsychoPy targets + iris recording)
- Pixel-to-degree transformation (affine or polynomial fit)
- Calibration accuracy test
- Calibration data persisted in SQLite per session

**Success criteria:** Mean calibration error < 2 degrees.

### Phase 3 — Stimulus Presentation (Gap Antisaccade Paradigm)

**Goal:** Frame-accurate stimulus presentation per CLAUDE.md protocol.

**Deliverables:**
- `stimulus.py` — PsychoPy Window, fixation cross, peripheral target
- `paradigm.py` — Gap paradigm: Fixation(1000ms) → Gap(200ms) → Target(1500ms) → ITI(1000-1500ms random)
- Trial sequencing: 40 antisaccade + 20 prosaccade, pseudo-randomized, left/right balanced
- Flip-based stimulus onset timestamp logging

**Success criteria:** Stimulus onset timing jitter < 1 frame (16.67ms @ 60Hz).

### Phase 4 — Saccade Detection + Latency

**Goal:** Implement velocity-threshold saccade detection algorithm from CLAUDE.md.

**Deliverables:**
- `signal_processing.py` — Savitzky-Golay smoothing (window=5, poly=2), velocity computation
- `saccade_detector.py` — Onset (>30 deg/s, 3+ frames), offset (<20 deg/s), direction classification
- Latency = saccade onset − stimulus onset
- Antisaccade/prosaccade classification (direction vs stimulus side)
- Synthetic saccade integration tests
- Literature validation: prosaccade latency in ~150-250ms range

**Success criteria:** Onset detection error < 1 frame on synthetic data, latency accuracy < 5ms.

### Phase 5 — Session Runner (End-to-End)

**Goal:** Single-command calibration → experiment → recording pipeline.

**Deliverables:**
- `session.py` — UUID-based session creation, flow management
- `run_session.py` — CLI entry point
- Concurrent camera capture + stimulus presentation (threading/multiprocessing)
- Raw iris data + computed metrics saved together in SQLite
- Post-session summary statistics

**Success criteria:** Full 60-trial session completes without interruption, all data persisted.

### Phase 6 — Clinical Metrics + Reporting

**Goal:** Generate clinical reports from recorded data.

**Deliverables:**
- `metrics.py` — Antisaccade error rate, mean/median latency, corrective saccade latency
- `reports.py` — Matplotlib charts: trial-by-trial latency, saccade traces, error distribution
- `analyze.py` — Offline re-analysis script
- PDF/PNG output
- Validation against known clinical datasets

**Success criteria:** Generated metrics consistent with published normative ranges.

### Phase 7 — Flask Dashboard + MariaDB

**Goal:** Web-based visualization and multi-station support.

**Deliverables:**
- `mariadb_repo.py` — same repository interface as SQLite
- Flask dashboard: session list, detail page, chart rendering
- Cross-session comparison (same patient over time)
- DB backend selection via `settings.yaml`

**Success criteria:** Sessions viewable via dashboard, SQLite↔MariaDB switch via config only.
