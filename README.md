# OcuTrace

Webcam-based saccade & antisaccade analysis system for neurological screening, focused on Parkinson's Disease monitoring.

OcuTrace provides an affordable alternative to commercial eye-tracking devices (Tobii, EyeLink) using standard webcams. It implements the gap antisaccade paradigm — a well-established clinical protocol for detecting oculomotor dysfunction associated with neurodegenerative diseases.

## What It Does

OcuTrace uses your laptop's webcam to track iris movements in real time and measure how quickly and accurately a person can control their eye movements. The system runs a standardized test consisting of 60 trials:

- **Antisaccade trials (40):** A dot appears on one side of the screen. The patient must look at the **opposite** side. This requires inhibitory control — the ability to suppress an automatic eye movement toward the stimulus.
- **Prosaccade trials (20):** A dot appears on one side. The patient looks **toward** it. This measures basic saccade speed.

A colored fixation dot at the center tells the patient what to do:
- **Red dot** = antisaccade (look opposite)
- **Green dot** = prosaccade (look toward)

## What It Measures

| Metric | What It Means | Normal Range |
|--------|---------------|--------------|
| **Antisaccade error rate** | % of trials where the patient looked toward the stimulus instead of away | < 20% in healthy adults |
| **Saccade latency** | Time from stimulus onset to first eye movement (ms) | 150–400 ms |
| **Antisaccade latency** | Typically longer than prosaccade latency | 200–400 ms |
| **Prosaccade latency** | Baseline saccade reaction time | 150–250 ms |

Elevated antisaccade error rates and increased latencies are associated with frontal lobe dysfunction, which is common in Parkinson's Disease and other neurodegenerative conditions.

## How It Works

### Pipeline

```
Webcam (30fps) → MediaPipe Iris Tracking → Calibration (pixel→degree)
    → PsychoPy Stimulus Presentation (flip-based timing)
    → Saccade Detection (velocity-threshold algorithm)
    → Clinical Metrics → Report (Matplotlib) / Dashboard (Flask)
```

### Technical Details

1. **Iris Tracking:** MediaPipe FaceMesh with `FaceLandmarker` detects iris centers (landmarks 468/473) at ~30fps from a standard 640x480 webcam.

2. **Calibration:** A 9-point calibration maps pixel coordinates to degrees of visual angle using independent X/Y affine transforms. Calibration accounts for webcam parallax by grouping measurements per target level.

3. **Stimulus Presentation:** PsychoPy presents the gap antisaccade paradigm with frame-accurate timing:
   - Fixation (1000ms) → Gap/blank (200ms) → Peripheral stimulus at ±10° (1500ms) → ITI (1000–1500ms random)

4. **Saccade Detection:** Savitzky-Golay smoothing (window=3, poly=2) → velocity computation (deg/s) → onset threshold (20 deg/s for 1+ frames) → offset threshold (10 deg/s). Minimum 2° amplitude to reject noise.

5. **Data Storage:** SQLite database stores all sessions, trials, raw gaze data, and calibration results. MariaDB backend also available via config.

## Installation

### Requirements
- Python 3.10 (PsychoPy is not compatible with 3.12+)
- Webcam (built-in laptop camera works)
- Windows 10/11

### Setup

```bash
git clone https://github.com/barisozyurt/OcuTrace.git
cd OcuTrace

# Create virtual environment with Python 3.10
python3.10 -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe model (one-time)
mkdir models
curl -L -o models/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

## Usage

### Run a Complete Session

```bash
# First time — calibrate + test
python scripts/run_session.py --participant "Patient Name" --calibrate

# Subsequent sessions (uses existing calibration)
python scripts/run_session.py --participant "Patient Name"
```

During calibration, look at each dot as it appears (9 points). During the test:
- **Red center dot** → look **opposite** to the white target
- **Green center dot** → look **toward** the white target

### Generate Reports

```bash
# Analyze the most recent session
python scripts/analyze.py --latest

# Analyze a specific session
python scripts/analyze.py --session SESSION_ID

# List all sessions
python scripts/analyze.py --list
```

Reports are saved as PNG files in `data/reports/`.

### Web Dashboard

```bash
python scripts/dashboard.py
# Open http://127.0.0.1:5000 in browser
```

The dashboard shows all sessions with metrics, plots, and trial-by-trial details.

### Run Tests

```bash
pytest tests/                            # All tests
pytest tests/test_metrics.py             # Single file
pytest tests/test_metrics.py::test_name  # Single test
```

## Project Structure

```
OcuTrace/
├── config/
│   └── settings.yaml           # All configurable parameters
├── src/
│   ├── tracking/
│   │   ├── iris_tracker.py     # MediaPipe iris coordinate extraction
│   │   ├── calibration.py      # Pixel-to-degree transform
│   │   ├── calibration_display.py
│   │   └── glasses_detector.py # Glasses detection + quality gate
│   ├── experiment/
│   │   ├── paradigm.py         # Trial sequence generation
│   │   ├── stimulus.py         # PsychoPy stimulus presentation
│   │   └── session.py          # Session lifecycle management
│   ├── analysis/
│   │   ├── signal_processing.py # Savitzky-Golay smoothing + velocity
│   │   ├── saccade_detector.py  # Velocity-threshold onset detection
│   │   └── metrics.py          # Clinical metrics computation
│   ├── storage/
│   │   ├── models.py           # Data models (Session, Trial, GazeData)
│   │   ├── repository.py       # Abstract storage interface
│   │   ├── sqlite_repo.py      # SQLite implementation
│   │   ├── mariadb_repo.py     # MariaDB implementation
│   │   └── factory.py          # Storage backend factory
│   ├── visualization/
│   │   └── reports.py          # Matplotlib clinical reports
│   └── dashboard/
│       ├── app.py              # Flask application
│       ├── views.py            # Routes + plot generation
│       └── templates/          # HTML templates
├── scripts/
│   ├── run_session.py          # Main experiment runner
│   ├── calibrate.py            # Standalone calibration
│   ├── analyze.py              # Offline analysis + reports
│   └── dashboard.py            # Web dashboard launcher
├── tests/                      # 119 tests
├── models/                     # MediaPipe model (not in git)
├── data/                       # Database + reports (not in git)
└── requirements.txt
```

## Configuration

All parameters are in `config/settings.yaml`. Key settings:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `camera` | `device_index` | 0 | Webcam index |
| `paradigm` | `n_antisaccade_trials` | 40 | Number of antisaccade trials |
| `paradigm` | `n_prosaccade_trials` | 20 | Number of prosaccade trials |
| `paradigm` | `stimulus_eccentricity_deg` | 10.0 | Target distance from center (degrees) |
| `saccade_detection` | `onset_velocity_threshold` | 20.0 | Saccade onset threshold (deg/s) |
| `calibration` | `max_acceptable_error_deg` | 3.0 | Maximum calibration error |
| `display` | `screen_width_cm` | 53.0 | Physical screen width |
| `display` | `viewing_distance_cm` | 60.0 | Subject-to-screen distance |
| `storage` | `backend` | sqlite | Database backend (sqlite/mariadb) |

## Tech Stack

- **Iris Tracking:** MediaPipe FaceLandmarker
- **Camera:** OpenCV
- **Stimulus:** PsychoPy (flip-based timing)
- **Analysis:** NumPy, SciPy, Pandas
- **Visualization:** Matplotlib, Plotly
- **Storage:** SQLite (default), MariaDB (optional)
- **Dashboard:** Flask
- **Testing:** pytest (119 tests)

## License

MIT
