# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for OcuTrace.

Build with: pyinstaller ocutrace.spec
Output: dist/OcuTrace/OcuTrace.exe
"""

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config/settings.yaml', 'config'),
        ('models/face_landmarker.task', 'models'),
        # MediaPipe native library
        ('.venv/lib/site-packages/mediapipe/tasks/c', 'mediapipe/tasks/c'),
    ],
    hiddenimports=[
        # wxPython
        'wx',
        'wx._core',
        'wx._adv',
        # PsychoPy core
        'psychopy',
        'psychopy.visual',
        'psychopy.core',
        'psychopy.event',
        'psychopy.monitors',
        'psychopy.visual.textbox2',
        # MediaPipe
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.c',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
        'mediapipe.tasks.python.vision.face_landmarker',
        # OpenCV
        'cv2',
        # Scientific
        'scipy.signal',
        'scipy.ndimage',
        'numpy',
        'pandas',
        # Visualization
        'matplotlib',
        'matplotlib.backends.backend_agg',
        # OcuTrace modules
        'src',
        'src.config',
        'src.paths',
        'src.orchestrator',
        'src.tracking',
        'src.tracking.iris_tracker',
        'src.tracking.calibration',
        'src.tracking.calibration_display',
        'src.tracking.glasses_detector',
        'src.experiment',
        'src.experiment.paradigm',
        'src.experiment.stimulus',
        'src.experiment.session',
        'src.analysis',
        'src.analysis.signal_processing',
        'src.analysis.saccade_detector',
        'src.analysis.metrics',
        'src.storage',
        'src.storage.models',
        'src.storage.repository',
        'src.storage.sqlite_repo',
        'src.visualization',
        'src.visualization.reports',
        'src.gui',
        'src.gui.launcher',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused PsychoPy components to reduce bundle size
        'psychopy.sound',
        'psychopy.iohub',
        'psychopy.hardware',
        # Exclude test frameworks
        'pytest',
        'pytest_cov',
        # Exclude Flask dashboard (not needed in exe)
        'flask',
        'jinja2',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OcuTrace',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window — GUI only
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OcuTrace',
)
