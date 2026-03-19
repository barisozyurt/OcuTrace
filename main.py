"""OcuTrace main entry point.

Launches the GUI application. This is the single entry point for both
development (python main.py) and the PyInstaller .exe bundle.
"""
import multiprocessing
import sys


def main() -> None:
    """Launch OcuTrace."""
    # Required for PyInstaller on Windows — must be called before
    # any other multiprocessing usage.
    multiprocessing.freeze_support()

    # Add project root to path for development mode
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from src.gui.launcher import run_app
    run_app()


if __name__ == "__main__":
    main()
