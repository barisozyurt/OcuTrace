"""Launch the OcuTrace web dashboard.

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --port 8080
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.app import create_app


def main() -> None:
    """Parse arguments and start the Flask development server."""
    parser = argparse.ArgumentParser(description="OcuTrace Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app = create_app()
    print(f"Dashboard: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
