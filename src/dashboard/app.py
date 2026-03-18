"""Flask dashboard for OcuTrace session browsing and reporting."""
from __future__ import annotations

from typing import Any, Optional

from flask import Flask

from src.dashboard.views import register_routes


def create_app(config: Optional[dict[str, Any]] = None) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    config : dict or None
        Optional configuration overrides for Flask.

    Returns
    -------
    Flask
        Configured Flask application instance.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    if config:
        app.config.update(config)
    register_routes(app)
    return app
