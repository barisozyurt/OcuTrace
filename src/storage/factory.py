"""Factory for creating storage repository instances."""
from __future__ import annotations

from src.storage.repository import Repository


def create_repository(config: dict) -> Repository:
    """Create a repository based on the config backend setting.

    Parameters
    ----------
    config : dict
        Application configuration dictionary. Must contain a ``storage``
        section with a ``backend`` key set to ``"sqlite"`` or ``"mariadb"``.

    Returns
    -------
    Repository
        Concrete repository instance.

    Raises
    ------
    ValueError
        If the backend is not recognized.
    """
    backend = config["storage"]["backend"]
    if backend == "sqlite":
        from src.storage.sqlite_repo import SQLiteRepository

        return SQLiteRepository(config["storage"]["sqlite"]["database_path"])
    elif backend == "mariadb":
        from src.storage.mariadb_repo import MariaDBRepository

        cfg = config["storage"]["mariadb"]
        return MariaDBRepository(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
        )
    raise ValueError(f"Unknown storage backend: {backend}")
