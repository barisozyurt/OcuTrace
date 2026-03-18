"""Tests for MariaDB repository and storage factory.

Since we cannot connect to a live MariaDB server in unit tests,
these tests verify imports and factory routing only.
"""
from __future__ import annotations

import pytest

from src.storage.factory import create_repository
from src.storage.repository import Repository
from src.storage.sqlite_repo import SQLiteRepository


class TestMariaDBImport:
    """Verify the MariaDB repository module can be imported."""

    def test_import_module(self) -> None:
        """MariaDB repo module should import without error."""
        from src.storage import mariadb_repo  # noqa: F401

    def test_import_class(self) -> None:
        """MariaDBRepository class should be importable."""
        from src.storage.mariadb_repo import MariaDBRepository

        assert issubclass(MariaDBRepository, Repository)


class TestStorageFactory:
    """Verify the storage factory creates the right backends."""

    def test_factory_creates_sqlite(self, tmp_path: object) -> None:
        """Factory should return SQLiteRepository for sqlite backend."""
        config = {
            "storage": {
                "backend": "sqlite",
                "sqlite": {
                    "database_path": str(tmp_path / "test.db"),  # type: ignore[operator]
                },
            }
        }
        repo = create_repository(config)
        assert isinstance(repo, SQLiteRepository)

    def test_factory_creates_mariadb(self) -> None:
        """Factory should return MariaDBRepository for mariadb backend."""
        from src.storage.mariadb_repo import MariaDBRepository

        config = {
            "storage": {
                "backend": "mariadb",
                "mariadb": {
                    "host": "localhost",
                    "port": 3306,
                    "user": "test",
                    "password": "test",
                    "database": "test",
                },
            }
        }
        repo = create_repository(config)
        assert isinstance(repo, MariaDBRepository)

    def test_factory_raises_for_unknown_backend(self) -> None:
        """Factory should raise ValueError for unrecognized backend."""
        config = {
            "storage": {
                "backend": "postgres",
            }
        }
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_repository(config)
