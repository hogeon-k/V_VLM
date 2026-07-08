from repository.db_manager import DBManager


def test_db_manager_uses_configured_path(tmp_path) -> None:
    db_path = tmp_path / "inspection.sqlite3"
    manager = DBManager(db_path)

    with manager.get_connection() as connection:
        enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert db_path.exists()
    assert enabled == 1
