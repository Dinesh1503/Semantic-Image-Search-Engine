"""Unit tests for ``engine/database/db.py``."""

from unittest.mock import MagicMock

from database.db import VectorDatabase


def _connected_db():
    db = VectorDatabase(db="mydb", password="pw", username="MixedCase", port="5432")
    db.conn = MagicMock(name="conn")
    db.cur = MagicMock(name="cur")
    return db


def test_build_db_url_lowercases_username():
    db = VectorDatabase(db="mydb", password="pw", username="MixedCase", port="5432")
    assert db.DB_URL == "postgresql://mixedcase:pw@localhost:5432/mydb"
    # Connection handles start out unset.
    assert db.conn is None
    assert db.cur is None


def test_connect_opens_connection_and_cursor():
    import database.db as db_module

    db = VectorDatabase(db="mydb", password="pw", username="user", port="5432")
    db.connect()

    db_module.psycopg.connect.assert_called_with(db.DB_URL)
    assert db.conn is not None
    assert db.cur is not None


def test_create_table_executes_ddl_and_commits():
    db = _connected_db()

    db.create_table()

    db.cur.execute.assert_called_once()
    ddl = db.cur.execute.call_args[0][0]
    assert "CREATE TABLE IF NOT EXISTS images" in ddl
    db.conn.commit.assert_called_once()


def test_create_table_swallows_errors_without_raising():
    db = _connected_db()
    db.cur.execute.side_effect = RuntimeError("boom")

    # Should not raise -- the method logs and continues.
    db.create_table()

    db.conn.commit.assert_not_called()


def test_check_data_queries_count_and_rows():
    db = _connected_db()
    db.cur.fetchone.return_value = (2,)
    db.cur.fetchall.return_value = [("a.jpg", "p/a.jpg"), ("b.jpg", "p/b.jpg")]

    db.check_data()

    executed = [c.args[0] for c in db.cur.execute.call_args_list]
    assert any("COUNT(*)" in q for q in executed)
    assert any("SELECT name,path from images" in q for q in executed)


def test_test_populate_db_inserts_row_with_512_dim_embedding():
    db = _connected_db()

    db.test_populate_db()

    query, data = db.cur.execute.call_args[0]
    assert "INSERT INTO images" in query
    name, path, embedding = data
    assert name == "img1.jpg"
    assert path == "test_data/img1.jpg"
    assert len(embedding) == 512
    db.conn.commit.assert_called_once()


def test_test_populate_db_swallows_insert_errors():
    db = _connected_db()
    db.cur.execute.side_effect = RuntimeError("insert failed")

    db.test_populate_db()  # must not raise

    db.conn.commit.assert_not_called()
