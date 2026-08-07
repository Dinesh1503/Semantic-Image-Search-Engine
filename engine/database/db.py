import logging
import os

import numpy as np
import psycopg
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class VectorDatabase():
    def __init__(self,db:str,password:str,username:str,port:int):
        missing = [
            name
            for name, value in (
                ("DB", db),
                ("PASSWORD", password),
                ("USERNAME", username),
                ("PORT", port),
            )
            if value in (None, "")
        ]
        if missing:
            raise ValueError(
                "Missing database configuration: " + ", ".join(missing)
            )

        self.db = db
        self.username = username
        self.port = str(port)
        self.password = password
        self.conn = None
        self.cur = None

        self.DB_URL = "postgresql://" + self.username.lower() + ":" + self.password + "@localhost:" + self.port + "/" + self.db

    def connect(self):
        try:
            self.conn = psycopg.connect(self.DB_URL)
        except psycopg.Error as e:
            raise ConnectionError(
                f"Could not connect to database '{self.db}' on port {self.port}"
            ) from e
        self.cur = self.conn.cursor()

    def _require_cursor(self):
        if self.cur is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self.cur

    def close(self):
        if self.cur is not None:
            self.cur.close()
            self.cur = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def create_table(self):
        cur = self._require_cursor()
        try:
            cur.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        id SERIAL PRIMARY KEY,
                        name text,
                        path text,
                        embedding vector(512) 
                    );
                """)
            self.conn.commit()

        except psycopg.Error as e:
            self.conn.rollback()
            raise RuntimeError("Failed to create the 'images' table") from e

    def check_data(self):
        cur = self._require_cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM images;")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("COUNT query returned no rows")
            count = row[0]

            cur.execute("SELECT name,path from images;")
            rows = cur.fetchall()
        except psycopg.Error as e:
            self.conn.rollback()
            raise RuntimeError("Failed to read rows from the 'images' table") from e

        logger.info("images table contains %d rows", count)
        return count, rows

    def test_populate_db(self):

        query = """INSERT INTO images (name,path,embedding) 
                    VALUES (%s,%s,%s)"""

        embedding = np.random.rand(512).tolist()
        data = ('img1.jpg','test_data/img1.jpg',embedding)

        cur = self._require_cursor()
        try:
            cur.execute(query,data)
            self.conn.commit()

        except psycopg.Error as e:
            self.conn.rollback()
            raise RuntimeError("Failed to insert row into the 'images' table") from e


def main():
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    with VectorDatabase(
        os.getenv("DB"),
        os.getenv("PASSWORD"),
        os.getenv("USERNAME"),
        os.getenv("PORT"),
    ) as database:
        database.create_table()
        database.test_populate_db()
        database.check_data()


if __name__ == "__main__":
    main()
