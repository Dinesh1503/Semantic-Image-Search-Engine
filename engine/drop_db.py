from database.db import VectorDatabase

if __name__ == "__main__":
    db = VectorDatabase.from_env()
    db.connect()
    db.drop_table()
    db.disconnect()
