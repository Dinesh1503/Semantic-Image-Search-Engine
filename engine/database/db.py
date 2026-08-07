import psycopg
from dotenv import load_dotenv
import os
import numpy as np
import json
class VectorDatabase():

    INSERT_QUERY = """INSERT INTO images (name,path,caption,embedding,metadata) 
                    VALUES (%s,%s,%s,%s,%s)"""

    def __init__(self,db:str,password:str,user:str,port:int):
        self.db = db
        self.user = user
        self.port = port
        self.password = password
        self.conn = None
        self.cur = None

        self.DB_URL = "postgresql://" + self.user.lower() + ":" + self.password + "@localhost:" + self.port + "/" + self.db;

    @classmethod
    def from_env(cls) -> "VectorDatabase":
        """Build a database from the DB/PASSWORD/DB_USER/PORT environment variables."""
        load_dotenv()
        return cls(
            os.getenv("DB"),
            os.getenv("PASSWORD"),
            os.getenv("DB_USER"),
            os.getenv("PORT"),
        )

    def connect(self):
        self.conn = psycopg.connect(self.DB_URL)
        self.cur = self.conn.cursor()
    
    def disconnect(self):
        self.cur.close()
        self.conn.close()

    def _execute(self,query:str,params=None,many:bool=False,error_message:str="Database Error",raise_on_error:bool=False):
        """Run a statement, commit it and roll back with a message when it fails.

        Returns True on success, False when the statement raised, unless
        `raise_on_error` is set and the caller cannot continue without it.
        """
        try:
            if many:
                self.cur.executemany(query,params)
            else:
                self.cur.execute(query,params)
            self.conn.commit()
            return True

        except Exception as e:
            self.conn.rollback()
            print(f"{error_message}: {e}")
            if raise_on_error:
                raise
            return False

    def _fetch(self,query:str,params=None,error_message:str="Database Error")->list:
        """Run a query and return its rows, rolling back and returning [] on failure."""
        try:
            self.cur.execute(query,params)
            return self.cur.fetchall()

        except Exception as e:
            self.conn.rollback()
            print(f"{error_message}: {e}")
            return []

    def delete_all_data(self):
        self._execute("DELETE FROM images",error_message="Error deleting data",raise_on_error=True)
    
    def create_table(self):
        self._execute("CREATE EXTENSION IF NOT EXISTS vector;",error_message="Database Creation Error")
        created = self._execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        id SERIAL PRIMARY KEY,
                        name text,
                        path text,
                        caption text,
                        embedding vector(1024),
                        metadata JSONB DEFAULT '{}'::jsonb
                    );
                """,error_message="Database Creation Error")
        if created:
            print("Table verified/created.")

    def drop_table(self):
        self._execute("DROP TABLE IF EXISTS images;",error_message="Error dropping table",raise_on_error=True)

    def check_data(self):
        count = self._fetch("SELECT COUNT(*) FROM images;",error_message="Error counting data")
        print(count[0][0] if count else 0)

        print(self._fetch("SELECT name,path from images;",error_message="Error reading data"))

    def test_populate_db(self):
        
        embedding = np.random.rand(1024).tolist()
        caption = "img1"
        metadata = json.dumps({"a":"b"})
        data = ('img1.jpg','test_data/img1.jpg',caption,embedding,metadata)

        self._execute(self.INSERT_QUERY,data,error_message="Error Entrying Data")

    def add_data(self,data:list)->str:

        if self._execute(self.INSERT_QUERY,data,many=True,error_message="Error adding data to database"):
            print("Data Added Sucessfully")

    def retrieve_data(self,query_embedding:list,limit:int=5):

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        query = """SELECT name,path,(1 - (embedding <=> %s::vector)) AS SIMILARITY 
        FROM images 
        ORDER BY embedding <=> %s::vector ASC 
        LIMIT %s"""

        results = self._fetch(query,(query_embedding,query_embedding,limit),error_message="Error during search")

        print(f"Found {len(results)} matches!")
        return results
