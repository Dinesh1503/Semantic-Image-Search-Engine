import psycopg
from dotenv import load_dotenv
import os
import numpy as np

class VectorDatabase():
    def __init__(self,db:str,password:str,username:str,port:int):
        self.db = db
        self.username = username
        self.port = port
        self.password = password
        self.conn = None
        self.cur = None

        self.DB_URL = "postgresql://" + self.username.lower() + ":" + self.password + "@localhost:" + self.port + "/" + self.db; 
       

    def connect(self):
        self.conn = psycopg.connect(self.DB_URL)
        self.cur = self.conn.cursor()
    
    def create_table(self):
        try:

            self.cur.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        id SERIAL PRIMARY KEY,
                        name text,
                        path text,
                        embedding vector(512) 
                    );
                """)
            self.conn.commit()
        
        except Exception as e:
            print(f"Database Creation Error: {e}")

    def check_data(self):
        self.cur.execute("SELECT COUNT(*) FROM images;")
        count = self.cur.fetchone()[0]
        print(count)

        self.cur.execute("SELECT name,path from images;")
        rows = self.cur.fetchall()
        print(rows) 

    def test_populate_db(self):
        
        query = """INSERT INTO images (name,path,embedding) 
                    VALUES (%s,%s,%s)"""
        
        embedding = np.random.rand(512).tolist()
        data = ('img1.jpg','test_data/img1.jpg',embedding)
        
        try:
            self.cur.execute(query,data)
            self.conn.commit()

        except Exception as e:
            print("Error Entrying Data",e)


load_dotenv()

db = os.getenv("DB")
username = os.getenv("USERNAME")
port = os.getenv("PORT")
password = os.getenv("PASSWORD")

inf = VectorDatabase(db,password,username,port)
inf.connect()
inf.create_table()
inf.test_populate_db()
inf.check_data()


        


