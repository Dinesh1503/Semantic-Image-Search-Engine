import psycopg
from dotenv import load_dotenv
import os
import numpy as np
import json
class VectorDatabase():
    def __init__(self,db:str,password:str,user:str,port:int):
        self.db = db
        self.user = user
        self.port = port
        self.password = password
        self.conn = None
        self.cur = None

        self.DB_URL = "postgresql://" + self.user.lower() + ":" + self.password + "@localhost:" + self.port + "/" + self.db; 
       

    def connect(self):
        self.conn = psycopg.connect(self.DB_URL)
        self.cur = self.conn.cursor()
    
    def disconnect(self):
        self.cur.close()
        self.conn.close()
    
    def delete_all_data(self):
        self.cur.execute("DELETE FROM images")
        self.conn.commit()
    
    def create_table(self):
        try:
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.cur.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        id SERIAL PRIMARY KEY,
                        name text,
                        path text,
                        caption text,
                        embedding vector(1024),
                        metadata JSONB DEFAULT '{}'::jsonb
                    );
                """)
            self.conn.commit()
            print("Table verified/created.")
        
        except Exception as e:
            self.conn.rollback()
            print(f"Database Creation Error: {e}")
        
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
        
        query = """INSERT INTO images (name,path,caption,embedding,metadata) 
                    VALUES (%s,%s,%s,%s,%s)"""
        
        embedding = np.random.rand(1024).tolist()
        caption = "img1"
        metadata = json.dumps({"a":"b"})
        data = ('img1.jpg','test_data/img1.jpg',caption,embedding,metadata)
        
        try:
            self.cur.execute(query,data)
            self.conn.commit()

        except Exception as e:
            print("Error Entrying Data",e)
    

    # def data_not_null_check(data:tuple)->bool:
    #     (name,path,caption,embedding,metadata) = data

    #     if name is None or name is "":


    def add_data(self,data:list)->str:
        # name, path, caption, embedding, metadata = data
        
        # # 2. Convert the metadata dictionary into a JSON string
        # if isinstance(metadata, dict):
        #     metadata = json.dumps(metadata)
            
        # 3. Repack the variables into a new tuple ready for the database
        # formatted_data = (name, path, caption, embedding, metadata)

        query = """INSERT INTO images (name,path,caption,embedding,metadata) 
                    VALUES (%s,%s,%s,%s,%s)"""

        try: 
            self.cur.executemany(query,data)
            self.conn.commit()
            print("Data Added Sucessfully")
        
        except Exception as e:
            print(f"Error adding data to database {e}")

    def retrieve_data(self,query_embedding:list,limit:int=5):

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        query = """SELECT name,path,(1 - (embedding <=> %s::vector)) AS SIMILARITY 
        FROM images 
        ORDER BY embedding <=> %s::vector ASC 
        LIMIT %s"""

        try: 
            self.cur.execute(query,(query_embedding,query_embedding,limit))
            results = self.cur.fetchall()
            
            print(f"Found {len(results)} matches!")
            return results

        except Exception as e:
            self.conn.rollback() 
            print(f"Error during search: {e}")
            return []



# if __name__ == "__main__":
#     load_dotenv()

#     db = os.getenv("DB")
#     user = os.getenv("DB_USER")
#     port = os.getenv("PORT")
#     password = os.getenv("PASSWORD")

#     inf = VectorDatabase(db,password,user,port)
#     inf.connect()
#     inf.create_table()
#     inf.test_populate_db()
#     inf.check_data()


        


