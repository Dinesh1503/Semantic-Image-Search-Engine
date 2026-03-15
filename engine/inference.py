import os
import numpy as np

from models.huggingface_embedding_model import HuggingFaceEmbeddingModel
from models.huggingface_model import HuggingFaceModel

from database.db import VectorDatabase

from PIL import Image
from PIL.ExifTags import TAGS

from datetime import datetime
from pathlib import Path
import json
from dotenv import load_dotenv
class Inference: 

    THRESHOLD_GOOD = 0.4
    THRESHOLD_BAD = 0.2

    def __init__(self):
        # self.caption_model = HuggingFaceModel()
        self.embedding_model = HuggingFaceEmbeddingModel()


        pass
    
    def get_index(self,dir_path:str):

        image_info = []
        for root,dir,files in os.walk(dir_path, topdown=True):
            for name in files: 
                if name.endswith((".jpg",".png",".jpeg",".JPG")):
                    image_path = root + "/" + name
                    image_info.append({"image_path" :image_path,"image_filename":name})
        
        return image_info 

    def get_image_captions(self,image_indexes:list)->list:

        captions = []
        for i in image_indexes:
            image_path = i["image_path"]
            response = self.caption_model.generate_captions(image_path)
            captions.append(response)
        return captions
    
    def vectorise_captions(self,caption:str):

        vector = self.embedding_model.get_caption_vectors(caption)

        return vector
    
    def vectorise_query(self,query:str):

        vector = self.embedding_model.get_query_vectors(query)

        return vector
    
    def get_image_datetime(self,image_path:str):
        stats = os.stat(image_path)
        try:
     
            return datetime.fromtimestamp(stats.st_birthtime).isoformat()
       
        except AttributeError:
  
            return datetime.fromtimestamp(stats.st_ctime).isoformat()
    
    def get_exif_metatdata(self,image_path:str):
    
        image = Image.open(image_path)
        exif_data = image.getexif()
        
        if not exif_data:
            print("\n No EXIF data found.")
            return None
        
        readable_exif = {}
        for tag_id, value in exif_data.items():

            tag_name = TAGS.get(tag_id, tag_id)
            readable_exif[tag_name] = value
        
        exif_ifd = exif_data.get_ifd(0x8769) 
    
        for tag_id, value in exif_ifd.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if "datetime" in tag_name or "DateTime" in tag_name or "DATETIME" in tag_name:
                dt = datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
                dt = dt.isoformat()
                readable_exif[tag_name] = dt
            else:
                readable_exif[tag_name] = value

        # print(f"\n\n Metadata: {readable_exif}")
            
        return readable_exif
    
    def classify_results(self, results: list):
        classifications = {"good": [], "bad": [], "irrelevant": []}
        
        for i, row in enumerate(results):
            name, path, similarity = row
            
            if similarity >= self.THRESHOLD_GOOD:
                classifications["good"].append((i,similarity,path,name))
            elif similarity > self.THRESHOLD_BAD:
                classifications["bad"].append((i,similarity,path,name))
            else:
                classifications["irrelevant"].append((i,similarity,path,name))
                
        for key in classifications:
            classifications[key].sort(key=lambda x: x[1], reverse=True)

        return classifications



    
    


# inf = Inference()
# path = "/Users/dinesh/Project/Semantic Image Search Engine/engine/test_data/a"
# indexes = inf.get_index(path)

# dates = []
# for i in indexes:
#     exif_metadata = inf.get_exif_metatdata(i["image_path"])
#     if exif_metadata == None:
#         dates.append(inf.get_image_datetime(i))

# print(dates)

# def init():
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

#     return inf

# def start():
#     inf = Inference()
#     db = init()
#     # Use a raw string or Path object for Windows/Mac compatibility
#     base_path = Path("/Users/dinesh/Project/Semantic Image Search Engine/engine/test_data/c")
    
#     # 1. Get Data
#     indexes = inf.get_index(str(base_path))
#     caption = inf.get_image_captions(indexes) # Assuming this returns a list of captions
    
#     # caption[0] is likely the specific caption for indexes[0]
#     caption_vector = inf.vectorise_captions(caption[0])
#     metadata = inf.get_exif_metatdata(indexes[0]["image_path"])

#     # 2. Build the Tuple
#     obj = (
#         indexes[0]["image_filename"],
#         indexes[0]["image_path"],
#         caption[0], 
#         caption_vector.tolist(),
#         metadata
#     )

#     # 3. SAVE (Using standard open)
#     # indent=4 makes the file look professional if you open it in a text editor
#     with open("data.json", "w") as f:
#         json.dump(obj, f, indent=4)
    
#     # 4. READ
#     with open("data.json", "r") as f:
#         # JSON converts tuples to lists. We cast it back to a tuple here.
#         loaded_data = tuple(json.load(f))
    
#     print(f"Loaded {len(loaded_data)} items from JSON.")
#     print(f"Image Name: {loaded_data[0]}")

#     db.add_data(obj)
#     db.check_data()


def test():
    inf = Inference()
    # db = init()
    # query = "Waves crashing against dark, jagged rocks; ocean surface extending to horizon; sky above sea."
    # query_vector = inf.vectorise_query(query)
    # results = db.retrieve_data(query_vector)

    # classified_results = inf.classify_results(results)



    mock_results = [
    ('golden_retriever_1.jpg', '/images/dogs/golden_retriever_1.jpg', 0.8842),  # Should be GOOD
    ('yellow_lab.jpg', '/images/dogs/yellow_lab.jpg', 0.5210),                # Should be GOOD
    ('brown_cat.jpg', '/images/cats/brown_cat.jpg', 0.3155),                 # Should be BAD
    ('park_bench.jpg', '/images/scenery/park_bench.jpg', 0.1201),            # Should be IRRELEVANT
    ('night_sky.jpg', '/images/space/night_sky.jpg', -0.0542)                # Should be IRRELEVANT
]
    classified_results = inf.classify_results(mock_results)

    print(classified_results)

    

        # print(f"Image: {row[0]}, Similarity Score: {row[3]:.4f}")
    
if __name__ == "__main__":
    
    test()


