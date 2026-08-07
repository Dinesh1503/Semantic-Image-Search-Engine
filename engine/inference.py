import os

from models.huggingface_embedding_model import HuggingFaceEmbeddingModel
from models.huggingface_model import HuggingFaceModel

from paths import IMAGE_EXTENSIONS

from PIL import Image
from PIL.ExifTags import TAGS

from datetime import datetime
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
                if name.endswith(IMAGE_EXTENSIONS):
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
