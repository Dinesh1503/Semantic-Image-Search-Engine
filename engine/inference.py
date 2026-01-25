import os
import numpy as np

from models.huggingface_embedding_model import HuggingFaceEmbeddingModel
from models.huggingface_model import HuggingFaceModel

class Inference: 

    def __init__(self):
        self.caption_model = HuggingFaceModel()
        self.embedding_model = HuggingFaceEmbeddingModel()
    
    def get_index(self,dir_path:str):

        image_info = []
        for root,dir,files in os.walk(dir_path, topdown=True):
            for name in files: 
                if name.endswith((".jpg",".png",".jpeg")):
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
    

inf = Inference()
index = inf.get_index("/Users/dinesh/Project/Semantic Image Search Engine/engine/test_data/a/b")

print("\n index: ",index)

captions = inf.get_image_captions(index)

print("\n captions:",captions)
caption_vector = inf.vectorise_captions(captions[0])

query = "a starry night in the forest"
query_vector = inf.vectorise_query(query)

print(len(caption_vector),len(query_vector))

inf.embedding_model.compute_similarity(caption_vector,query_vector)

