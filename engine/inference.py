import os
from models.models import VlmModel,MlxModel 
from models.embedding_model import EmbeddingModel
import numpy as np
class Inference: 

    def __init__(self):
        pass
    
    def get_index(self,dir_path:str):

        image_info = []
        for root,dir,files in os.walk(dir_path, topdown=True):
            for name in files: 
                if name.endswith((".jpg",".png",".jpeg")):
                    image_path = root + "/" + name
                    image_info.append({"image_path" :image_path,"image_filename":name})
        
        return image_info 

    def get_image_captions(self,model:VlmModel,image_indexes:list)->list:

        # import gc
        # import mlx.core as mx

        captions = []
        for i in image_indexes:
            image_path = i["image_path"]
            response = model.generate(image_path)
            captions.append(response.text)
            break
            
        print(captions)

        # del model
        # gc.collect()
        # mx.metal.clear_cache()

        return captions
    
    def vectorise_captions(self,embedding_model:EmbeddingModel,caption:str):

        vector = embedding_model.get_vector_embeddings(caption)

        return vector


# inf = Inference()
# model = MlxModel()

# print(image_info,type(image_info))
# inf.get_image_captions(model,image_info)

vector_model = EmbeddingModel()
# caption_model = MlxModel()

inf = Inference()

# image_info = inf.get_index(path)

# captions = inf.get_image_captions(caption_model,image_info)

captions = ['Main Subjects: Grass blades in the foreground, trees in the background, sun positioned centrally behind the trees. Spatial Layout: Grass occupies the lower portion of the frame; trees are in the mid-ground, silhouetted against the sky; the sun is behind the trees. Attributes: Grass is green with yellowish highlights from sunlight; trees are dark silhouettes; sun emits bright, warm light with lens flare and bokeh. Environment: Bright, direct sunlight; clear sky with minimal cloud cover; likely late afternoon or early morning based on sun angle and warm tones.']

vectors = []

for i in captions:
    vectors.append(inf.vectorise_captions(vector_model,i))
    break
print(vectors)
