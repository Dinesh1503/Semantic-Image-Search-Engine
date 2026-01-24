import os
from models.models import VlmModel,MlxModel
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

    def get_image_captions(self,model:MlxModel,image_indexes:list)->list:

        captions = []
        for i in image_indexes:
            image_path = i["image_path"]
            response = model.generate(image_path)
            captions.append(response.text)
            break

        print(captions)
        return captions


# inf = Inference()
# model = MlxModel()

# print(image_info,type(image_info))
# inf.get_image_captions(model,image_info)
