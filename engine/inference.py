class Inference: 

    def __init__(self):
        pass

    
    def get_index(self,dir_path:str):
        
        import os

        image_info = []
        for root,dir,files in os.walk(dir_path, topdown=True):
            for name in files: 
                if name.endswith((".jpg",".png",".jpeg")):
                    image_path = root + "/" + name
                    image_info.append({"iamge_path" :image_path,"image_file_name":name})
    

