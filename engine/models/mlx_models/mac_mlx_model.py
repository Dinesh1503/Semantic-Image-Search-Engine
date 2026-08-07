from mlx_vlm import load,generate

from models.abstract_model_class import VlmModel
from models.prompts import qwen_vl_chat_prompt

class MlxModel(VlmModel):

    def __init__(self):

        self.model_name = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
        self.model, self.processor = load(self.model_name)
        self.generate_captions_func = generate

        self.system_prompt = qwen_vl_chat_prompt()

    def generate_captions(self,image_path:str):
        
        response = self.generate_captions_func(self.model,
                                               self.processor,
                                               image=[image_path],
                                               prompt = self.system_prompt,
                                               temp=0,
                                               max_tokens=600,
                                               verbose=True) 
        return response


