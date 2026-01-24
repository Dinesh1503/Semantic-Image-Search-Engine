from abc import ABC,abstractmethod

class VlmModel(ABC):
    @abstractmethod
    def generate(self):
        pass 

class MlxModel(VlmModel):

    def __init__(self):
        
        from mlx_vlm import load,generate

        self.model_name = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
        self.model, self.processor = load(self.model_name)
        self.generate_captions_func = generate

        PROMPT_TEXT = """Analyze this image for an image retrieval database.
                        Do not use poetic, emotional, or subjective language.
                        Instead, provide a structured, factual breakdown of the visual elements.

                        Please describe the image using these four categories:
                        1. MAIN SUBJECTS: List the specific physical objects visible.
                        2. SPATIAL LAYOUT: Where are these objects located?
                        3. ATTRIBUTES: Describe colors, textures, and states.
                        4. ENVIRONMENT: Describe the lighting, weather, and time of day.

                        Start the description directly with the main subject. Output as a single concise paragraph."""
        
        self.system_prompt = (
                                "<|im_start|>user\n"
                                "<|vision_start|><|image_pad|><|vision_end|>" 
                                f"{PROMPT_TEXT}\n"
                                "<|im_end|>\n"
                                "<|im_start|>assistant\n"
                            )
                            
    def generate(self,image_path:str):
        
        response = self.generate_captions_func(self.model,
                                               self.processor,
                                               image=[image_path],
                                               prompt = self.system_prompt,
                                               temp=0,
                                               max_tokens=600,
                                               verbose=True) 
        return response


