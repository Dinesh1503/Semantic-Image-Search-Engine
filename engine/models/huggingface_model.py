from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import re
from pathlib import Path
from models.abstract_model_class import VlmModel
class HuggingFaceModel(VlmModel):

    def __init__(self):

        local_path = Path("/Users/dinesh/Project/Semantic Image Search Engine/engine/models/qwen3-vl-4b")

        self.model_id = "Qwen/Qwen3-VL-4B-Instruct"
        # self.processor = AutoProcessor.from_pretrained(self.model_id)
        # self.model = AutoModelForImageTextToText.from_pretrained(
        #     self.model_id,
        #     device_map="auto",       
        #     dtype=torch.float16
        # )

        if local_path.exists():
            self.processor = AutoProcessor.from_pretrained(
                local_path,
                local_files_only=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                local_path,
                local_files_only=True,
                device_map="mps",
                dtype=torch.float16
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                cache_dir=local_path,
                device_map="mps",
                dtype=torch.float16
            )

        self.PROMPT_TEXT = """Analyze this image for an image retrieval database.
                            Do not use poetic, emotional, or subjective language.
                            Instead, provide a structured, factual breakdown of the visual elements.

                            Please describe the image using these four categories:
                            1. MAIN SUBJECTS: List the specific physical objects visible.
                            2. SPATIAL LAYOUT: Where are these objects located?
                            3. ATTRIBUTES: Describe colors, textures, and states.
                            4. ENVIRONMENT: Describe the lighting, weather, and time of day.

                            Start the description directly with the main subject. Output as a single concise paragraph.
                            """
        
    def generate_captions(self, image_path: str):
        image = Image.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.PROMPT_TEXT}
                ]
            },
        ]

        with torch.no_grad():  # stops gradient tensors being stored
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                temperature=0.0
            )

        caption = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        caption = re.sub(r"<\|.*?\|>", "", caption).strip()

        # explicitly free memory after each image
        del inputs, outputs
        torch.mps.empty_cache()
        print("\n Caption: ",caption)
        return caption
    
    # def generate_captions(self,image_path:str):

    #     image = Image.open(image_path)
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": image},
    #                 {"type": "text", "text": self.PROMPT_TEXT}
    #             ]
    #         },
    #     ]
    #     inputs = self.processor.apply_chat_template(
    #         messages,
    #         add_generation_prompt=True,
    #         tokenize=True,
    #         return_dict=True,
    #         return_tensors="pt",
    #     ).to(self.model.device)

    #     outputs = self.model.generate(**inputs, max_new_tokens=600,
    #                                     do_sample=False,          # deterministic output reduces randomness
    #                                     eos_token_id=self.processor.tokenizer.eos_token_id,
    #                                     repetition_penalty=1.2,   # discourages repeating phrases
    #                                     temperature=0.0           )
        
    #     caption = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
    #     caption = re.sub(r"<\|.*?\|>", "", caption).strip()

    #     print("\n Captions: ",caption)
    #     torch.mps.empty_cache()
    #     return caption


# model = HuggingFaceTransformersModel()
# path = "/Users/dinesh/Project/Semantic Image Search Engine/engine/test_data/img_3.jpg"
# model.generate_captions(path)