from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import re
from models.abstract_model_class import VlmModel
from models.device import torch_device
from paths import resolve_model_source
from models.prompts import CAPTION_PROMPT
class HuggingFaceModel(VlmModel):

    def __init__(self):

        self.model_id = "Qwen/Qwen3-VL-4B-Instruct"

        source, local_only, cache_dir = resolve_model_source(self.model_id, "qwen3-vl-4b")
        device = torch_device()

        self.processor = AutoProcessor.from_pretrained(source, local_files_only=local_only)
        self.model = AutoModelForImageTextToText.from_pretrained(
            source,
            local_files_only=local_only,
            cache_dir=None if local_only else cache_dir,
            device_map=device,
            dtype=torch.float32 if device == "cpu" else torch.float16
        )

        self.PROMPT_TEXT = CAPTION_PROMPT
        
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
        if self.model.device.type == "mps":
            torch.mps.empty_cache()
        elif self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        print("\n Caption: ",caption)
        return caption
