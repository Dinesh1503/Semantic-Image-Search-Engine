import logging
import os

from models.embedding_model import EmbeddingModel
from models.models import VlmModel

logger = logging.getLogger(__name__)


class Inference: 

    def __init__(self):
        pass
    
    def get_index(self,dir_path:str):

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Image directory does not exist: {dir_path}")

        image_info = []
        for root,dir,files in os.walk(dir_path, topdown=True):
            for name in files: 
                if name.endswith((".jpg",".png",".jpeg")):
                    image_path = root + "/" + name
                    image_info.append({"image_path" :image_path,"image_filename":name})

        if not image_info:
            logger.warning("No images found under %s", dir_path)

        return image_info 

    def get_image_captions(self,model:VlmModel,image_indexes:list)->list:

        captions = []
        failures = []
        for i in image_indexes:
            image_path = i["image_path"]
            try:
                response = model.generate(image_path)
            except Exception:
                logger.exception("Caption generation failed for %s", image_path)
                failures.append(image_path)
                continue
            captions.append(response.text)

        if failures and not captions:
            raise RuntimeError(
                f"Caption generation failed for all {len(failures)} image(s)"
            )

        return captions
    
    def vectorise_captions(self,embedding_model:EmbeddingModel,caption:str):

        return embedding_model.get_vector_embeddings(caption)


def main():
    logging.basicConfig(level=logging.INFO)

    vector_model = EmbeddingModel()
    inf = Inference()

    captions = ['Main Subjects: Grass blades in the foreground, trees in the background, sun positioned centrally behind the trees. Spatial Layout: Grass occupies the lower portion of the frame; trees are in the mid-ground, silhouetted against the sky; the sun is behind the trees. Attributes: Grass is green with yellowish highlights from sunlight; trees are dark silhouettes; sun emits bright, warm light with lens flare and bokeh. Environment: Bright, direct sunlight; clear sky with minimal cloud cover; likely late afternoon or early morning based on sun angle and warm tones.']

    vectors = [inf.vectorise_captions(vector_model, caption) for caption in captions]
    logger.info("Generated %d vectors", len(vectors))


if __name__ == "__main__":
    main()
