from sentence_transformers import SentenceTransformer
from models.abstract_model_class import EmbeddingModel

class HuggingFaceEmbeddingModel(EmbeddingModel):

    def __init__(self):

        self.model_name = "Qwen/Qwen3-Embedding-0.6B"

        self.model = SentenceTransformer(self.model_name)
        
        self.user_query_prompt = "query"
    
    def get_caption_vectors(self,caption:str)->list:

        vector = self.model.encode(caption,normalize_embeddings=True)

        return vector

    def get_query_vectors(self,user_query:str)->list:

        vector = self.model.encode(user_query,prompt = self.user_query_prompt,normalize_embeddings=True)

        return vector
    
    def compute_similarity(self,a,b):
        x = self.model.similarity(a,b)
        print("\n\n Similarity: ",x)


# model = HuggingFaceEmbeddingModel()
# caption = "MAIN SUBJECTS: Silhouettes of coniferous trees, roofline of a building with chimney structure, distant mountain ridgeline under starry sky. SPATIAL LAYOUT: Trees occupy left-center foreground; roofline extends from right midground to lower-right corner; mountains appear faintly behind tree line on horizon; stars fill entire upper portion of frame. ATTRIBUTES: Trees and buildings rendered in solid black silhouette against dark purple-to-blue gradient night sky speckled with white pinprick stars; no discernible texture due to low light; static scene without motion blur. ENVIRONMENT: Nighttime setting indicated by absence of sunlight; clear atmospheric conditions allowing visibility of numerous stars; ambient illumination suggests minimal artificial light pollution near location; likely rural or semi-rural area based on natural landscape features."
# query = "starry night in the forest"

# vector = model.get_caption_vectors(caption)
# print(type(vector))
# print(len(vector))
# print(vector)

# query = "starry night in the forest, with trees in the right side"

# query_vector = model.get_query_vectors(query)
# print(type(query_vector))
# print(len(query_vector))
# print(query_vector)

# model.compute_similarity(vector,query_vector)
                