from sentence_transformers import SentenceTransformer
from models.abstract_model_class import EmbeddingModel
from paths import resolve_model_source
class HuggingFaceEmbeddingModel(EmbeddingModel):

    def __init__(self):

        self.model_name = "Qwen/Qwen3-Embedding-0.6B"

        source, local_only, _ = resolve_model_source(self.model_name, "qwen3-embedding-0.6b")

        self.model = SentenceTransformer(source, local_files_only=local_only, device="mps")

        self.user_query_prompt = "query"
    
    def get_caption_vectors(self,caption:list[str])->list:

        vector = self.model.encode(caption,normalize_embeddings=True,batch_size=32)

        return vector.tolist()

    def get_query_vectors(self,user_query:str)->list:

        vector = self.model.encode(user_query,prompt = self.user_query_prompt,normalize_embeddings=True)

        return vector
    
    def compute_similarity(self,a,b):
        x = self.model.similarity(a,b)
        print("\n\n Similarity: ",x)
