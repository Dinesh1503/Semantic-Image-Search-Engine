import numpy as np
class EmbeddingModel():
    
    def __init__(self):

        try:
            import mlx.core as mx
            from mlx_embeddings.utils import load
        except ImportError as e:
            raise ImportError(
                "mlx and mlx_embeddings are required for EmbeddingModel and are only "
                "available on Apple Silicon"
            ) from e

        self.model_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
        self.mx = mx
        try:
            self.model,self.tokenizer = load(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}'"
            ) from e

    def get_vector_embeddings(self,text:str)->list:

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        tokens = self.mx.array(self.tokenizer.encode(text))
        tokens = tokens[None, :] 

        output = self.model(tokens)

        hidden_states = output.last_hidden_state

        embedding = hidden_states[:, -1, :]

        norm = self.mx.linalg.norm(embedding,axis=-1,keepdims=True)
        normalized_vector = embedding / self.mx.maximum(norm, 1e-12)

        float_vector = normalized_vector.astype(self.mx.float32)
        flattened_vector = np.array(float_vector[0]).tolist()

        return flattened_vector


# model = EmbeddingModel()
# caption = "MAIN SUBJECTS: A vintage red car. SPATIAL LAYOUT: Centered. ENVIRONMENT: Sunset."
# vector = model.get_vector_embeddings(caption)

# print(f"Vector Dimensions: {vector.shape[1]}") # 1024 for 0.6B model
# print(f"First 5 values: {vector[0, :5]}")

