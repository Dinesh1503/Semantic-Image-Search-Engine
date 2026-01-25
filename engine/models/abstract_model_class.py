from abc import abstractmethod,ABC

class VlmModel(ABC):
    @abstractmethod
    def generate_captions(self):
        pass 

class EmbeddingModel(ABC):
    @abstractmethod
    def get_caption_vectors(self):
        pass
    
    @abstractmethod
    def get_query_vectors(self):
        pass