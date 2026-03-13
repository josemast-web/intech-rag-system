import os
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from src.config import Config, logger

class IntechVectorService:
    """
    Service class to manage vector indexing, persistence, and query execution
    using Google Gemini models.
    """

    def __init__(self):
        # Configure Global Settings for LlamaIndex
        Settings.llm = Gemini(model_name="models/gemini-pro", api_key=Config.GEMINI_API_KEY)
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=Config.GEMINI_API_KEY)
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 100
        
        self.index = None

    def build_or_load_index(self, documents: list = None):
        """
        Initializes the vector index. Loads from local storage if available; 
        otherwise, generates a new index from provided documents.
        """
        storage_dir = Config.STORAGE_DIR

        if os.path.exists(storage_dir) and os.listdir(storage_dir):
            logger.info("Loading existing vector index from local storage.")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            if not documents:
                logger.error("No index found and no documents provided for initialization.")
                return None
            
            logger.info("Creating new vector index from provided documents.")
            self.index = VectorStoreIndex.from_documents(documents)
            self.index.storage_context.persist(persist_dir=storage_dir)
            logger.info(f"Index successfully persisted to {storage_dir}.")
        
        return self.index

    def execute_query(self, user_query: str):
        """
        Performs a semantic search and generates a response based on retrieved context.
        """
        if not self.index:
            logger.warning("Query execution attempted on uninitialized index.")
            return "System not ready. Please index documents first."

        logger.info(f"Processing query: {user_query}")
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        
        try:
            response = query_engine.query(user_query)
            return response
        except Exception as e:
            logger.error(f"Error during query execution: {str(e)}")
            return "An error occurred while retrieving technical information."

if __name__ == "__main__":
    # Integration Test
    from src.drive_loader import IntechDriveLoader
    
    loader = IntechDriveLoader()
    # Note: This test requires a valid credentials.json and API Key
    try:
        docs = loader.load_documents()
        service = IntechVectorService()
        service.build_or_load_index(docs)
        print("Success: Vector Service is operational.")
    except Exception as e:
        print(f"Test Failed: {e}")
