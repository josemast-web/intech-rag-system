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
    using the latest Gemini 1.5 Pro infrastructure.
    """

    def __init__(self):
        # Global Settings Configuration (2026 Standards)
        # Using Gemini 1.5 Pro for advanced technical reasoning
        Settings.llm = Gemini(
            model_name="models/gemini-1.5-flash-latest", 
            api_key=Config.GEMINI_API_KEY
        )
        # Using text-embedding-004 for high-precision retrieval
        Settings.embed_model = GeminiEmbedding(
            model_name="models/text-embedding-004", 
            api_key=Config.GEMINI_API_KEY
        )
        
        # Optimized for engineering documentation
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 100
        
        self.index = None

    def build_or_load_index(self, documents: list = None):
        """
        Initializes the vector index. Prioritizes local storage to minimize API latency.
        """
        storage_dir = Config.STORAGE_DIR

        if os.path.exists(storage_dir) and os.listdir(storage_dir):
            logger.info("Local vector index detected. Loading from storage.")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            if not documents:
                logger.error("Initialization failed: No documents provided and no index found.")
                return None
            
            logger.info("Initializing new vector index. This process may take a moment.")
            self.index = VectorStoreIndex.from_documents(documents)
            self.index.storage_context.persist(persist_dir=storage_dir)
            logger.info(f"Indexing complete. Knowledge base persisted to {storage_dir}.")
        
        return self.index

    def execute_query(self, user_query: str):
        """
        Executes semantic retrieval and generates a response grounded in Intech data.
        """
        if not self.index:
            logger.warning("Query ignored: Vector index is not initialized.")
            return "The knowledge base is currently offline. Please refresh the system."

        logger.info(f"Technical Inquiry Received: {user_query}")
        
        # Configure the engine to retrieve the top 5 most relevant fragments
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        
        try:
            response = query_engine.query(user_query)
            return response
        except Exception as e:
            logger.error(f"Execution Error during RAG process: {str(e)}")
            return "A technical error occurred during data retrieval. Please verify API connectivity."
