import os
import logging
from dotenv import load_dotenv

# Technical logging configuration (Direct and Formal)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntechRAG")

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management for Intech RAG System."""
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    EXCLUDED_FILES = ["Data_Monday"]
    STORAGE_DIR = "./storage"
    
    @classmethod
    def validate_config(cls):
        """Validates the presence of required environmental variables."""
        if not cls.GEMINI_API_KEY:
            logger.error("Missing GEMINI_API_KEY. System initialization aborted.")
            return False
        if not cls.GOOGLE_DRIVE_FOLDER_ID:
            logger.error("Missing GOOGLE_DRIVE_FOLDER_ID. Drive integration disabled.")
            return False
        
        logger.info("System configuration validated successfully.")
        return True
