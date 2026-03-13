import os
import logging
from typing import List, Optional
from llama_index.readers.google import GoogleDriveReader
from src.config import Config, logger

class IntechDriveLoader:
    """
    Service class to manage data extraction from Google Drive repository.
    Handles authentication, filtering, and document loading for the RAG pipeline.
    """

    def __init__(self):
        self.folder_id = Config.GOOGLE_DRIVE_FOLDER_ID
        self.excluded_files = Config.EXCLUDED_FILES
        self.loader: Optional[GoogleDriveReader] = None

    def initialize_loader(self, credentials_path: str = "credentials.json"):
        """
        Initializes the GoogleDriveReader with specified credentials.
        
        Args:
            credentials_path (str): Path to the Google Cloud service account JSON file.
        """
        if not os.path.exists(credentials_path):
            logger.error(f"Credentials file not found at {credentials_path}")
            raise FileNotFoundError("Authentication credentials are required.")

        try:
            # The loader will use the service account to access shared folders
            self.loader = GoogleDriveReader(service_account_key_path=credentials_path)
            logger.info("Google Drive Reader initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Drive Reader: {str(e)}")
            raise

    def load_documents(self) -> List:
        """
        Fetches and filters documents from the specified Google Drive folder.
        
        Returns:
            List: A list of processed documents ready for indexing.
        """
        if not self.loader:
            self.initialize_loader()

        logger.info(f"Starting data ingestion from Folder ID: {self.folder_id}")

        try:
            # Recursive load to capture sub-folders (Projects, Clients, etc.)
            documents = self.loader.load_data(
                folder_id=self.folder_id,
                recursive=True
            )

            # Filtering logic to exclude operational files (e.g., Data_Monday)
            filtered_documents = [
                doc for doc in documents 
                if not any(excluded in doc.metadata.get('file_name', '') 
                          for excluded in self.excluded_files)
            ]

            logger.info(f"Ingestion complete. {len(filtered_documents)} documents loaded.")
            return filtered_documents

        except Exception as e:
            logger.error(f"Error during document ingestion: {str(e)}")
            return []

if __name__ == "__main__":
    # Unit test for the loader module
    loader_service = IntechDriveLoader()
    try:
        data = loader_service.load_documents()
        print(f"Status: Success - Documents retrieved: {len(data)}")
    except Exception as err:
        print(f"Status: Execution Failed - {err}")
