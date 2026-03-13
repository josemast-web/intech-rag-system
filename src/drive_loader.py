import os
import json
import logging
from typing import List, Optional

import streamlit as st
from google.oauth2 import service_account
from llama_index.readers.google import GoogleDriveReader
from src.config import Config, logger

class IntechDriveLoader:
    """
    Manages document extraction from Google Drive.
    Auth priority: st.secrets (Streamlit Cloud) > credentials.json (local).
    """

    def __init__(self):
        self.folder_id = Config.GOOGLE_DRIVE_FOLDER_ID
        self.excluded_files = Config.EXCLUDED_FILES
        self.loader: Optional[GoogleDriveReader] = None

    def _build_credentials(self):
        """Builds ServiceAccountCredentials from st.secrets or local file."""
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]

        # -- Streamlit Cloud: read from st.secrets --
        if "gcp_service_account" in st.secrets:
            logger.info("Loading credentials from st.secrets.")
            creds_dict = {k: v for k, v in st.secrets["gcp_service_account"].items()}
            # Unescape newlines in private_key (TOML flattens them)
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            return service_account.Credentials.from_service_account_info(
                creds_dict, scopes=scopes
            )

        # -- Local fallback: credentials.json --
        if os.path.exists("credentials.json"):
            logger.info("Loading credentials from credentials.json.")
            with open("credentials.json") as f:
                creds_dict = json.load(f)
            return service_account.Credentials.from_service_account_info(
                creds_dict, scopes=scopes
            )

        raise FileNotFoundError(
            "No credentials found. Configure [gcp_service_account] in "
            "Streamlit secrets or place credentials.json in project root."
        )

    def initialize_loader(self):
        """Initializes GoogleDriveReader with resolved service account credentials."""
        try:
            creds = self._build_credentials()
            # Pass credentials object directly — no file path needed
            self.loader = GoogleDriveReader(credentials=creds)
            logger.info("Google Drive Reader initialized successfully.")
        except Exception as e:
            logger.error("Drive Reader initialization failed: %s", str(e))
            raise

    def load_documents(self) -> List:
        """
        Loads and filters documents from the configured Drive folder.
        Returns list of LlamaIndex Document objects.
        """
        if not self.loader:
            self.initialize_loader()

        logger.info("Starting ingestion from folder ID: %s", self.folder_id)

        try:
            documents = self.loader.load_data(
                folder_id=self.folder_id,
                recursive=True
            )

            # Filter operational exports — check all possible metadata key names
            filtered = [
                doc for doc in documents
                if not any(
                    excl in doc.metadata.get("file_name", "")
                    or excl in doc.metadata.get("fileName", "")
                    or excl in doc.metadata.get("name", "")
                    for excl in self.excluded_files
                )
            ]

            logger.info("Ingestion complete. %d documents loaded.", len(filtered))
            return filtered

        except Exception as e:
            logger.error("Document ingestion failed: %s", str(e))
            return []
