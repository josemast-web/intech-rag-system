import os
import json
import logging
from typing import List, Optional

import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from llama_index.readers.google import GoogleDriveReader
from src.config import Config, logger

class IntechDriveLoader:
    """
    Manages document extraction from Google Drive.
    Authenticates via st.secrets (Streamlit Cloud) or credentials.json (local).
    """

    def __init__(self):
        self.folder_id = Config.GOOGLE_DRIVE_FOLDER_ID
        self.excluded_files = Config.EXCLUDED_FILES
        self.loader: Optional[GoogleDriveReader] = None

    def _get_credentials(self):
        """
        Builds ServiceAccountCredentials from st.secrets or local JSON file.
        Priority: st.secrets > credentials.json
        """
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]

        # --- Streamlit Cloud path: read from st.secrets ---
        if "gcp_service_account" in st.secrets:
            logger.info("Loading credentials from st.secrets.")
            creds_dict = dict(st.secrets["gcp_service_account"])
            # private_key newlines must be unescaped when coming from TOML
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            return service_account.Credentials.from_service_account_info(
                creds_dict, scopes=scopes
            )

        # --- Local fallback: read from credentials.json ---
        local_path = "credentials.json"
        if os.path.exists(local_path):
            logger.info("Loading credentials from credentials.json.")
            with open(local_path) as f:
                creds_dict = json.load(f)
            return service_account.Credentials.from_service_account_info(
                creds_dict, scopes=scopes
            )

        raise FileNotFoundError(
            "No credentials found. Add [gcp_service_account] to st.secrets "
            "or place credentials.json in the project root."
        )

    def initialize_loader(self):
        """Initializes GoogleDriveReader using resolved credentials."""
        try:
            creds = self._get_credentials()
            # GoogleDriveReader accepts a credentials object directly
            self.loader = GoogleDriveReader(credentials=creds)
            logger.info("Google Drive Reader initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Drive Reader: %s", str(e))
            raise

    def load_documents(self) -> List:
        """
        Fetches and filters documents from the configured Drive folder.
        Returns list of documents ready for indexing.
        """
        if not self.loader:
            self.initialize_loader()

        logger.info("Starting ingestion from folder ID: %s", self.folder_id)

        try:
            documents = self.loader.load_data(
                folder_id=self.folder_id,
                recursive=True
            )

            # Exclude operational files (e.g. Data_Monday exports)
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
            logger.error("Error during document ingestion: %s", str(e))
            return []
