import os
import json
from typing import List, Optional

import streamlit as st
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

    def _build_credentials_dict(self) -> dict:
        """Returns raw service account dict from st.secrets or local file."""

        # -- Streamlit Cloud: read from st.secrets --
        if "gcp_service_account" in st.secrets:
            logger.info("Loading credentials from st.secrets.")
            creds_dict = {k: v for k, v in st.secrets["gcp_service_account"].items()}
            # Unescape newlines in private_key (TOML encoding flattens them)
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            return creds_dict

        # -- Local fallback: credentials.json --
        if os.path.exists("credentials.json"):
            logger.info("Loading credentials from credentials.json.")
            with open("credentials.json") as f:
                return json.load(f)

        raise FileNotFoundError(
            "No credentials found. Configure [gcp_service_account] in "
            "Streamlit secrets or place credentials.json in project root."
        )

    def initialize_loader(self):
        """Initializes GoogleDriveReader using service_account_key dict."""
        try:
            creds_dict = self._build_credentials_dict()
            # GoogleDriveReader requires service_account_key as raw dict
            self.loader = GoogleDriveReader(service_account_key=creds_dict)
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

            # Exclude operational exports — check all possible metadata key names
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
