import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

PDF_PATH_DIABETES = os.getenv("PDF_PATH_DIABETES")
CHROMA_DIR = os.getenv("CHORMA_DIR", "data/processed/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "clinical_guidelines")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_GEMINI_FLASH = os.getenv("MODEL_GEMINI_FLASH")
DEFAULT_DOC = {
    "doc_id": "standards-of-care-2026.pdf",
    "url": "https://ada.silverchair-cdn.com/ada/content_public/journal/care/issue/49/supplement_1/6/standards-of-care-2026.pdf",
    "label": "Diabetes Guideline"
}

MAX_FILE_SIZE_MB = 20


def get_secrets():
    load_dotenv()
    model_name = MODEL_GEMINI_FLASH or st.secrets.get("MODEL_GEMINI_FLASH")
    key = GEMINI_API_KEY or st.secrets.get("GEMINI_API_KEY")
    embedding = EMBEDDING_MODEL or st.secrets.get("EMBEDDING_MODEL")

    return {
        "model_name": model_name,
        "api_key": key,
        "embedding_model": embedding
    }


def validate_runtime_config(config: dict):
    missing_fields = []

    if not config.get("model_name"):
        missing_fields.append("MODEL_GEMINI_FLASH")
    if not config.get("api_key"):
        missing_fields.append("GEMINI_API_KEY")
    if not config.get("embedding_model"):
        missing_fields.append("EMBEDDING_MODEL")

    return missing_fields
