import os
from dotenv import load_dotenv

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
