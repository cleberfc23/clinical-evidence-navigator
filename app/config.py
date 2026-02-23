import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH_DIABETES = os.getenv("PDF_PATH_DIABETES")
CHROMA_DIR = os.getenv("CHORMA_DIR", "data/processed/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "clinical_guidelines")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_GEMINI_FLASH = os.getenv("MODEL_GEMINI_FLASH")

