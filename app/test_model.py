from dotenv import load_dotenv
from google import genai
from config import GEMINI_API_KEY, MODEL_GEMINI_FLASH

load_dotenv()


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=GEMINI_API_KEY)

query = "Explain briefly what is Diabetes"
response = client.models.generate_content(
    model=MODEL_GEMINI_FLASH,
    contents=query
)
print(f"\nQuestion: {query}")
print(f"GenAI: {response.text}\n")
