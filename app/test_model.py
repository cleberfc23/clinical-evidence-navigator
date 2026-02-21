import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=gemini_api_key)

query = "Explain briefly what is Diabetes"
response = client.models.generate_content(
    model=os.getenv("MODEL_GEMINI_FLASH"),
    contents=query
)
print(f"\nQuestion: {query}")
print(f"GenAI: {response.text}\n")
