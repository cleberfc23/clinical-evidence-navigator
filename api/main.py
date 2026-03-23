from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.rag_service import answer_question
from app.generator import generate_answer, build_client
# from core.settings import embedding_model, client, model_gemini_flash, RETRIEVAL_TOP_K

from core.settings import (
    DEFAULT_DOC,
    get_secrets,
    validate_runtime_config,
    DEBUG_MODE,
    MAX_REQUESTS,
    RETRIEVAL_TOP_K,
)

run_time_config_dict = get_secrets()
model_gemini_flash = run_time_config_dict["model_name"]
gemini_api_key = run_time_config_dict["api_key"]
embedding_model = run_time_config_dict["embedding_model"]
client = build_client(gemini_api_key)


app = FastAPI(title="Clinical Evidence Navigator API")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = answer_question(
            user_question=request.question,
            embedding_model=embedding_model,
            client=client,
            model_name=model_gemini_flash,
            top_k=RETRIEVAL_TOP_K,
        )
        return AskResponse(answer=result["answer_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
