from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.hyteria import fetch_data as fetch_hyteria_data
from app.services.dining_code_fetcher import fetch_data as fetch_dining_code_data
from app.services.llm_handler import generate_prompt, generate_embeddings
from fastapi.responses import StreamingResponse
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health-check")
def health_check():
    return {"status": "good"}

@app.post("/refresh-menu")
def refresh_menu():
    fetch_dining_code_data()

    generate_embeddings()

    return {"status": "OK"}

@app.get("/prompt")
async def prompt(messages: str, user_id: str):
    # response = generate_prompt(messages, user_id)

    return  StreamingResponse(generate_prompt(messages, user_id), media_type="text/event-stream")