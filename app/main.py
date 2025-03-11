from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.data_fetcher import fetch_data
from app.services.llm_handler import generate_prompt, generate_embedding

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
    fetch_data()
    generate_embedding()

    return {"status": "OK"}

@app.post("/prompt")
def prompt(messages: str):
    response = generate_prompt(messages)

    return {"message": response}