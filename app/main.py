from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.menu_agent import fetch_data, generate_embedding, generate_prompt

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