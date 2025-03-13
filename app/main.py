from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.hyteria import fetch_data as fetch_hyteria_data
from app.services.dining_code_fetcher import fetch_data as fetch_dining_code_data
from app.services.llm_handler import generate_prompt, generate_embeddings
import asyncio

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
async def refresh_menu():
    # Create tasks for parallel execution
    
    # Create coroutines for data fetching
    hyteria_task = asyncio.create_task(fetch_hyteria_data())
    dining_code_task = asyncio.create_task(fetch_dining_code_data())
    
    # Wait for both tasks to complete
    await asyncio.gather(hyteria_task, dining_code_task)

    generate_embeddings()

    return {"status": "OK"}

@app.post("/prompt")
def prompt(messages: str, user_id: str):
    response = generate_prompt(messages, user_id)

    return {"messages": response}