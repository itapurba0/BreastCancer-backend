from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel
import uvicorn

from contextlib import asynccontextmanager
from chatbot.engine import generate_rag_response
from auth.routes import router as auth_router
from auth.deps import get_current_user
from database import sessions_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Chatbot API starting up...")
    yield


origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CLIENT_URL = os.getenv("CLIENT_URL")
if CLIENT_URL:
    origins.append(f"https://{CLIENT_URL}")

app = FastAPI(title="Chatbot API", lifespan=lifespan)
app.include_router(auth_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "service": "chatbot-api"}


class MessageItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[MessageItem]


class SaveChatRequest(BaseModel):
    messages: list


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        return StreamingResponse(
            generate_rag_response(request.messages),
            media_type="text/plain",
        )
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/chat/history")
def get_chat_history(email: str = Depends(get_current_user)):
    session = sessions_collection.find_one({"email": email})
    if not session or not session.get("messages"):
        return {"messages": []}
    return {"messages": session["messages"]}


@app.post("/chat/save")
def save_chat_history(body: SaveChatRequest, email: str = Depends(get_current_user)):
    sessions_collection.update_one(
        {"email": email},
        {"$set": {"messages": body.messages, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return {"status": "ok"}


if __name__ == "__main__":
    print("Starting Chatbot API...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, loop="asyncio", reload=True)
