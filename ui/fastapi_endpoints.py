from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import asyncio

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connected clients for broadcasting messages
active_connections = {}


# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class SessionResponse(BaseModel):
    session_id: str
    share_code: Optional[str] = None


@app.websocket("/ws/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Add the WebSocket connection to the active_connections dict
    if session_id not in active_connections:
        active_connections[session_id] = []
    active_connections[session_id].append(websocket)

    try:
        while True:
            # Listening for incoming messages, if needed (optional)
            data = await websocket.receive_text()
            # You can handle incoming data if necessary
    except WebSocketDisconnect:
        # Remove client from the active_connections when disconnected
        active_connections[session_id].remove(websocket)
        if not active_connections[session_id]:
            del active_connections[session_id]


async def notify_clients(session_id: str, message: str, timeout: int = 5):
    """Notify all active WebSocket clients of a new message."""
    if session_id not in active_connections:
        return

    to_remove = []

    for websocket in active_connections[session_id]:
        try:
            # Send the message with timeout
            await asyncio.wait_for(
                websocket.send_text(f"{message}"),
                timeout=timeout
            )
        except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError) as e:
            logger.info(f"Removing dead websocket for session {session_id}: {e}")
            to_remove.append(websocket)

    # Clean up any closed or broken connections
    for ws in to_remove:
        try:
            active_connections[session_id].remove(ws)
        except ValueError:
            pass

    # Clean up session if no active connections remain
    if not active_connections[session_id]:
        del active_connections[session_id]