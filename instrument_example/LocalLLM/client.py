import httpx
import json
from typing import List, Dict, Any, Optional, Tuple
import io
import base64
from PIL import Image
import numpy as np
import websockets
import asyncio
from urllib.parse import urlparse, urlunparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_http_to_ws(url: str) -> str:
    """Convert an HTTP/HTTPS URL to a WS/WSS URL for WebSocket use."""
    parsed = urlparse(url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((ws_scheme, parsed.netloc, parsed.path, "", "", ""))



def image_to_base64(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(img, str):
        img = Image.open(img)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return "data:image/png;base64," + img_base64



class ChatClient:
    """A client for interacting with the FastAPI + Gradio chat application."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        """Initialize the chat client.

        Args:
            base_url: The base URL of the chat API.
        """
        self.base_url = base_url
        self.session_id = None
        self.client = httpx.Client(timeout=30.0)

        self.listening_websocket = False


    async def listen_for_messages(self, callback: callable=None):
        """Listen for new messages from the server in real-time."""
        self.listening_websocket = True
        print("Connection start.")
        websocket = await websockets.connect(convert_http_to_ws(f"{self.base_url}/ws/chat/{self.session_id}"))
        while self.listening_websocket:
            try:
                response = await websocket.recv()
                print("agent response:", response)
                if callback:
                    callback(response)
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed.")
                break

        self.listening_websocket = False
        await websocket.close()
        print("Session closed.")

    def create_session(self, session_id=None) -> str:
        """Create a new chat session.

        Returns:
            The session ID.
        """
        params = {}
        if session_id:
            params["session_id"] = session_id

        response = self.client.post(f"{self.base_url}/create_session", params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to create session: {response.text}")

        data = response.json()
        self.session_id = data["session_id"]
        logger.info(f"Created new session: {self.session_id}")
        return self.session_id

    def use_session(self, session_id: str) -> None:
        """Use an existing session.

        Args:
            session_id: The session ID to use.
        """
        # Verify the session exists by getting its history
        response = self.client.get(f"{self.base_url}/chat_history/{session_id}")
        if response.status_code != 200:
            raise Exception(f"Session {session_id} not found")

        self.session_id = session_id
        logger.info(f"Using existing session: {self.session_id}")

    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the chat.

        Args:
            message: The message to send.

        Returns:
            The response from the server.
        """
        if not self.session_id:
            self.create_session()

        payload = {
            "session_id": self.session_id,
            "message": message
        }

        response = self.client.post(f"{self.base_url}/chat", json=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to send message: {response.text}")
        return response.json()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the chat history for the current session.

        Returns:
            The chat history.
        """
        if not self.session_id:
            raise Exception("No active session")

        response = self.client.get(f"{self.base_url}/chat_history/{self.session_id}")
        if response.status_code != 200:
            raise Exception(f"Failed to get history: {response.text}")

        return response.json()["history"]


    def get_active_sessions(self) -> List[str]:
        response = self.client.get(f"{self.base_url}/api/active_sessions")
        if response.status_code != 200:
            raise Exception(f"Failed to get history: {response.text}")

        return response.json()["sessions"]

    def print_history(self) -> None:
        """Print the chat history in a readable format."""
        history = self.get_history()

        print("\n=== Chat History ===")
        for msg in history:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                print(f"\nYou: {content}")
            else:
                print(f"Assistant: {content}")
        print("\n==================")

    def add_conversation(self, conversation_string: str) -> Dict[str, Any]:
        """Add a conversation to the session.

        Args:
            conversation_string: A string containing the conversation in the format
                                "user: message1\nassistant: response1\nuser: message2\nassistant: response2"

        Returns:
            The response from the server.
        """
        if not self.session_id:
            self.create_session()

        payload = {
            "session_id": self.session_id,
            "conversation": conversation_string
        }

        response = self.client.post(f"{self.base_url}/add_conversation", json=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to add conversation: {response.text}")

        return response.json()


if __name__ == '__main__':
    client = ChatClient(base_url="http://133.1.195.184:7860")  # Connects to localhost:7860 by default

    sessions = client.get_active_sessions()
    print(sessions)
    if len(sessions) > 0:
        # session_id = sessions[-1]
        client.use_session("3649eaed-5462-4d4d-b4f8-c082852e5117")


        """
        add custom message or image to server
        """
#         bob = """
# ░░░░░░▄▄▄░░▄██▄░░░
# ░░░░░▐▀█▀▌░░░░▀█▄░░░
# ░░░░░▐█▄█▌░░░░░░▀█▄░░
# ░░░░░░▀▄▀░░░▄▄▄▄▄▀▀░░
# ░░░░▄▄▄██▀▀▀▀░░░░░░░
# ░░░█▀▄▄▄█░▀▀░░
# ░░░▌░▄▄▄▐▌▀▀▀░░ THIS IS BOB
# ▄░▐░░░▄▄░█░▀▀ ░░
# ▀█▌░░░▄░▀█▀░▀ ░░ COPY AND PASTE HIM,
# ░░░░░░░▄▄▐▌▄▄░░░ SO, HE CAN TAKE
# ░░░░░░░▀███▀█░▄░░ OVER your Lab
# ░░░░░░▐▌▀▄▀▄▀▐▄░░
# ░░░░░░▐▀░░░░░░▐▌░░
# ░░░░░░█░░░░░░░░█░░░
#         """
#         client.add_conversation(bob)
        # client.add_conversation(f'test image is: <img src="{image_to_base64("51DbphCzbgL.png")}">')



        """
        chat
        """
        # print(client.send_message("start the scan")["response"])


        """
        get chat history
        """
        # client.print_history()



        """
        Event subscribe
        """
        def on_new_message(msg):
            # print("New messages received:", msg)
            history = client.get_history()

            assistant_msg_list = []
            for msg in history:
                role = msg["role"]
                content = msg["content"]
                if role == "assistant":
                    assistant_msg_list.append(content)
            print(assistant_msg_list[-1])
            return assistant_msg_list[-1]


        loop = asyncio.get_event_loop()
        loop.run_until_complete(client.listen_for_messages(on_new_message))

