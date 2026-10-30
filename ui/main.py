from ui.LLMChatBot import LLMChatBot

llm = LLMChatBot(max_seq_length=4000)


"""
Must import unsloth model firstly.
"""
from ui.fastapi_endpoints import *

import gradio as gr
import json
import os
import httpx
import uuid
import re

import logging
def cmd_to_md(text):
    cmd_pattern = re.compile(r"<cmd>\s*(.*?)\s*</cmd>", re.DOTALL)
    cmds = cmd_pattern.findall(text)
    non_cmd_text = cmd_pattern.sub("", text).strip()
    parts = []
    if non_cmd_text:
        parts.append(non_cmd_text)

    if cmds:
        merged_cmd = "\n".join(cmds)
        parts.append(f"```cmd\n{merged_cmd}\n```")

    return "\n".join(parts)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---- GLOBAL STORAGE ----
# Store chat history in memory (you could replace this with a database in production)
chat_histories = {}


def save_history(session_id: str):
    history_file = os.path.join("resources/history", f"{session_id}.json")
    if session_id in chat_histories:
        with open(history_file, "w") as f:
            json.dump(chat_histories[session_id], f)
        logger.info(f"Chat history for session {session_id} saved to {history_file}")
    else:
        logger.warning(f"No history found for session {session_id} to save")

# Function to load chat history from a file
def load_history(session_id: str):
    history_file = os.path.join("resources/history", f"{session_id}.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            chat_histories[session_id] = json.load(f)
        logger.info(f"Chat history for session {session_id} loaded from {history_file}")
    else:
        logger.warning(f"No saved history found for session {session_id}")


def render_history_for_ui(history):
    rendered = []
    for msg in history:
        content = msg["content"]
        if msg["role"] == "assistant":
            content = cmd_to_md(content)
        rendered.append({
            "role": msg["role"],
            "content": content
        })
    return rendered



@app.middleware("http")
async def extract_session_id(request: Request, call_next):
    query_sid = request.query_params.get("session_id")
    if query_sid:
        request.app.state.session_id_from_url = query_sid
    return await call_next(request)




# Create a new session
@app.post("/create_session", response_model=SessionResponse)
async def create_session(session_id: str = Query(None)):
    # If session_id is not provided, generate a new one
    if not session_id:
        session_id = str(uuid.uuid4())

    # Ensure the session exists in the chat_histories
    if session_id not in chat_histories:
        chat_histories[session_id] = []
        print(f"Created new session: {session_id}")
    else:
        print(f"Session already exists: {session_id}")

    return {"session_id": session_id}


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        message = data.get("message")
    except:
        # If parsing fails, assume it's a properly formatted ChatRequest
        chat_request = ChatRequest.parse_obj(await request.json())
        session_id = chat_request.session_id
        message = chat_request.message

    # If no session ID provided, create a new one
    if not session_id:
        session_id = str(uuid.uuid4())

    # Initialize chat history for this session if needed
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    # Add user message to history
    chat_histories[session_id].append({"role": "user", "content": message})

    def stream_response():
        full_response = ""
        for chunk, msg_type in llm.generate_response_stream(message, max_new_tokens=1000):
            full_response += chunk
            yield chunk

        chat_histories[session_id].append({"role": "assistant", "content": full_response, "msg_type": msg_type})

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    if session_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": chat_histories[session_id]}


@app.post("/add_conversation")
async def add_conversation(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        conversation_string = data.get("conversation")
    except:
        # If parsing fails, assume it's a properly formatted ChatRequest
        chat_request = ChatRequest.parse_obj(await request.json())
        session_id = chat_request.session_id
        conversation_string = chat_request.conversation_string

    logger.info(f"Session ID: {session_id}, Conversation: {conversation_string}")

    if not conversation_string:
        raise HTTPException(status_code=400, detail="Conversation string is required")

    # Initialize chat history for this session if needed
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_histories[session_id].append({"role": "assistant", "content": conversation_string.strip()})

    return {
        "session_id": session_id,
        "status": "success",
        "history": chat_histories[session_id]
    }


@app.get("/api/active_sessions")
async def get_active_sessions():
    """Get a list of all active session IDs (for admin purposes)"""
    return {"sessions": list(chat_histories.keys())}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history"""
    if session_id in chat_histories:
        del chat_histories[session_id]
        return {"status": "success", "message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")




def create_gradio_app_with_session_sharing():
    with gr.Blocks(title="SPM-GPT WebUI") as demo:
        with gr.Row():
            session_info = gr.JSON(label="Session Information", visible=True, elem_id="session_info")


        with gr.Row():
            chatbot = gr.Chatbot(height=900, elem_id="chatbot",
                                 # type="messages",
                                 avatar_images=["resources/user.png", "resources/bot.png"])

        with gr.Row():
            msg = gr.Textbox(placeholder="Type a message...", label="Message", scale=8)

        with gr.Row():
            with gr.Column():
                new_session_btn = gr.Button("New Session")
                load_session_btn = gr.Button("Load Session", elem_id="load_session_btn")
                auto_update_checkbox = gr.Checkbox(
                    label="Enable Auto Update",
                    value=False,  # default ON
                    elem_id="auto_update_checkbox",
                    interactive=True
                )

            session_id_input = gr.Textbox(label="Session ID (for sharing)", elem_id="current_session_id", interactive=True)
            session_id_input._state = True  # Persist across refreshes
            with gr.Column():
                save_history_btn = gr.Button("Save History")
                load_history_btn = gr.Button("Load History")



        # State variables
        current_session_id = gr.State("")
        # Debug output (can be hidden in production)
        debug_output = gr.Textbox(label="Debug Output", visible=True)

        # When the user sends a message
        async def user_message(message, history, session_id):
            if not message.strip():
                yield "", history, session_id, {}, session_id

            history = history or []
            history.append({"role": "user", "content": message})
            assistant_response = ""

            # Send to API with streaming
            async with httpx.AsyncClient() as client:
                async with client.stream(
                        "POST",
                        "http://localhost:7860/chat",  # <-- change this to your streaming endpoint
                        json={"message": message, "session_id": session_id}
                ) as response:
                    async for chunk in response.aiter_text():
                        assistant_response += chunk
                        rendered_response = cmd_to_md(assistant_response)
                        # Optionally update Gradio UI here (requires generator-style yield)
                        # For simplicity, we collect full message here
                        yield "", history + [{"role": "assistant", "content": rendered_response}], session_id, {}, session_id


            cmd_content = re.findall(r"<cmd>\s*(.*?)\s*</cmd>", assistant_response, re.DOTALL)
            non_cmd_parts = re.sub(r"<cmd>.*?</cmd>", "", assistant_response, flags=re.DOTALL).strip()

            history.append({"role": "assistant", "content": non_cmd_parts})
            history.append({"role": "assistant", "content": str(cmd_content)})

            session_info = {
                "session_id": session_id,
                "message_count": len(history) // 2,
                "active_connections": active_connections,
                "sessions_list": list(chat_histories.keys())
            }

            # Notify all connected clients of the new message
            send_message = {
                "response": assistant_response,
                "type": chat_histories[session_id][-1]["msg_type"],
            }

            await asyncio.create_task(notify_clients(session_id, json.dumps(send_message)))  # Non-blocking call

            yield "", history, session_id, session_info, session_id

        # Create a new session
        async def new_session(optional_session_id=None):
            params = {}
            if optional_session_id:
                params["session_id"] = optional_session_id

            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:7860/create_session", params=params)
                data = response.json()
                data["sessions_list"] = list(chat_histories.keys())
                session_id = data["session_id"]

                return (
                    data,  # session_info
                    [],  # chatbot
                    session_id,  # current_session_id
                    session_id,  # session_id_input
                    f"Created new session: {session_id}"  # debug_output
                )

        # Load session by ID
        async def load_session_by_id(session_id):
            logger.info(session_id_input.value)
            if not session_id.strip():
                return {"error": "No session ID provided"}, [], "", session_id, "Error: No session ID provided"

            async with httpx.AsyncClient() as client:
                # Get chat history
                response = await client.get(f"http://localhost:7860/chat_history/{session_id}")
                if response.status_code != 200:
                    return {"error": "Invalid session ID"}, [], "", session_id, f"Error: Session {session_id} not found"

                history_data = response.json()

                # Convert API format to Gradio format
                api_history = history_data.get("history", [])
                # gradio_history = history_data.get("history", [])
                gradio_history = render_history_for_ui(api_history)

                session_info = {
                    "session_id": session_id,
                    "message_count": len(api_history) // 2,
                    "sessions_list": list(chat_histories.keys())
                }

                return session_info, gradio_history, session_id, session_id, f"Loaded session: {session_id}"

        def save_history_button(session_id):
            save_history(session_id)
            return f"History saved for session {session_id}"

        # Load history function triggered by button
        def load_history_button(session_id):
            load_history(session_id)
            return f"History loaded for session {session_id}"

        # Handle message submission
        msg.submit(user_message, [msg, chatbot, current_session_id],
                   [msg, chatbot, current_session_id, session_info, session_id_input])

        # Handle session management
        new_session_btn.click(new_session, None,
                              [session_info, chatbot, current_session_id, session_id_input, debug_output])
        load_session_btn.click(load_session_by_id, [session_id_input],
                               [session_info, chatbot, current_session_id, session_id_input, debug_output])

        save_history_btn.click(save_history_button, inputs=[current_session_id], outputs=[debug_output])
        load_history_btn.click(load_history_button, inputs=[session_id_input], outputs=[debug_output])

        # JavaScript: click load_session_btn every 1s
        js_poll = """
        function() {
            let autoUpdateInterval = null;
            function startPolling() {
                if (!autoUpdateInterval) {
                    autoUpdateInterval = setInterval(() => {
                        document.querySelector('button#load_session_btn')?.click();
                    }, 5000);
                    console.log("Auto update started");
                }
            }

            function stopPolling() {
                if (autoUpdateInterval) {
                    clearInterval(autoUpdateInterval);
                    autoUpdateInterval = null;
                    console.log("Auto update stopped");
                }
            }

            // Start if checkbox is checked at load
            const checkbox = document.querySelector('#auto_update_checkbox input');
            if (checkbox && checkbox.checked) {
                startPolling();
            }

            // Listen for checkbox changes
            if (checkbox) {
                checkbox.addEventListener('change', function() {
                    if (this.checked) {
                        startPolling();
                    } else {
                        stopPolling();
                    }
                });
            }

            return [];
        }
        """

        async def auto_load_or_create():
            session_id = getattr(app.state, "session_id_from_url", None)
            logger.info(f"[auto_load_or_create] got session_id_input = {session_id!r}")
            if session_id:
                return await load_session_by_id(session_id)
            else:
                return await new_session()

        demo.load(auto_load_or_create,
                  [],
                  [session_info, chatbot, current_session_id, session_id_input, debug_output],
                  js=js_poll)

    return demo


gr_app = create_gradio_app_with_session_sharing()
app = gr.mount_gradio_app(app, gr_app, path="/")


if __name__ == "__main__":
    import uvicorn
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"open in local host:{local_ip}:7860", )

    # 133.1.195.183:7860/ws/chat/spm-agent20260204
    uvicorn.run(app, host="0.0.0.0", port=7860)
