/**
 * This file contains helper functions to integrate session sharing with the web UI.
 * Add this script to your HTML or include it in your Gradio app.
 */

// Function to set the current session ID and reload history
function loadExistingSession(sessionId) {
  // Validate session ID format (basic UUID format check)
  if (!sessionId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)) {
    alert("Invalid session ID format");
    return;
  }
  
  // Store session ID in the Gradio app
  window.gradioApp().querySelector("#current_session_id").value = sessionId;
  
  // Fetch and load history
  fetchSessionHistory(sessionId);
}

// Function to fetch session history
async function fetchSessionHistory(sessionId) {
  try {
    const response = await fetch(`/chat_history/${sessionId}`);
    
    if (!response.ok) {
      throw new Error("Session not found or server error");
    }
    
    const data = await response.json();
    
    // Update UI with history
    if (data && data.history) {
      updateChatbotUI(data.history);
      
      // Update session info display
      const sessionInfo = {
        session_id: sessionId,
        message_count: data.history.length / 2,
        status: "Loaded existing session"
      };
      
      updateSessionInfoUI(sessionInfo);
    }
  } catch (error) {
    console.error("Error loading session:", error);
    alert(`Failed to load session: ${error.message}`);
  }
}

// Function to update the chatbot UI with messages
function updateChatbotUI(history) {
  // This function needs to be adapted based on your specific Gradio implementation
  // This is a generic example
  const chatbot = window.gradioApp().querySelector("#chatbot");
  
  // Clear existing messages
  while (chatbot.firstChild) {
    chatbot.removeChild(chatbot.firstChild);
  }
  
  // Add messages from history
  for (let i = 0; i < history.length; i += 2) {
    if (i + 1 < history.length) {
      const userMsg = history[i].content;
      const assistantMsg = history[i + 1].content;
      
      // Add user message
      const userDiv = document.createElement("div");
      userDiv.className = "user-message";
      userDiv.textContent = userMsg;
      chatbot.appendChild(userDiv);
      
      // Add assistant message
      const assistantDiv = document.createElement("div");
      assistantDiv.className = "assistant-message";
      assistantDiv.textContent = assistantMsg;
      chatbot.appendChild(assistantDiv);
    }
  }
  
  // Scroll to bottom
  chatbot.scrollTop = chatbot.scrollHeight;
}

// Function to update session info display
function updateSessionInfoUI(sessionInfo) {
  // This function needs to be adapted based on your specific Gradio implementation
  const sessionInfoElement = window.gradioApp().querySelector("#session_info");
  sessionInfoElement.textContent = JSON.stringify(sessionInfo, null, 2);
}

// Add a button to the UI for loading sessions by ID
function addLoadSessionButton() {
  const container = document.createElement("div");
  container.className = "load-session-container";
  container.style.margin = "10px 0";
  
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Enter session ID";
  input.style.padding = "8px";
  input.style.marginRight = "10px";
  
  const button = document.createElement("button");
  button.textContent = "Load Session";
  button.style.padding = "8px 16px";
  button.onclick = () => loadExistingSession(input.value);
  
  container.appendChild(input);
  container.appendChild(button);
  
  // Insert after the session info element or another suitable location
  const targetElement = window.gradioApp().querySelector("#session_info");
  targetElement.parentNode.insertBefore(container, targetElement.nextSibling);
}

// Initialize when the page loads
window.addEventListener("DOMContentLoaded", () => {
  // Wait for Gradio app to be ready
  const checkGradioLoaded = setInterval(() => {
    if (window.gradioApp && window.gradioApp()) {
      clearInterval(checkGradioLoaded);
      addLoadSessionButton();
      
      // Add function to global scope for console access
      window.loadExistingSession = loadExistingSession;
    }
  }, 100);
});