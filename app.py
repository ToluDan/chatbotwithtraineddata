import streamlit as st
from Minichatbot import ChatBotAssistant, get_stocks

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Chatbot")

# =========================
# LOAD CHATBOT
# =========================
@st.cache_resource
def load_chatbot():

    assistant = ChatBotAssistant(
        'responseplusquestions.json',
        function_mapping={'stocks': get_stocks}
    )

    assistant.parse_intents()

    assistant.load_model(
        'chatbot_model.pth',
        'dimesions.json'
    )

    return assistant

assistant = load_chatbot()

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# DISPLAY OLD MESSAGES
# =========================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================
prompt = st.chat_input("Type your message")

if prompt:

    # Show user message
    st.chat_message("user").markdown(prompt)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Generate response
    response = assistant.to_process_message(prompt)

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save bot response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )