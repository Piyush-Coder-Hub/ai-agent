import streamlit as st
from db_connectivity_chatbot_backend import chatbot ,retrive_all_thread_id_from_db
from langchain_core.messages import HumanMessage
import uuid

#********************************************Utility functions*************************************************

def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

def reset_chat():
    st.session_state['message_history'] = []
    st.session_state['thread_id'] = generate_thread_id()
    add_thread_history(st.session_state['thread_id'])
    
def add_thread_history(thread_id):
    if thread_id not in st.session_state['chat_thread_history']:
        st.session_state['chat_thread_history'].append(thread_id)

def load_conversation_history(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_thread_history' not in st.session_state:
    st.session_state['chat_thread_history'] = retrive_all_thread_id_from_db()

add_thread_history(st.session_state['thread_id'])

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
st.sidebar.title("Chatbot UI")

if st.sidebar.button("New Chat"):
    reset_chat()

st.header("My Conversion with the AI Agent")

for thread_id in st.session_state['chat_thread_history'][::-1]:  # Display threads in reverse order (most recent first)
    if st.sidebar.button(thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation_history(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages


# Render history
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
   # first add the message to message_history
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= CONFIG,
                stream_mode= 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})