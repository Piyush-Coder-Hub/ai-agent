from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict,Annotated
from langchain_community.chat_models import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

# Initialize the Ollama model
generator_model = ChatOllama(model="llama3.1", temperature=0)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState):
    message = state["messages"]
    response = generator_model.invoke(message)
    return {"messages": [response]}

def retrive_all_thread_id_from_db():
    all_threads = set()
    for checkpointe in checkpointer.list(None):
        all_threads.add(checkpointe.config["configurable"]["thread_id"])
    
    return list(all_threads)


conn = sqlite3.connect(database="chatbot_conversations.db",check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
