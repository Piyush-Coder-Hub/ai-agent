from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict,Annotated
from langchain_community.chat_models import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

# Initialize the Ollama model
generator_model = ChatOllama(model="llama3.1", temperature=0)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState):
    message = state["messages"]
    response = generator_model.invoke(message)
    return {"messages": [response]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)