from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict,Annotated
from langchain_community.chat_models import ChatOllama
from langgraph.graph.message import add_messages

# Initialize the Ollama model
generator_model = ChatOllama(model="llama3.1", temperature=0)

class ChatState(TypedDict):
    conversation_history: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState):
    message= state["conversation_history"]
    response = generator_model.invoke(message)
    return {"conversation_history": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)
chatbot= graph.compile()

# initial_state = {
#     'conversation_history': [HumanMessage(content='What is the capital of india')]
# }

# print(chatbot.invoke(initial_state)['conversation_history'][-1].content)

while True:
    user_input = input("Type Here : ")
    print("User: ", user_input)

    if(user_input.strip().lower() in ['exit', 'quit', 'q',"close","end","bye"]):
        print("Exiting the chat.")
        break

    response = chatbot.invoke({'conversation_history': [HumanMessage(content=user_input)]})

    print("AI: ", response['conversation_history'][-1].content)