from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

class MemoryAgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]

llm = ChatOllama(
    model="llama3.1",
    temperature=0.7
)

def process(state:MemoryAgentState):
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\n AI :- {response.content}")
    return state

graph = StateGraph(MemoryAgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

conversation_history = []
user_input = input("Enter :: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result =  agent.invoke({"messages":conversation_history})
    print(result['messages'])
    conversation_history = result['messages']
    user_input = input("Enter: ")

with open("logging.txt","w") as file:
    file.write("Your Conversation history")

    for message in conversation_history:
        if(isinstance(message,HumanMessage)):
            file.write(f"You : {message.content} \n")
        elif(isinstance(message,AIMessage)):
            file.write(f"AI : {message.content} \n")
    
    file.write("End of Conversation")

print("Conversation saved to logging.txt file")