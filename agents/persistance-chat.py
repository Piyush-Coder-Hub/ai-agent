from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_community.chat_models import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from typer import prompt

# Initialize the Ollama model
generator_model = ChatOllama(model="llama3.1", temperature=0)

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state:JokeState):
   prompt = f"generate a joke on topic {state['topic']}"
   response = generator_model.invoke(prompt).content
   return {"joke": response}

def explain_joke(state:JokeState):
    prompt = f'write an explanation for the joke - {state["joke"]}'
    response = generator_model.invoke(prompt).content
    return {"explanation": response}

graph = StateGraph(JokeState)
graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', explain_joke)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)


config1 = {"configurable": {"thread_id": "1"}}
print(workflow.invoke({'topic':'pizza'}, config=config1))
print("\n\n")
print(list(workflow.get_state_history(config=config1)))