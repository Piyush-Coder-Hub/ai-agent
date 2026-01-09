from typing import TypedDict
from langgraph.graph import StateGraph,END

class TripleNodeAgent(TypedDict):
    age:int
    name:str
    skills:str
    result:str

def first_node(state:TripleNodeAgent) -> TripleNodeAgent:
    state['result'] = f"Hi there !! {state['name']}"
    return state

def second_node(state:TripleNodeAgent) -> TripleNodeAgent:
    state['result'] = state['result'] + " Your Age is " + f"{state['age']}"
    return state

def triple_node(state:TripleNodeAgent) -> TripleNodeAgent:
    state['result'] = state['result'] + " Your Skills are " + f"{state['skills']}"
    return state


graph = StateGraph(TripleNodeAgent)
graph.add_node("first_node",first_node)
graph.add_node("second_node",second_node)
graph.add_node("triple_node",triple_node)
graph.set_entry_point("first_node")
graph.add_edge("first_node","second_node")
graph.add_edge("second_node","triple_node")
graph.set_finish_point("triple_node")
app = graph.compile()
finalOutput = app.invoke({"name":"SLDAS" ,"age":45,"skills":"Python,Jaava,C#"})

print(finalOutput['result'])