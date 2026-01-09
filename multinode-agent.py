from typing import TypedDict
from langgraph.graph import StateGraph,END

class MultiNodeAgent(TypedDict):
    age:int
    name:str
    result:str

def first_node(state:MultiNodeAgent) -> MultiNodeAgent:
    state['result'] = f"Hi there !! {state['name']}"
    return state

def second_node(state:MultiNodeAgent) -> MultiNodeAgent:
    state['result'] = state['result'] + " Your Age is " + f"{[state['age']]}"
    return state

graph = StateGraph(MultiNodeAgent)
graph.add_node("first_node",first_node)
graph.add_node("second_node",second_node)
graph.set_entry_point("first_node")
graph.add_edge("first_node","second_node")
graph.set_finish_point("second_node")
app = graph.compile()
finalOutput = app.invoke({"name":"SLDAS" ,"age":45})

print(finalOutput['result'])