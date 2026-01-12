from typing import TypedDict,List
from langgraph.graph import StateGraph,END
import random

class LoopingGraphState(TypedDict):
    name:str
    number:List[int]
    counter:int

def greet_user(state:LoopingGraphState) -> LoopingGraphState:
    """Method used to Greet User"""
    state['name'] = f"Hi!! there ,{state['name']}"
    state['counter'] = 0
    return state

def generate_random_num(state:LoopingGraphState) -> LoopingGraphState:
    """Method to generate random num from 1 to 10"""
    state['number'].append(random.randint(0,10))
    state['counter'] +=1
    return state

def should_countinue(state:LoopingGraphState) -> LoopingGraphState:
    """Method to decide looping of generate_random_num ot exit"""
    if state['counter'] <5:
        print("Entering loop " ,state['counter'])
        return "loop"
    else:
        return "exit"
    
graph = StateGraph(LoopingGraphState)
graph.add_node("random",generate_random_num)
graph.add_node("greet",greet_user)
graph.add_edge("greet","random")
graph.add_conditional_edges(
    "random",
    should_countinue,
    {
        "loop":"random",
        "exit":END
    }
)

graph.set_entry_point("greet")
app = graph.compile()
app.invoke({"name":"PT","number":[],"counter":-1})