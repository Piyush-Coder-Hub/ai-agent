from typing import TypedDict
from langgraph.graph import StateGraph,END,START
from IPython.display import Image, display

class ConditionalState(TypedDict):
    number1:int
    number2:int
    operation:str
    finalOutput:str

def add_node(state : ConditionalState) -> ConditionalState:
    """This is node for addition of two numbers"""
    state['finalOutput'] = state['number1'] + state['number2']
    return state

def sub_node(state:ConditionalState) -> ConditionalState:
    """This is node for subtration of two numbers"""
    state['finalOutput'] = state['number1'] - state['number2']
    return state

def invalid_node(state:ConditionalState) -> ConditionalState:
    """This is node for invalid operation"""
    state['finalOutput'] = f"Ivalid operation {state['operation']} not supported."
    return state

def decide_next_node(state:ConditionalState) -> ConditionalState:
    """This node select the next node"""
    
    if(state['operation']) == "+":
        return "addition_operation"
    
    if(state['operation']) == "-":
        return "subtration_operation"
    
    else:
        return "invalid_operation"
    
graph = StateGraph(ConditionalState)
graph.add_node("add_node",add_node)
graph.add_node("sub_node",sub_node)
graph.add_node("invalid_node",invalid_node)
graph.add_node("router",lambda state:state)
graph.add_edge(START,"router")
graph.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "addition_operation":"add_node",
        "subtration_operation":"sub_node",
        "invalid_operation" : "invalid_node"
    }
)

graph.add_edge("add_node",END)
graph.add_edge("sub_node",END)
graph.add_edge("invalid_node",END)
app = graph.compile()
result = app.invoke(ConditionalState(number1=5,number2=10,operation="*"))
print(result['finalOutput'])

display(Image(app.get_graph().draw_mermaid_png()))
