from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
import re

# -------------------- State --------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -------------------- Tools --------------------
@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
}

# -------------------- LLM --------------------
llm = ChatOllama(model="llama3.1", temperature=0.7)

# -------------------- Tool Detection --------------------
def detect_tool_call(text: str):
    text = text.lower()
    numbers = list(map(int, re.findall(r"-?\d+", text)))

    if len(numbers) < 2:
        return None, None

    a, b = numbers[0], numbers[1]

    if "add" in text or "sum" in text or "plus" in text or "+" in text:
        return "add", {"a": a, "b": b}
    elif "subtract" in text or "minus" in text or "-" in text:
        return "subtract", {"a": a, "b": b}
    elif "multiply" in text or "times" in text or "*" in text or "x" in text:
        return "multiply", {"a": a, "b": b}

    return None, None

# -------------------- Agent Node --------------------
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    last_message = state["messages"][-1]
    # If user message → try tool
    if isinstance(last_message, HumanMessage):
        tool_name, tool_args = detect_tool_call(last_message.content)
        if tool_name:
            result = tools[tool_name].run(tool_args)
            # Send result back as AI message (not ToolMessage)
            return {"messages": [AIMessage(content=str(result))]}

    # Otherwise, let the LLM respond
    response = llm.invoke([system_prompt] + state["messages"])
    write_history_to_file(state["messages"])
    return {"messages": [response]}

# -------------------- Continue Logic --------------------
def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    # If we just produced a number → continue once so LLM can respond
    if isinstance(last_message, AIMessage) and last_message.content.isdigit():
        return "continue"
    else:
        return "end"

# -------------------- Graph --------------------
graph = StateGraph(AgentState)
graph.add_node("my_agent", model_call)
graph.set_entry_point("my_agent")

graph.add_conditional_edges(
    "my_agent",
    should_continue,
    {
        "continue": "my_agent",
        "end": END
    }
)

app = graph.compile()

# -------------------- Streaming Printer --------------------
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# -------------------- Run --------------------
inputs = {
    "messages": [
        ("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")
    ]
}

def write_history_to_file(messages, filename="conversation_log.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "")
            f.write(f"{role}: {msg.content}\n")
        f.write("\n" + "-" * 40 + "\n\n")
        
print_stream(app.stream(inputs, stream_mode="values"))

