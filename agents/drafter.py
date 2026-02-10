from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
import json
import uuid

# This is the global variable to store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [update, save]

# Use Ollama instead of OpenAI
model = ChatOllama(model="llama3.1", temperature=0.7)


def call_tool_manually(tool_call: dict) -> ToolMessage:
    """Execute tool manually from JSON tool call"""
    tool_name = tool_call.get("name")
    args = tool_call.get("arguments", {})
    
    tools_dict = {
        "update": update,
        "save": save,
    }
    
    if tool_name not in tools_dict:
        return ToolMessage(
            content=f"Unknown tool: {tool_name}", 
            tool_name=tool_name,
            tool_call_id=str(uuid.uuid4())
        )
    
    result = tools_dict[tool_name].invoke(args)
    return ToolMessage(
        content=result, 
        tool_name=tool_name,
        tool_call_id=str(uuid.uuid4())
    )


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
You are Drafter, a helpful writing assistant.

When the user wants to update or save, respond ONLY with valid JSON tool calls:
{{"name": "update", "arguments": {{"content": "..."}}}}
or
{{"name": "save", "arguments": {{"filename": "..."}}}}

If no tool needed, respond normally.

Current document:
{document_content}
""")

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    
    messages = list(state["messages"]) + [user_message, response]
    
    # Try to parse JSON tool call
    try:
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and "name" in parsed:
            print(f"🔧 USING TOOL: {parsed['name']}")
            tool_message = call_tool_manually(parsed)
            print(f"🛠️ TOOL RESULT: {tool_message.content}")
            messages.append(tool_message)
    except json.JSONDecodeError:
        pass

    return {"messages": messages}


def should_continue(state: AgentState) -> str:
    """Stop when save tool is executed"""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.tool_name == "save":
            return "end"
    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# LangGraph setup
graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER (OLLAMA) =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
