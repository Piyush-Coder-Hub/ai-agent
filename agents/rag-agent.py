from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import json


load_dotenv()

# --------------------
# LLM (Ollama)
# --------------------
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# --------------------
# Embeddings (Ollama)
# --------------------
# Note: Make sure to pull the model first with: ollama pull mxbai-embed-large
# Or use: ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")

pdf_path = r"d:\Piyush\Coding\AI\Agent\ai-agent\agents\Stock_Market_Performance_2024.pdf"

# Safety check
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

# Load PDF
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# --------------------
# Chunking
# --------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"d:\Piyush\Coding\AI\Agent\ai-agent\agents"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Created ChromaDB vector store!")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# --------------------
# Retriever
# --------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

# Bind tools to Ollama model
#llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools}


# --------------------
# LLM Agent
# --------------------
# def call_llm(state: AgentState) -> AgentState:
#     messages = list(state['messages'])
#     messages = [SystemMessage(content=system_prompt)] + messages
#     message = llm.invoke(messages)
#     return {'messages': [message]}


def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)

    print("\n🤖 RAW MODEL OUTPUT:\n", response.content)

    tool_messages = []
    try:
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and "name" in parsed:
            tool_name = parsed["name"]
            args = parsed.get("arguments", {})

            if tool_name in tools_dict:
                print(f"🔧 Calling tool: {tool_name}")
                result = tools_dict[tool_name].invoke(args.get("query", ""))
                tool_messages.append(ToolMessage(name=tool_name, content=str(result)))
    except json.JSONDecodeError:
        pass

    return {"messages": [response] + tool_messages}

# --------------------
# Retriever Agent
# --------------------
def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if t['name'] not in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


# --------------------
# LangGraph Setup
# --------------------
graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


# --------------------
# Run Loop
# --------------------
def running_agent():
    print("\n=== RAG AGENT (OLLAMA) ===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


if __name__ == "__main__":
    running_agent()