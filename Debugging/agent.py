from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages

from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import tools_condition


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Demo_Project"



llm=init_chat_model("groq:openai/gpt-oss-20b")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

## State Graph with tool call

def make_tool_graph():
    @tool
    def add(a:float,b:float):
        """Add two numbers"""
        return a+b

    tools = [add]
    tool_node = ToolNode(tools)

    llm_with_tool = llm.bind_tools([add])

    ## Node Definition
    def call_llm_model(state:State):
        return {"messages":[llm_with_tool.invoke(state['messages'])]}

    #Graph 

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools",ToolNode(tools))

    # Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",tools_condition
    )
    builder.add_edge("tools","tool_calling_llm"),
    builder.add_edge("tool_calling_llm",END)

    #compile the graph
    graph = builder.compile()

    return graph

tool_agent = make_tool_graph()