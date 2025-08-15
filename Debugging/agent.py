from typing import Annotated
from typing import TypedDict
from langchain_core.tools import tool
from langgraph.graph import START,StateGraph,END
from langgraph.graph.message import add_messages, BaseMessage
from langgraph.prebuilt import tools_condition , ToolNode
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="MultiAgentic_Langgraph"

from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:llama3-8b-8192")

class State(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]

from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)
tool.invoke("What is langgraph")

def make_tool_graph():
    from langchain_core.tools import tool
    from langgraph.prebuilt import tools_condition

    @tool
    def add(a:float , b:float):
        """Add two numbers"""
        return a+b
    tools = [add, TavilySearch(max_results=2).as_tool()]
    llm_with_tool = llm.bind_tools(tools)

    ## StateGraph

    from langgraph.graph import StateGraph,START,END
    from langgraph.prebuilt import ToolNode
    from langgraph.prebuilt import tools_condition

    ## Node Defination
    def tool_calling_llm(state:State):
        return {"messages":[llm_with_tool.invoke(state["messages"])]}

    ## Graph
    builder=StateGraph(State)
    builder.add_node("tool_calling_llm",tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    ##Add Edges
    builder.add_edge(START,"tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                                tools_condition
                                )
    builder.add_edge("tools","tool_calling_llm")

    ## Compile the graph

    graph=builder.compile()

    return graph

    # from IPython.display import Image, display

    # display(Image(graph.get_graph().draw_mermaid_png()))

tool_agent = make_tool_graph()






