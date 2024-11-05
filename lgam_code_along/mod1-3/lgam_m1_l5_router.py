from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()  # Load the .env file
#
# Define tool calling LLM 
#

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools([multiply])

#
# Create tool calling llm node
#

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

#
# Create graph
#

# Build graph
builder = StateGraph(MessagesState)
# Should I call the tool?  If so, what arguments/values do I need?
builder.add_node("tool_calling_llm", tool_calling_llm)
# the actual tool itself
builder.add_node("tools", ToolNode([multiply]))
# START to tool calling decision maker
builder.add_edge(START, "tool_calling_llm")
# if the tool_calling_llm says to run the tool then run the tool, otherwise END
# tools_condition = Use in the conditional_edge to route to the ToolNode if the last message has tool calls. Otherwise, route to the end.
builder.add_conditional_edges(
    "tool_calling_llm", # the starting node
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
# tool to END
builder.add_edge("tools", END)
graph = builder.compile()

# View
# display(Image(graph.get_graph().draw_mermaid_png()))

#
# Results
#

# messages = [HumanMessage(content="Hello world.")]
messages = [HumanMessage(content="multiply 5 * 6.")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()