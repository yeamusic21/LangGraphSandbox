from langchain_openai import ChatOpenAI

from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()  # Load the .env file

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)



# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)

# Show
# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# Input
initial_input = {"messages": "Multiply 2 and 3"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

print("------------------------------------------------")
print("------------------------------------------------")
print("------------------------------------Stop before assistant")

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

state = graph.get_state(thread)
print("state: ", state)

print("------------------------------------------------")
print("------------------------------------------------")
print("------------------------------------Tee up state to push forward")

graph.update_state(
    thread,
    {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
)

new_state = graph.get_state(thread).values
for m in new_state['messages']:
    m.pretty_print()

print("------------------------------------------------")
print("------------------------------------------------")
print("------------------------------------Push forward")

for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

print("------------------------------------------------")
print("------------------------------------------------")
print("------------------------------------Stopped at assistant again")
print("------------------------------------Push forward one more time to complete graph")

for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()