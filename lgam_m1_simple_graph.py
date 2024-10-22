# langchain-academy-main
# module-1
# simple-graph.ipynb

from typing_extensions import TypedDict
import random
from typing import Literal
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

#
# First, define the State of the graph. 
# The State schema serves as the input schema for all Nodes and Edges in the graph.
# Let's use the `TypedDict` class from python's `typing` module as our schema, which provides type hints for the keys.
#

class State(TypedDict):
    graph_state: str

#
# By default, the new value returned by each node will override the prior state value.
#

def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}

#
## Edges
# Edges connect the nodes.
# Normal Edges are used if you want to *always* go from, for example, `node_1` to `node_2`.
# Conditional Edges are used want to *optionally* route between nodes.
# Conditional edges are implemented as functions that return the next node to visit based upon some logic.
#

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"

#
# Build Graph
#

# Add Nodes
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Add Edges
builder.add_edge(START, "node_1") # START --> node_1
builder.add_conditional_edges("node_1", decide_mood) # node_1 --> node_2 OR node_3
builder.add_edge("node_2", END) # node_2 --> END
builder.add_edge("node_3", END) # node_3 --> END

# Compile Graph
graph = builder.compile()

# View Graph
# display(Image(graph.get_graph().draw_mermaid_png()))

# 
# Invoke Graph 
#

res = graph.invoke({"graph_state" : "Hi, this is Lance."})
print(res)