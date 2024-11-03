from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()  # Load the .env file

#
# Messages
#

messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Lance"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

print("--------------------------Messages")
for m in messages:
    # m.pretty_print()
    print(m)

#
# Invoke LLM with Messages
#

llm = ChatOpenAI(model="gpt-3.5-turbo")
# llm = ChatOllama(model="gemma2:2b")
# result = llm.invoke(messages)

# print("--------------------------Invoke LLM with Messages")
# print(type(result))
# print(result)
# print(result.response_metadata)

#
# Tools
# !!!!!!!!!!!!!!!!!!! ollama._types.ResponseError: gemma2:2b does not support tools

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

# tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multiplied by 3", name="Lance")])
# print("--------------------------Tools")
# print(tool_call)
# print(tool_call.additional_kwargs['tool_calls'])

#
# Using Messages as State
#

# class MessagesState(TypedDict):
#     messages: list[AnyMessage]

#
# Reducers
#

# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# class MessagesState(MessagesState):
#     # Add any keys needed beyond messages, which is pre-built 
#     pass

# # Initial state
# initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
#                     HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
#                    ]

# # New message to add
# new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# # Test
# am_res = add_messages(initial_messages , new_message)
# print("--------------------------Messages")
# print(am_res)

#
# Graph
#
    
# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# View
# display(Image(graph.get_graph().draw_mermaid_png()))

print("--------------------------Graph")
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    # m.pretty_print()
    print(m)

print("--------------------------Graph")
messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
for m in messages['messages']:
    # m.pretty_print()
    print(m)