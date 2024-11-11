from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})

# we return the result of the LLM chain as a 'HumanMessage' to trick the LLM
# seems to work without '-> List[BaseMessage]' ??
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # def reflection_node(messages: Sequence[BaseMessage]):
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 2:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
# print(graph.get_graph().draw_mermaid())
# graph.get_graph().print_ascii()

if __name__ == "__main__":
    # tracing
    tracer_provider = register(
        project_name="langgraph-udemy",
        endpoint="http://127.0.0.1:6006/v1/traces"
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    # run
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)
    print(response[-1].content)