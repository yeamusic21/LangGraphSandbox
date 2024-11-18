from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph

from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT


flow = StateGraph(AgentState)

flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.add_node(ACT, execute_tools)

flow.set_entry_point(AGENT_REASON)

flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":

    # tracing
    tracer_provider = register(
        project_name="langgraph-udemy",
        endpoint="http://127.0.0.1:6006/v1/traces"
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    print("Hello ReAct with LangGraph")
    res = app.invoke(
        input={
            "input": "what is the weather in sf? List it and then Triple it ",
        }
    )
    print(res["agent_outcome"].return_values["output"])
