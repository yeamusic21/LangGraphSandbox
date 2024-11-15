# IMPORTS
import json
import random
from collections import defaultdict
from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from chains import parser

# TOOLS
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tool_executor = ToolExecutor([tavily_tool])

# EXECUTE TOOLS
def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    tool_invocation: AIMessage = state[-1] # get the latest message?
    parsed_tool_calls = parser.invoke(tool_invocation) # BaseMessage to Json???
    ids = []
    tool_invocations = []
    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="tavily_search_results_json",
                    tool_input=query,
                ) # this seems old
            )
            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations) # run all tool calls at the same time in different threads

    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    tool_messages = []
    for id_, query_outputs in outputs_map.items():
        tool_messages.append(
            ToolMessage(content=json.dumps(query_outputs), tool_call_id=id_)
        )

    return tool_messages # return tool messages to LLM???



if __name__ == "__main__":

    # tracing
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    tracer_provider = register(
        project_name="langgraph-udemy",
        endpoint="http://127.0.0.1:6006/v1/traces"
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    # invoke
    from langchain_core.messages import HumanMessage
    from cool_classes import AnswerQuestion, Reflection

    ################################# Execute Tools DEBUGGING FROM EDEN MARCO

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_KpYHichFFEmLitHFvFhKy1Ra",
    )
    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content=answer.json(),
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                    }
                ],
            ),
        ]
    )
    print(raw_res)
    res = json.loads(raw_res[0].content)
    print(res)