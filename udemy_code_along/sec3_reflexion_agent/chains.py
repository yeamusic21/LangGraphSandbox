# IMPORTS
from dotenv import load_dotenv
load_dotenv()
import datetime
from langchain_core.output_parsers.openai_tools import (
    PydanticToolsParser,
    JsonOutputToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from cool_classes import AnswerQuestion, ReviseAnswer

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# USED IN TOOL_EXECUTOR.PY
parser = JsonOutputToolsParser(return_id=True)

# ACTOR PROMPT
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# FIRST RESPONDER CHAIN
first_responder = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

# USED IN MAIN.PY ... OR NOWHERE???
# validator = PydanticToolsParser(tools=[AnswerQuestion])

# REVISER PROMPT
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# REVISER CHAIN
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")



# DEBUGGING AND LEARNING..............
if __name__=="__main__":

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
    
    ################################# First Responder
    
    inputs = HumanMessage(content="Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital.")
    res = first_responder.invoke([inputs])
    print("===============================")
    print(res)
    print(type(res)) # <class 'langchain_core.messages.ai.AIMessage'>

    # NOTES:
    # https://python.langchain.com/docs/how_to/tool_calling/
    # - Need to remember that "tool calling" doesn't actually call a tool, it simply returns WHAT tool and WHAT arguments to 
    #   pass to that tool.  By using tool_choice="AnswerQuestion", we TRICK the chain/llm to return specific information
    #   in a specific format.
    # - Chain returns <class 'langchain_core.messages.ai.AIMessage'> with content='' and metadata.  The metadata identifies the 
    #   tool called being AnswerQuestion tool, and the returned arguments.. which is the AI results in the format of 
    #   AnswerQuestion class, and then there is a key of 'name' with value 'AnswerQuestion'. 