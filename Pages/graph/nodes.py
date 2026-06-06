from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState
from typing import Literal
from .tools import complete_python_task
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

tools = [complete_python_task]

model = llm.bind_tools(tools)

with open(
    os.path.join(os.path.dirname(__file__), "../prompts/main_prompt.md"), "r"
) as file:
    prompt = file.read()

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("placeholder", "{messages}"),
    ]
)
model = chat_template | model


def create_data_summary(state: AgentState) -> str:
    summary = ""
    variables = []
    for d in state["input_data"]:
        variables.append(d.variable_name)
        summary += f"\n\nVariable: {d.variable_name}\n"
        summary += f"Description: {d.data_description}"

    if "current_variables" in state:
        remaining_variables = [
            v for v in state["current_variables"] if v not in variables
        ]
        for v in remaining_variables:
            summary += f"\n\nVariable: {v}"
    return summary


def route_to_tools(
    state: AgentState,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route back to the agent.
    """

    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


def call_model(state: AgentState):

    current_data_template = """The following data is available:\n{data_summary}"""
    current_data_message = HumanMessage(
        content=current_data_template.format(data_summary=create_data_summary(state))
    )
    model_input = {**state, "messages": [current_data_message] + state["messages"]}

    llm_outputs = model.invoke(model_input)

    return {
        "messages": [llm_outputs],
        "intermediate_outputs": [current_data_message.content],
    }
