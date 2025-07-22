import copy
from datetime import datetime
from typing import Annotated, Any
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import add_messages # Import add_messages for proper message merging
from langgraph.types import Command
from langchain_openai import ChatOpenAI
import os

ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Generic handoff tool to transfer control and the full current context to another agent

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"

    description = description or f"Transfer control and the full current context to the {agent_name} agent to continue the task."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }

        new_state_for_update = state.copy()

        current_messages = new_state_for_update.get("messages", [])
        updated_messages = add_messages(current_messages, [tool_message])
        new_state_for_update["messages"] = updated_messages

        return Command(
            goto=agent_name,
            update=new_state_for_update,
            graph=Command.PARENT,
        )
    return handoff_tool


# Specific handoff tools for each agent

transfer_to_researcher =  create_handoff_tool (
    agent_name = "researcher",
    description = "Transfer control and the full current context to the researcher agent to continue the task."
)

transfer_to_processor =  create_handoff_tool (
    agent_name = "processor",
    description = "Transfer control and the full current context to the processor agent to continue the task."
)

transfer_to_retriever =  create_handoff_tool (
    agent_name = "retriever",
    description = "Transfer control and the full current context to the retriever agent to continue the task."
)

transfer_to_analyzer =  create_handoff_tool (
    agent_name = "analyzer",
    description = "Transfer control and the full current context to the analyzer agent to continue the task."
)

transfer_to_writer =  create_handoff_tool (
    agent_name = "writer",
    description = "Transfer control and the full current context to the writer agent to continue the task."
)

transfer_to_reviewer =  create_handoff_tool (
    agent_name = "reviewer",
    description = "Transfer control and the full current context to the reviewer agent to continue the task."
)

transfer_to_editor =  create_handoff_tool (
    agent_name = "editor",
    description = "Transfer control and the full current context to the editor agent to continue the task."
)

# PICO tool

@tool
def create_pico_for_meta_analysis(
    state: Annotated[dict[str, Any], InjectedState]
) -> Command:
    """
    Make the PICO scientific elements for a given state.
    """


    pico_prompt = """
    You are a helpful assistant that makes the PICO elements based on the current state, specially the user_request key, for making a meta-analysis.
    You MUST return ONLY a dictionary with the PICO elements, no other text or comments.
    The dictionary must have the following keys: population, intervention, comparison, outcome.
    For example:
    {
        "population": "Patients with hypertension",
        "intervention": "Treatment with a new drug",
        "comparison": "Treatment with a placebo",
        "outcome": "Blood pressure reduction"
    }
    """

    pico_model = ChatOpenAI(model="o3")
    pico_chain = pico_prompt | pico_model.with_structured_output(dict)

    user_request = state.get("user_request", "")
    pico_result = pico_chain.invoke({"user_request": user_request})

    return Command(update={"meta_analysis_pico": pico_result})
