import copy
from datetime import datetime
from typing import Annotated, Any
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import add_messages # Import add_messages for proper message merging
from langgraph.types import Command
from langchain_openai import ChatOpenAI
import os
import json

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
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Make the PICO scientific elements for a given state.
    """


    # Grab the original user request to craft the prompt
    user_request = state.get("user_request", "")

    pico_prompt = f"""
You are a helpful assistant that generates the PICO (Population, Intervention, Comparison, Outcome) elements for a meta-analysis based on the following user request:

\"\"\"{user_request}\"\"\"

Return ONLY a JSON object with exactly the keys "population", "intervention", "comparison", and "outcome" and no additional keys, text or comments.
Example format:
{{
  "population": "Patients with hypertension",
  "intervention": "Treatment with a new drug",
  "comparison": "Treatment with a placebo",
  "outcome": "Blood pressure reduction"
}}
""".strip()

    pico_model = ChatOpenAI(model="o3")
    response = pico_model.invoke(pico_prompt)

    # ChatOpenAI returns an AIMessage; extract the text content
    json_str = response.content if hasattr(response, "content") else str(response)
    try:
        pico_result: dict[str, str] = json.loads(json_str)
    except json.JSONDecodeError:
        # If parsing fails, gracefully fallback to an empty dict to avoid breaking the pipeline
        pico_result = {}

    # Build the ToolMessage required by LangGraph
    tool_message = {
        "role": "tool",
        "content": "PICO created successfully",
        "name": "create_pico_for_meta_analysis",
        "tool_call_id": tool_call_id,
    }

    # Merge new message and PICO into state
    new_state_for_update = state.copy()
    current_messages = new_state_for_update.get("messages", [])
    updated_messages = add_messages(current_messages, [tool_message])
    new_state_for_update["messages"] = updated_messages
    new_state_for_update["meta_analysis_pico"] = pico_result

    return Command(update=new_state_for_update)
