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

# Clean context tool

@tool
def clean_context(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Cleans the context by:
    1. Clearing urls_to_process and processed_urls lists, leaving only a timestamp placeholder
    2. Summarizing all messages into a single organized message to reduce context size
    
    This tool helps prevent context overload by cleaning accumulated URLs and condensing message history.
    """
    
    try:
        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create placeholder messages for URL lists
        url_placeholder = f"Cleaned at {current_time}"
        
        # Get current messages for summarization
        current_messages = state.get("messages", [])
        
        if len(current_messages) > 10:  # Only summarize if there are many messages
            # Prepare content for LLM summarization
            messages_text = ""
            for i, msg in enumerate(current_messages):
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
                else:
                    content = str(msg)
                
                role = getattr(msg, 'role', 'unknown') if hasattr(msg, 'role') else (msg.get('role', 'unknown') if isinstance(msg, dict) else 'unknown')
                name = getattr(msg, 'name', '') if hasattr(msg, 'name') else (msg.get('name', '') if isinstance(msg, dict) else '')
                
                messages_text += f"Message {i+1} ({role}" + (f" - {name}" if name else "") + f"): {content}\n\n"
            
            # Create summarization prompt
            summary_prompt = f"""
You are tasked with creating a comprehensive summary of a meta-analysis conversation history.

CONVERSATION HISTORY TO SUMMARIZE:
{messages_text}

INSTRUCTIONS:
Create a single, well-organized summary that captures:
1. The main research topic and objectives
2. Key actions taken by each agent (researcher, processor, retriever, analyzer, etc.)
3. Important findings, data collected, or insights generated
4. Current status and progress made
5. Any issues encountered and how they were resolved
6. Next steps or pending tasks mentioned

Keep the summary concise but comprehensive. Focus on factual information and actionable insights.
Return ONLY the summary text, no additional formatting or comments.
""".strip()

            # Use LLM to create summary
            summary_model = ChatOpenAI(model="gpt-4,1", temperature=0.1)
            summary_response = summary_model.invoke(summary_prompt)
            
            # Extract summary content
            summary_content = summary_response.content if hasattr(summary_response, "content") else str(summary_response)
            
            # Create the summary message
            from langchain_core.messages import AIMessage
            summary_message = AIMessage(
                content=f"📋 CONVERSATION SUMMARY (Generated at {current_time}):\n\n{summary_content}",
                name="supervisor"
            )
            
            summarized_messages = [summary_message]
        else:
            # If not many messages, keep them as is
            summarized_messages = current_messages
        
        # Create success tool message
        tool_message = {
            "role": "tool",
            "content": f"Context cleaned successfully at {current_time}. "
                      f"URLs cleared and messages {'summarized' if len(current_messages) > 10 else 'maintained'}.",
            "name": "clean_context",
            "tool_call_id": tool_call_id,
        }
        
        # Create informative AI message
        from langchain_core.messages import AIMessage
        info_message = AIMessage(
            content=f"🧹 Context cleanup completed! Cleared URL lists and "
                   f"{'condensed {len(current_messages)} messages into a summary' if len(current_messages) > 10 else f'maintained {len(current_messages)} messages'}. "
                   f"This should help reduce context overload and improve performance.",
            name="supervisor"
        )
        
        # Prepare updated state
        new_state_for_update = state.copy()
        
        # Add tool message and info message to the summarized messages
        updated_messages = add_messages(summarized_messages, [tool_message, info_message])
        
        # Update state with cleaned data
        new_state_for_update.update({
            "urls_to_process": [url_placeholder],
            "processed_urls": [url_placeholder], 
            "messages": updated_messages
        })
        
        return Command(update=new_state_for_update)
        
    except Exception as e:
        # Handle errors gracefully
        error_message = {
            "role": "tool",
            "content": f"Error cleaning context: {str(e)}",
            "name": "clean_context",
            "tool_call_id": tool_call_id,
        }
        
        new_state_for_update = state.copy()
        current_messages = new_state_for_update.get("messages", [])
        updated_messages = add_messages(current_messages, [error_message])
        new_state_for_update["messages"] = updated_messages
        
        return Command(update=new_state_for_update)


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
