from langchain_anthropic import ChatAnthropic
from langgraph_supervisor import create_supervisor
from prompts.supervisor_prompt import supervisor_prompt
from agents.researcher import researcher_agent
from agents.processor import processor_agent
from agents.retriever import retriever_agent
from agents.analyzer import analyzer_agent
from agents.writer import writer_agent
import os
from tools.supervisor_agent_tools import create_pico_for_meta_analysis, create_handoff_tool, clean_context
from datetime import datetime
from state.state import MetaAnalysisState
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

date_time = datetime.now().strftime("%Y-%m-%d")

# Create the main agent 
# Its function is to choose, via handoff, which agent should be called recursively
# Using only available agents: researcher_agent and processor_agent

supervisor_agent = create_supervisor (
    agents = [researcher_agent, processor_agent, retriever_agent, analyzer_agent, writer_agent],
    tools = [
        create_handoff_tool(agent_name="researcher"),
        create_handoff_tool(agent_name="processor"),
        create_handoff_tool(agent_name="retriever"),
        create_handoff_tool(agent_name="analyzer"),
        create_handoff_tool(agent_name="writer"),
        create_pico_for_meta_analysis,
        clean_context,
    ],
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    ),
    prompt = supervisor_prompt,
    state_schema = MetaAnalysisState
).compile().with_config(recursion_limit=500)
