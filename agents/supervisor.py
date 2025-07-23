from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_supervisor import create_supervisor
from prompts.supervisor_prompt import supervisor_prompt
from agents.researcher import researcher_agent
from agents.processor import processor_agent
import os
from tools.supervisor_agent_tools import create_pico_for_meta_analysis, create_handoff_tool
from datetime import datetime
from state.state import MetaAnalysisState
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

date_time = datetime.now().strftime("%Y-%m-%d")

# Create the main agent 
# Its function is to choose, via handoff, which agent should be called recursively
# Using only available agents: researcher_agent and processor_agent

supervisor_agent = create_supervisor (
    agents = [researcher_agent, processor_agent],
    tools = [
        create_handoff_tool(agent_name="researcher"),
        create_handoff_tool(agent_name="processor"),
        create_pico_for_meta_analysis,
    ],
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    ),
    prompt = supervisor_prompt,
    state_schema = MetaAnalysisState
).compile().with_config(recursion_limit=100)
