from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from prompts.researcher_prompt import researcher_prompt
from tools.researcher_agent_tools import literature_search
from state.state import MetaAnalysisState
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

researcher_agent = create_react_agent(
    model = model,
    tools = [literature_search],
    prompt = researcher_prompt,
    name = "researcher",
    state_schema = MetaAnalysisState,
)
