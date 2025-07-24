from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from state.state import MetaAnalysisState
from prompts.analyzer_prompt import analyzer_prompt
from tools.analyzer_agent_tools import analyze_chunks, python_repl
import os
from dotenv import load_dotenv

# Loads environment variables from the .env file
load_dotenv()


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

analyzer_agent = create_react_agent(
    model = model,
    tools = [analyze_chunks, python_repl],
    prompt = analyzer_prompt,
    name = "analyzer",
    state_schema = MetaAnalysisState,
)
