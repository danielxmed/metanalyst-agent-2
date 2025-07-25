from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from state.state import MetaAnalysisState
from prompts.reviewer_prompt import reviewer_prompt
import os
from dotenv import load_dotenv

# Loads environment variables from the .env file
load_dotenv()


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

reviewer_agent = create_react_agent(
    model = model,
    tools = [],
    prompt = reviewer_prompt,
    name = "reviewer",
    state_schema = MetaAnalysisState,
)
