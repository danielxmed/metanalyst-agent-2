from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from prompts.researcher_prompt import researcher_prompt
from tools.researcher_agent_tools import literature_search
from state.state import MetanalysisState
import os


ChatAnthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
model = ChatAnthropic(model="claude-sonnet-4-20250514")

researcher_agent = create_react_agent(
    model = model,
    tools = [literature_search],
    prompt = researcher_prompt,
    name = "researcher",
    state_schema = MetanalysisState,
).with_config(recursion_limit=10)
