from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from state.state import MetaAnalysisState
import os
from dotenv import load_dotenv


load_dotenv()


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

writer_agent = create_react_agent(
    model=model,
    tools=[write_draft],
    prompt=writer_prompt,
    name="writer",
    state_schema=MetaAnalysisState,
)