from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from state.state import MetaAnalysisState
from prompts.retriever_prompt import retriever_prompt
from tools.retriever_agent_tools import retrieve_chunks
import os
from dotenv import load_dotenv

# Loads environment variables from the .env file
load_dotenv()


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

retriever_agent = create_react_agent(
    model = model,
    tools = [retrieve_chunks],
    prompt = retriever_prompt,
    name = "retriever",
    state_schema = MetaAnalysisState,
)
