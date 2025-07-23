from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from state.state import MetaAnalysisState
from prompts.processor_prompt import processor_prompt
from tools.process_urls import process_urls
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

processor_agent = create_react_agent(
    model = model,
    tools = [process_urls],
    prompt = processor_prompt,
    name = "processor",
    state_schema = MetaAnalysisState,
)
