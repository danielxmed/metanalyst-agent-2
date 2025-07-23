from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from state.state import MetaAnalysisState
from prompts.processor_prompt import processor_prompt
from tools.process_urls import process_urls
import os


ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="o3")

processor_agent = create_react_agent(
    model = model,
    tools = [process_urls],
    prompt = processor_prompt,
    name = "processor",
    state_schema = MetaAnalysisState,
).with_config(recursion_limit=50)