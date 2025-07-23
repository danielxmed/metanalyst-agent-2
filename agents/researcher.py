from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from prompts.researcher_prompt import researcher_prompt
from tools.researcher_agent_tools import literature_search
from state.state import MetaAnalysisState
import os


ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="o3")

researcher_agent = create_react_agent(
    model = model,
    tools = [literature_search],
    prompt = researcher_prompt,
    name = "researcher",
    state_schema = MetaAnalysisState,
).with_config(recursion_limit=50)
