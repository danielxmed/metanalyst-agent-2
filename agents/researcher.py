from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from prompts.researcher_prompt import researcher_prompt
from tools.researcher_agent_tools import literature_search
from state.state import MetaAnalysisState
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

researcher_agent = create_react_agent(
    model = model,
    tools = [literature_search],
    prompt = researcher_prompt,
    name = "researcher",
    state_schema = MetaAnalysisState,
)
