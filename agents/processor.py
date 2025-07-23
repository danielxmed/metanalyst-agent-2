from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from state.state import MetaAnalysisState
from prompts.processor_prompt import processor_prompt
from tools.process_urls import process_urls
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

processor_agent = create_react_agent(
    model = model,
    tools = [process_urls],
    prompt = processor_prompt,
    name = "processor",
    state_schema = MetaAnalysisState,
)
