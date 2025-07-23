from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from state.state import MetaAnalysisState
import os
from dotenv import load_dotenv
from tools.retriever_agent_tools import retrieve_chunks
from prompts.retriever_prompt import retriever_prompt

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

retriever_agent = create_react_agent(
    model=model,
    tools=[retrieve_chunks],
    prompt=retriever_prompt,
    name="retriever",
    state_schema=MetaAnalysisState,
)