from langchain_anthropic import ChatOpenAI
from langgraph_supervisor import create_supervisor
from prompts.supervisor_prompt import supervisor_prompt
from agents.researcher import researcher_agent
import os
from tools.supervisor_agent_tools import create_pico_for_meta_analysis, create_handoff_tool
from datetime import datetime

date_time = datetime.now().strftime("%Y-%m-%d")

ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Create the main agent 
# Its function is to choose, via handoff, which agent should be called recursively

supervisor_agent = create_supervisor (
    agents = [researcher_agent, processor_agent, retriever_agent, analyzer_agent, writer_agent, reviewer_agent, editor_agent],
    tools = [create_handoff_tool(agent_name="researcher_agent"), create_handoff_tool(agent_name="processor_agent"), create_handoff_tool(agent_name="retriever_agent"), create_handoff_tool(agent_name="analyzer_agent"), create_handoff_tool(agent_name="writer_agent"), create_handoff_tool(agent_name="reviewer_agent"), create_handoff_tool(agent_name="editor_agent"), create_pico_for_meta_analysis()],
    model = ChatOpenAI(model="o3"),
    prompt = supervisor_prompt,
    name = "supervisor",
    add_handoff_back_messages = True
).with_config(recursion_limit=100)
