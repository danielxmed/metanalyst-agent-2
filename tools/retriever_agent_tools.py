from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Annotated
import os


@tool
def retrieve_chunks(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Performs a semantic search on a local vector database for gathering chunks of
    referenced medical literature in order to make a meta-analysis.
    """
    
    
    