from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from typing import Annotated, List, Dict, Any
import os
import json
from datetime import datetime


@tool
def review_draft():
    """
    Review the draft of the meta-analysis and provide feedback to the supervisor and the writer agent to improve the meta-analysis.
    """
    return "I'm a reviewer tool"