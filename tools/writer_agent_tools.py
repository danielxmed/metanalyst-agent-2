import os
import json
import glob
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from typing import Annotated
from state.state import MetaAnalysisState
import tempfile
import subprocess
import sys

tool_prompt= """
You are a scientific writer.
Your responsability is, for the given PICO, chunks of scientific literature and the results of the analyzer agent, to write a draft of the meta-analysis. 
The draft should be written in a markdown format and saved in a file called current_draft.md in data/current_draft directory. 
You will recieve the metanalysis_pico from the state, just like analysis_results. The chunks will be given to you as a list of JSON files directly from the data/retrieved_chunks directory.
You you use mostly the chunks, metanalysis_pico and analysis_results to write the draft, but you wil recieve the whole state.
You must write the meta-analysis in the best way possible with the given information. This means there is no rigid structure to follow - you must use your judgment to chose the topics and subtopics.
Some structures are required, though:
- Introduction: Contextualize the topic and the PICO elements
- Methods: You will read the state and use the information for explaining how the meta-analysis was conducted, being totally transparent about the automated nature of the process. Cite the queries used, the agent calling sequence, everything.
- Results: Use mostly the analysis_results to write the results section. You must prioritize the objective metrics and calculations. Use tables as needed and try the plot the data in a way that can be turned into charts and graphs, wich will be done by the editor agent. Do not plan anything too much graphically complex, as you don`t know how much can the editor agent handle.
- Discussion: In here, you can use not only the analysis_results, but also you own conclusions and interpretations. You can also use the metanalysis_pico to guide your discussion.
- Limitations: Again, be transparent. This is a automated process made mostly by many LLMs, wich can hallucinate, also, the data gathering was not very traditional, as we used agentic web scrapping for the literature review. The whole process may be seen as kind of a "black box" for the reader.
- Conclusion: Summarize the main findings and conclusions.
- References (APA style): Use the references from the chunks and the metanalysis_pico. In the text body itself, you should cite every citable information in APA style. Exemple:  Bla blab lbla bla[Nobrega, et al, 2025], buh reh beuear [Marx, 2024]. Hub nuir tenga puh [Xui, 2022][Coxes, 2023]

You MUST return ONLY the draft, in markdown format, without any other text or comments.
If there are reviewer_feedbacks in the state, you MUST use them to improve the draft. Do not ignore them.

"""


@tool
def write_draft(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Write a draft of the meta-analysis based on the chunks and the results of the Analyzer agent.
    """
    return Command(update={"current_draft": "Draft written"})