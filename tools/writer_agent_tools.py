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

IMPORTANT: If there is a "Current Draft" section provided, it means this is a revision. You must:
1. Read and understand the existing draft content
2. Apply the reviewer feedbacks to improve it
3. Keep the good parts while addressing the specific feedback points
4. Do NOT start completely from scratch - build upon the existing work

If no current draft exists, this is the first version - create it from scratch.

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
    try:
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=1
        )
        
        # Load all chunks from data/retrieved_chunks directory
        chunks_dir = "data/retrieved_chunks"
        chunks_data = []
        
        if os.path.exists(chunks_dir):
            chunk_files = glob.glob(os.path.join(chunks_dir, "*.json"))
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        chunks_data.append(chunk_data)
                except Exception as e:
                    print(f"Error loading chunk file {chunk_file}: {e}")
        
        # Prepare context for the model
        chunks_text = "\n\n---\n\n".join([
            f"Chunk ID: {chunk.get('id', 'N/A')}\n"
            f"Source: {chunk.get('source', 'N/A')}\n"
            f"Content: {chunk.get('content', '')}"
            f"Reference: {chunk.get('reference', '')}"
            for chunk in chunks_data
        ])
        
        # Get state information
        meta_analysis_pico = state.get('meta_analysis_pico', {})
        analysis_results = state.get('analysis_results', [])
        reviewer_feedbacks = state.get('reviewer_feedbacks', [])
        
        # Load existing draft if it exists
        draft_dir = "data/current_draft"
        draft_path = os.path.join(draft_dir, "current_draft.md")
        current_draft_content = ""
        
        if os.path.exists(draft_path):
            try:
                with open(draft_path, 'r', encoding='utf-8') as f:
                    current_draft_content = f.read()
            except Exception as e:
                print(f"Error loading existing draft: {e}")
        
        # Prepare the complete prompt for the model
        current_draft_section = f"""
## Current Draft (if exists):
{current_draft_content if current_draft_content else "No existing draft - this is the first version."}

""" if current_draft_content or reviewer_feedbacks else ""
        
        complete_prompt = f"""
{tool_prompt}

## PICO Elements:
{json.dumps(meta_analysis_pico, indent=2)}

## Analysis Results:
{json.dumps(analysis_results, indent=2)}

## Reviewer Feedbacks (if any):
{json.dumps(reviewer_feedbacks, indent=2)}

{current_draft_section}## Complete State (for transparency in methods section):
{json.dumps(state, indent=2, default=str)}

## Retrieved Chunks:
{chunks_text}

Please write the meta-analysis draft in markdown format.
"""
        
        # Call the model
        response = llm.invoke([HumanMessage(content=complete_prompt)])
        draft_content = response.content
        
        # Ensure the data/current_draft directory exists
        draft_dir = "data/current_draft"
        os.makedirs(draft_dir, exist_ok=True)
        
        # Save the draft to current_draft.md
        draft_path = os.path.join(draft_dir, "current_draft.md")
        with open(draft_path, 'w', encoding='utf-8') as f:
            f.write(draft_content)
        
        # Update current_draft_iteration
        current_iteration = state.get('current_draft_iteration', 0) + 1
        
        return Command(
            update={
                "current_draft_iteration": current_iteration,
                "messages": [ToolMessage(
                    content=f"Draft successfully written and saved to {draft_path}. Current draft iteration: {current_iteration}",
                    tool_call_id=tool_call_id
                )]
            }
        )
        
    except Exception as e:
        return Command(
            update={
                "messages": [ToolMessage(
                    content=f"Error writing draft: {str(e)}",
                    tool_call_id=tool_call_id
                )]
            }
        )
