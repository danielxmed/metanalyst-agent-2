"""
Tools for the analyzer agent.

This module contains the tools used by the analyzer agent to:
1. Analyze retrieved chunks using flexible structured output with Gemini
2. Execute Python calculations dynamically written by the model
"""

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

# Simple and flexible Pydantic model to structure only the essentials
class AnalysisInsight(BaseModel):
    """An individual insight extracted from chunk analysis."""
    
    insight: str = Field(description="The insight text or extracted conclusion")
    references: List[str] = Field(description="List of references that support this insight (authors, studies, etc.)")

class FlexibleAnalysisOutput(BaseModel):
    """Flexible structured output from chunk analysis."""
    
    insights: List[AnalysisInsight] = Field(description="List of insights extracted by the model")

@tool
def analyze_chunks(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Analyzes all retrieved chunks using Gemini with flexible structured output.
    
    The model has complete freedom to decide which data to extract and which insights to generate.
    The structuring only serves to properly index references with insights.
    
    Returns:
        Command: Command to update the LangGraph state
    """
    
    try:
        # Check if there are chunks to analyze
        chunks_dir = "data/retrieved_chunks"
        if not os.path.exists(chunks_dir):
            return {
                "analysis_results": [{
                    "insight": "Chunks directory not found",
                    "references": ["System"],
                    "timestamp": "current",
                    "analysis_type": "error"
                }]
            }
        
        # Read all JSON files from chunks
        chunk_files = glob.glob(os.path.join(chunks_dir, "*.json"))
        if not chunk_files:
            return {
                "analysis_results": [{
                    "insight": "No chunks found for analysis",
                    "references": ["System"],
                    "timestamp": "current",
                    "analysis_type": "error"
                }]
            }
        
        all_chunks_content = []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    all_chunks_content.append(chunk_data)
            except Exception as e:
                print(f"Error reading chunk {chunk_file}: {e}")
                continue
        
        if not all_chunks_content:
            return {
                "analysis_results": [{
                    "insight": "No valid chunks found",
                    "references": ["System"],
                    "timestamp": "current", 
                    "analysis_type": "error"
                }]
            }
        
        # Configure Gemini model with structured output
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        
        # Apply flexible structured output schema
        structured_llm = llm.with_structured_output(FlexibleAnalysisOutput)
        
        # Prepare context
        pico_context = state.get("meta_analysis_pico", {})
        user_request = state.get("user_request", "")
        
        # Flexible prompt that gives freedom to the model
        analysis_prompt = f"""
        You are a medical meta-analysis expert. Carefully analyze all the scientific literature chunks provided.

        META-ANALYSIS CONTEXT:
        - User request: {user_request}
        - Defined PICO: {json.dumps(pico_context, indent=2, ensure_ascii=False)}

        CHUNKS FOR ANALYSIS:
        {json.dumps(all_chunks_content, indent=2, ensure_ascii=False)}

        INSTRUCTIONS:
        You have TOTAL FREEDOM to decide:
        - Which data is important to extract
        - Which calculations to perform
        - Which insights to generate
        - Which conclusions to draw
        - Which patterns to identify

        Extract any information you deem relevant for the meta-analysis, including but not limited to:
        - Quantitative data (sample sizes, effect sizes, p-values, etc.)
        - Study characteristics (types, quality, limitations)
        - Important clinical findings
        - Methodological patterns
        - Inconsistencies or gaps in the literature
        - Any other insight you consider valuable

        For each insight you generate, make sure to:
        - Properly reference the source (authors, studies, specific chunks)
        - Be specific and quantitative when possible
        - Include your critical analysis

        IMPORTANT: You decide what is important. Be creative and comprehensive in your analysis.
        """
        
        # Execute structured analysis
        structured_result = structured_llm.invoke(analysis_prompt)
        
        # Convert result to state format
        formatted_results = []
        
        for insight_obj in structured_result.insights:
            formatted_results.append({
                "insight": insight_obj.insight,
                "references": insight_obj.references,
                "timestamp": "current",
                "analysis_type": "flexible_analysis"
            })
        
        # Analysis feedback message
        analysis_message = ToolMessage(
            content=f"🔬 Analyzer Agent completed chunk analysis.\n"
                   f"Insights extracted: {len(formatted_results)}\n"
                   f"Chunks analyzed: {len(all_chunks_content)}\n"
                   f"PICO context: {pico_context.get('population', 'N/A')} | {pico_context.get('intervention', 'N/A')}",
            tool_call_id=tool_call_id
        )
        
        # Friendly agent message
        friendly_message = AIMessage(
            content=f"Chunk analysis completed! I extracted {len(formatted_results)} relevant insights "
                   f"from {len(all_chunks_content)} available chunks. The insights include quantitative data, "
                   f"methodological characteristics, and clinical conclusions based on the analyzed literature.",
            name="analyzer"
        )
        
        return Command(
            update={
                "analysis_results": formatted_results,
                "messages": [analysis_message, friendly_message]
            }
        )
        
    except Exception as e:
        error_message = ToolMessage(
            content=f"❌ Error in chunk analysis: {str(e)}",
            tool_call_id=tool_call_id
        )
        
        return Command(
            update={
                "analysis_results": [{
                    "insight": f"Error in chunk analysis: {str(e)}",
                    "references": ["System"],
                    "timestamp": "current",
                    "analysis_type": "error"
                }],
                "messages": [error_message]
            }
        )

@tool  
def python_repl(
    code: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Executes Python code written by the model for advanced analysis.
    
    The model writes the code it deems necessary, having access to analysis_results
    from the state and configured Python modules.
    
    Args:
        code: Python code written by the model to execute
    
    Returns:
        Command: Command to update the LangGraph state
    """
    
    try:
        # Get data from state
        analysis_results = state.get("analysis_results", [])
        pico_context = state.get("meta_analysis_pico", {})
        
        # Prepare Python environment with allowed modules
        setup_code = f"""
import json
import math
import statistics
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

# Available data for analysis
analysis_results = {json.dumps(analysis_results)}
pico_context = {json.dumps(pico_context)}

# Helper function to extract numbers from text
def extract_numbers(text):
    \"\"\"Extracts numbers from a string\"\"\"
    return [float(x) for x in re.findall(r'\\d+(?:\\.\\d+)?', text)]

# Helper function to count keywords
def count_keywords(texts, keywords):
    \"\"\"Counts keyword occurrences in a list of texts\"\"\"
    counts = {{}}
    for keyword in keywords:
        counts[keyword] = sum(1 for text in texts if keyword.lower() in text.lower())
    return counts

# Your custom code:
"""
        
        # Combine setup code with model code
        full_code = setup_code + "\n" + code
        
        # Execute code in secure environment
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            # Execute Python script
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                # Capture stdout as result
                output = result.stdout.strip()
                
                # Success message
                success_message = ToolMessage(
                    content=f"🐍 Python REPL executed successfully.\n"
                           f"Code executed: {len(code.split())} lines\n"
                           f"Data analyzed: {len(analysis_results)} insights",
                    tool_call_id=tool_call_id
                )
                
                # Friendly agent message
                friendly_message = AIMessage(
                    content=f"Python code executed successfully! I performed custom calculations "
                           f"based on {len(analysis_results)} insights available in the state.",
                    name="analyzer"
                )
                
                # Try to parse as JSON if possible, otherwise return as text
                try:
                    if output.startswith('[') or output.startswith('{'):
                        parsed_output = json.loads(output)
                        if isinstance(parsed_output, list):
                            return Command(
                                update={
                                    "analysis_results": parsed_output,
                                    "messages": [success_message, friendly_message]
                                }
                            )
                        else:
                            return Command(
                                update={
                                    "analysis_results": [{
                                        "insight": json.dumps(parsed_output, ensure_ascii=False),
                                        "references": ["Custom Python calculation"],
                                        "timestamp": "current",
                                        "analysis_type": "python_calculation"
                                    }],
                                    "messages": [success_message, friendly_message]
                                }
                            )
                    else:
                        return Command(
                            update={
                                "analysis_results": [{
                                    "insight": output,
                                    "references": ["Custom Python calculation"],
                                    "timestamp": "current",
                                    "analysis_type": "python_calculation"
                                }],
                                "messages": [success_message, friendly_message]
                            }
                        )
                except json.JSONDecodeError:
                    # If not JSON, return as text insight
                    return Command(
                        update={
                            "analysis_results": [{
                                "insight": output,
                                "references": ["Custom Python calculation"],
                                "timestamp": "current",
                                "analysis_type": "python_calculation"
                            }],
                            "messages": [success_message, friendly_message]
                        }
                    )
            else:
                error_message = ToolMessage(
                    content=f"❌ Error in Python code execution: {result.stderr}",
                    tool_call_id=tool_call_id
                )
                
                return Command(
                    update={
                        "analysis_results": [{
                            "insight": f"Error in Python code execution: {result.stderr}",
                            "references": ["Python System"],
                            "timestamp": "current", 
                            "analysis_type": "python_error"
                        }],
                        "messages": [error_message]
                    }
                )
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
        
    except Exception as e:
        error_message = ToolMessage(
            content=f"❌ General error in Python execution: {str(e)}",
            tool_call_id=tool_call_id
        )
        
        return Command(
            update={
                "analysis_results": [{
                    "insight": f"General error in Python execution: {str(e)}",
                    "references": ["Python System"],
                    "timestamp": "current",
                    "analysis_type": "python_error"
                }],
                "messages": [error_message]
            }
        )
