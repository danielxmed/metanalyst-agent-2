"""
Tools for the analyzer agent.

This module contains the tools used by the analyzer agent to:
1. Analyze the retrieved chunks using flexible structured output with Gemini
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
from datetime import datetime
import tempfile
import subprocess
import sys

# Modelo Pydantic simples e flexível para estruturar apenas o essencial
class AnalysisInsight(BaseModel):
    """An individual insight extracted from the analysis of the chunks."""
    
    insight: str = Field(description="The text of the extracted insight or conclusion")
    references: List[str] = Field(description="List of references supporting this insight (authors, studies, etc.)")

class FlexibleAnalysisOutput(BaseModel):
    """Flexible structured output from the analysis of the chunks."""
    
    insights: List[AnalysisInsight] = Field(description="List of insights extracted by the model")

@tool
def analyze_chunks(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Analisa todos os chunks recuperados usando Gemini com output estruturado flexível.
    
    O modelo tem liberdade total para decidir quais dados extrair e quais insights gerar.
    A estruturação serve apenas para indexar adequadamente as referências com os insights.
    
    Returns:
        Command: Comando para atualizar o estado do LangGraph
    """
    
    try:
        # Verificar se há chunks para analisar
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
        
        # Ler todos os arquivos JSON dos chunks
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
                    "insight": "No valid chunk found",
                    "references": ["System"],
                    "timestamp": "current", 
                    "analysis_type": "error"
                }]
            }
        
        # Configurar o modelo Gemini com output estruturado
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        
        # Aplicar schema de output estruturado flexível
        structured_llm = llm.with_structured_output(FlexibleAnalysisOutput)
        
        # Preparar o contexto
        pico_context = state.get("meta_analysis_pico", {})
        user_request = state.get("user_request", "")
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Prompt aprimorado com foco em rastreabilidade científica
        analysis_prompt = f"""
        You are a medical meta-analysis expert. Carefully analyze all provided scientific literature chunks.

        Today's date is {current_date}.

        META-ANALYSIS CONTEXT:
        - User request: {user_request}
        - Defined PICO: {json.dumps(pico_context, indent=2, ensure_ascii=False)}

        CHUNKS FOR ANALYSIS:
        {json.dumps(all_chunks_content, indent=2, ensure_ascii=False)}

        CRITICAL INSTRUCTIONS FOR SCIENTIFIC RIGOR:

        🔬 EXTRACT ANY RELEVANT INSIGHTS:
        - Quantitative data (sample sizes, effect sizes, p-values, confidence intervals)
        - Study characteristics (RCT design, patient populations, follow-up periods)
        - Clinical findings (efficacy, safety, outcomes)
        - Statistical measures (OR, RR, HR with 95% CI)
        - Methodological quality assessment
        - Comparative effectiveness data
        - Safety profiles and adverse events
        - Guidelines and recommendations

        📚 REFERENCE REQUIREMENTS (CRITICAL):
        For EVERY insight you generate, you MUST:
        1. Extract the EXACT reference information from each chunk's "reference" field
        2. Use the citation format provided in the chunk (author names, year, journal, etc.)
        3. If insight comes from multiple chunks, include ALL relevant references
        4. NEVER use generic references like "Study" or "Research" - always use the specific citation
        5. Ensure each insight is traceable to its original scientific source

        📊 SCIENTIFIC RIGOR:
        - Be specific with numbers, percentages, and statistical measures
        - Quote exact findings from the studies
        - Mention study designs and sample sizes
        - Include confidence intervals when available
        - Note any limitations or heterogeneity

        🎯 COMPREHENSIVE ANALYSIS:
        Extract insights covering:
        - Primary efficacy endpoints
        - Secondary outcomes
        - Safety and tolerability
        - Patient subgroups
        - Duration of treatment effects
        - Clinical practice implications

        REMEMBER: Every insight must be scientifically traceable through proper referencing!
        """
        
        # Executar análise estruturada
        structured_result = structured_llm.invoke(analysis_prompt)
        
        # Converter resultado para formato do estado
        formatted_results = []
        
        for insight_obj in structured_result.insights:
            formatted_results.append({
                "insight": insight_obj.insight,
                "references": insight_obj.references,
                "timestamp": "current",
                "analysis_type": "flexible_analysis"
            })
        
        # Mensagem de feedback sobre a análise
        analysis_message = ToolMessage(
            content=f"🔬 Analyzer Agent completed chunk analysis.\n"
                   f"Extracted insights: {len(formatted_results)}\n"
                   f"Chunks analyzed: {len(all_chunks_content)}\n"
                   f"PICO context: {pico_context.get('population', 'N/A')} | {pico_context.get('intervention', 'N/A')}",
            tool_call_id=tool_call_id
        )
        
        # Mensagem amigável do agente
        friendly_message = AIMessage(
            content=f"Chunk analysis completed! I extracted {len(formatted_results)} relevant insights "
                   f"from the {len(all_chunks_content)} available chunks. The insights include quantitative data, "
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
            content=f"❌ Error analyzing chunks: {str(e)}",
            tool_call_id=tool_call_id
        )
        
        return Command(
            update={
                "analysis_results": [{
                    "insight": f"Error analyzing chunks: {str(e)}",
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
    Executes Python code written by the model for advanced analyses.
    
    The model writes whatever code it deems necessary, having access to the state's analysis_results
    and the configured Python modules.
    
    Args:
        code: Python code written by the model to execute
    
    Returns:
        Command: Command to update the LangGraph state
    """
    
    try:
        # Get data from state
        analysis_results = state.get("analysis_results", [])
        pico_context = state.get("meta_analysis_pico", {})
        
        # Prepare Python environment with allowed modules and reference preservation
        setup_code = f"""
import json
import math
import statistics
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

# Data available for analysis
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

# Helper function to collect all unique references from analysis_results
def collect_all_references():
    \"\"\"Collects all unique references from existing analysis results\"\"\"
    all_refs = set()
    for result in analysis_results:
        if 'references' in result and result['references']:
            for ref in result['references']:
                if ref and ref not in ['Custom Python calculation', 'Python System', 'System']:
                    all_refs.add(ref)
    return list(all_refs)

# Helper function to get references from specific insights
def get_references_from_insights(insight_indices=None):
    \"\"\"Gets references from specific insights or all insights if indices not provided\"\"\"
    refs = set()
    if insight_indices is None:
        insight_indices = range(len(analysis_results))
    
    for i in insight_indices:
        if i < len(analysis_results) and 'references' in analysis_results[i]:
            for ref in analysis_results[i]['references']:
                if ref and ref not in ['Custom Python calculation', 'Python System', 'System']:
                    refs.add(ref)
    return list(refs)

# Function to create properly referenced insight
def create_insight_with_references(insight_text, source_insight_indices=None):
    \"\"\"Creates an insight with proper references from source insights\"\"\"
    references = get_references_from_insights(source_insight_indices)
    if not references:
        references = collect_all_references()
    if not references:
        references = ["Derived from analysis of retrieved literature"]
    
    return {{
        "insight": insight_text,
        "references": references,
        "timestamp": "current",
        "analysis_type": "python_calculation"
    }}

# Your custom code:
"""
        
        # Combine setup code with model code
        full_code = setup_code + "\n" + code
        
        # Execute code in a safe environment
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            # Execute the Python script
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=60,  # 60 seconds timeout
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
                           f"based on the {len(analysis_results)} insights available in the state.",
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
                    content=f"❌ Error executing Python code: {result.stderr}",
                    tool_call_id=tool_call_id
                )
                
                return Command(
                    update={
                        "analysis_results": [{
                            "insight": f"Error executing Python code: {result.stderr}",
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
            content=f"❌ General error executing Python: {str(e)}",
            tool_call_id=tool_call_id
        )
        
        return Command(
            update={
                "analysis_results": [{
                    "insight": f"General error executing Python: {str(e)}",
                    "references": ["Python System"],
                    "timestamp": "current",
                    "analysis_type": "python_error"
                }],
                "messages": [error_message]
            }
        )
