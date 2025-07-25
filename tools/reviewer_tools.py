from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from typing import Annotated, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from datetime import datetime

review_draft_prompt = """
You are a metana-analysis scientific reviewer. 
Your responsability is to review the draft of the meta-analysis given to you and provide feedbacks to the writers in order to improve it. You will also be given the metanalysis_pico elements and previous feedbacks that you gave.
Provide feedbacks in structured format, as a json object, in the following format:
{
    "feedback": [
        {
            "feedback": "feedback",
            "feedback_type": "feedback_type",
        },
        {
            "feedback": "feedback",
            "feedback_type": "feedback_type",
        },
        ...
    ]
    "is_complete": True or False
}

The final goal is to make the meta-analysis robust, scientific accurate, not very short, analytical, well-structured, complete and with enough references.
Check the previous feedbacks and make sure you don't repeat the same feedbacks.
Along with the feedbacks, you will return an object that represents if the meta-analysis is complete or not. If you decide it`s complete, you will return a boolean value of True, otherwise, you will return a boolean value of False. This is important for the supervisor agent to know when to stop iterating.

The final ouput is something like this:
{
    "feedback": [
        {
            "feedback": "feedback",
            "feedback_type": "feedback_type",
        },
        {
            "feedback": "feedback",
            "feedback_type": "feedback_type",
        },
        ...
    ],
    "is_complete": True
}

The system`s architectecture is a multi-agents supervisor-workers workflow:

- Supervisor: This agent is responsible for coordinating the team and making sure the meta-analysis is complete.
- Researcher: This agent is responsible for gathering URLs from the medical literature to help the team make a meta-analysis for the given PICO elements.
- Retriever: This agent is responsible for retrieving vectorized chunks of medical literature that are semantically relevant to the meta-analysis.
- Analyzer: This agent is responsible for analyzing the chunks of medical literature and providing feedback to the writer agent to improve the meta-analysis.
- Writer: This agent is responsible for writing the meta-analysis.
- Reviewer: This agent is responsible for reviewing the meta-analysis and providing feedback to the writer agent to improve it.
- Editor: This agent is responsible for taking the current_draft after its completion (after reviewer says it's complete) and writing a robust html with tables, graphs, etc. It is supposed to be the last agent to run. Stores the final_draft.html in the data/final_draft directory.

"""

# Pydantic models for structured output
class ReviewFeedback(BaseModel):
    """Individual feedback item from the reviewer"""
    feedback: str = Field(description="The specific feedback or suggestion for improvement")
    feedback_type: str = Field(description="The category/type of feedback (e.g., 'methodology', 'analysis', 'writing', 'structure', 'references')")

class ReviewResult(BaseModel):
    """Complete review result with feedback and completion status"""
    feedback: List[ReviewFeedback] = Field(description="List of feedback items for improving the meta-analysis")
    is_complete: bool = Field(description="Whether the meta-analysis is complete and ready for final editing")


@tool
def review_draft(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Uses gemini-2.5-pro to review the draft of the meta-analysis and provide feedbacks to the writers in order to improve it.
    """
    try:
        # Initialize Gemini model with structured output
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3  # Lower temperature for more consistent reviews
        )
        
        # Configure model for structured output
        structured_llm = llm.with_structured_output(ReviewResult)
        
        # Load current draft
        draft_dir = "data/current_draft"
        draft_path = os.path.join(draft_dir, "current_draft.md")
        current_draft_content = ""
        
        if os.path.exists(draft_path):
            try:
                with open(draft_path, 'r', encoding='utf-8') as f:
                    current_draft_content = f.read()
            except Exception as e:
                return Command(
                    update={
                        "messages": [ToolMessage(
                            content=f"Error loading current draft: {str(e)}",
                            tool_call_id=tool_call_id
                        )]
                    }
                )
        else:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content="No current draft found to review",
                        tool_call_id=tool_call_id
                    )]
                }
            )
        
        # Get state information
        meta_analysis_pico = state.get('meta_analysis_pico', {})
        reviewer_feedbacks = state.get('reviewer_feedbacks', [])
        messages = state.get('messages', [])
        
        # Prepare the complete prompt for the model
        complete_prompt = f"""
{review_draft_prompt}

## PICO Elements:
{json.dumps(meta_analysis_pico, indent=2)}

## Previous Reviewer Feedbacks (avoid repeating these):
{json.dumps(reviewer_feedbacks, indent=2)}

## Recent Messages/Context:
{json.dumps([str(msg) for msg in messages[-10:]], indent=2)}

## Current Draft to Review:
{current_draft_content}

PlAease provide structured feedback for this meta-analysis draft.
"""
        
        # Call the model with structured output
        review_result = structured_llm.invoke([HumanMessage(content=complete_prompt)])
        
        # Convert the structured result to dictionary format for state storage
        feedback_dict = {
            "timestamp": datetime.now().isoformat(),
            "feedback": [
                {
                    "feedback": item.feedback,
                    "feedback_type": item.feedback_type
                }
                for item in review_result.feedback
            ],
            "is_complete": review_result.is_complete
        }
        
        # Prepare summary message
        feedback_count = len(review_result.feedback)
        completion_status = "COMPLETE" if review_result.is_complete else "NEEDS IMPROVEMENT"
        
        summary_message = f"""Review completed! Status: {completion_status}
        
Generated {feedback_count} feedback items:
{chr(10).join([f"- [{fb.feedback_type}] {fb.feedback}" for fb in review_result.feedback[:3]])}
{'...' if feedback_count > 3 else ''}

Meta-analysis is {'ready for final editing' if review_result.is_complete else 'not yet complete and needs further iterations'}.
"""
        
        tool_message = ToolMessage(
            content=summary_message,
            tool_call_id=tool_call_id
        )
        
        return Command(
            update={
                "reviewer_feedbacks": [feedback_dict],
                "is_complete": review_result.is_complete,
                "messages": [tool_message]
            }
        )
        
    except Exception as e:
        error_message = ToolMessage(
            content=f"Error during review process: {str(e)}",
            tool_call_id=tool_call_id
        )
        
        return Command(
            update={
                "is_complete": False,  # Set to False on error
                "messages": [error_message]
            }
        )