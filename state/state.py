from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import operator

def add_limited_queries(left: List[str], right: List[str]) -> List[str]:
    """
    Custom reducer for query lists that maintains only the last 20 queries.
    This prevents the state from being overloaded with too many queries.
    """
    if right is None:
        return left if left is not None else []
    
    if left is None:
        combined = right
    else:
        combined = left + right
    
    # Keep only the last 20 queries
    return combined[-20:] if len(combined) > 20 else combined

class MetaAnalysisState(TypedDict):
    """
    Represents the state of the meta-analysis graph.

    This state contains all the necessary information to execute
    the complete meta-analysis pipeline, from the definition of PICO elements
    to the production of the final document.
    """

    # Current iteration of the process
    current_iteration: int

    # History of exchanged messages
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Remaining steps (required by create_react_agent)
    remaining_steps: int

    # Original user request
    user_request: str

    # PICO elements defined for the meta-analysis
    meta_analysis_pico: Optional[Dict[str, str]]

    # Search queries already performed
    previous_search_queries: Annotated[List[str], add_limited_queries]

    # Retrieve queries already performed
    previous_retrieve_queries: Annotated[List[str], add_limited_queries]

    # Number of URLs currently queued for processing (stored in data/urls/urls_to_process.json)
    urls_to_process_count: int

    # Number of URLs already processed (stored in data/urls/processed_urls.json)
    processed_urls_count: int

    # Number of chunks retrieved and stored in data/retrieved_chunks directory
    retrieved_chunks_count: int

    # Results of the analyses performed
    analysis_results: Annotated[List[Dict[str, Any]], operator.add]

    # Iteration of the current draft
    current_draft_iteration: int

    # Reviewer feedbacks on drafts
    reviewer_feedbacks: Annotated[List[Dict[str, Any]], operator.add]
