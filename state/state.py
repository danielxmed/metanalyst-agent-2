from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import operator

def replace_urls_to_process_limited(left: List[str], right: List[str]) -> List[str]:
    """
    Custom reducer for urls_to_process that replaces the list instead of adding to it,
    but limits to the last 100 URLs to prevent state overload.
    This allows the processor to remove processed URLs from the list.
    """
    # If right (new value) is provided, use it; otherwise keep left (current value)
    result = right if right is not None else left
    
    # Limit to last 100 URLs
    if result and len(result) > 100:
        result = result[-100:]
    
    return result if result is not None else []

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

def add_limited_urls(left: List[str], right: List[str]) -> List[str]:
    """
    Custom reducer for URL lists that maintains only the last 100 URLs.
    This prevents the state from being overloaded with too many URLs.
    """
    if right is None:
        return left if left is not None else []
    
    if left is None:
        combined = right
    else:
        combined = left + right
    
    # Keep only the last 100 URLs
    return combined[-100:] if len(combined) > 100 else combined

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

    # URLs to be processed
    urls_to_process: Annotated[List[str], replace_urls_to_process_limited]

    # URLs already processed
    processed_urls: Annotated[List[str], add_limited_urls]

    # Number of chunks retrieved and stored in data/retrieved_chunks directory
    retrieved_chunks_count: int

    # Results of the analyses performed
    analysis_results: Annotated[List[Dict[str, Any]], operator.add]

    # Iteration of the current draft
    current_draft_iteration: int

    # Reviewer feedbacks on drafts
    reviewer_feedbacks: Annotated[List[Dict[str, Any]], operator.add]
