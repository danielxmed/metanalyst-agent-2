from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import operator

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
    previous_search_queries: Annotated[List[str], operator.add]

    # Retrieve queries already performed
    previous_retrieve_queries: Annotated[List[str], operator.add]

    # URLs to be processed
    urls_to_process: Annotated[List[str], operator.add]

    # URLs already processed
    processed_urls: Annotated[List[str], operator.add]

    # Chunks retrieved from documents
    retrieved_chunks: Annotated[List[Dict[str, Any]], operator.add]

    # Results of the analyses performed
    analysis_results: Annotated[List[Dict[str, Any]], operator.add]

    # Current draft of the document
    current_draft: Optional[str]

    # Iteration of the current draft
    current_draft_iteration: int

    # Reviewer feedbacks on drafts
    reviewer_feedbacks: Annotated[List[Dict[str, Any]], operator.add]

    # Final approved document
    final_draft: Optional[str]
