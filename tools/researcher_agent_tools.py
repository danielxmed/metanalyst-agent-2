from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Annotated
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.url_manager import add_urls_to_process, get_urls_to_process_count, get_processed_urls_count


@tool
def literature_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Performs scientific literature search using Tavily API with focus on medical journals.

    This tool searches for scientific articles in reliable medical databases using the
    Tavily API. Returns URLs of relevant articles formatted as text.
    
    Args:
        query (str): Search query for scientific literature

    Returns:
        str: Search results formatted as text
    """
    
    # List of medical journal and scientific database domains
    medical_domains = [
        "pubmed.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov/pmc", 
        "www.cochranelibrary.com",
        "lilacs.bvsalud.org",
        "scielo.org",
        "www.embase.com",
        "www.webofscience.com",
        "www.scopus.com",
        "www.epistemonikos.org",
        "www.ebscohost.com",
        "www.tripdatabase.com",
        "pedro.org.au",
        "doaj.org",
        "scholar.google.com",
        "clinicaltrials.gov",
        "apps.who.int/trialsearch",
        "www.clinicaltrialsregister.eu",
        "www.isrctn.com",
        "www.thelancet.com",
        "www.nejm.org",
        "jamanetwork.com",
        "www.bmj.com",
        "www.nature.com/nm",
        "www.acpjournals.org/journal/aim",
        "journals.plos.org/plosmedicine",
        "www.jclinepi.com",
        "systematicreviewsjournal.biomedcentral.com",
        "ascopubs.org/journal/jco",
        "www.ahajournals.org/journal/circ",
        "www.gastrojournal.org",
        "academic.oup.com/eurheartj",
        "www.archives-pmr.org",
        "www.jacc.org",
        "www.scielo.br",
        "nejm.org",
        "thelancet.com",
        "bmj.com",
        "acpjournals.org/journal/aim",
        "cacancerjournal.com",
        "nature.com/nm",
        "cell.com/cell-metabolism/home",
        "thelancet.com/journals/langlo/home",
        "cochranelibrary.com",
        "memorias.ioc.fiocruz.br",
        "scielo.br/j/csp/",
        "cadernos.ensp.fiocruz.br",
        "scielo.br/j/rsp/",
        "scielo.org/journal/rpsp/",
        "journal.paho.org",
        "rbmt.org.br",
        "revistas.usp.br/rmrp",
        "ncbi.nlm.nih.gov/pmc",
        "scopus.com",
        "webofscience.com",
        "bvsalud.org",
        "jbi.global",
        "tripdatabase.com",
        "gov.br",
        "droracle.ai",
        "wolterskluwer.com",
        "semanticscholar.org",
        "globalindexmedicus.net",
        "sciencedirect.com"
    ]
    
    try:
        # Import and configure Tavily client
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content="Error: TAVILY_API_KEY not found in environment variables.",
                        tool_call_id=tool_call_id
                    )]
                }
            )
        
        # Initialize Tavily client
        tavily_client = TavilyClient(api_key=api_key)
        
        # Perform search with specified parameters
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=20,
            include_domains=medical_domains
        )
        
        # Extract URLs from results
        urls = []
        if "results" in response:
            for result in response["results"]:
                if "url" in result:
                    urls.append(result["url"])
        
        # Add URLs to JSON file with deduplication
        urls_added = add_urls_to_process(urls)

        # Get current counts for reporting
        current_to_process_count = get_urls_to_process_count()
        current_processed_count = get_processed_urls_count()
        
        # Create informative message about the search performed
        search_message = ToolMessage(
            content=f"🔍 Researcher Agent executed scientific literature search.\n"
                   f"Query: '{query}'\n" 
                   f"URLs found: {len(urls)}\n"
                   f"URLs added (after deduplication): {urls_added}\n"
                   f"Domains searched: medical journals and scientific databases",
            tool_call_id=tool_call_id
        )
        
        # ADICIONA MENSAGEM AMIGÁVEL DO AGENTE (para evitar Issue #5548)
        from langchain_core.messages import AIMessage
        friendly_message = AIMessage(
            content=f"Completei a busca por literatura científica sobre '{query}'. "
                   f"Encontrei {len(urls)} artigos e adicionei {urls_added} novos URLs únicos. "
                   f"Total de URLs para processar: {current_to_process_count}. "
                   f"Pronto para processar os artigos ou fazer mais buscas se necessário.",
            name="researcher"
        )
        
        # Return command to update state with counters
        return Command(
            update={
                "urls_to_process_count": current_to_process_count,
                "processed_urls_count": current_processed_count,
                "previous_search_queries": [query],  # Add single query, let reducer handle limiting
                "messages": [search_message, friendly_message]  # ToolMessage + AIMessage amigável
            }
        )
        
    except ImportError:
        return Command(
            update={
                "messages": [ToolMessage(
                    content="Error: tavily-python library not installed. Run: pip install tavily-python",
                    tool_call_id=tool_call_id
                )]
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [ToolMessage(
                    content=f"Error in literature search: {str(e)}",
                    tool_call_id=tool_call_id
                )]
            }
        )
