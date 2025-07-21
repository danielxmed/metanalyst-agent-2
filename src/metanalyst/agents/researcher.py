"""
Researcher Agent for scientific literature search and URL collection.

This agent uses Tavily Search API to find relevant scientific literature
based on PICO framework and research criteria.
"""

from typing import Dict, Any, List
from tavily import TavilyClient

from .base import BaseAgent
from ..state import MetanalysisState
from ..config import LLMConfig, TavilyConfig


class ResearcherAgent(BaseAgent):
    """
    Researcher agent responsible for searching scientific literature.
    
    Uses Tavily Search API to find relevant publications, clinical trials,
    and other scientific resources based on the research question and PICO framework.
    """
    
    def __init__(self, name: str, llm, config: LLMConfig, tavily_config: TavilyConfig):
        """
        Initialize the researcher agent.
        
        Args:
            name: Agent name
            llm: Language model instance
            config: LLM configuration
            tavily_config: Tavily search configuration
        """
        super().__init__(name, llm, config)
        self.tavily_config = tavily_config
        
        # Initialize Tavily client
        if tavily_config.api_key:
            self.tavily_client = TavilyClient(api_key=tavily_config.api_key)
        else:
            self.tavily_client = None
            self.logger.warning("No Tavily API key provided")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the researcher agent."""
        return """
        You are a specialized researcher agent for medical meta-analysis. Your role is to:
        
        1. Analyze research questions and PICO frameworks
        2. Generate targeted search queries for scientific literature
        3. Search medical databases and repositories
        4. Collect relevant URLs and publication information
        5. Filter results based on inclusion/exclusion criteria
        
        Focus on:
        - High-quality peer-reviewed publications
        - Systematic reviews and meta-analyses
        - Randomized controlled trials (RCTs)
        - Clinical studies and case reports
        - Cochrane reviews
        
        Prioritize sources from:
        - PubMed/MEDLINE
        - Cochrane Library
        - Google Scholar
        - ClinicalTrials.gov
        - Medical journals and repositories
        
        Always consider the PICO framework when generating search strategies.
        """
    
    def execute(self, state: MetanalysisState) -> Dict[str, Any]:
        """
        Execute literature search based on current state.
        
        Args:
            state: Current metanalysis state
            
        Returns:
            Dictionary with search results and URLs
        """
        # Validate required state
        if not self.validate_state_requirements(state, ["pico"]):
            return {"error": "Missing required PICO framework"}
        
        # Generate search queries if not already present
        if not state["search_queries"]:
            search_queries = self._generate_search_queries(state)
        else:
            search_queries = state["search_queries"]
        
        # Perform searches
        urls_found = []
        search_results = {}
        
        for query in search_queries:
            try:
                results = self._search_literature(query)
                search_results[query] = results
                
                # Extract URLs from results
                for result in results:
                    if result.get("url") and result["url"] not in urls_found:
                        urls_found.append(result["url"])
                        
            except Exception as e:
                self.logger.error(
                    "Search failed for query",
                    query=query,
                    error=str(e)
                )
        
        # Filter URLs based on criteria
        filtered_urls = self._filter_urls(urls_found, state)
        
        self.logger.info(
            "Literature search completed",
            queries_executed=len(search_queries),
            urls_found=len(urls_found),
            urls_filtered=len(filtered_urls)
        )
        
        return {
            "search_queries": search_queries,
            "urls_to_process": filtered_urls,
            "search_results": search_results,
            "current_step": "literature_search_completed"
        }
    
    def _generate_search_queries(self, state: MetanalysisState) -> List[str]:
        """
        Generate targeted search queries based on PICO and research context.
        
        Args:
            state: Current metanalysis state
            
        Returns:
            List of search query strings
        """
        pico = state["pico"]
        
        # Create context for LLM
        context = f"""
        PICO Framework:
        - Population: {pico.get('population', 'Not specified')}
        - Intervention: {pico.get('intervention', 'Not specified')}
        - Comparison: {pico.get('comparison', 'Not specified')}
        - Outcome: {pico.get('outcome', 'Not specified')}
        
        Inclusion Criteria: {state.get('inclusion_criteria', [])}
        Exclusion Criteria: {state.get('exclusion_criteria', [])}
        Study Types: {state.get('study_types', [])}
        """
        
        task = """
        Generate 5-8 targeted search queries for scientific literature search.
        Each query should be optimized for medical databases and focus on different
        aspects of the research question. Include both broad and specific terms.
        
        Format: Return only the search queries, one per line.
        """
        
        prompt = self.create_llm_prompt(context, task)
        response = self.llm.invoke(prompt)
        
        # Parse queries from response
        queries = [
            line.strip() 
            for line in response.content.split('\n') 
            if line.strip() and not line.startswith('#')
        ]
        
        return queries[:8]  # Limit to 8 queries
    
    def _search_literature(self, query: str) -> List[Dict[str, Any]]:
        """
        Search literature using Tavily API.
        
        Args:
            query: Search query string
            
        Returns:
            List of search results
        """
        if not self.tavily_client:
            self.logger.warning("Tavily client not available, returning empty results")
            return []
        
        try:
            # Perform search with medical domain focus
            results = self.tavily_client.search(
                query=query,
                search_depth=self.tavily_config.search_depth,
                max_results=self.tavily_config.max_results,
                include_domains=self.tavily_config.include_domains
            )
            
            return results.get("results", [])
            
        except Exception as e:
            self.logger.error(
                "Tavily search failed",
                query=query,
                error=str(e)
            )
            return []
    
    def _filter_urls(self, urls: List[str], state: MetanalysisState) -> List[str]:
        """
        Filter URLs based on inclusion/exclusion criteria and quality.
        
        Args:
            urls: List of URLs to filter
            state: Current metanalysis state
            
        Returns:
            Filtered list of URLs
        """
        filtered_urls = []
        
        # Priority domains (higher quality sources)
        priority_domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "cochranelibrary.com", 
            "clinicaltrials.gov",
            "nejm.org",
            "bmj.com",
            "thelancet.com",
            "jamanetwork.com"
        ]
        
        # Separate high and low priority URLs
        high_priority = []
        low_priority = []
        
        for url in urls:
            if any(domain in url for domain in priority_domains):
                high_priority.append(url)
            else:
                low_priority.append(url)
        
        # Add high priority URLs first
        filtered_urls.extend(high_priority)
        
        # Add low priority URLs up to a limit
        max_total_urls = 50  # Configurable limit
        remaining_slots = max_total_urls - len(high_priority)
        
        if remaining_slots > 0:
            filtered_urls.extend(low_priority[:remaining_slots])
        
        return filtered_urls
    
    def should_execute(self, state: MetanalysisState) -> bool:
        """
        Determine if researcher should execute based on state.
        
        Args:
            state: Current metanalysis state
            
        Returns:
            True if should execute, False otherwise
        """
        # Execute if no search has been done yet or if PICO has been updated
        return (
            len(state.get("search_queries", [])) == 0 or
            len(state.get("urls_to_process", [])) == 0
        )
    
    def estimate_execution_time(self, state: MetanalysisState) -> int:
        """
        Estimate execution time based on search scope.
        
        Args:
            state: Current metanalysis state
            
        Returns:
            Estimated time in seconds
        """
        # Base time for search query generation
        base_time = 30
        
        # Additional time per search query
        query_count = len(state.get("search_queries", [])) or 6  # Default estimate
        search_time = query_count * 10  # 10 seconds per query
        
        return base_time + search_time