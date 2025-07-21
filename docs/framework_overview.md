# Metanalyst Agent Framework

## Overview

The **Metanalyst Agent** is a multi-system agent framework designed for autonomous meta-analysis generation. It implements a **supervisor-workers architecture** using LangGraph, where a central orchestrator coordinates specialized agents to perform different aspects of the meta-analysis process.

## Architecture

### Hub-and-Spoke Design

The system follows a "sun" architecture with a central orchestrator that invokes specialized agents as tools:

```
                    RESEARCHER
                         │
                         │
            EDITOR ──────┼────── PROCESSOR
                │        │        │
                │        │        │
    ANALYST ───┼────────●────────┼─── RETRIEVER
                │   ORCHESTRATOR  │
                │        │        │
                │        │        │
           REVIEWER ──────┼────── WRITER
                         │
                         │
                     (PROCESSOR combines
                      extraction + vectorization)

    ● = Central Orchestrator (Hub)
    │ = Direct Connections (Agents-as-a-Tool)
```

### Core Principles

1. **Central Hub**: Orchestrator maintains global state and decision logic
2. **Agents as Tools**: Each agent is a specialized tool invoked by the orchestrator
3. **Contextual Decision**: Orchestrator analyzes state and chooses next agent dynamically
4. **Direct Communication**: All agents communicate only with the orchestrator
5. **Shared State**: Single point of truth maintained by the orchestrator

## State Management

### MetanalysisState Schema

The central state contains all necessary information for the meta-analysis process:

```python
class MetanalysisState(TypedDict):
    # Core identification
    metanalysis_id: str
    
    # Communication and interaction
    human_messages: List[Dict[str, Any]]
    agents_messages: List[Dict[str, Any]]
    
    # Research and data collection
    urls_to_process: List[str]
    processed_urls: List[str]
    search_queries: List[str]
    
    # Analysis and insights
    objective_metrics: Dict[str, Any]
    insights: Dict[str, Any]
    feedbacks: Dict[str, Any]
    
    # Report generation
    report_drafts: Dict[str, Any]
    final_report_not_edited: Optional[str]
    final_report_edited: Optional[str]
    
    # Process control and metadata
    current_step: Optional[str]
    next_agent: Optional[str]
    status: str  # "initialized", "in_progress", "completed", "error", "paused"
    
    # PICO framework for medical research
    pico: Dict[str, str]  # Population, Intervention, Comparison, Outcome
    
    # Additional fields for methodology, quality assessment, etc.
```

### What's NOT in State

- **Full URL/publication contents**: Stored in PostgreSQL database
- **Vector embeddings**: Stored in vector store (FAISS)
- **Large binary data**: Handled by external storage systems

## Specialized Agents

### 1. Orchestrator Agent (Central Hub)
- **Role**: Conductor orchestrating the entire meta-analysis symphony
- **Responsibilities**:
  - Define and refine PICO of the research
  - Analyze current state and decide next agent
  - Manage global state and progress
  - Implement retry logic and quality control
  - Maintain decision history and justifications

### 2. Researcher Agent
- **Tools**: Tavily Search API
- **Responsibilities**:
  - Search scientific literature using specific domains
  - Generate queries based on PICO
  - Return list of candidate URLs
  - Filter results by relevance and quality

### 3. Processor Agent
- **Tools**: Tavily Extract API, OpenAI Embeddings, FAISS, GPT-4
- **Responsibilities**:
  - Extract full content from URLs using Tavily Extract
  - Process markdown to structured JSON
  - Generate Vancouver references
  - Create objective summaries with statistical data
  - Chunk scientific publications intelligently
  - Generate vector embeddings and store in FAISS

### 4. Retriever Agent
- **Tools**: FAISS, Cosine Similarity
- **Responsibilities**:
  - Search for relevant information using PICO
  - Return high-similarity chunks
  - Maintain reference context

### 5. Writer Agent
- **Responsibilities**:
  - Analyze retrieved chunks
  - Generate structured HTML report
  - Include methodology and results
  - Appropriately cite references

### 6. Reviewer Agent
- **Responsibilities**:
  - Review report quality
  - Generate improvement feedback
  - Suggest additional searches if necessary
  - Validate compliance with medical standards

### 7. Analyst Agent
- **Tools**: Matplotlib, Plotly, SciPy, NumPy
- **Responsibilities**:
  - Statistical analyses (meta-analysis, forest plots)
  - Generate graphs and tables
  - Calculate metrics (OR, RR, CI)
  - Create HTML visualizations

### 8. Editor Agent
- **Responsibilities**:
  - Integrate report + analyses
  - Generate final HTML
  - Structure final document
  - Ensure appropriate formatting

## Configuration Management

### Easy LLM Provider Switching

The framework supports multiple LLM providers with easy switching:

```python
from src.metanalyst.config import MetanalystConfig, LLMProvider

# Create configuration
config = MetanalystConfig.from_env()

# Switch all agents to OpenAI
config.update_llm_provider(LLMProvider.OPENAI, "gpt-4-turbo-preview")

# Switch all agents to Google
config.update_llm_provider(LLMProvider.GOOGLE, "gemini-pro")

# Switch all agents to Azure
config.update_llm_provider(LLMProvider.AZURE, "gpt-4")
```

### Supported Providers

- **Anthropic** (Default): Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **OpenAI**: GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
- **Google**: Gemini Pro, Gemini Pro Vision
- **Azure OpenAI**: GPT-4, GPT-3.5 Turbo

### Configuration Structure

```python
class MetanalystConfig:
    # Core system configuration
    system_name: str = "Metanalyst Agent"
    version: str = "0.1.0"
    debug: bool = False
    
    # Agent configurations (individual LLM configs for each agent)
    agents: AgentConfig
    
    # External service configurations
    database: DatabaseConfig  # PostgreSQL
    tavily: TavilyConfig      # Search API
    vectorstore: VectorStoreConfig  # FAISS/Chroma/Pinecone
    
    # Processing limits and timeouts
    max_concurrent_agents: int = 3
    agent_timeout: int = 300
    max_retries: int = 3
    
    # Human-in-the-loop settings
    require_human_approval: bool = True
    auto_approve_low_risk: bool = False
```

## Usage Examples

### Basic Setup

```python
import asyncio
from src.metanalyst.config import MetanalystConfig
from src.metanalyst.orchestrator import MetanalystOrchestrator
from src.metanalyst.state import create_initial_state

# 1. Create configuration from environment
config = MetanalystConfig.from_env()

# 2. Initialize orchestrator
orchestrator = MetanalystOrchestrator(config)

# 3. Define research question
human_request = "Meta-analysis on mindfulness interventions for anxiety"
pico = {
    "population": "Adults with anxiety disorders",
    "intervention": "Mindfulness-based interventions",
    "comparison": "Control groups",
    "outcome": "Anxiety reduction"
}

# 4. Run meta-analysis
final_state = await orchestrator.run_metanalysis(
    human_request=human_request,
    pico=pico
)
```

### Manual Agent Execution

```python
# Get specific agent
researcher = orchestrator.agent_registry.get_agent("researcher")

# Execute manually
state = create_initial_state("test-001", "Research question", pico)
result = researcher.execute(state)
```

### Configuration Validation

```python
from src.metanalyst.config import validate_config

config = MetanalystConfig.from_env()
issues = validate_config(config)

if issues:
    print("Configuration issues:")
    for issue in issues:
        print(f"  - {issue}")
```

## Persistence and Checkpointing

### PostgreSQL Integration

The framework uses PostgreSQL for:
- **State persistence**: Complete state saved at each step
- **Checkpointing**: Resume from any point in the process
- **Human-in-the-loop**: Pause for human feedback
- **Error recovery**: Retry from last successful step
- **Audit trail**: Complete history of decisions and actions

### Vector Store Integration

FAISS (or other vector stores) for:
- **Document embeddings**: Processed scientific papers
- **Semantic search**: Retrieve relevant information
- **Reference tracking**: Maintain source citations

## Development Guidelines

### Adding New Agents

1. Inherit from `BaseAgent`
2. Implement `get_system_prompt()` and `execute()` methods
3. Register in `AgentRegistry`
4. Add to orchestrator graph
5. Define dependencies in registry

```python
from src.metanalyst.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "You are a custom agent for..."
    
    def execute(self, state: MetanalysisState) -> Dict[str, Any]:
        # Custom logic here
        return {"custom_field": "value"}
```

### Testing

```bash
# Run basic framework tests
python tests/test_basic_framework.py

# Run with pytest for full test suite
pytest tests/

# Run example usage
python example_usage.py
```

### Environment Setup

1. Copy `.env.example` to `.env`
2. Add your API keys:
   - `ANTHROPIC_API_KEY`
   - `TAVILY_API_KEY`
   - `OPENAI_API_KEY` (for embeddings)
3. Configure PostgreSQL database
4. Install dependencies: `pip install -r requirements.txt`

## Future Development

### Planned Enhancements

1. **Full Agent Implementation**: Complete functionality for all agents
2. **Advanced Statistics**: Forest plots, heterogeneity analysis
3. **Quality Assessment**: GRADE, Cochrane risk of bias tools
4. **Web Interface**: Dashboard for monitoring and control
5. **API Endpoints**: RESTful API for external integration
6. **Distributed Processing**: Scale across multiple machines
7. **Advanced Human-in-the-Loop**: Rich feedback interfaces

### Extensibility

The framework is designed for easy extension:
- **New LLM Providers**: Add to `llm_factory.py`
- **New Search Engines**: Extend researcher agent
- **New Vector Stores**: Update processor/retriever agents
- **Custom Workflows**: Modify orchestrator decision logic
- **Additional Output Formats**: Extend writer/editor agents

## Architecture Benefits

1. **Modularity**: Each agent is independent and replaceable
2. **Scalability**: Easy to distribute agents across machines
3. **Flexibility**: Dynamic routing based on current state
4. **Reliability**: Checkpointing and error recovery
5. **Observability**: Complete audit trail and logging
6. **Maintainability**: Clear separation of concerns
7. **Testability**: Each agent can be tested independently

## Conclusion

The Metanalyst Agent framework provides a solid foundation for autonomous meta-analysis generation. The hub-and-spoke architecture with LangGraph ensures scalability, reliability, and maintainability while making it easy to swap LLM providers and extend functionality.

The framework is ready for development of the actual agent implementations, with all the necessary infrastructure in place for state management, configuration, persistence, and orchestration.