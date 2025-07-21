# Metanalyst Agent System Architecture

## Overview

The Metanalyst Agent System is a multi-agent framework designed for autonomous meta-analysis generation. It implements a **supervisor-workers architecture** where a central orchestrator coordinates specialized agents to perform different aspects of the meta-analysis workflow.

## Core Architecture

### Hub-and-Spoke Design

The system follows a "sun" architecture pattern:

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
                     (HUMAN APPROVAL)

    ● = Central Orchestrator (Hub)
    │ = Direct Connections (Handoff Tools)
```

### Key Principles

1. **Central Control**: The orchestrator maintains global state and decision logic
2. **Agents as Tools**: Each specialized agent is invoked as a tool by the orchestrator
3. **Contextual Decisions**: The orchestrator analyzes state to choose the next agent
4. **Single Source of Truth**: All state is managed centrally
5. **Dynamic Workflow**: No fixed pipeline - decisions are made dynamically

## System Components

### 1. State Management (`metanalyst.core.state`)

The state system maintains comprehensive information throughout the meta-analysis process:

```python
class MetanalysisState(TypedDict):
    # Core identification
    metanalysis_id: str
    
    # Communication history
    human_messages: List[HumanMessage]
    agents_messages: List[AgentMessage]
    
    # URL processing
    urls_to_process: List[URLToProcess]
    processed_urls: List[ProcessedURL]
    
    # Analysis data
    objective_metrics: List[ObjectiveMetric]
    insights: List[Insight]
    
    # PICO framework
    pico_population: Optional[str]
    pico_intervention: Optional[str]
    pico_comparison: Optional[str]
    pico_outcome: Optional[str]
    
    # Process control
    current_step: str
    next_agent: Optional[str]
    process_status: str
    # ... and more
```

**Key Features:**
- **Comprehensive**: Captures all aspects of the meta-analysis process
- **Persistent**: Stored in PostgreSQL with full history
- **Structured**: Uses Pydantic models for type safety
- **Reducers**: Supports list accumulation with `add` operator

### 2. Configuration System (`metanalyst.core.config`)

Flexible configuration supporting multiple LLM providers and environments:

```python
class MetanalystConfig:
    llm: LLMConfig              # LLM provider and settings
    database: DatabaseConfig    # PostgreSQL connection
    tavily: TavilyConfig       # Search API configuration
    vector_store: VectorStoreConfig  # Embeddings settings
    system: SystemConfig        # General system settings
```

**Supported LLM Providers:**
- **Anthropic** (default): Claude 3.5 Sonnet
- **OpenAI**: GPT-4 and variants
- **Azure OpenAI**: Enterprise deployments
- **Google**: Gemini models

### 3. Orchestrator Agent (`metanalyst.core.orchestrator`)

The central coordinator that manages the entire workflow:

**Responsibilities:**
- Analyze current state and context
- Decide which agent should handle the next step
- Coordinate handoffs between agents
- Request human feedback when needed
- Maintain workflow continuity

**Key Methods:**
- `start_metanalysis()`: Initialize new meta-analysis
- `continue_metanalysis()`: Resume existing process
- `get_metanalysis_status()`: Check current status
- `pause_metanalysis()`: Pause for human review

### 4. Handoff System (`metanalyst.agents.handoffs`)

The communication mechanism between agents:

```python
class HandoffPayload:
    target_agent: HandoffTarget
    task_description: str
    priority: int
    context: Dict[str, Any]
    expected_output: Optional[str]
    timeout: Optional[int]
```

**Available Handoff Tools:**
- `researcher_handoff`: Literature search and query generation
- `processor_handoff`: Content extraction and processing
- `retriever_handoff`: Vector similarity search
- `writer_handoff`: Report generation
- `reviewer_handoff`: Quality control and validation
- `analyst_handoff`: Statistical analysis and visualization
- `editor_handoff`: Final document assembly
- `human_approval_handoff`: Human-in-the-loop workflows

## Workflow Stages

### 1. Initialization
- Parse human request
- Define PICO framework
- Set up initial state

### 2. Literature Search
- Generate search queries
- Find relevant publications
- Collect URLs for processing

### 3. Content Processing
- Extract content from URLs
- Generate embeddings
- Structure data for analysis

### 4. Analysis Phase
- Statistical meta-analysis
- Content analysis and synthesis
- Generate insights and findings

### 5. Report Generation
- Create structured reports
- Include visualizations
- Format for different outputs

### 6. Quality Review
- Validate methodology
- Check compliance with standards
- Request human feedback

### 7. Finalization
- Integrate all components
- Generate final report
- Mark process as complete

## Persistence and Memory

### PostgreSQL Integration
- **Checkpointing**: Full state persistence at each step
- **Thread Management**: Multiple concurrent meta-analyses
- **History Tracking**: Complete audit trail
- **Recovery**: Resume from any point in the process

### Vector Storage
- **Embeddings**: Text chunks with semantic search
- **FAISS Integration**: Fast similarity search
- **Reference Tracking**: Maintain source attribution

## Human-in-the-Loop

### Interaction Points
- **Critical Decisions**: PICO framework definition
- **Quality Gates**: Report review and approval
- **Error Recovery**: Manual intervention when needed
- **Guidance**: Direction for complex cases

### Interrupt System
- **Pause Execution**: Wait for human input
- **Resume Capability**: Continue from exact point
- **Timeout Handling**: Automatic fallbacks

## Scalability and Reliability

### Concurrent Processing
- **Multiple Agents**: Parallel execution where possible
- **Resource Management**: Configurable limits
- **Load Balancing**: Distribute work appropriately

### Error Handling
- **Retry Logic**: Automatic retry with backoff
- **Graceful Degradation**: Continue with partial results
- **Error Recovery**: Resume from last successful checkpoint

### Monitoring
- **State Tracking**: Complete visibility into process
- **Logging**: Comprehensive audit trail
- **Metrics**: Performance and quality indicators

## Extension Points

### Adding New Agents
1. Define agent-specific tools and capabilities
2. Create handoff tool for orchestrator
3. Implement agent logic and integration
4. Update configuration and documentation

### Custom Workflows
1. Define workflow-specific handoff tools
2. Implement conditional routing logic
3. Add workflow templates and patterns
4. Test and validate new workflows

## Security Considerations

### API Key Management
- Environment variable configuration
- Secure key rotation support
- Provider-specific authentication

### Data Protection
- Encrypted state storage options
- Secure database connections
- Audit trail maintenance

### Access Control
- Human approval requirements
- Role-based permissions (future)
- Secure API endpoints (future)

## Future Enhancements

### Planned Features
1. **Web Interface**: User-friendly dashboard
2. **API Server**: RESTful API for integration
3. **Advanced Analytics**: Process optimization
4. **Template System**: Pre-configured workflows
5. **Collaboration**: Multi-user support

### Scalability Improvements
1. **Distributed Processing**: Multi-node deployment
2. **Queue Management**: Advanced task scheduling
3. **Caching**: Intelligent result caching
4. **Performance Monitoring**: Real-time metrics

This architecture provides a solid foundation for autonomous meta-analysis generation while maintaining flexibility for future enhancements and customizations.