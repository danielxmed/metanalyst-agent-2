# 🩺 Meta-Analyst Agent 2

An intelligent multi-agent system for automated generation of medical meta-analyses using a supervisor-based architecture.

## 📋 Overview

Meta-Analyst Agent 2 is an open-source project that automates the complex process of conducting medical meta-analyses through a sophisticated multi-agent system. This system orchestrates specialized AI agents to handle each step of the meta-analysis workflow, from literature search to final HTML report generation.

The system is designed to handle the full pipeline of meta-analysis creation, including PICO framework definition, literature search, content processing, statistical analysis, and report writing - all coordinated by a supervisor agent that intelligently manages the workflow.

## 🏗️ Architecture

### Supervisor-Based Multi-Agent System

The core of the system is a **Supervisor Agent** that coordinates a team of specialized agents, each responsible for different aspects of the meta-analysis process:

#### 🎯 **Supervisor Agent**
- **Role**: Orchestrates the entire workflow and decides which agent to call next
- **Responsibilities**: 
  - Defines PICO (Population, Intervention, Comparison, Outcome) elements
  - Manages state transitions between agents
  - Ensures optimal workflow progression within 100 iterations
- **Intelligence**: Adapts workflow based on current state and results quality

#### 🔍 **Researcher Agent**
- **Role**: Literature search and URL collection
- **Tools**: Tavily search integration
- **Output**: Curated list of relevant research paper URLs

#### ⚙️ **Processor Agent**
- **Role**: Content extraction and vectorization
- **Functions**: 
  - Extracts content from research URLs
  - Chunks content for optimal processing
  - Creates vector embeddings
  - Stores in vectorstore for retrieval

#### 📚 **Retriever Agent**
- **Role**: Context-aware content retrieval
- **Method**: Retrieves relevant chunks based on PICO criteria
- **Output**: Focused content segments for analysis

#### 📊 **Analyzer Agent**
- **Role**: Statistical analysis and metrics calculation
- **Capabilities**:
  - Calculates Odds Ratios (OR)
  - Computes Risk Ratios (RR)
  - Generates statistical insights
  - Stores quantitative results

#### ✍️ **Writer Agent**
- **Role**: Meta-analysis draft creation
- **Output**: Structured markdown meta-analysis draft
- **Integration**: Uses analyzer results and retrieved content

#### 🔍 **Reviewer Agent**
- **Role**: Quality assurance and feedback
- **Functions**:
  - Reviews writer drafts for completeness
  - Provides actionable feedback
  - Signals when draft quality is sufficient

#### 🎨 **Editor Agent**
- **Role**: Final report generation
- **Output**: Polished HTML version of meta-analysis
- **Trigger**: Called only when reviewer approves draft

### Intelligent Workflow Management

The supervisor doesn't follow a rigid sequence but intelligently decides the next step based on:
- Current state analysis
- Content quality assessment
- Token limit management (200,000 tokens)
- Iteration efficiency (100 iterations max)

**Key Decision Points:**
- Always starts with PICO definition
- Calls Researcher when URLs are needed
- Activates Processor when URLs need processing
- Engages Retriever when processed content is available
- Triggers Analyzer when sufficient content is retrieved
- Initiates Writer when analysis results are ready
- Calls Reviewer after each draft
- Finalizes with Editor only when draft is approved

## 🚀 Recommended Setup

### Model Recommendation
We **strongly recommend using Gemini 2.5 Pro** due to its:
- **Larger context window** - Essential for processing extensive medical literature
- **Superior reasoning capabilities** - Critical for complex meta-analysis logic
- **Better handling of structured data** - Important for statistical calculations

### Prerequisites
- Python 3.8+
- API access to your preferred LLM provider (Gemini 2.5 Pro recommended)
- Tavily API key for literature search

### Installation

```bash
# Clone the repository
git clone https://github.com/danielxmed/metanalyst-agent-2.git
cd metanalyst-agent-2

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## 📁 Project Structure

```
metanalyst-agent-2/
├── agents/                     # Core agent implementations
│   ├── supervisor.py          # Supervisor agent - workflow orchestrator
│   ├── researcher.py          # Literature search agent
│   ├── processor.py           # Content processing agent
│   ├── retriever.py           # Content retrieval agent
│   └── ...                    # Additional specialized agents
├── prompts/                   # Agent prompt templates
│   ├── supervisor_prompt.py   # Supervisor agent instructions
│   └── ...                    # Other agent prompts
├── tools/                     # Agent-specific tools and utilities
├── state/                     # State management system
│   └── state.py              # Centralized state handling
├── tests/                     # Comprehensive test suite
├── data/                      # Processing and storage directories
│   ├── chunks/               # Processed content chunks
│   ├── publications_vectorstore/ # Vector embeddings
│   └── full_json_referenced/ # Complete publication data
└── requirements.txt          # Python dependencies
```

## 🎯 Key Features

- **🤖 Intelligent Workflow**: Supervisor agent adapts strategy based on results
- **📖 Comprehensive Literature Search**: Automated research paper discovery
- **🧠 Advanced Content Processing**: Smart chunking and vectorization
- **📊 Statistical Analysis**: Automated calculation of meta-analysis metrics
- **📝 Professional Reporting**: Markdown and HTML output generation
- **🔄 Quality Assurance**: Built-in review and feedback loops
- **⚡ Scalable Architecture**: Handles large volumes of literature efficiently

## 🔬 Development Status

This project is currently under active development. The core multi-agent architecture is implemented, and we're continuously improving the system's capabilities and reliability.

## 🤝 Contributing

We welcome contributions from the medical and AI communities! Whether you're interested in:
- Improving agent prompts
- Adding new statistical methods
- Enhancing the user interface
- Expanding test coverage
- Documentation improvements

Feel free to open issues or submit pull requests on our [GitHub repository](https://github.com/danielxmed/metanalyst-agent-2).

## 👨‍⚕️ About the Author

**Daniel Nobrega Medeiros**
- Physician and Programmer
- Specialized in AI applications for healthcare
- Contact: daniel@nobregamedtech.com.br
- Passionate about Python and medical AI automation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

When using this software, please cite:
- **Author**: Daniel Nobrega Medeiros
- **Repository**: https://github.com/danielxmed/metanalyst-agent-2
- **Contact**: daniel@nobregamedtech.com.br

---

*Empowering medical research through intelligent automation* 🚀
