# 🤖 Meta-Analyst Agent

An intelligent agent system for conducting automated meta-analyses of scientific literature.

## 📋 Overview

The Meta-Analyst Agent is a multi-agent system that automates the process of conducting meta-analyses, from literature search to final document generation. The system uses specialized agents for different stages of the process.

## ⚙️ Setup

### 1. Prerequisites

- Python 3.8+
- API keys for:
  - OpenAI (for GPT)
  - Anthropic (for Claude)
  - Tavily (for web search)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/danielxmed/metanalyst-agent-2.git
cd metanalyst-agent-2

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables Configuration

1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit the `.env` file and add your API keys:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
TAVILY_API_KEY=your_tavily_key_here
```

**⚠️ IMPORTANT: Never commit your real API keys! The `.env` file is in `.gitignore` to protect your credentials.**

## 🏗️ Architecture

The system consists of:

- **Supervisor Agent**: Coordinates the workflow
- **Researcher Agent**: Performs literature searches
- **State Manager**: Maintains process state

## 📊 Current Process State

| State Key                      | Description                                    |
|------------------------------- |------------------------------------------------|
| **current_iteration**          | Current flow iteration                         |
| **messages**                   | Messages exchanged so far                      |
| **meta_analysis_pico**         | Defined PICO elements                          |
| **user_request**               | Original user request                          |
| **previous_search_queries**    | Previously performed searches                  |
| **urls_to_process**            | URLs to be processed                           |
| **processed_urls**             | Already processed URLs                         |
| **retrieved_chunks**           | Chunks retrieved from repository               |
| **previous_retrieve_queries**  | Previous retrieval queries                     |
| **analysis_results**           | Analysis results                               |
| **current_draft**              | Current meta-analysis draft                    |
| **current_draft_iteration**    | Current draft iteration                        |
| **reviewer_feedbacks**         | Reviewer feedbacks                             |
| **final_draft**                | Final meta-analysis version                    |

> _Each row represents a state key maintained during meta-analysis pipeline execution._

## 🚀 How to Use

```python
# Basic usage example
from agents.supervisor import SupervisorAgent
from state.state import State

# Initialize state
state = State()

# Create supervisor agent
supervisor = SupervisorAgent()

# Execute a meta-analysis
result = supervisor.run("Meta-analysis on effectiveness of mental health interventions")
```

## 📁 Project Structure

```
metanalyst-agent-2/
├── agents/                 # System agents
│   ├── supervisor.py      # Supervisor agent
│   └── researcher.py      # Researcher agent
├── prompts/               # Prompt templates
├── state/                 # State management
├── tools/                 # Agent tools
├── tests/                 # Automated tests
├── .env.example          # Configuration example
└── requirements.txt      # Python dependencies
```

## 🧪 Tests

Run tests to verify everything is working:

```bash
python -m pytest tests/
```

## 🛡️ Security

- Never commit `.env` files with real keys
- Use environment variables for sensitive configurations
- Keep your API keys secure and don't share them

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributions

Contributions are welcome! Please open an issue or pull request.
