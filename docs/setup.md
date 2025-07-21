# Metanalyst Agent System Setup Guide

## Prerequisites

### System Requirements
- Python 3.9 or higher
- PostgreSQL 12 or higher
- 4GB+ RAM recommended
- Internet connection for API calls

### API Keys Required
- **Anthropic API Key** (primary LLM provider)
- **Tavily API Key** (search functionality)
- **OpenAI API Key** (optional, for alternative LLM)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd metanalyst-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Configuration

### 1. Environment Setup

Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` file with your API keys:

```bash
# Required API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional API Keys (for alternative LLM providers)
OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_API_KEY=your_azure_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Database Setup

#### Option A: Local PostgreSQL

Install PostgreSQL locally:

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS (with Homebrew)
brew install postgresql
brew services start postgresql

# Create database
createdb metanalyst
```

Update your `.env` file:
```bash
POSTGRES_URL=postgresql://postgres:password@localhost:5432/metanalyst
```

#### Option B: Docker PostgreSQL

```bash
docker run -d \
  --name metanalyst-postgres \
  -e POSTGRES_DB=metanalyst \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15
```

### 4. LLM Provider Configuration

The system defaults to Anthropic's Claude 3.5 Sonnet. To use a different provider:

```bash
# For OpenAI
METANALYST_LLM_PROVIDER=openai
METANALYST_LLM_MODEL_NAME=gpt-4

# For Azure OpenAI
METANALYST_LLM_PROVIDER=azure
METANALYST_LLM_MODEL_NAME=gpt-4
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here

# For Google Gemini
METANALYST_LLM_PROVIDER=google
METANALYST_LLM_MODEL_NAME=gemini-pro
```

## Verification

### 1. Run Basic Tests

```bash
python -m pytest tests/ -v
```

### 2. Run Example Script

```bash
python examples/basic_usage.py
```

Expected output:
```
🔬 Metanalyst Agent System - Basic Usage Example
==================================================
✅ Loaded environment variables from .env file

🧪 Testing configuration system...
   Default LLM provider: anthropic
   Default model: claude-3-5-sonnet-latest
   ✅ Configuration system working correctly

🧪 Testing handoff tools...
   Available tools: 8
   ✅ Handoff tool execution successful

🔧 Initializing configuration...
   LLM Provider: anthropic
   LLM Model: claude-3-5-sonnet-latest
   Database URL: postgresql://postgres:password@localhost:5432/metanalyst

🤖 Initializing orchestrator agent...
   ✅ Orchestrator agent initialized successfully

🎉 Example completed successfully!
```

## Getting API Keys

### Anthropic API Key
1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

### Tavily API Key
1. Visit [tavily.com](https://tavily.com)
2. Sign up for an account
3. Navigate to API settings
4. Generate an API key
5. Copy the key to your `.env` file

### OpenAI API Key (Optional)
1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Go to API Keys section
4. Create a new secret key
5. Copy the key to your `.env` file

## Configuration Options

### System Settings

```bash
# Logging
METANALYST_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Concurrency
METANALYST_MAX_CONCURRENT_AGENTS=5

# Human-in-the-loop
METANALYST_REQUIRE_HUMAN_APPROVAL=true
METANALYST_AUTO_APPROVE_LOW_RISK=false
```

### LLM Settings

```bash
# Model parameters
METANALYST_LLM_TEMPERATURE=0.1      # 0.0-2.0
METANALYST_LLM_MAX_TOKENS=4096      # Maximum tokens per response

# Provider-specific
ANTHROPIC_API_URL=https://api.anthropic.com  # Custom endpoint
OPENAI_ORGANIZATION=your-org-id              # OpenAI organization
```

### Search Settings

```bash
# Tavily configuration
METANALYST_TAVILY_MAX_RESULTS=10        # Max search results
METANALYST_TAVILY_SEARCH_DEPTH=advanced # basic or advanced
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```
Error: could not connect to server: Connection refused
```

**Solution:**
- Ensure PostgreSQL is running
- Check connection string in `.env`
- Verify database exists

#### 2. API Key Error
```
Error: Invalid API key
```

**Solution:**
- Verify API key is correct in `.env`
- Check API key permissions
- Ensure sufficient API credits

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'metanalyst'
```

**Solution:**
- Ensure virtual environment is activated
- Install package: `pip install -e .`
- Check Python path

#### 4. Permission Errors
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
- Check file permissions
- Ensure virtual environment ownership
- Run with appropriate privileges

### Debug Mode

Enable debug logging:
```bash
METANALYST_LOG_LEVEL=DEBUG
```

Run with verbose output:
```bash
python examples/basic_usage.py --verbose
```

### Health Check

Create a simple health check script:

```python
# health_check.py
from metanalyst.core.config import MetanalystConfig
from metanalyst.core.state import create_initial_state

def health_check():
    try:
        # Test configuration
        config = MetanalystConfig()
        print(f"✅ Configuration loaded")
        
        # Test state creation
        state = create_initial_state("test", "test request")
        print(f"✅ State management working")
        
        # Test database connection (if configured)
        if config.database.postgres_url:
            from metanalyst.core.config import get_postgres_checkpointer
            checkpointer = get_postgres_checkpointer(config.database)
            print(f"✅ Database connection successful")
        
        print("🎉 System health check passed!")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    health_check()
```

## Development Setup

### Additional Dependencies

```bash
pip install -e ".[dev]"
```

### Pre-commit Hooks

```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_orchestrator.py

# Run with coverage
python -m pytest --cov=metanalyst tests/
```

### Code Formatting

```bash
# Format code
black metanalyst/ tests/

# Sort imports
isort metanalyst/ tests/

# Type checking
mypy metanalyst/
```

## Production Deployment

### Environment Configuration

Use production configuration:
```python
from metanalyst.core.config import PRODUCTION_CONFIG
```

### Security Considerations

1. **API Key Management**
   - Use environment variables
   - Rotate keys regularly
   - Limit key permissions

2. **Database Security**
   - Use SSL connections
   - Implement proper authentication
   - Regular backups

3. **Network Security**
   - Use HTTPS endpoints
   - Implement rate limiting
   - Monitor API usage

### Monitoring

Set up logging and monitoring:
```bash
# Configure logging
METANALYST_LOG_LEVEL=INFO

# Monitor database
# Monitor API usage
# Track performance metrics
```

## Next Steps

1. **Explore Examples**: Check the `examples/` directory
2. **Read Architecture**: Review `docs/architecture.md`
3. **Run Tests**: Validate your setup with the test suite
4. **Start Development**: Begin implementing additional agents

For questions or issues, please check the documentation or create an issue in the repository.