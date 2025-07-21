"""
Configuration management for the Metanalyst Agent.

This module provides configuration classes and utilities for easy setup
and customization of the metanalyst system, including LLM selection.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"


class LLMConfig(BaseModel):
    """Configuration for LLM models."""
    
    provider: LLMProvider = Field(default=LLMProvider.ANTHROPIC)
    model_name: str = Field(default="claude-3-5-sonnet-latest")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4000, gt=0)
    api_key: Optional[str] = None
    
    class Config:
        use_enum_values = True


class DatabaseConfig(BaseModel):
    """Configuration for PostgreSQL database."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="metanalyst")
    username: str = Field(default="postgres")
    password: str = Field(default="")
    
    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class TavilyConfig(BaseModel):
    """Configuration for Tavily search."""
    
    api_key: Optional[str] = None
    max_results: int = Field(default=10, ge=1, le=50)
    search_depth: str = Field(default="advanced")  # "basic" or "advanced"
    include_domains: List[str] = Field(default_factory=lambda: [
        "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com",
        "cochranelibrary.com",
        "clinicaltrials.gov"
    ])


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    
    provider: str = Field(default="faiss")  # "faiss", "chroma", "pinecone"
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_provider: str = Field(default="openai")
    dimension: int = Field(default=1536)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=100)


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    
    orchestrator: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.1
    ))
    
    researcher: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest", 
        temperature=0.2
    ))
    
    processor: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.1
    ))
    
    retriever: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.0
    ))
    
    writer: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.3
    ))
    
    reviewer: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.1
    ))
    
    analyst: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.0
    ))
    
    editor: LLMConfig = Field(default_factory=lambda: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.2
    ))


class MetanalystConfig(BaseModel):
    """Main configuration for the Metanalyst Agent system."""
    
    # Core system configuration
    system_name: str = Field(default="Metanalyst Agent")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    
    # Agent configurations
    agents: AgentConfig = Field(default_factory=AgentConfig)
    
    # External service configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    
    # Processing limits and timeouts
    max_concurrent_agents: int = Field(default=3, ge=1, le=10)
    agent_timeout: int = Field(default=300, ge=30)  # seconds
    max_retries: int = Field(default=3, ge=0)
    
    # Human-in-the-loop settings
    require_human_approval: bool = Field(default=True)
    auto_approve_low_risk: bool = Field(default=False)
    
    @classmethod
    def from_env(cls) -> "MetanalystConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Load API keys from environment
        config.agents.orchestrator.api_key = os.getenv("ANTHROPIC_API_KEY")
        config.agents.researcher.api_key = os.getenv("ANTHROPIC_API_KEY")
        config.agents.processor.api_key = os.getenv("ANTHROPIC_API_KEY")
        config.agents.retriever.api_key = os.getenv("ANTHROPIC_API_KEY")
        config.agents.writer.api_key = os.getenv("ANTHROPIC_API_KEY")
        config.agents.reviewer.api_key = os.getenv("ANTHROPIC_API_KEY")
        config.agents.analyst.api_key = os.getenv("ANTHROPIC_API_KEY")
        config.agents.editor.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        config.tavily.api_key = os.getenv("TAVILY_API_KEY")
        
        # Database configuration from environment
        if os.getenv("DATABASE_URL"):
            # Parse DATABASE_URL if provided
            config.database.host = os.getenv("DB_HOST", "localhost")
            config.database.port = int(os.getenv("DB_PORT", "5432"))
            config.database.database = os.getenv("DB_NAME", "metanalyst")
            config.database.username = os.getenv("DB_USER", "postgres")
            config.database.password = os.getenv("DB_PASSWORD", "")
        
        # Debug mode
        config.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        return config
    
    def get_llm_config(self, agent_name: str) -> LLMConfig:
        """Get LLM configuration for a specific agent."""
        return getattr(self.agents, agent_name, self.agents.orchestrator)
    
    def update_llm_provider(self, provider: LLMProvider, model_name: Optional[str] = None) -> None:
        """Update LLM provider for all agents."""
        model_map = {
            LLMProvider.ANTHROPIC: "claude-3-5-sonnet-latest",
            LLMProvider.OPENAI: "gpt-4-turbo-preview",
            LLMProvider.GOOGLE: "gemini-pro",
            LLMProvider.AZURE: "gpt-4"
        }
        
        default_model = model_name or model_map.get(provider, "claude-3-5-sonnet-latest")
        
        for agent_config in [
            self.agents.orchestrator,
            self.agents.researcher,
            self.agents.processor,
            self.agents.retriever,
            self.agents.writer,
            self.agents.reviewer,
            self.agents.analyst,
            self.agents.editor
        ]:
            agent_config.provider = provider
            if model_name:
                agent_config.model_name = default_model


def create_default_config() -> MetanalystConfig:
    """Create default configuration with Anthropic and Tavily as defaults."""
    return MetanalystConfig.from_env()


def validate_config(config: MetanalystConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check for required API keys
    if not config.agents.orchestrator.api_key:
        issues.append("Missing ANTHROPIC_API_KEY environment variable")
    
    if not config.tavily.api_key:
        issues.append("Missing TAVILY_API_KEY environment variable")
    
    # Validate database configuration
    if not config.database.password and config.database.host != "localhost":
        issues.append("Database password required for remote connections")
    
    # Check model availability
    for agent_name in ["orchestrator", "researcher", "processor", "retriever", 
                       "writer", "reviewer", "analyst", "editor"]:
        agent_config = config.get_llm_config(agent_name)
        if agent_config.provider == LLMProvider.ANTHROPIC and not agent_config.api_key:
            issues.append(f"Missing API key for {agent_name} agent")
    
    return issues