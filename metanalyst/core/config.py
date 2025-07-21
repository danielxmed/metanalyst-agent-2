"""
Configuration management for the metanalyst agent system.

This module handles all configuration settings including LLM selection,
database connections, and system parameters.
"""

import os
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from pathlib import Path
import json


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    GOOGLE = "google"


class LLMConfig(BaseModel):
    """Configuration for LLM settings."""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model_name: str = "claude-3-5-sonnet-latest"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4096, gt=0)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Provider-specific settings
    anthropic_version: str = "2023-06-01"
    openai_organization: Optional[str] = None
    azure_deployment_name: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    google_project: Optional[str] = None
    
    @validator('api_key', always=True)
    def get_api_key(cls, v, values):
        """Auto-detect API key from environment if not provided."""
        if v is not None:
            return v
        
        provider = values.get('provider')
        if provider == LLMProvider.ANTHROPIC:
            return os.getenv('ANTHROPIC_API_KEY')
        elif provider == LLMProvider.OPENAI:
            return os.getenv('OPENAI_API_KEY')
        elif provider == LLMProvider.AZURE:
            return os.getenv('AZURE_OPENAI_API_KEY')
        elif provider == LLMProvider.GOOGLE:
            return os.getenv('GOOGLE_API_KEY')
        
        return None


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    postgres_url: str = Field(
        default_factory=lambda: os.getenv(
            'POSTGRES_URL', 
            'postgresql://postgres:password@localhost:5432/metanalyst'
        )
    )
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


class TavilyConfig(BaseModel):
    """Configuration for Tavily search."""
    api_key: str = Field(
        default_factory=lambda: os.getenv('TAVILY_API_KEY', '')
    )
    max_results: int = 10
    search_depth: str = "advanced"  # basic, advanced
    include_domains: Optional[list[str]] = None
    exclude_domains: Optional[list[str]] = None


class VectorStoreConfig(BaseModel):
    """Configuration for vector store settings."""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chunk_size: int = 1000
    chunk_overlap: int = 100
    similarity_threshold: float = 0.7


class SystemConfig(BaseModel):
    """General system configuration."""
    log_level: str = "INFO"
    max_concurrent_agents: int = 5
    default_timeout: int = 300  # seconds
    retry_attempts: int = 3
    retry_delay: int = 1  # seconds
    
    # Human-in-the-loop settings
    require_human_approval: bool = True
    auto_approve_low_risk: bool = False
    human_timeout: int = 3600  # seconds (1 hour)


class MetanalystConfig(BaseModel):
    """Main configuration class for the metanalyst system."""
    
    # Core configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    # Agent-specific configurations
    orchestrator_config: Dict[str, Any] = Field(default_factory=dict)
    researcher_config: Dict[str, Any] = Field(default_factory=dict)
    processor_config: Dict[str, Any] = Field(default_factory=dict)
    retriever_config: Dict[str, Any] = Field(default_factory=dict)
    writer_config: Dict[str, Any] = Field(default_factory=dict)
    reviewer_config: Dict[str, Any] = Field(default_factory=dict)
    analyst_config: Dict[str, Any] = Field(default_factory=dict)
    editor_config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        env_prefix = "METANALYST_"
        case_sensitive = False


def load_config(config_file: Optional[Union[str, Path]] = None) -> MetanalystConfig:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_file: Path to JSON configuration file (optional)
    
    Returns:
        MetanalystConfig: Loaded configuration
    """
    config_data = {}
    
    # Load from file if provided
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
    
    # Create config with file data (environment variables will override)
    return MetanalystConfig(**config_data)


def save_config(config: MetanalystConfig, config_file: Union[str, Path]) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration to save
        config_file: Path to save the configuration file
    """
    config_path = Path(config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config.dict(), f, indent=2, default=str)


def get_llm_instance(config: LLMConfig):
    """
    Create an LLM instance based on configuration.
    
    Args:
        config: LLM configuration
    
    Returns:
        Configured LLM instance
    """
    if config.provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_key=config.api_key,
            anthropic_api_url=config.base_url,
        )
    
    elif config.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=config.api_key,
            openai_api_base=config.base_url,
            openai_organization=config.openai_organization,
        )
    
    elif config.provider == LLMProvider.AZURE:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            azure_endpoint=config.base_url,
            api_key=config.api_key,
            api_version=config.azure_api_version,
            azure_deployment=config.azure_deployment_name,
        )
    
    elif config.provider == LLMProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            google_api_key=config.api_key,
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")


def get_tavily_instance(config: TavilyConfig):
    """
    Create a Tavily search instance based on configuration.
    
    Args:
        config: Tavily configuration
    
    Returns:
        Configured Tavily instance
    """
    from langchain_tavily import TavilySearchResults
    
    return TavilySearchResults(
        api_key=config.api_key,
        max_results=config.max_results,
        search_depth=config.search_depth,
        include_domains=config.include_domains,
        exclude_domains=config.exclude_domains,
    )


def get_postgres_checkpointer(config: DatabaseConfig):
    """
    Create a PostgreSQL checkpointer based on configuration.
    
    Args:
        config: Database configuration
    
    Returns:
        Configured PostgreSQL checkpointer
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    
    checkpointer = PostgresSaver.from_conn_string(config.postgres_url)
    checkpointer.setup()  # Create tables if they don't exist
    
    return checkpointer


# Default configuration instance
default_config = MetanalystConfig()


# Configuration presets for different environments
DEVELOPMENT_CONFIG = MetanalystConfig(
    llm=LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.1
    ),
    system=SystemConfig(
        log_level="DEBUG",
        require_human_approval=False,
        auto_approve_low_risk=True
    )
)

PRODUCTION_CONFIG = MetanalystConfig(
    llm=LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.05
    ),
    system=SystemConfig(
        log_level="INFO",
        require_human_approval=True,
        auto_approve_low_risk=False,
        max_concurrent_agents=10
    )
)

TESTING_CONFIG = MetanalystConfig(
    llm=LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.0,
        max_tokens=1024
    ),
    system=SystemConfig(
        log_level="DEBUG",
        require_human_approval=False,
        auto_approve_low_risk=True,
        default_timeout=60
    )
)