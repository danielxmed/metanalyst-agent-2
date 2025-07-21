"""
LLM Factory for creating language model instances.

This module provides a factory pattern for creating LLM instances
based on configuration, making it easy to switch between providers.
"""

from typing import Any, Dict, Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from .config import LLMConfig, LLMProvider


def create_llm(config: LLMConfig) -> BaseChatModel:
    """
    Create an LLM instance based on configuration.
    
    Args:
        config: LLM configuration specifying provider, model, and parameters
        
    Returns:
        Configured LLM instance
        
    Raises:
        ValueError: If provider is not supported
        ValueError: If required API key is missing
    """
    if not config.api_key:
        raise ValueError(f"API key required for {config.provider} provider")
    
    common_params = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    
    if config.provider == LLMProvider.ANTHROPIC:
        return ChatAnthropic(
            model=config.model_name,
            anthropic_api_key=config.api_key,
            **common_params
        )
    
    elif config.provider == LLMProvider.OPENAI:
        return ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.api_key,
            **common_params
        )
    
    elif config.provider == LLMProvider.GOOGLE:
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.api_key,
            **common_params
        )
    
    elif config.provider == LLMProvider.AZURE:
        # Azure OpenAI requires additional configuration
        return ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.api_key,
            azure_endpoint=config.api_key,  # This would need to be configured properly
            api_version="2024-02-15-preview",
            **common_params
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")


def create_embedding_model(provider: str, api_key: str, model_name: str = "text-embedding-3-small"):
    """
    Create an embedding model instance.
    
    Args:
        provider: Embedding provider ("openai", "anthropic", etc.)
        api_key: API key for the provider
        model_name: Name of the embedding model
        
    Returns:
        Configured embedding model
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
    
    elif provider == "anthropic":
        # Anthropic doesn't have embedding models, fall back to OpenAI
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_model_info(provider: LLMProvider) -> Dict[str, Any]:
    """
    Get information about available models for a provider.
    
    Args:
        provider: LLM provider
        
    Returns:
        Dictionary with model information
    """
    model_info = {
        LLMProvider.ANTHROPIC: {
            "models": [
                "claude-3-5-sonnet-latest",
                "claude-3-opus-latest", 
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            "context_length": {
                "claude-3-5-sonnet-latest": 200000,
                "claude-3-opus-latest": 200000,
                "claude-3-sonnet-20240229": 200000,
                "claude-3-haiku-20240307": 200000
            },
            "pricing_tier": "premium"
        },
        
        LLMProvider.OPENAI: {
            "models": [
                "gpt-4-turbo-preview",
                "gpt-4-0125-preview",
                "gpt-4-1106-preview",
                "gpt-3.5-turbo-0125"
            ],
            "context_length": {
                "gpt-4-turbo-preview": 128000,
                "gpt-4-0125-preview": 128000,
                "gpt-4-1106-preview": 128000,
                "gpt-3.5-turbo-0125": 16385
            },
            "pricing_tier": "standard"
        },
        
        LLMProvider.GOOGLE: {
            "models": [
                "gemini-pro",
                "gemini-pro-vision"
            ],
            "context_length": {
                "gemini-pro": 32768,
                "gemini-pro-vision": 32768
            },
            "pricing_tier": "competitive"
        },
        
        LLMProvider.AZURE: {
            "models": [
                "gpt-4",
                "gpt-35-turbo"
            ],
            "context_length": {
                "gpt-4": 8192,
                "gpt-35-turbo": 4096
            },
            "pricing_tier": "enterprise"
        }
    }
    
    return model_info.get(provider, {})


def validate_model_config(config: LLMConfig) -> bool:
    """
    Validate that the model configuration is valid.
    
    Args:
        config: LLM configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    model_info = get_model_info(config.provider)
    
    if not model_info:
        return False
    
    if config.model_name not in model_info.get("models", []):
        return False
    
    if config.temperature < 0 or config.temperature > 2:
        return False
    
    if config.max_tokens and config.max_tokens <= 0:
        return False
    
    return True


def get_recommended_model(provider: LLMProvider, use_case: str = "general") -> str:
    """
    Get recommended model for a specific use case.
    
    Args:
        provider: LLM provider
        use_case: Use case ("general", "analysis", "creative", "fast")
        
    Returns:
        Recommended model name
    """
    recommendations = {
        LLMProvider.ANTHROPIC: {
            "general": "claude-3-5-sonnet-latest",
            "analysis": "claude-3-opus-latest",
            "creative": "claude-3-5-sonnet-latest",
            "fast": "claude-3-haiku-20240307"
        },
        
        LLMProvider.OPENAI: {
            "general": "gpt-4-turbo-preview",
            "analysis": "gpt-4-0125-preview",
            "creative": "gpt-4-turbo-preview",
            "fast": "gpt-3.5-turbo-0125"
        },
        
        LLMProvider.GOOGLE: {
            "general": "gemini-pro",
            "analysis": "gemini-pro",
            "creative": "gemini-pro",
            "fast": "gemini-pro"
        },
        
        LLMProvider.AZURE: {
            "general": "gpt-4",
            "analysis": "gpt-4",
            "creative": "gpt-4",
            "fast": "gpt-35-turbo"
        }
    }
    
    provider_recs = recommendations.get(provider, {})
    return provider_recs.get(use_case, provider_recs.get("general", ""))


def create_agent_llms(config_dict: Dict[str, LLMConfig]) -> Dict[str, BaseChatModel]:
    """
    Create LLM instances for multiple agents.
    
    Args:
        config_dict: Dictionary mapping agent names to LLM configs
        
    Returns:
        Dictionary mapping agent names to LLM instances
    """
    llms = {}
    
    for agent_name, llm_config in config_dict.items():
        try:
            llms[agent_name] = create_llm(llm_config)
        except Exception as e:
            raise ValueError(f"Failed to create LLM for agent {agent_name}: {e}")
    
    return llms