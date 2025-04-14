"""
LLM API module for Datapresso framework.

This module provides a unified interface for interacting with various LLM providers.
"""

from datapresso.llm_api.llm_provider import LLMProvider
from datapresso.llm_api.llm_api_manager import LLMAPIManager
from datapresso.llm_api.openai_provider import OpenAIProvider
from datapresso.llm_api.anthropic_provider import AnthropicProvider

# Import local provider if transformers is available
try:
    from datapresso.llm_api.local_provider import LocalProvider
    __all__ = ["LLMProvider", "LLMAPIManager", "OpenAIProvider", "AnthropicProvider", "LocalProvider"]
except ImportError:
    __all__ = ["LLMProvider", "LLMAPIManager", "OpenAIProvider", "AnthropicProvider"]
