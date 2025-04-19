"""
DeepSeek provider implementation for Datapresso framework.

This module provides an implementation of the LLMProvider interface for DeepSeek's API.
DeepSeek's API is largely compatible with OpenAI's API.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import requests
from pathlib import Path

from datapresso.llm_api.llm_provider import LLMProvider
from datapresso.llm_api.generic_openai_provider import GenericOpenAIProvider # Inherit common logic

# Default DeepSeek API base URL
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# Default cost per 1M tokens for DeepSeek models (Needs verification with actual pricing)
# Using placeholders based on common knowledge - VERIFY THESE
DEFAULT_DEEPSEEK_COSTS = {
    "deepseek-chat": {"prompt": 0.14, "completion": 0.28}, # Example: $0.14/1M input, $0.28/1M output
    "deepseek-coder": {"prompt": 0.14, "completion": 0.28}, # Example
}

class DeepSeekProvider(GenericOpenAIProvider): # Inherit from GenericOpenAIProvider
    """
    DeepSeek provider implementation.

    Leverages the GenericOpenAIProvider due to API compatibility.
    Requires a DeepSeek API key.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the DeepSeek provider.

        Parameters
        ----------
        config : Dict[str, Any]
            Provider configuration. Should include 'api_key' (or DEEPSEEK_API_KEY env var) and 'model'.
            'api_base' defaults to DeepSeek's official endpoint if not provided.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        # Set DeepSeek specific defaults before calling super().__init__
        config['api_base'] = config.get('api_base', DEFAULT_DEEPSEEK_API_BASE)
        config['model'] = config.get('model', 'deepseek-chat') # Default DeepSeek model

        # API key handling specific to DeepSeek
        api_key = config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not provided in config or DEEPSEEK_API_KEY environment variable")
        config['api_key'] = api_key # Ensure api_key is in config for superclass

        # Cost handling specific to DeepSeek (per 1M tokens -> per 1k tokens)
        cost_per_1m_tokens = config.get("cost_per_1m_tokens", DEFAULT_DEEPSEEK_COSTS)
        cost_per_1k_tokens = {
            model: {
                "prompt": costs.get("prompt", 0.0) / 1000.0,
                "completion": costs.get("completion", 0.0) / 1000.0
            } for model, costs in cost_per_1m_tokens.items()
        }
        config['cost_per_1k_tokens'] = cost_per_1k_tokens # Override/set for superclass

        # Initialize using the GenericOpenAIProvider's logic
        super().__init__(config, logger)

        self.logger.info(f"Initialized DeepSeekProvider for model '{self.model}' at base URL: {self.api_base}")

    # Most methods (generate, generate_with_structured_output, generate_batch,
    # _make_api_request, _process_response, _process_structured_response, _calculate_cost)
    # are inherited from GenericOpenAIProvider and should work directly
    # due to API compatibility.

    # We might only need to override list_available_models if DeepSeek's endpoint differs
    # or provides different information. The GenericOpenAIProvider's implementation
    # already tries the standard /v1/models endpoint.

    # Example override if needed:
    # def list_available_models(self) -> List[str]:
    #     """
    #     List available models from the DeepSeek API.
    #     (Overrides GenericOpenAIProvider if DeepSeek's endpoint is different)
    #     """
    #     # If DeepSeek uses the standard endpoint, the parent method is fine.
    #     # If not, implement specific logic here.
    #     self.logger.debug("Using inherited list_available_models for DeepSeek.")
    #     return super().list_available_models()

# No need to redefine methods unless DeepSeek has specific deviations from OpenAI standard API
# that are not covered by GenericOpenAIProvider configuration.