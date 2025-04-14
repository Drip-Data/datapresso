"""
LLM API Manager for Datapresso framework.

This module provides a unified interface for interacting with various LLM providers.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from pathlib import Path

from datapresso.llm_api.llm_provider import LLMProvider
from datapresso.llm_api.openai_provider import OpenAIProvider
from datapresso.llm_api.anthropic_provider import AnthropicProvider

# Import local provider if transformers is available
try:
    from datapresso.llm_api.local_provider import LocalProvider
    HAS_LOCAL_PROVIDER = True
except ImportError:
    HAS_LOCAL_PROVIDER = False


class LLMAPIManager:
    """
    LLM API Manager for Datapresso framework.
    
    Provides a unified interface for interacting with various LLM providers.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the LLM API Manager.

        Parameters
        ----------
        config : Dict[str, Any]
            Manager configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize providers
        self.providers = {}
        self._initialize_providers()
        
        # Set default provider
        self.default_provider = config.get("default_provider", "openai")
        if self.default_provider not in self.providers:
            available_providers = list(self.providers.keys())
            if available_providers:
                self.default_provider = available_providers[0]
                self.logger.warning(f"Default provider '{config.get('default_provider')}' not available. Using '{self.default_provider}' instead.")
            else:
                raise ValueError("No LLM providers available")
                
        self.logger.info(f"Initialized LLM API Manager with default provider: {self.default_provider}")

    def _initialize_providers(self) -> None:
        """
        Initialize LLM providers based on configuration.
        """
        provider_configs = self.config.get("providers", {})
        
        # Initialize OpenAI provider if configured
        if "openai" in provider_configs:
            try:
                self.providers["openai"] = OpenAIProvider(provider_configs["openai"], self.logger)
                self.logger.info("Initialized OpenAI provider")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
                
        # Initialize Anthropic provider if configured
        if "anthropic" in provider_configs:
            try:
                self.providers["anthropic"] = AnthropicProvider(provider_configs["anthropic"], self.logger)
                self.logger.info("Initialized Anthropic provider")
            except Exception as e:
                self.logger.error(f"Failed to initialize Anthropic provider: {str(e)}")
                
        # Initialize local provider if configured and available
        if "local" in provider_configs and HAS_LOCAL_PROVIDER:
            try:
                self.providers["local"] = LocalProvider(provider_configs["local"], self.logger)
                self.logger.info("Initialized local provider")
            except Exception as e:
                self.logger.error(f"Failed to initialize local provider: {str(e)}")
                
        # Log available providers
        if not self.providers:
            self.logger.warning("No LLM providers initialized")
        else:
            self.logger.info(f"Available providers: {', '.join(self.providers.keys())}")

    def get_provider(self, provider_name: Optional[str] = None) -> LLMProvider:
        """
        Get a provider instance.

        Parameters
        ----------
        provider_name : Optional[str], optional
            Provider name, by default None (uses default provider)

        Returns
        -------
        LLMProvider
            Provider instance.

        Raises
        ------
        ValueError
            If the provider is not available.
        """
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available. Available providers: {', '.join(self.providers.keys())}")
            
        return self.providers[provider_name]

    def generate(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Input prompt.
        provider_name : Optional[str], optional
            Provider name, by default None (uses default provider)
        **kwargs : Dict[str, Any]
            Additional parameters for the provider.

        Returns
        -------
        Dict[str, Any]
            Response containing generated text and metadata.
        """
        provider = self.get_provider(provider_name)
        return provider.generate(prompt, **kwargs)

    def generate_with_structured_output(self, prompt: str, output_schema: Dict[str, Any], provider_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate structured output from a prompt.

        Parameters
        ----------
        prompt : str
            Input prompt.
        output_schema : Dict[str, Any]
            Schema defining the expected output structure.
        provider_name : Optional[str], optional
            Provider name, by default None (uses default provider)
        **kwargs : Dict[str, Any]
            Additional parameters for the provider.

        Returns
        -------
        Dict[str, Any]
            Response containing structured output and metadata.
        """
        provider = self.get_provider(provider_name)
        return provider.generate_with_structured_output(prompt, output_schema, **kwargs)

    def generate_batch(self, prompts: List[str], provider_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts.

        Parameters
        ----------
        prompts : List[str]
            List of input prompts.
        provider_name : Optional[str], optional
            Provider name, by default None (uses default provider)
        **kwargs : Dict[str, Any]
            Additional parameters for the provider.

        Returns
        -------
        List[Dict[str, Any]]
            List of responses for each prompt.
        """
        provider = self.get_provider(provider_name)
        return provider.generate_batch(prompts, **kwargs)

    def get_metrics(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage metrics for a provider.

        Parameters
        ----------
        provider_name : Optional[str], optional
            Provider name, by default None (uses default provider)

        Returns
        -------
        Dict[str, Any]
            Usage metrics.
        """
        provider = self.get_provider(provider_name)
        return provider.get_metrics()

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage metrics for all providers.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Usage metrics for each provider.
        """
        return {name: provider.get_metrics() for name, provider in self.providers.items()}

    def save_metrics(self, output_dir: Union[str, Path]) -> None:
        """
        Save metrics for all providers.

        Parameters
        ----------
        output_dir : Union[str, Path]
            Output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics for each provider
        for name, provider in self.providers.items():
            file_path = output_dir / f"{name}_metrics.json"
            provider.save_metrics(file_path)
            
        # Save combined metrics
        combined_metrics = {
            "timestamp": time.time(),
            "providers": self.get_all_metrics()
        }
        
        combined_path = output_dir / "combined_metrics.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_metrics, f, indent=2)
            
        self.logger.info(f"Saved metrics to {output_dir}")
