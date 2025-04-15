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
from datapresso.llm_api.generic_openai_provider import GenericOpenAIProvider
# Import other providers as they are created
# from datapresso.llm_api.gemini_provider import GeminiProvider
# from datapresso.llm_api.deepseek_provider import DeepSeekProvider
# from datapresso.llm_api.local_api_provider import LocalAPIProvider

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
            Manager configuration loaded from a config file (e.g., llm_api_config.yaml).
            Should contain 'default_provider' and a 'providers' dictionary.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize providers
        self.providers = {}
        self._initialize_providers()
        
        # Set default provider
        # Set default provider - ensure it exists after initialization
        self.default_provider = self.config.get("default_provider")
        if not self.default_provider:
             available_providers = list(self.providers.keys())
             if available_providers:
                 self.default_provider = available_providers[0]
                 self.logger.warning(f"No 'default_provider' specified in config. Using first available provider: '{self.default_provider}'")
             else:
                 raise ValueError("No LLM providers initialized and no default provider specified.")
        elif self.default_provider not in self.providers:
             available_providers = list(self.providers.keys())
             original_default = self.default_provider
             if available_providers:
                 self.default_provider = available_providers[0]
                 self.logger.warning(f"Configured default provider '{original_default}' not found or failed to initialize. Using first available provider: '{self.default_provider}' instead.")
             else:
                 raise ValueError(f"Configured default provider '{original_default}' not found or failed to initialize, and no other providers are available.")
                
        self.logger.info(f"Initialized LLM API Manager with default provider: {self.default_provider}")

    def _initialize_providers(self) -> None:
        """
        Initialize LLM providers based on the 'providers' section of the configuration.
        Dynamically loads and instantiates providers based on 'provider_type'.
        """
        provider_configs = self.config.get("providers", {})
        global_settings = self.config.get("global_settings", {})

        if not provider_configs:
            self.logger.warning("No providers found in the 'providers' section of the configuration.")
            return

        # Map provider types to classes
        provider_class_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "generic_openai": GenericOpenAIProvider,
            # Add future providers here
            # "gemini": GeminiProvider,
            # "deepseek": DeepSeekProvider,
            # "local_api": LocalAPIProvider,
        }
        if HAS_LOCAL_PROVIDER:
            provider_class_map["local"] = LocalProvider

        for name, config in provider_configs.items():
            # Merge global settings with provider-specific config
            # Provider-specific settings override global settings
            merged_config = {**global_settings, **config}

            # Determine provider type
            provider_type = merged_config.get("provider_type")
            if not provider_type:
                # Infer type from name if possible (e.g., name 'openai' implies type 'openai')
                if name in provider_class_map:
                    provider_type = name
                    self.logger.debug(f"Inferred provider type '{provider_type}' for provider '{name}'")
                else:
                    self.logger.error(f"Skipping provider '{name}': 'provider_type' not specified and cannot be inferred.")
                    continue

            if provider_type not in provider_class_map:
                self.logger.error(f"Skipping provider '{name}': Unknown provider_type '{provider_type}'. Supported types: {list(provider_class_map.keys())}")
                continue

            ProviderClass = provider_class_map[provider_type]

            # Handle optional dependencies (like 'local')
            if provider_type == "local" and not HAS_LOCAL_PROVIDER:
                 self.logger.warning(f"Skipping provider '{name}' (type 'local'): 'transformers' package not installed.")
                 continue

            try:
                # Instantiate the provider
                self.providers[name] = ProviderClass(merged_config, self.logger.getChild(name))
                self.logger.info(f"Successfully initialized provider '{name}' (type: {provider_type})")
            except ImportError as e:
                 self.logger.error(f"Failed to initialize provider '{name}' (type: {provider_type}): Missing dependency - {str(e)}")
            except ValueError as e:
                 self.logger.error(f"Failed to initialize provider '{name}' (type: {provider_type}): Configuration error - {str(e)}")
            except Exception as e:
                 self.logger.error(f"Failed to initialize provider '{name}' (type: {provider_type}): Unexpected error - {str(e)}", exc_info=True)


        # Log final list of successfully initialized providers
        if not self.providers:
            self.logger.warning("No LLM providers were successfully initialized.")
        else:
            self.logger.info(f"Successfully initialized providers: {', '.join(self.providers.keys())}")

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

    def generate(self, messages: List[Dict[str, str]], provider_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate text based on a list of messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries, e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
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
        return provider.generate(messages, **kwargs)

    def generate_with_structured_output(self, messages: List[Dict[str, str]], output_schema: Dict[str, Any], provider_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate structured output based on a list of messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries.
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
        return provider.generate_with_structured_output(messages, output_schema, **kwargs)

    def generate_batch(self, messages_list: List[List[Dict[str, str]]], provider_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple sets of messages.

        Parameters
        ----------
        messages_list : List[List[Dict[str, str]]]
            List of message lists. Each inner list is a conversation history.
        provider_name : Optional[str], optional
            Provider name, by default None (uses default provider)
        **kwargs : Dict[str, Any]
            Additional parameters for the provider.

        Returns
        -------
        List[Dict[str, Any]]
            List of responses for each message list.
        """
        provider = self.get_provider(provider_name)
        return provider.generate_batch(messages_list, **kwargs)

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

    def get_available_models(self, provider_name: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        """
        Get the list of available models for one or all providers.

        Parameters
        ----------
        provider_name : Optional[str], optional
            If specified, returns models only for this provider.
            If None, returns a dictionary mapping all provider names to their models.

        Returns
        -------
        Union[Dict[str, List[str]], List[str]]
            Either a list of model names (if provider_name is specified)
            or a dictionary mapping provider names to lists of model names.

        Raises
        ------
        ValueError
            If a specific provider_name is requested but not found.
        """
        if provider_name:
            provider = self.get_provider(provider_name) # Handles ValueError if not found
            try:
                return provider.list_available_models()
            except Exception as e:
                self.logger.error(f"Failed to get available models for provider '{provider_name}': {str(e)}")
                return [] # Return empty list on error for a single provider
        else:
            all_models = {}
            for name, provider in self.providers.items():
                try:
                    all_models[name] = provider.list_available_models()
                except Exception as e:
                    self.logger.error(f"Failed to get available models for provider '{name}': {str(e)}")
                    all_models[name] = [] # Include provider in dict but with empty list on error
            return all_models
