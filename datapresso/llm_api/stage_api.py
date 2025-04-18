"""
Provides a stage-specific wrapper around LLMAPIManager for simplified usage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Assuming config_loader and LLMAPIManager are in the same directory or accessible
from .config_loader import load_llm_config
from .llm_api_manager import LLMAPIManager

logger = logging.getLogger(__name__)

class StageLLMApi:
    """
    A wrapper class providing a simplified interface for a specific Datapresso stage
    to interact with the LLM API layer.

    It loads the stage-specific configuration, initializes an LLMAPIManager,
    and provides methods that handle template lookups.
    """
    def __init__(self, stage_name: str, config_dir: Path):
        """
        Initializes the StageLLMApi for a given stage.

        Parameters
        ----------
        stage_name : str
            The unique identifier for the stage (e.g., 'data_generator').
            Used to find the configuration file '{stage_name}.yaml'.
        config_dir : Path
            The directory where stage-specific configuration files are stored
            (e.g., 'datapresso/datapresso/llm_api/configs/').
        """
        self.stage_name = stage_name
        self.config_dir = config_dir
        self.config: Dict[str, Any] = {}
        self.llm_manager: Optional[LLMAPIManager] = None
        self.system_prompt_templates: Dict[str, str] = {}
        self.output_schema_templates: Dict[str, Dict[str, Any]] = {}
        self.logger = logger.getChild(f"StageLLMApi.{self.stage_name}")

        self._initialize()

    def _initialize(self):
        """Loads configuration and initializes the LLM manager."""
        config_path = self.config_dir / f"{self.stage_name}.yaml"
        try:
            self.config = load_llm_config(config_path)
            self.llm_manager = LLMAPIManager(config=self.config, logger=self.logger.getChild("LLMAPIManager"))

            # Load templates from the loaded config
            self.system_prompt_templates = self.config.get('system_prompt_templates', {})
            self.output_schema_templates = self.config.get('output_schema_templates', {})

            self.logger.info(f"Successfully initialized using config: {config_path}")

        except FileNotFoundError:
            self.logger.error(f"Configuration file not found at {config_path}. LLM calls for stage '{self.stage_name}' will fail.")
            # Keep self.llm_manager as None
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM API for stage '{self.stage_name}': {e}", exc_info=True)
            # Keep self.llm_manager as None

    def is_available(self) -> bool:
        """Checks if the LLM manager was successfully initialized."""
        return self.llm_manager is not None

    def generate(self,
                 user_prompt: str,
                 system_prompt_template: Optional[str] = None,
                 system_prompt_override: Optional[str] = None,
                 provider_name: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Generates text using the configured LLM provider for this stage.

        Parameters
        ----------
        user_prompt : str
            The main user query or instruction.
        system_prompt_template : Optional[str], optional
            Name of the system prompt template defined in the stage's config file.
        system_prompt_override : Optional[str], optional
            A specific system prompt string to use, overriding any template.
        provider_name : Optional[str], optional
            Name of the provider (defined in config) to use, overriding the default.
        **kwargs : Dict[str, Any]
            Additional parameters to pass to the LLM provider (e.g., temperature, max_tokens),
            overriding defaults set in the configuration.

        Returns
        -------
        Dict[str, Any]
            The response dictionary from the LLMAPIManager.
        """
        if not self.is_available():
            return {"error": {"message": f"LLM API for stage '{self.stage_name}' is not available due to initialization errors.", "type": "config_error"}}

        # Determine final system prompt
        final_system_prompt = system_prompt_override
        if final_system_prompt is None and system_prompt_template:
            final_system_prompt = self.system_prompt_templates.get(system_prompt_template)
            if final_system_prompt is None:
                self.logger.warning(f"System prompt template '{system_prompt_template}' not found for stage '{self.stage_name}'.")

        # Construct messages
        messages = [{"role": "user", "content": user_prompt}]
        if final_system_prompt:
            messages.insert(0, {"role": "system", "content": final_system_prompt})

        # Call the manager
        try:
            # Ensure llm_manager is not None (checked by is_available, but for type hinting)
            assert self.llm_manager is not None
            return self.llm_manager.generate(messages=messages, provider_name=provider_name, **kwargs)
        except Exception as e:
            self.logger.error(f"Error during LLM generate call for stage '{self.stage_name}': {e}", exc_info=True)
            return {"error": {"message": f"LLM generate call failed: {e}", "type": "runtime_error"}}


    def generate_with_structured_output(self,
                                        user_prompt: str,
                                        output_schema_template: str,
                                        output_schema_override: Optional[Dict[str, Any]] = None,
                                        system_prompt_template: Optional[str] = None,
                                        system_prompt_override: Optional[str] = None,
                                        provider_name: Optional[str] = None,
                                        **kwargs) -> Dict[str, Any]:
        """
        Generates structured output using the configured LLM provider for this stage.

        Parameters
        ----------
        user_prompt : str
            The main user query or instruction.
        output_schema_template : str
            Name of the output schema template defined in the stage's config file.
        output_schema_override : Optional[Dict[str, Any]], optional
            A specific JSON schema dictionary to use, overriding the template.
        system_prompt_template : Optional[str], optional
            Name of the system prompt template.
        system_prompt_override : Optional[str], optional
            A specific system prompt string to use, overriding any template.
        provider_name : Optional[str], optional
            Name of the provider to use, overriding the default.
        **kwargs : Dict[str, Any]
            Additional parameters for the LLM provider (e.g., temperature, function_description).

        Returns
        -------
        Dict[str, Any]
            The response dictionary from the LLMAPIManager.
        """
        if not self.is_available():
            return {"error": {"message": f"LLM API for stage '{self.stage_name}' is not available due to initialization errors.", "type": "config_error"}}

        # Determine final output schema
        final_schema = output_schema_override
        if final_schema is None:
            final_schema = self.output_schema_templates.get(output_schema_template)
            if final_schema is None:
                err_msg = f"Output schema template '{output_schema_template}' not found for stage '{self.stage_name}'."
                self.logger.error(err_msg)
                return {"error": {"message": err_msg, "type": "config_error"}}

        # Determine final system prompt
        final_system_prompt = system_prompt_override
        if final_system_prompt is None and system_prompt_template:
            final_system_prompt = self.system_prompt_templates.get(system_prompt_template)
            if final_system_prompt is None:
                self.logger.warning(f"System prompt template '{system_prompt_template}' not found for stage '{self.stage_name}'.")

        # Construct messages
        messages = [{"role": "user", "content": user_prompt}]
        if final_system_prompt:
            messages.insert(0, {"role": "system", "content": final_system_prompt})

        # Add default function description if not provided in kwargs
        if 'function_description' not in kwargs:
            kwargs['function_description'] = f"Extract information based on schema '{output_schema_template}' for stage '{self.stage_name}'"

        # Call the manager
        try:
            # Ensure llm_manager is not None
            assert self.llm_manager is not None
            return self.llm_manager.generate_with_structured_output(
                messages=messages,
                output_schema=final_schema,
                provider_name=provider_name,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error during LLM structured generate call for stage '{self.stage_name}': {e}", exc_info=True)
            return {"error": {"message": f"LLM structured generate call failed: {e}", "type": "runtime_error"}}

    def get_available_models(self, provider_name: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
         """Gets available models for a specific provider or all providers configured for this stage."""
         if not self.is_available():
              self.logger.warning(f"LLM API for stage '{self.stage_name}' not available, cannot list models.")
              return {} if provider_name is None else []
         assert self.llm_manager is not None
         return self.llm_manager.get_available_models(provider_name)

    def get_metrics(self, provider_name: Optional[str] = None) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
         """Gets metrics for a specific provider or all providers used by this stage."""
         if not self.is_available():
              self.logger.warning(f"LLM API for stage '{self.stage_name}' not available, cannot get metrics.")
              return {}
         assert self.llm_manager is not None
         if provider_name:
              return self.llm_manager.get_metrics(provider_name)
         else:
              return self.llm_manager.get_all_metrics()