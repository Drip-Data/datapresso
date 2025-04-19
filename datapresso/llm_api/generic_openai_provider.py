"""
Generic OpenAI-compatible provider implementation for Datapresso framework.

This module provides an implementation of the LLMProvider interface for APIs
that adhere to the OpenAI API specification (e.g., vLLM, TGI, OpenRouter, custom endpoints).
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import requests
from pathlib import Path

from datapresso.llm_api.llm_provider import LLMProvider


class GenericOpenAIProvider(LLMProvider):
    """
    Generic OpenAI-compatible provider implementation.

    Provides access to any OpenAI-compatible API endpoint for text generation.
    Requires 'api_base' and 'model' to be specified in the configuration.
    'api_key' is optional.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Generic OpenAI-compatible provider.

        Parameters
        ----------
        config : Dict[str, Any]
            Provider configuration. Must include 'api_base' and 'model'.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        super().__init__(config, logger)

        # API configuration - Required
        self.api_base = config.get("api_base")
        if not self.api_base:
            raise ValueError("GenericOpenAIProvider requires 'api_base' in config")
        self.model = config.get("model") # Model name expected by the endpoint
        if not self.model:
             raise ValueError("GenericOpenAIProvider requires 'model' in config")

        # API key - Optional
        self.api_key = config.get("api_key") or os.environ.get(config.get("api_key_env_var", "GENERIC_OPENAI_API_KEY")) # Allow specifying env var name

        # Default parameters
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 1024)
        self.timeout = config.get("timeout", 60)

        # Retry configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

        # Cost tracking - Defaults to 0 if not provided
        self.cost_per_1k_tokens = config.get("cost_per_1k_tokens", {
             self.model: {"prompt": 0.0, "completion": 0.0}
        })
        # Ensure the configured model has a cost entry, default to 0
        if self.model not in self.cost_per_1k_tokens:
             self.cost_per_1k_tokens[self.model] = {"prompt": 0.0, "completion": 0.0}


        self.logger.info(f"Initialized GenericOpenAIProvider for model '{self.model}' at base URL: {self.api_base}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate text based on a list of messages using a generic OpenAI-compatible API.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries.
        **kwargs : Dict[str, Any]
            Additional parameters for the API.

        Returns
        -------
        Dict[str, Any]
            Response containing generated text and metadata.
        """
        # Prepare request
        model = kwargs.get("model", self.model) # Use specific model if passed, else the default for this provider
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Add optional parameters common to OpenAI API
        for param in ["top_p", "n", "stop", "presence_penalty", "frequency_penalty", "stream"]:
            if param in kwargs:
                request_data[param] = kwargs[param]

        # Make API request
        start_time = time.time()
        response = self._make_api_request("chat/completions", request_data)
        latency = time.time() - start_time

        # Process response
        processed_response = self._process_response(response, model, latency)

        # Update metrics and history
        self._update_metrics(processed_response, latency)
        self._add_to_history(request_data, processed_response, latency)

        return processed_response

    def generate_with_structured_output(self, messages: List[Dict[str, str]], output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output using function calling (if supported by the endpoint).

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries.
        output_schema : Dict[str, Any]
            Schema defining the expected output structure (used for function calling).
        **kwargs : Dict[str, Any]
            Additional parameters for the API.

        Returns
        -------
        Dict[str, Any]
            Response containing structured output and metadata.
        """
        # Prepare request
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        # Create function definition (assuming OpenAI-style function calling)
        function_name = kwargs.get("function_name", "generate_structured_output")
        function_description = kwargs.get("function_description", "Generate structured output based on the prompt")

        functions = [{
            "name": function_name,
            "description": function_description,
            "parameters": output_schema
        }]

        request_data = {
            "model": model,
            "messages": messages,
            "functions": functions,
            "function_call": {"name": function_name},
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Add optional parameters
        for param in ["top_p", "n", "stop", "presence_penalty", "frequency_penalty"]:
            if param in kwargs:
                request_data[param] = kwargs[param]

        # Make API request
        start_time = time.time()
        response = self._make_api_request("chat/completions", request_data)
        latency = time.time() - start_time

        # Process response
        processed_response = self._process_structured_response(response, model, latency)

        # Update metrics and history
        self._update_metrics(processed_response, latency)
        self._add_to_history(request_data, processed_response, latency)

        return processed_response

    def generate_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple sets of messages (sequentially).

        Parameters
        ----------
        messages_list : List[List[Dict[str, str]]]
            List of message lists.
        **kwargs : Dict[str, Any]
            Additional parameters for the API.

        Returns
        -------
        List[Dict[str, Any]]
            List of responses for each message list.
        """
        self.logger.info(f"Generating responses for {len(messages_list)} message lists (sequentially) using {self.api_base}")

        responses = []
        model = kwargs.get("model", self.model) # Use consistent model for batch

        # Process each message list
        for i, messages in enumerate(messages_list):
            self.logger.debug(f"Processing message list {i+1}/{len(messages_list)}")
            try:
                response = self.generate(messages, model=model, **kwargs) # Pass model explicitly
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to generate response for message list {i+1}: {str(e)}")
                responses.append({
                    "text": "", "structured_output": {},
                    "error": {"message": f"Generation failed: {str(e)}", "type": "batch_error"},
                    "model": model, "latency": 0.0, "usage": {}, "cost": 0.0
                })

            # Add small delay between requests
            if i < len(messages_list) - 1:
                time.sleep(self.retry_delay * 0.5)

        return responses

    def _make_api_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the generic OpenAI-compatible API.
        """
        url = f"{self.api_base.rstrip('/')}/{endpoint}"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API request to {url} failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    return {
                        "error": {
                            "message": f"API request failed after {self.max_retries} attempts: {str(e)}",
                            "type": "api_error",
                            "details": str(e) # Include details from the exception
                        }
                    }
        # Should not be reached if max_retries >= 1
        return {"error": {"message": "API request failed unexpectedly", "type": "internal_error"}}


    def _process_response(self, response: Dict[str, Any], model: str, latency: float) -> Dict[str, Any]:
        """
        Process the API response. (Identical to OpenAIProvider's implementation)
        """
        if "error" in response:
            return {
                "text": "",
                "error": response["error"],
                "model": model, "latency": latency, "usage": {}, "cost": 0.0
            }

        text = ""
        usage = {}
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            text = message.get("content", "")
        usage = response.get("usage", {}) # May be missing or incomplete from some endpoints

        cost = self._calculate_cost(model, usage)

        return {
            "text": text, "error": None, "model": model, "latency": latency,
            "usage": usage, "cost": cost, "raw_response": response
        }

    def _process_structured_response(self, response: Dict[str, Any], model: str, latency: float) -> Dict[str, Any]:
        """
        Process the API response for structured output. (Identical to OpenAIProvider's implementation)
        """
        if "error" in response:
            return {
                "structured_output": {},
                "error": response["error"],
                "model": model, "latency": latency, "usage": {}, "cost": 0.0
            }

        structured_output = {}
        usage = {}
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            function_call = message.get("function_call", {})
            if function_call and "arguments" in function_call:
                try:
                    # Handle potential string escaping issues from different servers
                    args_str = function_call["arguments"]
                    if isinstance(args_str, str):
                        structured_output = json.loads(args_str)
                    else:
                        structured_output = args_str # Assume it's already parsed if not a string
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse function call arguments: {str(e)}. Raw args: '{function_call.get('arguments')}'")
                    structured_output = {"error": f"Failed to parse function call arguments: {e}"}
            # Fallback: Check if content itself is JSON (some models might do this without function calling)
            elif message.get("content"):
                 try:
                     potential_json = json.loads(message["content"])
                     if isinstance(potential_json, dict):
                          structured_output = potential_json
                          self.logger.debug("Parsed structured output from message content instead of function call.")
                 except json.JSONDecodeError:
                     pass # Content was not JSON

        usage = response.get("usage", {})
        cost = self._calculate_cost(model, usage)

        return {
            "structured_output": structured_output, "error": None, "model": model, "latency": latency,
            "usage": usage, "cost": cost, "raw_response": response
        }

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Calculate the cost of the API request based on configured rates.
        """
        # Use the specific model requested, or the provider's default model for cost lookup
        cost_model_key = model if model in self.cost_per_1k_tokens else self.model
        model_costs = self.cost_per_1k_tokens.get(cost_model_key, {"prompt": 0.0, "completion": 0.0})

        prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
        completion_tokens = usage.get("completion_tokens", 0) if usage else 0

        prompt_cost = (prompt_tokens / 1000) * model_costs.get("prompt", 0.0)
        completion_cost = (completion_tokens / 1000) * model_costs.get("completion", 0.0)

        return prompt_cost + completion_cost

    def list_available_models(self) -> List[str]:
        """
        List available models from the generic OpenAI-compatible API endpoint.
        Attempts to use the standard /v1/models endpoint.
        """
        endpoint = "models"
        url = f"{self.api_base.rstrip('/')}/{endpoint}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            # Some endpoints might return 4xx/5xx but still be valid,
            # try to parse JSON even if status is not 200
            if response.status_code >= 400:
                 self.logger.warning(f"Received status code {response.status_code} when listing models from {url}. Attempting to parse anyway.")

            models_data = response.json()

            # Handle different possible response structures for model listing
            if isinstance(models_data, dict) and "data" in models_data and isinstance(models_data["data"], list):
                # Standard OpenAI format
                model_ids = [model.get("id") for model in models_data["data"] if model.get("id")]
            elif isinstance(models_data, list):
                 # Maybe a simple list of strings or objects with 'id'
                 if all(isinstance(m, str) for m in models_data):
                      model_ids = models_data
                 elif all(isinstance(m, dict) and "id" in m for m in models_data):
                      model_ids = [m["id"] for m in models_data]
                 else:
                      model_ids = []
            else:
                 # Unknown format
                 self.logger.warning(f"Unknown format received for model list from {url}: {models_data}")
                 model_ids = []

            # Always include the specifically configured model for this provider
            if self.model not in model_ids:
                model_ids.append(self.model)

            return sorted(list(set(model_ids)))

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch available models from {url}: {str(e)}")
            # Return the configured model as a fallback
            return [self.model]
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse models response from {url}: {str(e)}")
            # Return the configured model as a fallback
            return [self.model]