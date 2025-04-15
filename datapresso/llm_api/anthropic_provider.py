"""
Anthropic provider implementation for Datapresso framework.

This module provides an implementation of the LLMProvider interface for Anthropic's API.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import requests
from pathlib import Path

from datapresso.llm_api.llm_provider import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider implementation.
    
    Provides access to Anthropic's API for text generation.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Anthropic provider.

        Parameters
        ----------
        config : Dict[str, Any]
            Provider configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        super().__init__(config, logger)
        
        # Get API key
        self.api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided in config or environment variables")
            
        # API configuration
        self.api_base = config.get("api_base", "https://api.anthropic.com/v1")
        self.default_model = config.get("model", "claude-3-opus-20240229")
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 1024)
        self.timeout = config.get("timeout", 60)
        
        # Retry configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # Cost tracking
        self.cost_per_1k_tokens = config.get("cost_per_1k_tokens", {
            "claude-3-opus-20240229": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet-20240229": {"prompt": 0.003, "completion": 0.015},
            "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125}
        })
        
        self.logger.info(f"Initialized Anthropic provider with model: {self.default_model}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate text based on a list of messages using Anthropic's API.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries, e.g., [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
            Anthropic API expects alternating user/assistant roles. A system message can be provided via kwargs['system_prompt'].
        **kwargs : Dict[str, Any]
            Additional parameters for the API, including 'system_prompt'.

        Returns
        -------
        Dict[str, Any]
            Response containing generated text and metadata.
        """
        # Prepare request
        model = kwargs.get("model", self.default_model)
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Add system prompt if provided
            **({"system": kwargs["system_prompt"]} if "system_prompt" in kwargs else {})
        }
        
        # Add optional parameters
        for param in ["top_p", "top_k", "stop_sequences"]:
            if param in kwargs:
                request_data[param] = kwargs[param]
                
        # Make API request
        start_time = time.time()
        response = self._make_api_request("messages", request_data)
        latency = time.time() - start_time
        
        # Process response
        processed_response = self._process_response(response, model, latency)
        
        # Update metrics and history
        self._update_metrics(processed_response, latency)
        self._add_to_history(request_data, processed_response, latency)
        
        return processed_response

    def generate_with_structured_output(self, messages: List[Dict[str, str]], output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output based on a list of messages using Anthropic's API.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries, e.g., [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
        output_schema : Dict[str, Any]
            Schema defining the expected output structure.
        **kwargs : Dict[str, Any]
            Additional parameters for the API.

        Returns
        -------
        Dict[str, Any]
            Response containing structured output and metadata.
        """
        # Prepare request
        model = kwargs.get("model", self.default_model)
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        
        # Create system prompt with schema instructions
        schema_str = json.dumps(output_schema, indent=2)
        system_prompt = kwargs.get("system_prompt", "")
        
        # If no system prompt provided in kwargs, create one with schema instructions
        if not system_prompt:
            system_prompt = (
                f"You are a helpful assistant that generates structured data. "
                f"Please provide your response as a valid JSON object that conforms to the following schema:\n\n"
                f"{schema_str}\n\n"
                f"Ensure your response is valid JSON and follows the schema exactly. Only output the JSON object."
            )
        
        request_data = {
            "model": model,
            "messages": messages,
            "system": system_prompt, # Use the dedicated system parameter
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add optional parameters
        for param in ["top_p", "top_k", "stop_sequences"]:
            if param in kwargs:
                request_data[param] = kwargs[param]
                
        # Make API request
        start_time = time.time()
        response = self._make_api_request("messages", request_data)
        latency = time.time() - start_time
        
        # Process response
        processed_response = self._process_structured_response(response, model, latency)
        
        # Update metrics and history
        self._update_metrics(processed_response, latency)
        self._add_to_history(request_data, processed_response, latency)
        
        return processed_response

    def generate_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple sets of messages using Anthropic's API.
        Note: This currently makes sequential calls.

        Parameters
        ----------
        messages_list : List[List[Dict[str, str]]]
            List of message lists. Each inner list is a conversation history.
        **kwargs : Dict[str, Any]
            Additional parameters for the API.

        Returns
        -------
        List[Dict[str, Any]]
            List of responses for each message list.
        """
        self.logger.info(f"Generating responses for {len(messages_list)} message lists (sequentially)")
        
        responses = []
        
        # Process each message list
        for i, messages in enumerate(messages_list):
            self.logger.debug(f"Processing message list {i+1}/{len(messages_list)}")

            # Generate response
            try:
                # Note: system_prompt from kwargs will apply to all batch items if provided
                response = self.generate(messages, **kwargs)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to generate response for message list {i+1}: {str(e)}")
                responses.append({
                    "text": "",
                    "error": {"message": f"Generation failed: {str(e)}", "type": "batch_error"},
                    "model": kwargs.get("model", self.default_model),
                    "latency": 0.0,
                    "usage": {},
                    "cost": 0.0
                })

            # Add small delay between requests to potentially avoid rate limiting
            if i < len(messages_list) - 1:
                time.sleep(self.retry_delay * 0.5) # Use a fraction of retry delay
                
        return responses

    def _make_api_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Anthropic API.

        Parameters
        ----------
        endpoint : str
            API endpoint.
        data : Dict[str, Any]
            Request data.

        Returns
        -------
        Dict[str, Any]
            API response.
        """
        url = f"{self.api_base}/{endpoint}"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                # Check for successful response
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                
                # Check if we should retry
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Return error response
                    return {
                        "error": {
                            "message": f"API request failed after {self.max_retries} attempts: {str(e)}",
                            "type": "api_error"
                        }
                    }

    def _process_response(self, response: Dict[str, Any], model: str, latency: float) -> Dict[str, Any]:
        """
        Process the API response.

        Parameters
        ----------
        response : Dict[str, Any]
            API response.
        model : str
            Model used for generation.
        latency : float
            Request latency in seconds.

        Returns
        -------
        Dict[str, Any]
            Processed response.
        """
        # Check for error
        if "error" in response:
            return {
                "text": "",
                "error": response["error"],
                "model": model,
                "latency": latency,
                "usage": {},
                "cost": 0.0
            }
            
        # Extract generated text
        text = ""
        if "content" in response and len(response["content"]) > 0:
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    text += content_block.get("text", "")
            
        # Extract usage statistics
        usage = {
            "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
            "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0)
        }
        
        # Calculate cost
        cost = self._calculate_cost(model, usage)
        
        return {
            "text": text,
            "error": None,
            "model": model,
            "latency": latency,
            "usage": usage,
            "cost": cost,
            "raw_response": response
        }

    def _process_structured_response(self, response: Dict[str, Any], model: str, latency: float) -> Dict[str, Any]:
        """
        Process the API response for structured output.

        Parameters
        ----------
        response : Dict[str, Any]
            API response.
        model : str
            Model used for generation.
        latency : float
            Request latency in seconds.

        Returns
        -------
        Dict[str, Any]
            Processed response with structured output.
        """
        # Check for error
        if "error" in response:
            return {
                "structured_output": {},
                "error": response["error"],
                "model": model,
                "latency": latency,
                "usage": {},
                "cost": 0.0
            }
            
        # Extract text
        text = ""
        if "content" in response and len(response["content"]) > 0:
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    text += content_block.get("text", "")
            
        # Try to parse JSON from the text
        structured_output = {}
        try:
            # Find JSON in the text (look for content between ``` or just try to parse the whole text)
            json_match = None
            if "```json" in text and "```" in text.split("```json", 1)[1]:
                json_match = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text and "```" in text.split("```", 1)[1]:
                json_match = text.split("```", 1)[1].split("```", 1)[0].strip()
                
            if json_match:
                structured_output = json.loads(json_match)
            else:
                # Try to parse the whole text
                structured_output = json.loads(text)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from response: {str(e)}")
            structured_output = {"error": "Failed to parse JSON from response"}
            
        # Extract usage statistics
        usage = {
            "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
            "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0)
        }
        
        # Calculate cost
        cost = self._calculate_cost(model, usage)
        
        return {
            "structured_output": structured_output,
            "error": None,
            "model": model,
            "latency": latency,
            "usage": usage,
            "cost": cost,
            "raw_response": response
        }

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Calculate the cost of the API request.

        Parameters
        ----------
        model : str
            Model used for generation.
        usage : Dict[str, int]
            Usage statistics.

        Returns
        -------
        float
            Cost in USD.
        """
        # Get cost rates for the model
        model_costs = self.cost_per_1k_tokens.get(model, {"prompt": 0.0, "completion": 0.0})
        
        # Calculate cost
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        prompt_cost = (prompt_tokens / 1000) * model_costs["prompt"]
        completion_cost = (completion_tokens / 1000) * model_costs["completion"]
        
        return prompt_cost + completion_cost

    def list_available_models(self) -> List[str]:
        """
        List available models for Anthropic.
        Note: Anthropic does not have a dedicated API endpoint for listing models.
        This method returns models defined in the cost configuration or a default list.

        Returns
        -------
        List[str]
            A list of known Anthropic model names.
        """
        # Primarily rely on models defined in the cost configuration
        known_models = list(self.cost_per_1k_tokens.keys())

        # Add default model if not already present
        if self.default_model not in known_models:
            known_models.append(self.default_model)

        # Fallback to a hardcoded list if cost config is empty (shouldn't happen with defaults)
        if not known_models:
            known_models = [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-2.1",
                "claude-2.0",
                "claude-instant-1.2"
            ]
            self.logger.warning("No models found in cost configuration, returning hardcoded list.")

        return sorted(list(set(known_models))) # Ensure uniqueness and sort
