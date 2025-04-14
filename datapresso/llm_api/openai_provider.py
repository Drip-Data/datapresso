"""
OpenAI provider implementation for Datapresso framework.

This module provides an implementation of the LLMProvider interface for OpenAI's API.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import requests
from pathlib import Path

from datapresso.llm_api.llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation.
    
    Provides access to OpenAI's API for text generation.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the OpenAI provider.

        Parameters
        ----------
        config : Dict[str, Any]
            Provider configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        super().__init__(config, logger)
        
        # Get API key
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided in config or environment variables")
            
        # API configuration
        self.api_base = config.get("api_base", "https://api.openai.com/v1")
        self.default_model = config.get("model", "gpt-4-turbo")
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 1024)
        self.timeout = config.get("timeout", 60)
        
        # Retry configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # Cost tracking
        self.cost_per_1k_tokens = config.get("cost_per_1k_tokens", {
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002}
        })
        
        self.logger.info(f"Initialized OpenAI provider with model: {self.default_model}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text from a prompt using OpenAI's API.

        Parameters
        ----------
        prompt : str
            Input prompt.
        **kwargs : Dict[str, Any]
            Additional parameters for the API.

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
            "messages": [{"role": "user", "content": prompt}],
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
        processed_response = self._process_response(response, model, latency)
        
        # Update metrics and history
        self._update_metrics(processed_response, latency)
        self._add_to_history(request_data, processed_response, latency)
        
        return processed_response

    def generate_with_structured_output(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output from a prompt using OpenAI's API.

        Parameters
        ----------
        prompt : str
            Input prompt.
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
        
        # Check if model supports function calling
        if "gpt-3.5" not in model and "gpt-4" not in model:
            self.logger.warning(f"Model {model} may not support function calling")
        
        # Create function definition
        function_name = kwargs.get("function_name", "generate_structured_output")
        function_description = kwargs.get("function_description", "Generate structured output based on the prompt")
        
        functions = [{
            "name": function_name,
            "description": function_description,
            "parameters": output_schema
        }]
        
        request_data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts using OpenAI's API.

        Parameters
        ----------
        prompts : List[str]
            List of input prompts.
        **kwargs : Dict[str, Any]
            Additional parameters for the API.

        Returns
        -------
        List[Dict[str, Any]]
            List of responses for each prompt.
        """
        self.logger.info(f"Generating responses for {len(prompts)} prompts")
        
        responses = []
        
        # Process each prompt
        for i, prompt in enumerate(prompts):
            self.logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
            
            # Generate response
            response = self.generate(prompt, **kwargs)
            responses.append(response)
            
            # Add small delay between requests to avoid rate limiting
            if i < len(prompts) - 1:
                time.sleep(0.5)
                
        return responses

    def _make_api_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the OpenAI API.

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
            "Authorization": f"Bearer {self.api_key}",
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
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            text = message.get("content", "")
            
        # Extract usage statistics
        usage = response.get("usage", {})
        
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
            
        # Extract function call
        structured_output = {}
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            function_call = message.get("function_call", {})
            
            if function_call and "arguments" in function_call:
                try:
                    structured_output = json.loads(function_call["arguments"])
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse function call arguments: {str(e)}")
                    structured_output = {"error": "Failed to parse function call arguments"}
            
        # Extract usage statistics
        usage = response.get("usage", {})
        
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
