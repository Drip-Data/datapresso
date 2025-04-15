"""
LLM Provider interface for Datapresso framework.

This module provides a unified interface for interacting with various LLM providers.
"""

import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import requests
from pathlib import Path


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM provider implementations should inherit from this class.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the LLM provider.

        Parameters
        ----------
        config : Dict[str, Any]
            Provider configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0,
            "total_latency": 0.0
        }
        
        # Initialize request history (limited size)
        self.max_history_size = config.get("max_history_size", 100)
        self.request_history = []

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate text based on a list of messages (system, user, assistant roles).

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries, e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        **kwargs : Dict[str, Any]
            Additional parameters for the provider.

        Returns
        -------
        Dict[str, Any]
            Response containing generated text and metadata.
        """
        pass

    @abstractmethod
    def generate_with_structured_output(self, messages: List[Dict[str, str]], output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output based on a list of messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries, e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        output_schema : Dict[str, Any]
            Schema defining the expected output structure.
        **kwargs : Dict[str, Any]
            Additional parameters for the provider.

        Returns
        -------
        Dict[str, Any]
            Response containing structured output and metadata.
        """
        pass

    @abstractmethod
    def generate_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple sets of messages.

        Parameters
        ----------
        messages_list : List[List[Dict[str, str]]]
            List of message lists. Each inner list is a conversation history.
        **kwargs : Dict[str, Any]
            Additional parameters for the provider.

        Returns
        -------
        List[Dict[str, Any]]
            List of responses for each message list.
        """
        pass

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """
        List available models for this provider.

        Returns
        -------
        List[str]
            A list of model names supported by the provider.
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics.

        Returns
        -------
        Dict[str, Any]
            Usage metrics.
        """
        return self.metrics.copy()

    def get_request_history(self) -> List[Dict[str, Any]]:
        """
        Get request history.

        Returns
        -------
        List[Dict[str, Any]]
            Request history.
        """
        return self.request_history.copy()

    def _update_metrics(self, response: Dict[str, Any], latency: float) -> None:
        """
        Update usage metrics based on response.

        Parameters
        ----------
        response : Dict[str, Any]
            Response from the provider.
        latency : float
            Request latency in seconds.
        """
        self.metrics["total_requests"] += 1
        
        if response.get("error") is None:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
            
        # Update token counts if available
        usage = response.get("usage", {})
        self.metrics["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.metrics["total_completion_tokens"] += usage.get("completion_tokens", 0)
        self.metrics["total_tokens"] += usage.get("total_tokens", 0)
        
        # Update cost if available
        self.metrics["total_cost"] += response.get("cost", 0.0)
        
        # Update latency
        self.metrics["total_latency"] += latency

    def _add_to_history(self, request: Dict[str, Any], response: Dict[str, Any], latency: float) -> None:
        """
        Add request and response to history.

        Parameters
        ----------
        request : Dict[str, Any]
            Request data.
        response : Dict[str, Any]
            Response data.
        latency : float
            Request latency in seconds.
        """
        # Create history entry
        entry = {
            "timestamp": time.time(),
            "request": request,
            "response": response,
            "latency": latency
        }
        
        # Add to history
        self.request_history.append(entry)
        
        # Trim history if needed
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]

    def save_metrics(self, file_path: Union[str, Path]) -> None:
        """
        Save metrics to a file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to save metrics.
        """
        file_path = Path(file_path)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current metrics
        metrics = self.get_metrics()
        
        # Add timestamp
        metrics["timestamp"] = time.time()
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.info(f"Saved metrics to {file_path}")
