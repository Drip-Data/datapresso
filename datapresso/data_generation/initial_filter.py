"""
Initial filter for Datapresso framework.

This module provides initial filtering of generated data samples.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union
import json


class InitialFilter:
    """
    Initial filter for Datapresso framework.
    
    Provides basic filtering of generated data samples to remove low-quality samples.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the initial filter.

        Parameters
        ----------
        config : Dict[str, Any]
            Filter configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.enabled = config.get("enabled", True)
        self.min_length = config.get("min_length", 50)
        self.max_length = config.get("max_length", 2000)
        self.banned_patterns = config.get("banned_patterns", [])
        
        self.logger.info(f"Initialized initial filter (enabled: {self.enabled})")

    def filter_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of generated samples.

        Parameters
        ----------
        samples : List[Dict[str, Any]]
            Generated samples to filter.

        Returns
        -------
        List[Dict[str, Any]]
            Filtered samples.
        """
        if not self.enabled:
            self.logger.info("Initial filtering disabled, returning all samples")
            return samples
            
        self.logger.info(f"Filtering {len(samples)} samples")
        
        filtered_samples = []
        rejected_count = 0
        
        for sample in samples:
            if self._is_valid_sample(sample):
                filtered_samples.append(sample)
            else:
                rejected_count += 1
                
        self.logger.info(f"Filtering results: {len(filtered_samples)} accepted, {rejected_count} rejected")
        
        return filtered_samples

    def _is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Check if a sample is valid according to filtering criteria.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample to check.

        Returns
        -------
        bool
            True if the sample is valid, False otherwise.
        """
        # Check required fields
        if not all(key in sample for key in ["instruction", "response"]):
            return False
            
        instruction = sample.get("instruction", "")
        response = sample.get("response", "")
        
        # Check length constraints
        if len(instruction) < self.min_length or len(instruction) > self.max_length:
            return False
            
        if len(response) < self.min_length or len(response) > self.max_length:
            return False
            
        # Check for banned patterns
        for pattern in self.banned_patterns:
            if re.search(pattern, instruction, re.IGNORECASE) or re.search(pattern, response, re.IGNORECASE):
                return False
                
        # Check for duplicate instruction and response
        if instruction.strip() == response.strip():
            return False
            
        # Check for JSON formatting in response if it claims to be JSON
        if "json" in instruction.lower() and response.strip().startswith("{"):
            try:
                json.loads(response)
            except json.JSONDecodeError:
                return False
                
        # Check for code formatting if it claims to contain code
        if any(code_word in instruction.lower() for code_word in ["code", "function", "program"]):
            if "```" not in response and not self._contains_code_indentation(response):
                return False
                
        return True

    def _contains_code_indentation(self, text: str) -> bool:
        """
        Check if text contains code-like indentation patterns.

        Parameters
        ----------
        text : str
            Text to check.

        Returns
        -------
        bool
            True if the text contains code-like indentation, False otherwise.
        """
        lines = text.split("\n")
        indented_lines = 0
        
        for line in lines:
            if line.startswith("    ") or line.startswith("\t"):
                indented_lines += 1
                
        # If at least 3 lines are indented, it's likely code
        return indented_lines >= 3

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the filter configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            New configuration.
        """
        self.config.update(config)
        
        # Update configuration values
        self.enabled = self.config.get("enabled", self.enabled)
        self.min_length = self.config.get("min_length", self.min_length)
        self.max_length = self.config.get("max_length", self.max_length)
        self.banned_patterns = self.config.get("banned_patterns", self.banned_patterns)
        
        self.logger.info("Updated initial filter configuration")

    def _verify_answer(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify the generated results.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Generated results to verify.

        Returns
        -------
        List[Dict[str, Any]]
            Verified results.
        """
        # Placeholder implementation
        # In a real implementation, this would use a verifier
        
        verified_samples = None 
        return verified_samples