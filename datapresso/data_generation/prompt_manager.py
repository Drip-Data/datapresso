"""
Prompt manager for Datapresso framework.

This module manages prompt templates for data generation.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import yaml
import random


class PromptManager:
    """
    Prompt manager for Datapresso framework.
    
    Manages prompt templates for data generation.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the prompt manager.

        Parameters
        ----------
        config : Dict[str, Any]
            Prompt manager configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Load templates
        self.templates = self._load_templates()
        
        self.logger.info(f"Initialized prompt manager with {len(self.templates)} templates")

    def _load_templates(self) -> Dict[str, str]:
        """
        Load prompt templates from files.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping template names to template strings.
        """
        templates = {}
        
        # Check if template path is provided
        template_path = self.config.get("path")
        if not template_path:
            # Use default templates
            templates = self._get_default_templates()
            self.logger.info("Using default prompt templates")
            return templates
            
        # Load templates from path
        template_path = Path(template_path)
        if not template_path.exists():
            self.logger.warning(f"Template path not found: {template_path}")
            templates = self._get_default_templates()
            return templates
            
        # Load templates from files
        for file_path in template_path.glob("*.{json,yaml,yml,txt}"):
            try:
                template_name = file_path.stem
                
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                else:  # .txt
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_data = {"template": f.read()}
                        
                if isinstance(template_data, dict) and "template" in template_data:
                    templates[template_name] = template_data["template"]
                    self.logger.info(f"Loaded template: {template_name}")
                else:
                    self.logger.warning(f"Invalid template format in {file_path}")
                    
            except Exception as e:
                self.logger.error(f"Error loading template {file_path}: {str(e)}")
                
        # If no templates were loaded, use defaults
        if not templates:
            templates = self._get_default_templates()
            self.logger.info("No valid templates found, using defaults")
            
        return templates

    def _get_default_templates(self) -> Dict[str, str]:
        """
        Get default prompt templates.

        Returns
        -------
        Dict[str, str]
            Dictionary of default templates.
        """
        return {
            "basic_generation": """
You are an AI assistant helping to generate high-quality instruction-response pairs for AI training.
Based on the example below, generate a new, unique instruction-response pair that follows a similar pattern but is not a duplicate.

Example:
Instruction: {instruction}
Response: {response}

Generate a new instruction-response pair:
""",
            "domain_specific": """
You are an AI assistant helping to generate high-quality instruction-response pairs for AI training in the domain of {domain}.
The difficulty level should be approximately {difficulty} on a scale of 0 to 1, where 0 is very easy and 1 is very challenging.

Based on the example below, generate a new, unique instruction-response pair that follows a similar pattern but is not a duplicate.

Example:
Instruction: {instruction}
Response: {response}

Generate a new instruction-response pair in the {domain} domain:
""",
            "complex_reasoning": """
You are an AI assistant helping to generate high-quality instruction-response pairs that require complex reasoning.
The generated pair should require multi-step reasoning, careful analysis, and detailed explanation.

Based on the example below, generate a new, unique instruction-response pair that follows a similar pattern but is not a duplicate.

Example:
Instruction: {instruction}
Response: {response}

Generate a new instruction-response pair that requires complex reasoning:
"""
        }

    def create_prompt(self, seed_sample: Dict[str, Any]) -> str:
        """
        Create a prompt for generation based on a seed sample.

        Parameters
        ----------
        seed_sample : Dict[str, Any]
            Seed sample to use as reference.

        Returns
        -------
        str
            Generated prompt.
        """
        # Get metadata from seed sample
        domain = seed_sample.get("metadata", {}).get("domain", "general")
        difficulty = seed_sample.get("metadata", {}).get("difficulty", 0.5)
        
        # Select template based on domain and difficulty
        template_name = self._select_template(domain, difficulty)
        template = self.templates.get(template_name, self.templates["basic_generation"])
        
        # Format the template
        prompt = template.format(
            instruction=seed_sample.get("instruction", ""),
            response=seed_sample.get("response", ""),
            domain=domain,
            difficulty=difficulty
        )
        
        return prompt

    def _select_template(self, domain: str, difficulty: float) -> str:
        """
        Select an appropriate template based on domain and difficulty.

        Parameters
        ----------
        domain : str
            Domain of the sample.
        difficulty : float
            Difficulty level of the sample.

        Returns
        -------
        str
            Selected template name.
        """
        # This is a simple selection strategy
        # In a real implementation, this would be more sophisticated
        
        if difficulty > 0.7:
            return "complex_reasoning"
        elif domain != "general":
            return "domain_specific"
        else:
            return "basic_generation"

    def get_template_names(self) -> List[str]:
        """
        Get the names of available templates.

        Returns
        -------
        List[str]
            List of template names.
        """
        return list(self.templates.keys())

    def get_template(self, name: str) -> Optional[str]:
        """
        Get a specific template by name.

        Parameters
        ----------
        name : str
            Template name.

        Returns
        -------
        Optional[str]
            Template string if found, None otherwise.
        """
        return self.templates.get(name)
