"""
Prompt manager for Datapresso framework.

This module manages prompt templates for data generation, supporting both standard instruction-response
and reasoning-based templates. Templates can be loaded from files or used from default configurations.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TypedDict
import json
import yaml


class TemplateConfig(TypedDict):
    """Type definition for template configuration."""
    system_message: str
    user_template: str


class PromptOutput(TypedDict):
    """Type definition for prompt generation output."""
    system_message: str
    user_message: str


class SeedSample(TypedDict, total=False):
    """
    Type definition for seed sample input.
    
    Attributes
    ----------
    question : str
        The input question or instruction.
    response : Dict[str, str]
        Response containing rationale and/or final answer.
        May include:
        - rationale: Step-by-step reasoning or code
        - final_answer: The final answer
        - origin_text: Original response text
    metadata : Dict[str, Any]
        Additional metadata about the sample.
    """
    question: str
    response: Dict[str, str]
    metadata: Dict[str, Any]


class GenerationPromptManager:
    """
    Prompt manager for few-shot generation in Datapresso framework.
    
    This class manages the loading and creation of prompts for generating new
    (question, rationale, answer) triples based on seed examples.
    
    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary for the prompt manager.
    logger : logging.Logger
        Logger instance for tracking operations.
    templates : Dict[str, TemplateConfig]
        Dictionary of loaded templates.
    """
    
    DEFAULT_TEMPLATE_TYPES = {"standard", "reasoning"}
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the prompt manager.

        Parameters
        ----------
        config : Dict[str, Any]
            Prompt manager configuration containing template settings.
        logger : Optional[logging.Logger]
            Logger instance for tracking operations, by default None.

        Raises
        ------
        TypeError
            If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")
            
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.templates = self._load_templates()
        
        self.logger.info(f"Initialized prompt manager with {len(self.templates)} templates")

    def _load_templates(self) -> Dict[str, TemplateConfig]:
        """
        Load prompt templates from files or use defaults based on configuration.
        
        Returns
        -------
        Dict[str, TemplateConfig]
            Dictionary mapping template names to template configurations.
        
        Raises
        ------
        ValueError
            If template type is invalid or no valid templates found.
        """
        template_type = self.config.get("template_type")
        template_path = self.config.get("path")
        
        #TODO check 
        if template_path:
            templates = self._load_templates_from_path(Path(template_path))
            if not templates:
                raise ValueError(f"No valid templates found in {template_path}")
            return templates
        
        if template_type not in self.DEFAULT_TEMPLATE_TYPES:
            raise ValueError(
                f"Template type must be one of {self.DEFAULT_TEMPLATE_TYPES}, "
                f"got: {template_type}"
            )
        
        return (
            self._get_default_reasoning_templates() if template_type == "reasoning"
            else self._get_default_standard_templates()
        )

    def _load_templates_from_path(self, template_path: Path) -> Dict[str, TemplateConfig]:
        """
        Load templates from files in the specified path.
        
        Parameters
        ----------
        template_path : Path
            Path to template directory.
            
        Returns
        -------
        Dict[str, TemplateConfig]
            Dictionary of loaded templates.
        """
        if not template_path.exists():
            self.logger.warning(f"Template path not found: {template_path}")
            return {}
            
        templates = {}
        for file_path in template_path.glob("*.{json,yaml,yml,txt}"):
            if not os.access(file_path, os.R_OK):
                self.logger.error(f"Permission denied: Cannot read {file_path}")
                continue
                
            template = self._load_single_template(file_path)
            if template:
                templates[file_path.stem] = template
                
        return templates

    def _load_single_template(self, file_path: Path) -> Optional[TemplateConfig]:
        """
        Load a single template file.
        
        Parameters
        ----------
        file_path : Path
            Path to template file.
            
        Returns
        -------
        Optional[TemplateConfig]
            Loaded template or None if loading failed.
        """
        try:
            if file_path.suffix.lower() in {'.yaml', '.yml'}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
            else:  # .txt
                with open(file_path, 'r', encoding='utf-8') as f:
                    template_data = {
                        "system_message": "Default system message for text template",
                        "user_template": f.read()
                    }
                    
            if self._is_valid_template(template_data):
                self.logger.info(f"Loaded template: {file_path.stem}")
                return template_data
            
            self.logger.warning(f"Invalid template format in {file_path}")
            return None
                    
        except Exception as e:
            self.logger.error(f"Error loading template {file_path}: {str(e)}")
            return None

    def _is_valid_template(self, template_data: Any) -> bool:
        """
        Validate template data structure.
        
        Parameters
        ----------
        template_data : Any
            Template data to validate.
            
        Returns
        -------
        bool
            True if template is valid, False otherwise.
        """
        return (
            isinstance(template_data, dict) 
            and "system_message" in template_data 
            and "user_template" in template_data
            and isinstance(template_data["system_message"], str)
            and isinstance(template_data["user_template"], str)
        )
    
    def _get_default_reasoning_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Get default reasoning prompt templates with separate system and user messages.
        Includes templates for code and math reasoning tasks.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary of default templates, each containing system_message and user_template.
        """
        # Define template categories
        CODE_TEMPLATE = {
            "system_message": """**You are an advanced data generation assistant.**  
                    Your goal is to generate high-quality synthetic data points based on 
                    provided examples. Your output must be well-structured, 
                    logically sound, and formatted correctly. 

                    **Instructions:**
                    1. **Follow the Structure**  
                    Each data point must include:  
                    - **Question**: A clear, well-formed query.  
                    - **Rationale**: A step-by-step, executable reasoning process ending 
                    with `print(final_answer)`.  
                    - **Final Answer**: The correct, concise result.  

                    2. **Ensure Logical Consistency**  
                    - The `rationale` must be code that runs correctly.  
                    - The `final_answer` should match the printed output.  

                    3. **Output Format (Strict)**  
                    ```
                    Question: [Generated question]
                    Rationale: [Code that solves the question, ending in a print statement,
                    outputting the answer.]
                    Final Answer: [The Final Answer]

                    **Now, generate a new data point based on the given examples.**""",
        
            "user_template": """
                            Question: {question}
                            Rationale: {rationale}
                            Final Answer: {answer}
                            New datapoint:
                            """
        }

        MATH_TEMPLATE = {
            "system_message": """You are an advanced mathematics reasoning assistant.
                Your role is to generate high-quality mathematical problems and their step-by-step solutions.
                
                Focus on:
                - Mathematical rigor and accuracy
                - Clear step-by-step derivations
                - Proper mathematical notation
                - Logical progression of reasoning""",
            
            "user_template": """
                Example for reference:
                Question: {question}
                Rationale: {rationale}
                Final Answer: {answer}
                
                Now generate a new mathematical reasoning example:"""
        }

        # Return organized templates
        return {
            "code_generation": CODE_TEMPLATE,
            "math_generation": MATH_TEMPLATE
        }
    
    def _get_default_standard_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Get default standard prompt templates with separate system and user messages.
        Includes templates for basic generation and domain-specific tasks.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary of default templates, each containing system_message and user_template.
        """
        # Define template categories
        BASIC_TEMPLATE = {
            "system_message": """You are an AI assistant helping to generate high-quality instruction-response pairs for AI training.
                Your task is to generate new, unique pairs that maintain consistent quality and format while avoiding duplicates.""",
            "user_template": """Based on the example below, generate a new instruction-response pair:

                Example:
                Instruction: {question}
                Response: {answer}

                Generate a new instruction-response pair:"""
        }

        DOMAIN_TEMPLATE = {
            "system_message": """You are an AI assistant specialized in generating high-quality instruction-response pairs for AI training.""",
            "user_template": """ 
                You should generating high-quality instruction-response pairs for {domain} domain training.
                Ensure all generated content is accurate and domain-appropriate.
            
            Based on the example below, generate a new domain-specific pair:
                Example:
                Instruction: {question}
                Response: {answer}

                Generate a new {domain} instruction-response pair:"""
        }

        # Return organized templates
        return {
            "basic_generation": BASIC_TEMPLATE,
            "domain_specific": DOMAIN_TEMPLATE
        }
    
    
    def create_prompt(self, seed_sample: SeedSample) -> PromptOutput:
        """
        Create a prompt for generation based on a seed sample.

        Parameters
        ----------
        seed_sample : SeedSample
            Seed sample containing question, response (with optional rationale), and metadata.

        Returns
        -------
        PromptOutput
            Dictionary containing system_message and user_message.

        Raises
        ------
        ValueError
            If template_name is not found or seed_sample lacks required fields.
        """
        template_type = self.config.get("template_type")
        template_name = self.config.get("template_name")

        template = self.templates.get(template_name)
        if not template:
            raise ValueError(
                f"Template '{template_name}' not found in available templates: "
                f"{list(self.templates.keys())}"
            )

        question = seed_sample.get("question", "")
        response_dict = seed_sample.get("response", {})
        print(response_dict)
        
        # Get final answer from response dict, with fallbacks
        final_answer = (
            response_dict.get("final_answer")
            or response_dict.get("origin_text")
            or ""
        )
        
        if template_type == "reasoning":
            # Get rationale from response dict, with fallback to empty string
            rationale = response_dict.get("rationale", "")
            user_message = template["user_template"].format(
                question=question,
                rationale=rationale,
                answer=final_answer
            )
        else:
            domain = seed_sample.get("metadata", {}).get("domain", "general")
            user_message = template["user_template"].format(
                question=question,
                answer=final_answer,
                domain=domain
            )

        return {
            "system_message": template["system_message"],
            "user_message": user_message
        }
        
      
    def get_template_names(self) -> List[str]:
        """
        Get the names of available templates.

        Returns
        -------
        List[str]
            List of template names.
        """
        return list(self.templates.keys())

    def get_template(self, name: str) -> Optional[TemplateConfig]:
        """
        Get a specific template by name.

        Parameters
        ----------
        name : str
            Template name.

        Returns
        -------
        Optional[TemplateConfig]
            Template configuration if found, None otherwise.
        """
        return self.templates.get(name)



# class DistillationPromptManager():
#     """
#     Prompt manager for distillation tasks in Datapresso framework.
#     Focuses on generating reasoning process and answer for given questions.
#     """

#     def create_prompt(self, question: str, domain: str = "general") -> Dict[str, str]:
#         """
#         Create a prompt for distillation based on a question.

#         Parameters
#         ----------
#         question : str
#             The question to be solved
#         domain : str, optional
#             Domain of the question (e.g., "math", "code"), by default "general"

#         Returns
#         -------
#         Dict[str, str]
#             Dictionary containing system_message and user_message
#         """
#         template_name = self._select_distillation_template(domain)
#         template = self.templates.get(template_name, self.templates["math_reasoning"])
        
#         return {
#             "system_message": template["system_message"],
#             "user_message": template["user_template"].format(question=question)
#         }

#     def _load_templates(self) -> Dict[str, Dict[str, str]]:
#         """Load distillation-specific templates."""
#         template_path = self.config.get("path")
        
#         if template_path:
#             # TODO: Implement custom template loading from path
#             pass
            
#         return self._get_default_distillation_templates()

#     def _get_default_distillation_templates(self) -> Dict[str, Dict[str, str]]:
#         """Get default distillation templates."""
#         return {
#             "code_reasoning": {
#                 "system_message": """You are an advanced code reasoning assistant.
#                     Your task is to solve programming problems by:
#                     1. Writing executable Python code that solves the problem
#                     2. Ensuring the code prints the final answer
#                     3. Providing clear explanations for your solution
                    
#                     Format your response as:
#                     Rationale:
#                     [Your step-by-step reasoning and code explanation]
                    
#                     Code Solution:
#                     ```python
#                     [Your executable Python code]
#                     ```
                    
#                     Final Answer:
#                     [The answer printed by your code]""",
                
#                 "user_template": "Solve this programming problem:\n{question}\n"
#             },
            
#             "math_reasoning": {
#                 "system_message": """You are an advanced mathematics reasoning assistant.
#                     Your task is to solve mathematical problems by:
#                     1. Breaking down the problem into clear steps
#                     2. Showing your work with proper mathematical notation
#                     3. Providing a clear final answer
                    
#                     Format your response as:
#                     Rationale:
#                     [Your step-by-step mathematical reasoning]
                    
#                     Final Answer:
#                     [Your final answer in LaTeX format: \boxed{{answer}}]""",
                
#                 "user_template": "Solve this mathematical problem:\n{question}\n"
#             }
#         }

#     def _select_distillation_template(self, domain: str) -> str:
#         """Select appropriate template based on question domain."""
#         domain_template_mapping = {
#             "code": "code_reasoning",
#             "math": "math_reasoning",
#         }
#         return domain_template_mapping.get(domain.lower(), "math_reasoning")
