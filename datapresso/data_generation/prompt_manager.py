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

import camel


class TemplateConfig(TypedDict):
    """Type definition for template configuration."""
    system_message: str
    user_template: str

class PromptOutput(TypedDict):
    """Type definition for prompt generation output."""
    system_message: str
    user_message: str

class SeedSample(TypedDict, total=False): # total=False --> 某些字段可能在特定情况下不需要
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
    response: Dict[str, Any]
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

        # Load templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, TemplateConfig]:
        """Load templates from path or use defaults."""
        template_path = self.config.get("template_path")
        load_path = self.config.get("self_load")
        if template_path and load_path:
            self.template_loader = UserTemplateLoader(self.logger)
            templates = self.template_loader.load_templates(template_path, "generation")
            if not templates:
                raise ValueError( f"No valid templates found in {template_path}." )
            return templates
        
        else:
            template_type = self.config.get("template_type")
            if template_type == "reasoning":
                return self._get_default_reasoning_templates()
            elif template_type == "standard":
                return self._get_default_standard_templates()
            else:
                raise ValueError(
                    f"Template type must be one of {self.DEFAULT_TEMPLATE_TYPES}, "
                    f"got: {template_type}"
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
                - Logical progression of reasoning
                
                 **Instructions:**
                    1. **Follow the Structure**  
                    Each data point must include:   
                    - **Question**: A clear, well-formed query.  
                    - **Rationale**: A step-by-step, executable reasoning process ending 
                    with `The final answer is (final_answer)`.  
                    - **Final Answer**: The correct, concise result.  

                **Output Format (Strict)**  
                    ```
                    Question: [Generated question]
                    Rationale: [Steps that solve the question, outputting the answer.]
                    Final Answer: [The Final Answer]""",
            
            "user_template": """
                Example for reference:
                Question: {question}
                Rationale: {rationale}
                Final Answer: {answer}
                
                Now generate a new mathematical reasoning example:"""
        }

        # Return organized templates
        return {
            "code": CODE_TEMPLATE,
            "math": MATH_TEMPLATE
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
            "basic": BASIC_TEMPLATE,
            "domain": DOMAIN_TEMPLATE
        }
       
    def create_prompt(self, seed_sample: SeedSample) -> PromptOutput:
        """
        Create a prompt for generation based on one seed sample.

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

        question = seed_sample.get("instruction", "")
        response_dict = seed_sample.get("response", {})

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


class DistillPromptManager:
    """
    Prompt manager for distillation tasks in Datapresso framework.
    Focuses on generating reasoning process and answer for given questions.
    
    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary for the prompt manager.
    logger : logging.Logger
        Logger instance for tracking operations.
    templates : Dict[str, TemplateConfig]
        Dictionary of loaded templates.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the distillation prompt manager.

        Parameters
        ----------
        config : Dict[str, Any]
            Prompt manager configuration containing template settings.
        logger : Optional[logging.Logger]
            Logger instance for tracking operations, by default None.
        """
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")
            
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
        # Load templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, TemplateConfig]:
        """Load templates from path or use defaults."""
        template_path = self.config.get("template_path")
        load_path = self.config.get("self_load")
        if template_path and load_path:
            self.template_loader = UserTemplateLoader(self.logger)
            templates = self.template_loader.load_templates(template_path, "generation")
            if not templates:
                raise ValueError( f"No valid templates found in {template_path}." )
            return templates
        else:
            return self._get_default_distillation_templates()

    def _get_default_distillation_templates(self) -> Dict[str, TemplateConfig]:
        """Get default distillation templates."""

        CODE_TEMPLATE = {
            "system_message": """\
                You are an advanced code reasoning assistant.
                Your task is to solve programming problems by:
                1. Writing executable Python code that solves the problem
                2. Ensuring the code prints the final answer

                Format your response as:
                Rationale:
                [Your step-by-step reasoning and code explanation]

                Code Solution:
                ```python
                [Your executable Python code]
                ```

                Final Answer:
                [The answer printed by your code]""",
            
            "user_template": "Solve this programming problem:\n{question}\n"
        }

        CODE_THINK_TEMPLATE = {
            "system_message": """\
                You are an advanced code reasoning assistant.
                Your task is to solve programming problems by providing both internal reasoning and final solution.

                Format your response as:
                <think>
                [Your internal step-by-step thought process]
                </think>

                Rationale:
                [Your step-by-step code explanation]

                Code Solution:
                ```python
                [Your executable Python code]
                ```

                Final Answer:
                [The answer printed by your code]""",
            
            "user_template": "Solve this programming problem:\n{question}\n"
        }

        MATH_TEMPLATE = {
            "system_message": """\
                You are an advanced mathematics reasoning assistant.
                Your task is to solve mathematical problems by:
                1. Breaking down the problem into clear steps
                2. Showing your work with proper mathematical notation
                3. Providing a clear final answer
                
                Format your response as:
                Rationale:
                [Your step-by-step mathematical reasoning]
                
                Final Answer:
                [Your final answer in LaTeX format: \boxed{{answer}}]""",
            
            "user_template": "Solve this mathematical problem:\n{question}\n"
        }

        MATH_THINK_TEMPLATE = {
            "system_message": """\
                You are an advanced mathematics reasoning assistant.
                Your task is to solve mathematical problems by providing both internal reasoning and final solution.

                Format your response as:
                <think>
                [Your internal mathematical thinking process]
                </think>

                Rationale:
                [Your formal step-by-step mathematical derivation]
                
                Final Answer:
                [Your final answer in LaTeX format: \boxed{{answer}}]""",
            
            "user_template": "Solve this mathematical problem:\n{question}\n"
        }

        templates = {
            "code_reasoning": CODE_TEMPLATE,
            "math_reasoning": MATH_TEMPLATE,
        }

        # Add think templates if required
        if self.config.get("include_think_process", False):
            templates.update({
                "code_reasoning_think": CODE_THINK_TEMPLATE,
                "math_reasoning_think": MATH_THINK_TEMPLATE
            })

        return templates

    def create_prompt(self, seed_sample: Dict[str, str]) -> PromptOutput:
        """
        Create a prompt for distillation based on a seed sample.

        Parameters
        ----------
        seed_sample : Dict[str, str]
            Dictionary containing the question and optional domain metadata.

        Returns
        -------
        PromptOutput
            Dictionary containing system_message and user_message.
            If include_think_process is True in config, the template will include
            the <think> tags for capturing internal reasoning process.
        """
        template_name = self.config.get("template_name")
        include_think = self.config.get("include_think_process", False)

        # Modify template name if think process is required
        if include_think:
            template_name = template_name+"_think"

        template = self.templates.get(template_name)
        if not template:    
            raise ValueError(
                f"Template '{template_name}' not found in available templates: "
                f"{list(self.templates.keys())}"
            )

        # Clean template format
        system_message = _clean_template_format(template["system_message"])
        user_template = _clean_template_format(template["user_template"])

        question = seed_sample.get("instruction", "")
        user_message = user_template.format(question=question)

        return {
            "system_message": system_message,
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


class VerificationPromptManager:
    """
    Prompt manager for verification tasks in Datapresso framework.
    Focuses on generating prompts for verifying the correctness of generated answers.
    
    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary for the prompt manager.
    logger : logging.Logger
        Logger instance for tracking operations.
    templates : Dict[str, TemplateConfig]
        Dictionary of loaded templates.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the verification prompt manager.

        Parameters
        ----------
        config : Dict[str, Any]
            Prompt manager configuration containing template settings.
        logger : Optional[logging.Logger]
            Logger instance for tracking operations, by default None.
        """

        self.config = config
        # Load templates
        self.templates = self._load_templates()
        

    def create_prompt(self, seed_sample: Dict[str, str]) -> PromptOutput:
        question = seed_sample.get("instruction", "")
        template_name = self.config.get("template_name")
        template = self.templates.get(template_name)

        question = seed_sample.get("instruction", "")
        
        user_message = template["user_template"].format(
            question=question)
        
        return {
            "system_message": template["system_message"],
            "user_message": user_message
        }
    
    def _load_templates(self) -> Dict[str, TemplateConfig]:
        """Load templates from path or use defaults."""
        template_path = self.config.get("template_path")
        load_path = self.config.get("self_load")
        if template_path and load_path:
            self.template_loader = UserTemplateLoader(self.logger)
            templates = self.template_loader.load_templates(template_path, "generation")
            if not templates:
                raise ValueError( f"No valid templates found in {template_path}." )

        else:
            templates = self.get_templates()
        return templates

    def get_templates(self) -> PromptOutput:
        VERIFICATION_TEMPLATE = {
            "system_message": """\
                You are an agent designed to answer mathematical questions with clarity and precision.
                Your task is to provide a step-by-step explanation for any mathematical problem posed by the user, ensuring the response is easy to follow. Adhere to these guidelines:
                Analyze the mathematical question carefully and break down the solution process into clear, logical steps.
                Use natural language to explain each step, incorporating LaTeX notation (e.g., $x + 2$)
                for mathematical expressions when helpful. Conclude your response with the final answer enclosed
                in a LaTeX \boxed{} environment (e.g., \boxed{5}).
                Place this at the end of your explanation as a standalone statement.
                It should be a Python expression, for example "[1, 2, 3]" for a list. """,
            
            "user_template": "The question you should answer is:\n{question}\n"
        }
       
        templates = {
            "cot_verification": VERIFICATION_TEMPLATE,
        }
        return templates



# TODO:check the code , currently do not support self-loading 
class UserTemplateLoader:
    """
    Handles loading of user-defined templates from local files.
    Supports both generation and distillation templates.
    
    Attributes
    ----------
    logger : logging.Logger
        Logger instance for tracking operations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the template loader.

        Parameters
        ----------
        logger : Optional[logging.Logger]
            Logger instance for tracking operations.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def load_templates(self, template_path: Union[str, Path], template_type: str) -> Dict[str, TemplateConfig]:
        """
        Load templates from the specified path.
        
        Parameters
        ----------
        template_path : Union[str, Path]
            Path to template directory.
        template_type : str
            Type of templates to load ('generation' or 'distillation').
            
        Returns
        -------
        Dict[str, TemplateConfig]
            Dictionary of loaded templates.
        """
        if isinstance(template_path, str):
            template_path = Path(template_path)
            
        if not template_path.exists():
            self.logger.warning(f"Template path not found: {template_path}")
            return {}

        templates = {}
        
        # Load templates based on type
        if template_type == "generation":
            templates.update(self._load_generation_templates(template_path))
        elif template_type == "distillation":
            templates.update(self._load_distillation_templates(template_path))
        else:
            self.logger.warning(f"Unknown template type: {template_type}")
            
        return templates

    def _load_generation_templates(self, template_path: Path) -> Dict[str, TemplateConfig]:
        """Load generation-specific templates."""
        templates = {}
        generation_path = template_path / "generation"
        
        if generation_path.exists():
            for file_path in generation_path.glob("*.{json,yaml,yml}"):
                template = self._load_single_template(file_path)
                if template:
                    templates[file_path.stem] = template
                    
        return templates

    def _load_distillation_templates(self, template_path: Path) -> Dict[str, TemplateConfig]:
        """Load distillation-specific templates."""
        templates = {}
        distillation_path = template_path / "distillation"
        
        if distillation_path.exists():
            for file_path in distillation_path.glob("*.{json,yaml,yml}"):
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
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in {'.yaml', '.yml'}:
                    template_data = yaml.safe_load(f)
                else:
                    template_data = json.load(f)
                    
            if self._is_valid_template(template_data):
                self.logger.info(f"Loaded template: {file_path.stem}")
                return template_data
                
            self.logger.warning(f"Invalid template format in {file_path}")
            
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

def _clean_template_format(template: str) -> str:
    """
    Clean template format by removing extra indentation while preserving newlines.
    
    Parameters
    ----------
    template : str
        Input template string with potential extra indentation.
        
    Returns
    -------
    str
        Cleaned template string.
    """
    # Split into lines and remove empty strings
    lines = [line for line in template.split('\n') if line.strip()]
    
    # Find minimum indentation (excluding empty lines)
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    
    # Remove common indentation and join lines
    cleaned_lines = [line[min_indent:] if line.strip() else '' for line in lines]
    return '\n'.join(cleaned_lines)
