"""
Configuration validator for Datapresso framework.

This module provides validation for user configurations.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import os
from pathlib import Path


class ConfigValidator:
    """
    Configuration validator for Datapresso framework.
    
    Validates user configurations and provides helpful error messages.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration validator.

        Parameters
        ----------
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.

        Returns
        -------
        Tuple[bool, List[str]]
            Tuple of (is_valid, error_messages).
        """
        errors = []
        
        # Validate project settings
        self._validate_project_settings(config, errors)
        
        # Validate seed database settings
        self._validate_seed_db(config, errors)
        
        # Validate data generation settings if enabled
        if config.get("data_generation", {}).get("enabled", True):
            self._validate_data_generation(config, errors)
            
        # Validate quality assessment settings
        self._validate_quality_assessment(config, errors)
        
        # Validate data filtering settings
        self._validate_data_filtering(config, errors)
        
        # Validate advanced assessment settings if enabled
        if config.get("advanced_assessment", {}).get("enabled", False):
            self._validate_advanced_assessment(config, errors)
            
        # Validate training settings if enabled
        if config.get("training", {}).get("enabled", True):
            self._validate_training(config, errors)
            
        # Validate evaluation settings
        self._validate_evaluation(config, errors)
        
        # Validate logging settings
        self._validate_logging(config, errors)
        
        # Log validation results
        if errors:
            self.logger.error(f"Configuration validation failed with {len(errors)} errors")
            for i, error in enumerate(errors, 1):
                self.logger.error(f"Error {i}: {error}")
        else:
            self.logger.info("Configuration validation successful")
            
        return len(errors) == 0, errors

    def _validate_project_settings(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate project settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "project_name" not in config:
            errors.append("Missing required configuration: project_name")
        elif not isinstance(config["project_name"], str):
            errors.append("project_name must be a string")
            
        if "output_dir" in config and not isinstance(config["output_dir"], str):
            errors.append("output_dir must be a string")

    def _validate_seed_db(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate seed database settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "seed_db" not in config:
            errors.append("Missing required configuration: seed_db")
            return
            
        seed_db = config["seed_db"]
        
        if not isinstance(seed_db, dict):
            errors.append("seed_db must be a dictionary")
            return
            
        if "path" not in seed_db:
            errors.append("Missing required configuration: seed_db.path")
        elif not isinstance(seed_db["path"], str):
            errors.append("seed_db.path must be a string")
            
        if "format" in seed_db and seed_db["format"] not in ["jsonl", "json", "csv", "tsv"]:
            errors.append("seed_db.format must be one of: jsonl, json, csv, tsv")
            
        # Check if validation is enabled and properly configured
        if seed_db.get("validation", {}).get("enabled", True):
            validation = seed_db.get("validation", {})
            if "schema_check" in validation and not isinstance(validation["schema_check"], bool):
                errors.append("seed_db.validation.schema_check must be a boolean")

    def _validate_data_generation(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate data generation settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "data_generation" not in config:
            errors.append("Missing required configuration: data_generation")
            return
            
        data_gen = config["data_generation"]
        
        if not isinstance(data_gen, dict):
            errors.append("data_generation must be a dictionary")
            return
            
        if "model" not in data_gen:
            errors.append("Missing required configuration: data_generation.model")
        elif not isinstance(data_gen["model"], str):
            errors.append("data_generation.model must be a string")
            
        if "target_count" in data_gen and not isinstance(data_gen["target_count"], int):
            errors.append("data_generation.target_count must be an integer")
            
        if "temperature" in data_gen:
            temp = data_gen["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append("data_generation.temperature must be a number between 0 and 2")
                
        if "prompt_templates" in data_gen:
            templates = data_gen["prompt_templates"]
            if "path" in templates and not isinstance(templates["path"], str):
                errors.append("data_generation.prompt_templates.path must be a string")

    def _validate_quality_assessment(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate quality assessment settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "quality_assessment" not in config:
            errors.append("Missing required configuration: quality_assessment")
            return
            
        qa = config["quality_assessment"]
        
        if not isinstance(qa, dict):
            errors.append("quality_assessment must be a dictionary")
            return
            
        if "metrics" in qa:
            metrics = qa["metrics"]
            if not isinstance(metrics, list):
                errors.append("quality_assessment.metrics must be a list")
            elif not all(isinstance(m, str) for m in metrics):
                errors.append("All items in quality_assessment.metrics must be strings")
                
        if "verification_methods" in qa:
            methods = qa["verification_methods"]
            if not isinstance(methods, list):
                errors.append("quality_assessment.verification_methods must be a list")
            elif not all(isinstance(m, str) for m in methods):
                errors.append("All items in quality_assessment.verification_methods must be strings")
                
        if "llm_evaluator" in qa:
            evaluator = qa["llm_evaluator"]
            if "model" in evaluator and not isinstance(evaluator["model"], str):
                errors.append("quality_assessment.llm_evaluator.model must be a string")

    def _validate_data_filtering(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate data filtering settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "data_filtering" not in config:
            errors.append("Missing required configuration: data_filtering")
            return
            
        df = config["data_filtering"]
        
        if not isinstance(df, dict):
            errors.append("data_filtering must be a dictionary")
            return
            
        if "quality_threshold" in df:
            threshold = df["quality_threshold"]
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                errors.append("data_filtering.quality_threshold must be a number between 0 and 1")
                
        if "diversity_weight" in df:
            weight = df["diversity_weight"]
            if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
                errors.append("data_filtering.diversity_weight must be a number between 0 and 1")
                
        if "target_size" in df and not isinstance(df["target_size"], int):
            errors.append("data_filtering.target_size must be an integer")
            
        if "difficulty_distribution" in df:
            dist = df["difficulty_distribution"]
            if not isinstance(dist, dict):
                errors.append("data_filtering.difficulty_distribution must be a dictionary")
            else:
                total = sum(dist.values())
                if not (0.99 <= total <= 1.01):  # Allow for floating point imprecision
                    errors.append("data_filtering.difficulty_distribution values must sum to 1.0")

    def _validate_advanced_assessment(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate advanced assessment settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "advanced_assessment" not in config:
            return  # Optional section
            
        aa = config["advanced_assessment"]
        
        if not isinstance(aa, dict):
            errors.append("advanced_assessment must be a dictionary")
            return
            
        if aa.get("enabled", False):
            if "model" not in aa:
                errors.append("Missing required configuration: advanced_assessment.model")
            elif not isinstance(aa["model"], str):
                errors.append("advanced_assessment.model must be a string")
                
            if "methods" in aa:
                methods = aa["methods"]
                if not isinstance(methods, list):
                    errors.append("advanced_assessment.methods must be a list")
                elif not all(isinstance(m, str) for m in methods):
                    errors.append("All items in advanced_assessment.methods must be strings")
                elif not methods:
                    errors.append("advanced_assessment.methods cannot be empty if enabled")

    def _validate_training(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate training settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "training" not in config:
            return  # Optional section
            
        training = config["training"]
        
        if not isinstance(training, dict):
            errors.append("training must be a dictionary")
            return
            
        if training.get("enabled", True):
            if "model" not in training:
                errors.append("Missing required configuration: training.model")
            elif not isinstance(training["model"], str):
                errors.append("training.model must be a string")
                
            if "batch_size" in training and not isinstance(training["batch_size"], int):
                errors.append("training.batch_size must be an integer")
                
            if "learning_rate" in training:
                lr = training["learning_rate"]
                if not isinstance(lr, (int, float)) or lr <= 0:
                    errors.append("training.learning_rate must be a positive number")
                    
            if "epochs" in training and not isinstance(training["epochs"], int):
                errors.append("training.epochs must be an integer")

    def _validate_evaluation(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate evaluation settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "evaluation" not in config:
            return  # Optional section
            
        eval_config = config["evaluation"]
        
        if not isinstance(eval_config, dict):
            errors.append("evaluation must be a dictionary")
            return
            
        if "benchmark_datasets" in eval_config:
            datasets = eval_config["benchmark_datasets"]
            if not isinstance(datasets, list):
                errors.append("evaluation.benchmark_datasets must be a list")
            elif not all(isinstance(d, str) for d in datasets):
                errors.append("All items in evaluation.benchmark_datasets must be strings")
                
        if "metrics" in eval_config:
            metrics = eval_config["metrics"]
            if not isinstance(metrics, list):
                errors.append("evaluation.metrics must be a list")
            elif not all(isinstance(m, str) for m in metrics):
                errors.append("All items in evaluation.metrics must be strings")

    def _validate_logging(self, config: Dict[str, Any], errors: List[str]) -> None:
        """
        Validate logging settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.
        errors : List[str]
            List to append error messages to.
        """
        if "logging" not in config:
            return  # Optional section
            
        logging_config = config["logging"]
        
        if not isinstance(logging_config, dict):
            errors.append("logging must be a dictionary")
            return
            
        if "level" in logging_config:
            level = logging_config["level"]
            if not isinstance(level, str) or level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                errors.append("logging.level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
                
        if "save_path" in logging_config and not isinstance(logging_config["save_path"], str):
            errors.append("logging.save_path must be a string")
            
        for bool_key in ["console_output", "file_output"]:
            if bool_key in logging_config and not isinstance(logging_config[bool_key], bool):
                errors.append(f"logging.{bool_key} must be a boolean")
