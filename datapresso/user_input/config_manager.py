"""
Configuration manager for Datapresso framework.

This module handles loading, validating, and managing user configurations.
"""

import os
import yaml
import json
from typing import Dict, Any, Union, Optional
from pathlib import Path
import logging
from datetime import datetime


class ConfigManager:
    """
    Configuration manager for Datapresso framework.
    
    Handles loading, validating, and managing user configurations.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration manager.

        Parameters
        ----------
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                               "config", "default.yaml")

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the configuration file (YAML or JSON).

        Returns
        -------
        Dict[str, Any]
            Loaded and validated configuration.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the configuration file format is invalid.
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        self.logger.info(f"Loading configuration from: {config_path}")
        
        # Load configuration based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
        # Load default configuration
        default_config = self._load_default_config()
        
        # Merge configurations
        merged_config = self._merge_configs(default_config, user_config)
        
        # Validate the merged configuration
        validated_config = self.validate_config(merged_config)
        
        # Save a copy of the merged configuration
        self._save_config_copy(validated_config, config_path)
        
        return validated_config

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and set default values.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate.

        Returns
        -------
        Dict[str, Any]
            Validated configuration.

        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        # Ensure required top-level keys exist
        required_keys = ["project_name", "seed_db", "quality_assessment", "data_filtering"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        # Validate project name
        if not isinstance(config.get("project_name"), str):
            raise ValueError("Project name must be a string")
            
        # Validate seed database configuration
        if not isinstance(config.get("seed_db"), dict):
            raise ValueError("seed_db configuration must be a dictionary")
            
        if "path" not in config["seed_db"]:
            raise ValueError("Missing required seed_db.path configuration")
            
        # Set default values for optional configurations
        config.setdefault("output_dir", f"outputs/{config['project_name']}")
        
        # Ensure data_generation is properly configured if enabled
        if config.get("data_generation", {}).get("enabled", True):
            if "model" not in config.get("data_generation", {}):
                raise ValueError("Missing required data_generation.model configuration")
                
        # Ensure advanced_assessment is properly configured if enabled
        if config.get("advanced_assessment", {}).get("enabled", False):
            if "model" not in config.get("advanced_assessment", {}):
                raise ValueError("Missing required advanced_assessment.model configuration")
                
        # Ensure training is properly configured if enabled
        if config.get("training", {}).get("enabled", True):
            if "model" not in config.get("training", {}):
                raise ValueError("Missing required training.model configuration")
                
        return config

    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load the default configuration.

        Returns
        -------
        Dict[str, Any]
            Default configuration.
        """
        try:
            with open(self.default_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load default configuration: {str(e)}")
            return {}

    def _merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge default and user configurations.

        Parameters
        ----------
        default_config : Dict[str, Any]
            Default configuration.
        user_config : Dict[str, Any]
            User configuration.

        Returns
        -------
        Dict[str, Any]
            Merged configuration.
        """
        merged = default_config.copy()
        
        def _merge_dict(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge dictionaries."""
            for key, value in overlay.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = _merge_dict(base[key], value)
                else:
                    base[key] = value
            return base
            
        return _merge_dict(merged, user_config)

    def _save_config_copy(self, config: Dict[str, Any], source_path: Path) -> None:
        """
        Save a copy of the merged configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to save.
        source_path : Path
            Path to the source configuration file.
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create directory for config copies if it doesn't exist
            output_dir = Path(config.get("output_dir", "outputs"))
            config_dir = output_dir / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON for consistency
            config_copy_path = config_dir / f"config_{timestamp}.json"
            with open(config_copy_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Saved configuration copy to: {config_copy_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save configuration copy: {str(e)}")

    def generate_default_config(self, output_path: Union[str, Path]) -> None:
        """
        Generate a default configuration file.

        Parameters
        ----------
        output_path : Union[str, Path]
            Path to save the default configuration.
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load default configuration
        default_config = self._load_default_config()
        
        # Save to the specified path
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {output_path.suffix}")
            
        self.logger.info(f"Generated default configuration at: {output_path}")
