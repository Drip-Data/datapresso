"""
Configuration loading utilities for the LLM API module.
"""

import os
import yaml # Requires PyYAML - should be added to dependencies
import logging
from pathlib import Path
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

def _resolve_env_vars(cfg: Any) -> Any:
    """Recursively resolves environment variables specified in `env(VAR_NAME)` format."""
    if isinstance(cfg, dict):
        return {k: _resolve_env_vars(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [_resolve_env_vars(i) for i in cfg]
    elif isinstance(cfg, str) and cfg.startswith('env(') and cfg.endswith(')'):
        var_name = cfg[4:-1]
        value = os.environ.get(var_name)
        if value is None:
            # Log a warning but return None, allowing for optional env vars
            logger.warning(f"Environment variable '{var_name}' referenced in config but not found.")
            return None
        return value
    return cfg

def load_llm_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a YAML configuration file for the LLM API, resolving environment variables.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        The loaded configuration dictionary with environment variables resolved.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the file cannot be parsed as YAML.
    Exception
        For other potential file reading errors.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"LLM API config file not found at: {config_path.resolve()}")

    logger.debug(f"Loading LLM API configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        if not isinstance(raw_config, dict):
             logger.warning(f"Configuration file {config_path} does not contain a valid YAML dictionary.")
             return {} # Return empty dict if file is empty or invalid top level

        resolved_config = _resolve_env_vars(raw_config)
        logger.debug(f"Successfully loaded and resolved config from {config_path}")
        return resolved_config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise # Re-raise the YAML error
    except Exception as e:
        logger.error(f"Error reading configuration file {config_path}: {e}")
        raise # Re-raise other file reading errors