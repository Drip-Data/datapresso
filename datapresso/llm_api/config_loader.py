"""
Configuration loading utilities for the LLM API module.
"""

import os
import yaml # Requires PyYAML - should be added to dependencies
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

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

def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a single YAML file and resolve environment variables.

    Parameters
    ----------
    file_path : Path
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        The loaded and resolved configuration dictionary.
    """
    try:
        if not file_path.is_file():
            logger.debug(f"YAML file not found: {file_path}")
            return {}

        logger.debug(f"Loading YAML file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        if not isinstance(raw_config, dict):
            logger.warning(f"YAML file {file_path} does not contain a valid dictionary.")
            return {}

        resolved_config = _resolve_env_vars(raw_config)
        return resolved_config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e}")
        return {}

def _merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Parameters
    ----------
    configs : List[Dict[str, Any]]
        List of configuration dictionaries to merge.

    Returns
    -------
    Dict[str, Any]
        The merged configuration dictionary.
    """
    merged_config = {}

    for config in configs:
        for key, value in config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged_config[key] = {**merged_config[key], **value}
            else:
                # For non-dict values or keys not in merged_config, just update
                merged_config[key] = value

    return merged_config

def load_llm_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads LLM API configuration, supporting both single file and directory-based configurations.

    If config_path points to a file, loads that file directly.
    If config_path points to a directory, loads and merges config.yaml, prompt.yaml, and model.yaml.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the configuration file or directory.

    Returns
    -------
    Dict[str, Any]
        The loaded and merged configuration dictionary with environment variables resolved.

    Raises
    ------
    FileNotFoundError
        If neither the configuration file nor directory exists.
    """
    config_path = Path(config_path)

    # Check if path exists
    if not (config_path.exists()):
        raise FileNotFoundError(f"LLM API config path not found: {config_path.resolve()}")

    # Case 1: config_path is a file - legacy single file mode
    if config_path.is_file():
        logger.debug(f"Loading LLM API configuration from single file: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            if not isinstance(raw_config, dict):
                logger.warning(f"Configuration file {config_path} does not contain a valid YAML dictionary.")
                return {}

            resolved_config = _resolve_env_vars(raw_config)
            logger.debug(f"Successfully loaded and resolved config from {config_path}")
            return resolved_config

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading configuration file {config_path}: {e}")
            raise

    # Case 2: config_path is a directory - new multi-file mode
    elif config_path.is_dir():
        logger.debug(f"Loading LLM API configuration from directory: {config_path}")

        # Define paths to individual config files
        main_config_path = config_path / "config.yaml"
        prompt_config_path = config_path / "prompt.yaml"
        model_config_path = config_path / "model.yaml"

        # Load individual config files
        main_config = _load_yaml_file(main_config_path)
        prompt_config = _load_yaml_file(prompt_config_path)
        model_config = _load_yaml_file(model_config_path)

        # Check if main config exists
        if not main_config:
            logger.warning(f"Main configuration file not found or empty: {main_config_path}")

        # Merge configurations
        # Extract system_prompt_templates and output_schema_templates from prompt_config
        system_prompt_templates = prompt_config.get("system_prompt_templates", {})
        output_schema_templates = prompt_config.get("output_schema_templates", {})

        # Extract provider configurations from model_config
        providers = model_config.get("providers", {})

        # Add templates and providers to main_config
        if system_prompt_templates:
            main_config["system_prompt_templates"] = system_prompt_templates

        if output_schema_templates:
            main_config["output_schema_templates"] = output_schema_templates

        if providers:
            main_config["providers"] = providers

        logger.debug(f"Successfully loaded and merged configurations from {config_path}")
        return main_config

    else:
        # This should never happen as we already checked existence
        raise ValueError(f"Unexpected path type: {config_path}")

def create_config_directory(stage_name: str, config_dir: Path, template_dir: Path) -> Path:
    """
    Create a new configuration directory structure for a stage.

    Parameters
    ----------
    stage_name : str
        Name of the stage (e.g., 'data_generator').
    config_dir : Path
        Base directory for configurations.
    template_dir : Path
        Directory containing template files.

    Returns
    -------
    Path
        Path to the created configuration directory.

    Raises
    ------
    FileNotFoundError
        If template directory does not exist.
    """
    # Ensure template directory exists
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    # Create stage config directory
    stage_config_dir = config_dir / stage_name
    stage_config_dir.mkdir(parents=True, exist_ok=True)

    # Define template file paths
    config_template = template_dir / "config.yaml.template"
    prompt_template = template_dir / "prompt.yaml.template"
    model_template = template_dir / "model.yaml.template"

    # Define target file paths
    config_target = stage_config_dir / "config.yaml"
    prompt_target = stage_config_dir / "prompt.yaml"
    model_target = stage_config_dir / "model.yaml"

    # Copy template files if they don't exist
    if config_template.is_file() and not config_target.is_file():
        shutil.copy(config_template, config_target)
        logger.info(f"Created config file: {config_target}")

    if prompt_template.is_file() and not prompt_target.is_file():
        shutil.copy(prompt_template, prompt_target)
        logger.info(f"Created prompt file: {prompt_target}")

    if model_template.is_file() and not model_target.is_file():
        shutil.copy(model_template, model_target)
        logger.info(f"Created model file: {model_target}")

    logger.info(f"Configuration directory created for stage '{stage_name}' at {stage_config_dir}")
    return stage_config_dir

def migrate_legacy_config(legacy_config_path: Path, target_dir: Path) -> bool:
    """
    Migrate a legacy single-file configuration to the new directory structure.

    Parameters
    ----------
    legacy_config_path : Path
        Path to the legacy configuration file.
    target_dir : Path
        Target directory for the new configuration structure.

    Returns
    -------
    bool
        True if migration was successful, False otherwise.
    """
    try:
        if not legacy_config_path.is_file():
            logger.error(f"Legacy config file not found: {legacy_config_path}")
            return False

        # Load legacy config
        with open(legacy_config_path, 'r', encoding='utf-8') as f:
            legacy_config = yaml.safe_load(f)

        if not isinstance(legacy_config, dict):
            logger.error(f"Legacy config file does not contain a valid YAML dictionary: {legacy_config_path}")
            return False

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract sections
        main_config = {
            key: value for key, value in legacy_config.items()
            if key not in ["system_prompt_templates", "output_schema_templates", "providers"]
        }

        prompt_config = {
            "system_prompt_templates": legacy_config.get("system_prompt_templates", {}),
            "output_schema_templates": legacy_config.get("output_schema_templates", {})
        }

        model_config = {
            "providers": legacy_config.get("providers", {})
        }

        # Save new config files
        with open(target_dir / "config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(main_config, f, default_flow_style=False, sort_keys=False)

        with open(target_dir / "prompt.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(prompt_config, f, default_flow_style=False, sort_keys=False)

        with open(target_dir / "model.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Successfully migrated legacy config {legacy_config_path} to {target_dir}")
        return True

    except Exception as e:
        logger.error(f"Error migrating legacy config {legacy_config_path}: {str(e)}")
        return False