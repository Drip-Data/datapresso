"""
Datapresso LLM API Layer Initialization

This module provides a simplified, stage-specific interface for interacting with
various LLM providers configured centrally within this package.

Usage:
    from datapresso.llm_api import data_generator_api, quality_assessor_api

    # Simple generation using data_generator's config
    response = data_generator_api.generate(user_prompt="Hello!")

    # Structured generation using quality_assessor's config and templates
    response_structured = quality_assessor_api.generate_with_structured_output(
        user_prompt="Assess this text.",
        output_schema_template="quality_score",
        system_prompt_template="quality_assessor"
    )

To add support for a new stage (e.g., 'my_analyzer'):
1. Add 'my_analyzer' to the KNOWN_STAGES list below (or call register_llm_stage).
2. A configuration directory will be created at `datapresso/datapresso/llm_api/configs/my_analyzer/`
   with separate files for main configuration, prompts, and model schemas.
3. Users can then `from datapresso.llm_api import my_analyzer_api`.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

# Core components
from .llm_provider import LLMProvider
from .llm_api_manager import LLMAPIManager
from .stage_api import StageLLMApi
from .config_loader import load_llm_config, create_config_directory, migrate_legacy_config # Keep loaders accessible if needed elsewhere

# Provider implementations
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider # Requires google-generativeai
from .deepseek_provider import DeepSeekProvider
from .generic_openai_provider import GenericOpenAIProvider
try:
    from .local_provider import LocalProvider
    HAS_LOCAL_PROVIDER = True
except ImportError:
    LocalProvider = None # type: ignore # Make type checkers happy
    HAS_LOCAL_PROVIDER = False

logger = logging.getLogger(__name__)

# --- Configuration ---
_current_dir = Path(__file__).parent
CONFIG_DIR = _current_dir / "configs"
TEMPLATE_DIR = _current_dir / "templates"
DEFAULT_TEMPLATE_PATH = TEMPLATE_DIR / "llm_api_config.yaml.template"

# --- Known Datapresso Stages ---
# List the identifiers for stages that require LLM API access.
# This list determines which config files are checked/created and which
# module-level API objects are pre-initialized on import.
# Use lowercase identifiers, typically matching the module or class name.
KNOWN_STAGES: List[str] = [
    "data_generator",
    "quality_assessor",
    # Add other known stages here as needed, e.g.:
    # "data_filter",
    # "evaluation_runner",
]

# --- Initialization and API Object Creation ---
_initialized_apis: Dict[str, Optional[StageLLMApi]] = {}

def _initialize_stage_api(stage_name: str) -> Optional[StageLLMApi]:
    """Initializes and returns StageLLMApi for a given stage name."""
    global _initialized_apis
    if stage_name in _initialized_apis:
        return _initialized_apis[stage_name] # Return cached instance (or None if init failed)

    # StageLLMApi now handles both legacy and new config formats
    # It will create a new config directory if neither exists
    try:
        stage_api_instance = StageLLMApi(stage_name=stage_name, config_dir=CONFIG_DIR)
        if stage_api_instance.is_available():
             logger.debug(f"Successfully initialized LLM API object for stage: {stage_name}")
             _initialized_apis[stage_name] = stage_api_instance
             return stage_api_instance
        else:
             logger.error(f"Initialization of StageLLMApi for '{stage_name}' completed but manager is not available (check logs for config/init errors).")
             _initialized_apis[stage_name] = None
             return None
    except Exception as e:
        logger.error(f"Failed to initialize LLM API for stage '{stage_name}': {e}", exc_info=True)
        _initialized_apis[stage_name] = None
        return None

# Pre-initialize APIs for known stages and expose them at module level
for _stage in KNOWN_STAGES:
    _api_object_name = f"{_stage}_api"
    globals()[_api_object_name] = _initialize_stage_api(_stage)

# --- Registration for New Stages ---
_exported_api_names = [f"{s}_api" for s in KNOWN_STAGES]

def register_llm_stage(stage_name: str) -> bool:
    """
    Registers a new stage for LLM API usage.

    This ensures its configuration file exists (creating from template if needed)
    and makes its API object (`{stage_name}_api`) available for import from this module.

    Should ideally be called early, e.g., when the stage's module is imported.

    Parameters
    ----------
    stage_name : str
        The unique identifier for the new stage.

    Returns
    -------
    bool
        True if registration (including potential API object initialization)
        was successful or if the stage was already registered, False otherwise.
    """
    global _initialized_apis, _exported_api_names
    api_object_name = f"{stage_name}_api"

    if stage_name in KNOWN_STAGES and api_object_name in globals():
        logger.debug(f"Stage '{stage_name}' is already known/registered.")
        # Ensure it's initialized if it failed previously but config might be fixed now
        if globals()[api_object_name] is None:
             globals()[api_object_name] = _initialize_stage_api(stage_name)
        return globals()[api_object_name] is not None

    logger.info(f"Registering new LLM API stage: '{stage_name}'")
    api_instance = _initialize_stage_api(stage_name)

    if api_instance:
        globals()[api_object_name] = api_instance
        if stage_name not in KNOWN_STAGES: # Avoid duplicates if called multiple times
             KNOWN_STAGES.append(stage_name)
        if api_object_name not in _exported_api_names:
             _exported_api_names.append(api_object_name)
             # Dynamically update __all__ if needed, though modifying __all__ dynamically is tricky
             # It's generally better to pre-define __all__ or rely on direct import
             # __all__.append(api_object_name) # Be cautious with this
        return True
    else:
        # Initialization failed, but mark as attempted
        globals()[api_object_name] = None # Ensure the name exists but is None
        if api_object_name not in _exported_api_names:
             _exported_api_names.append(api_object_name)
        return False


# --- Public API Export ---
__all__ = [
    # Core Classes
    "LLMProvider",
    "LLMAPIManager",
    "StageLLMApi",
    # Provider Implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "DeepSeekProvider",
    "GenericOpenAIProvider",
    "LocalProvider", # Will be None if dependencies missing
    # Utilities
    "load_llm_config",
    "register_llm_stage",
    # Stage-specific API Objects (Dynamically added based on KNOWN_STAGES)
] + _exported_api_names

# --- Cleanup ---
# Remove temporary variables from module scope
del _stage, _api_object_name, _current_dir, Path, logging, shutil
# Keep logger, CONFIG_DIR, TEMPLATE_DIR, KNOWN_STAGES, _initialized_apis, _exported_api_names ?
# Maybe keep KNOWN_STAGES public?
# Let's remove most internal vars for cleaner namespace
# del CONFIG_DIR, TEMPLATE_DIR, DEFAULT_TEMPLATE_PATH, KNOWN_STAGES, _initialized_apis, _exported_api_names
