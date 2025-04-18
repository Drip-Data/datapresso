"""
Unit tests for the Datapresso LLM API layer stage-specific interfaces.

NOTE:
- These tests assume that the corresponding configuration files
  (e.g., configs/data_generator.yaml, configs/quality_assessor.yaml)
  exist and are configured to use the 'gemini' provider with a valid API key
  set via the GOOGLE_API_KEY environment variable.
- These tests perform actual API calls and may incur costs and take time.
- Ensure the required templates ('default', 'person_info', 'quality_score', etc.)
  exist in the respective YAML configuration files.
- Run tests from the root directory of the 'datapresso' project.
"""

import unittest
import os
import logging
from pathlib import Path
import time

# Configure logging for tests
logging.basicConfig(level=logging.WARNING) # Set to DEBUG for more verbose output
test_logger = logging.getLogger("TestLLMApi")
test_logger.setLevel(logging.INFO) # Or DEBUG

# --- IMPORTANT ---
# These imports rely on the __init__.py successfully initializing the API objects.
# If __init__.py fails (e.g., cannot create default configs), these might be None.
try:
    from datapresso.llm_api import (
        data_generator_api,
        quality_assessor_api,
        register_llm_stage # Optional: Can be used to register test-specific stages
    )
    API_OBJECTS_LOADED = True
except ImportError as e:
    test_logger.error(f"Failed to import API objects from datapresso.llm_api: {e}. Tests will be skipped.")
    data_generator_api = None
    quality_assessor_api = None
    API_OBJECTS_LOADED = False

# --- Test Configuration ---
# Ensure GOOGLE_API_KEY is set in the environment for these tests to run.
GEMINI_API_KEY_SET = os.environ.get("GOOGLE_API_KEY") is not None

# Decorator to skip tests if API objects didn't load or Gemini key is missing
skip_reason_api = "LLM API objects failed to load from __init__.py"
skip_reason_key = "GOOGLE_API_KEY environment variable not set"

def skip_if_api_unavailable(test_func):
    """Decorator to skip test if API objects are not loaded."""
    return unittest.skipUnless(API_OBJECTS_LOADED, skip_reason_api)(test_func)

def skip_if_gemini_key_missing(test_func):
    """Decorator to skip test if Gemini API key is not set."""
    return unittest.skipUnless(GEMINI_API_KEY_SET, skip_reason_key)(test_func)

def skip_if_stage_unavailable(stage_api_object, stage_name):
    """Decorator factory to skip test if a specific stage API is unavailable."""
    def decorator(test_func):
        reason = f"Stage API object '{stage_name}_api' is None or not available (check config/init logs)."
        return unittest.skipUnless(stage_api_object and stage_api_object.is_available(), reason)(test_func)
    return decorator

class TestLLMApiUsage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_logger.info("Starting LLM API Tests...")
        if not API_OBJECTS_LOADED:
            test_logger.warning(skip_reason_api)
        if not GEMINI_API_KEY_SET:
            test_logger.warning(skip_reason_key + ". Gemini tests will be skipped.")
        if data_generator_api and not data_generator_api.is_available():
             test_logger.warning(f"data_generator_api object loaded but not available. Check configs/data_generator.yaml and logs.")
        if quality_assessor_api and not quality_assessor_api.is_available():
             test_logger.warning(f"quality_assessor_api object loaded but not available. Check configs/quality_assessor.yaml and logs.")


    # --- Data Generator Tests ---

    @skip_if_api_unavailable
    @skip_if_gemini_key_missing
    @skip_if_stage_unavailable(data_generator_api, "data_generator")
    def test_data_generator_generate_simple(self):
        """Test simple generation using data_generator_api (expects Gemini)."""
        test_logger.info("Running test_data_generator_generate_simple...")
        start_time = time.time()
        response = data_generator_api.generate(
            user_prompt="Generate a short description for a sci-fi movie.",
            system_prompt_template="default", # Assumes 'default' template exists
            provider_name="gemini" # Explicitly use gemini if default isn't set
        )
        duration = time.time() - start_time
        test_logger.info(f"API call duration: {duration:.2f}s")
        test_logger.debug(f"Response: {response}")

        self.assertIsNotNone(response, "API response should not be None")
        self.assertIsNone(response.get("error"), f"API call failed: {response.get('error')}")
        self.assertIn("text", response)
        self.assertTrue(isinstance(response.get("text"), str))
        self.assertGreater(len(response.get("text", "")), 0, "Generated text should not be empty")
        self.assertIn("model", response)
        # self.assertIn("gemini", response.get("model", ""), "Model name should indicate Gemini") # Model name might vary

    @skip_if_api_unavailable
    @skip_if_gemini_key_missing
    @skip_if_stage_unavailable(data_generator_api, "data_generator")
    def test_data_generator_generate_structured(self):
        """Test structured generation using data_generator_api (expects Gemini)."""
        test_logger.info("Running test_data_generator_generate_structured...")
        start_time = time.time()
        response = data_generator_api.generate_with_structured_output(
            user_prompt="Create a profile for a detective named Alex Mercer.",
            output_schema_template="person_info", # Assumes 'person_info' schema exists
            system_prompt_template="persona_generator", # Assumes 'persona_generator' prompt exists
            provider_name="gemini"
        )
        duration = time.time() - start_time
        test_logger.info(f"API call duration: {duration:.2f}s")
        test_logger.debug(f"Response: {response}")

        self.assertIsNotNone(response, "API response should not be None")
        self.assertIsNone(response.get("error"), f"API call failed: {response.get('error')}")
        self.assertIn("structured_output", response)
        output = response.get("structured_output")
        self.assertTrue(isinstance(output, dict))
        self.assertIn("name", output, "Structured output should contain 'name'")
        self.assertGreater(len(output), 0, "Structured output should not be empty")
        # self.assertIn("gemini", response.get("model", ""), "Model name should indicate Gemini")

    # --- Quality Assessor Tests ---

    @skip_if_api_unavailable
    @skip_if_gemini_key_missing
    @skip_if_stage_unavailable(quality_assessor_api, "quality_assessor")
    def test_quality_assessor_generate_simple(self):
        """Test simple generation using quality_assessor_api (expects Gemini)."""
        test_logger.info("Running test_quality_assessor_generate_simple...")
        start_time = time.time()
        response = quality_assessor_api.generate(
            user_prompt="What are the key aspects of good technical writing?",
            system_prompt_template="default",
            provider_name="gemini"
        )
        duration = time.time() - start_time
        test_logger.info(f"API call duration: {duration:.2f}s")
        test_logger.debug(f"Response: {response}")

        self.assertIsNotNone(response, "API response should not be None")
        self.assertIsNone(response.get("error"), f"API call failed: {response.get('error')}")
        self.assertIn("text", response)
        self.assertTrue(isinstance(response.get("text"), str))
        self.assertGreater(len(response.get("text", "")), 0, "Generated text should not be empty")
        # self.assertIn("gemini", response.get("model", ""), "Model name should indicate Gemini")

    @skip_if_api_unavailable
    @skip_if_gemini_key_missing
    @skip_if_stage_unavailable(quality_assessor_api, "quality_assessor")
    def test_quality_assessor_generate_structured(self):
        """Test structured generation using quality_assessor_api (expects Gemini)."""
        test_logger.info("Running test_quality_assessor_generate_structured...")
        text_to_assess = "The code calculates the sum of two numbers. It is clear and works correctly."
        start_time = time.time()
        response = quality_assessor_api.generate_with_structured_output(
            user_prompt=f"Assess the quality of this text: '{text_to_assess}'",
            output_schema_template="quality_score", # Assumes 'quality_score' schema exists
            system_prompt_template="quality_assessor", # Assumes 'quality_assessor' prompt exists
            provider_name="gemini"
        )
        duration = time.time() - start_time
        test_logger.info(f"API call duration: {duration:.2f}s")
        test_logger.debug(f"Response: {response}")

        self.assertIsNotNone(response, "API response should not be None")
        self.assertIsNone(response.get("error"), f"API call failed: {response.get('error')}")
        self.assertIn("structured_output", response)
        output = response.get("structured_output")
        self.assertTrue(isinstance(output, dict))
        self.assertIn("clarity", output, "Structured output should contain 'clarity'")
        self.assertIn("coherence", output)
        self.assertIn("accuracy", output)
        self.assertIn("relevance", output)
        self.assertIn("justification", output)
        self.assertGreater(len(output), 0, "Structured output should not be empty")
        # self.assertIn("gemini", response.get("model", ""), "Model name should indicate Gemini")


if __name__ == '__main__':
    # Ensure the tests run from the correct directory context if needed
    # Example: Change directory to project root if tests are run directly
    # os.chdir(Path(__file__).parent.parent.parent) # Adjust based on actual structure
    print("\n--- Running LLM API Tests ---")
    print("IMPORTANT: Ensure configs/*.yaml are set up for 'gemini' provider and GOOGLE_API_KEY is set.")
    unittest.main()