"""
Google Gemini provider implementation for Datapresso framework.

This module provides an implementation of the LLMProvider interface for Google's Gemini API.
Requires the 'google-generativeai' package to be installed.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from pathlib import Path
import warnings

from datapresso.llm_api.llm_provider import LLMProvider

# Import optional dependency
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold, ContentDict, PartDict
    from google.api_core import exceptions as google_exceptions
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    # Define dummy types for when the library is not installed
    ContentDict = Dict[str, Any]
    PartDict = Dict[str, Any]


# Mapping from Datapresso roles to Gemini roles
# Gemini uses 'user' and 'model'
ROLE_MAP = {
    "system": "user", # Gemini doesn't have a distinct system role, often prepended to the first user message or handled via system_instruction
    "user": "user",
    "assistant": "model",
    "tool": "function", # For tool/function calling results
}

# Default cost per 1k characters (as Gemini pricing is often character-based)
# Needs verification against actual Gemini pricing. Using placeholder values.
# Prices are highly model-dependent.
DEFAULT_GEMINI_COSTS = {
    "gemini-1.5-pro-latest": {"prompt_char": 0.0000125 * 1000, "completion_char": 0.0000375 * 1000}, # Example based on $/1k chars
    "gemini-1.5-flash-latest": {"prompt_char": 0.00000125 * 1000, "completion_char": 0.0000025 * 1000}, # Example
    "gemini-1.0-pro": {"prompt_char": 0.000125 * 1000, "completion_char": 0.000375 * 1000}, # Example
    # Add other models as needed
}
# We will calculate cost based on characters for Gemini

class GeminiProvider(LLMProvider):
    """
    Google Gemini provider implementation.

    Provides access to Google's Gemini API for text generation.
    Requires 'google-generativeai' package and an API key.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Gemini provider.

        Parameters
        ----------
        config : Dict[str, Any]
            Provider configuration. Must include 'api_key' (or GOOGLE_API_KEY env var) and 'model'.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        super().__init__(config, logger)

        if not HAS_GEMINI:
            raise ImportError("google-generativeai package is required for GeminiProvider. Install with 'pip install google-generativeai'")

        # Get API key
        self.api_key = config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided in config or GOOGLE_API_KEY environment variable")

        try:
             genai.configure(api_key=self.api_key)
        except Exception as e:
             self.logger.error(f"Failed to configure Gemini API: {e}")
             raise ValueError(f"Failed to configure Gemini API: {e}") from e

        # Model configuration
        self.default_model = config.get("model", "gemini-1.5-pro-latest")
        # Gemini uses 'temperature', 'top_p', 'top_k', 'max_output_tokens'
        self.default_temperature = config.get("temperature", 0.7)
        self.default_top_p = config.get("top_p", 1.0)
        self.default_top_k = config.get("top_k", None) # Gemini default is often dynamic
        self.default_max_tokens = config.get("max_tokens", 2048) # Maps to max_output_tokens

        # Safety settings (configure as needed)
        self.safety_settings = config.get("safety_settings", {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        })

        # Retry configuration (handled by google-api-core library, but we keep ours for consistency)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.timeout = config.get("timeout", 120) # Timeout for the request itself

        # Cost tracking (character-based for Gemini)
        self.cost_per_1k_chars = config.get("cost_per_1k_chars", DEFAULT_GEMINI_COSTS)
        # Ensure the default model has a cost entry
        if self.default_model not in self.cost_per_1k_chars:
             self.cost_per_1k_chars[self.default_model] = {"prompt_char": 0.0, "completion_char": 0.0}
             self.logger.warning(f"Cost for default model '{self.default_model}' not found in config, defaulting to 0.")

        # Initialize the generative model client
        try:
            self.model_client = genai.GenerativeModel(
                model_name=self.default_model,
                # system_instruction can be set here if needed globally
            )
            self.logger.info(f"Initialized Gemini provider with model: {self.default_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini GenerativeModel client for {self.default_model}: {e}")
            raise RuntimeError(f"Failed to initialize Gemini client: {e}") from e


    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[ContentDict]]:
        """Converts Datapresso message format to Gemini's ContentDict list and extracts system prompt."""
        gemini_history: List[ContentDict] = []
        system_prompt: Optional[str] = None

        # Handle potential system message first
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content", "")
            messages = messages[1:] # Remove system message from main list

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            gemini_role = ROLE_MAP.get(role)

            if not gemini_role:
                self.logger.warning(f"Unsupported role '{role}' in message, skipping.")
                continue

            # Simple text content
            # TODO: Handle multi-modal content if needed in the future
            parts: List[PartDict] = [{"text": content}]
            gemini_history.append({"role": gemini_role, "parts": parts})

        # Ensure conversation starts with 'user' role if no history or starts with 'model'
        if not gemini_history or gemini_history[0]['role'] == 'model':
             # Prepend an empty user message if needed, though ideally the input should be valid
             # This might indicate an issue with the input message list structure
             self.logger.debug("Prepending empty user message to ensure valid Gemini conversation start.")
             gemini_history.insert(0, {"role": "user", "parts": [{"text": ""}]})


        # Gemini requires alternating user/model roles. We might need to merge consecutive messages
        # of the same role, although the underlying library might handle this.
        # For simplicity here, we assume the input `messages` list is already reasonably structured.
        # A more robust implementation would merge consecutive messages.

        return system_prompt, gemini_history

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate text based on a list of messages using the Gemini API.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries.
        **kwargs : Dict[str, Any]
            Additional parameters (temperature, max_tokens, top_p, top_k, stop_sequences, system_prompt).

        Returns
        -------
        Dict[str, Any]
            Response containing generated text and metadata.
        """
        start_time = time.time()
        error_response = None
        model_name = kwargs.get("model", self.default_model)

        try:
            # Get generation config parameters from kwargs or defaults
            temperature = kwargs.get("temperature", self.default_temperature)
            max_output_tokens = kwargs.get("max_tokens", self.default_max_tokens)
            top_p = kwargs.get("top_p", self.default_top_p)
            top_k = kwargs.get("top_k", self.default_top_k)
            stop_sequences = kwargs.get("stop_sequences", None) # Gemini uses 'stop_sequences'

            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences
            )

            # Convert messages and extract system prompt
            system_prompt_from_messages, gemini_history = self._convert_messages_to_gemini_format(messages)
            # Allow overriding system prompt via kwargs
            system_instruction = kwargs.get("system_prompt", system_prompt_from_messages)


            # Use the correct model client (handle if model differs from default)
            client = self.model_client
            if model_name != self.default_model:
                 try:
                      client = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
                 except Exception as e:
                      raise ValueError(f"Failed to initialize client for model {model_name}: {e}") from e
            elif system_instruction and not client.system_instruction:
                 # Re-initialize default client if system_instruction is provided now but wasn't at init
                 client = genai.GenerativeModel(model_name=self.default_model, system_instruction=system_instruction)


            # Make the API call with retry logic embedded in the library
            # The library handles retries for common transient errors.
            response = client.generate_content(
                contents=gemini_history,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                request_options={"timeout": self.timeout}
            )

            # Extract text - handle potential blocks or errors
            generated_text = ""
            prompt_feedback = None
            candidates = []
            if response.candidates:
                 candidates = response.candidates
                 # Check for finish reason other than STOP
                 finish_reason = candidates[0].finish_reason.name
                 if finish_reason != "STOP" and finish_reason != "MAX_TOKENS":
                      self.logger.warning(f"Gemini generation finished with reason: {finish_reason}")
                      # Potentially blocked by safety settings
                      if finish_reason == "SAFETY":
                           safety_ratings = candidates[0].safety_ratings
                           self.logger.warning(f"Safety Ratings: {safety_ratings}")
                           error_response = {"message": f"Content blocked due to safety settings: {finish_reason}", "type": "safety_error", "details": safety_ratings}
                      else:
                           error_response = {"message": f"Generation stopped unexpectedly: {finish_reason}", "type": "generation_error"}

                 # Extract text from the first candidate's content
                 if candidates[0].content and candidates[0].content.parts:
                      generated_text = "".join(part.text for part in candidates[0].content.parts if hasattr(part, 'text'))

            if hasattr(response, 'prompt_feedback'):
                 prompt_feedback = response.prompt_feedback


        except google_exceptions.GoogleAPIError as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            error_response = {"message": f"Gemini API error: {str(e)}", "type": "api_error", "details": str(e)}
            generated_text = ""
        except Exception as e:
            self.logger.error(f"Gemini generation failed: {str(e)}", exc_info=True)
            error_response = {"message": f"Gemini generation failed: {str(e)}", "type": "generation_error", "details": str(e)}
            generated_text = ""

        latency = time.time() - start_time

        # Calculate usage (characters)
        prompt_chars = sum(len(msg.get("content", "")) for msg in messages)
        completion_chars = len(generated_text)
        total_chars = prompt_chars + completion_chars

        # Gemini API doesn't directly return token counts in the standard response.
        # We'll use character counts for cost and report tokens as 0 or estimate if needed.
        usage = {
            "prompt_tokens": 0, # Or estimate: prompt_chars // 4
            "completion_tokens": 0, # Or estimate: completion_chars // 4
            "total_tokens": 0, # Or estimate: total_chars // 4
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
            "total_chars": total_chars,
        }

        cost = self._calculate_cost(model_name, usage)

        processed_response = {
            "text": generated_text,
            "error": error_response,
            "model": model_name,
            "latency": latency,
            "usage": usage,
            "cost": cost,
            "raw_response": { # Store key details, avoid storing large raw objects directly if possible
                 "candidates": [{"finish_reason": c.finish_reason.name, "safety_ratings": c.safety_ratings} for c in candidates] if candidates else [],
                 "prompt_feedback": str(prompt_feedback) if prompt_feedback else None,
            }
        }

        # Update metrics and history
        self._update_metrics(processed_response, latency)
        # History stores original messages format
        history_request = {"messages": messages, "model": model_name, **kwargs}
        self._add_to_history(history_request, processed_response, latency)

        return processed_response


    def generate_with_structured_output(self, messages: List[Dict[str, str]], output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output using Gemini's Tool/Function Calling.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries.
        output_schema : Dict[str, Any]
            Schema defining the expected output structure (OpenAPI format).
        **kwargs : Dict[str, Any]
            Additional parameters.

        Returns
        -------
        Dict[str, Any]
            Response containing structured output and metadata.
        """
        start_time = time.time()
        error_response = None
        structured_output = {}
        model_name = kwargs.get("model", self.default_model)

        try:
            # Prepare generation config (same as generate)
            temperature = kwargs.get("temperature", self.default_temperature)
            max_output_tokens = kwargs.get("max_tokens", self.default_max_tokens)
            top_p = kwargs.get("top_p", self.default_top_p)
            top_k = kwargs.get("top_k", self.default_top_k)
            # Note: stop_sequences might interfere with function calling

            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
            )

            # Convert messages and extract system prompt
            system_prompt_from_messages, gemini_history = self._convert_messages_to_gemini_format(messages)
            system_instruction = kwargs.get("system_prompt", system_prompt_from_messages)

            # Define the tool based on the output schema
            function_name = kwargs.get("function_name", "extract_information")
            function_description = kwargs.get("function_description", "Extracts structured information based on the user query.")

            # Gemini expects FunctionDeclaration format
            tool = genai.Tool(
                function_declarations=[
                    genai.FunctionDeclaration(
                        name=function_name,
                        description=function_description,
                        parameters=output_schema # Gemini uses OpenAPI schema directly
                    )
                ]
            )

            # Get the client
            client = self.model_client
            if model_name != self.default_model:
                 client = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
            elif system_instruction and not client.system_instruction:
                 client = genai.GenerativeModel(model_name=self.default_model, system_instruction=system_instruction)


            # Make the API call requesting the specific function
            response = client.generate_content(
                contents=gemini_history,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                tools=[tool],
                # Force the tool call - check Gemini docs for the exact mechanism
                # This might involve tool_config={"function_calling_config": "ANY"} or similar
                # tool_config=genai.ToolConfig(function_calling_config=genai.FunctionCallingConfig(
                #      mode=genai.FunctionCallingConfig.Mode.ANY, # Or .REQUIRED? Check docs
                #      allowed_function_names=[function_name]
                # ))
                request_options={"timeout": self.timeout}
            )

            # Process the response to find the function call
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call and part.function_call.name == function_name:
                        # Found the function call, extract arguments
                        try:
                            # Arguments are already a dict in google-generativeai >= 0.3.0
                            structured_output = dict(part.function_call.args)
                        except Exception as e:
                             self.logger.error(f"Failed to extract/parse function call arguments: {e}")
                             error_response = {"message": f"Failed to parse function call arguments: {e}", "type": "parsing_error"}
                        break # Stop after finding the first matching function call
                if not structured_output and not error_response:
                     # Function call was expected but not found
                     self.logger.warning(f"Function call '{function_name}' not found in Gemini response.")
                     error_response = {"message": f"Function call '{function_name}' not found in response", "type": "generation_error"}

            elif response.candidates and response.candidates[0].finish_reason.name != "STOP":
                 # Handle non-stop finish reasons like SAFETY
                 finish_reason = response.candidates[0].finish_reason.name
                 self.logger.warning(f"Gemini structured generation finished with reason: {finish_reason}")
                 if finish_reason == "SAFETY":
                      safety_ratings = response.candidates[0].safety_ratings
                      error_response = {"message": f"Content blocked due to safety settings: {finish_reason}", "type": "safety_error", "details": safety_ratings}
                 else:
                      error_response = {"message": f"Generation stopped unexpectedly: {finish_reason}", "type": "generation_error"}
            else:
                 # No function call found and finish reason was STOP or MAX_TOKENS
                 self.logger.warning(f"No function call '{function_name}' found in Gemini response. Finish Reason: {response.candidates[0].finish_reason.name if response.candidates else 'N/A'}")
                 # Check if the text content itself is the JSON (less reliable)
                 text_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')) if response.candidates and response.candidates[0].content else ""
                 try:
                      parsed_content = json.loads(text_content)
                      if isinstance(parsed_content, dict):
                           structured_output = parsed_content
                           self.logger.debug("Parsed structured output from text content as fallback.")
                      else:
                           error_response = {"message": "No function call found and text content is not a valid JSON object", "type": "generation_error"}
                 except json.JSONDecodeError:
                      error_response = {"message": "No function call found and text content is not valid JSON", "type": "generation_error"}


        except google_exceptions.GoogleAPIError as e:
            self.logger.error(f"Gemini API error during structured generation: {str(e)}")
            error_response = {"message": f"Gemini API error: {str(e)}", "type": "api_error", "details": str(e)}
        except Exception as e:
            self.logger.error(f"Gemini structured generation failed: {str(e)}", exc_info=True)
            error_response = {"message": f"Gemini structured generation failed: {str(e)}", "type": "generation_error", "details": str(e)}

        latency = time.time() - start_time

        # Calculate usage (characters)
        prompt_chars = sum(len(msg.get("content", "")) for msg in messages)
        # Completion chars are harder to estimate accurately for function calls, use 0 or estimate based on output dict size
        completion_chars = len(json.dumps(structured_output)) if structured_output else 0
        total_chars = prompt_chars + completion_chars

        usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "prompt_chars": prompt_chars, "completion_chars": completion_chars, "total_chars": total_chars,
        }
        cost = self._calculate_cost(model_name, usage)

        processed_response = {
            "structured_output": structured_output if not error_response else {},
            "error": error_response,
            "model": model_name,
            "latency": latency,
            "usage": usage,
            "cost": cost,
            "raw_response": { # Store key details
                 "candidates": [{"finish_reason": c.finish_reason.name, "safety_ratings": c.safety_ratings} for c in response.candidates] if hasattr(response, 'candidates') and response.candidates else [],
                 "prompt_feedback": str(response.prompt_feedback) if hasattr(response, 'prompt_feedback') else None,
            }
        }

        # Update metrics and history
        self._update_metrics(processed_response, latency)
        history_request = {"messages": messages, "output_schema": output_schema, "model": model_name, **kwargs}
        self._add_to_history(history_request, processed_response, latency)

        return processed_response


    def generate_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple sets of messages using Gemini API (sequentially).
        Gemini library might support async/batching, but this implements sequential calls.

        Parameters
        ----------
        messages_list : List[List[Dict[str, str]]]
            List of message lists.
        **kwargs : Dict[str, Any]
            Additional parameters for the API.

        Returns
        -------
        List[Dict[str, Any]]
            List of responses for each message list.
        """
        self.logger.info(f"Generating responses for {len(messages_list)} message lists (sequentially) using Gemini")
        responses = []
        model_name = kwargs.get("model", self.default_model) # Use consistent model for batch

        for i, messages in enumerate(messages_list):
            self.logger.debug(f"Processing message list {i+1}/{len(messages_list)}")
            try:
                # Pass model explicitly to generate
                response = self.generate(messages, model=model_name, **kwargs)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to generate response for message list {i+1}: {str(e)}")
                responses.append({
                    "text": "",
                    "error": {"message": f"Generation failed: {str(e)}", "type": "batch_error"},
                    "model": model_name, "latency": 0.0, "usage": {}, "cost": 0.0
                })

            # Add small delay between requests
            if i < len(messages_list) - 1:
                time.sleep(self.retry_delay * 0.5)

        return responses

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Calculate the cost based on character counts and configured rates.
        """
        cost_model_key = model if model in self.cost_per_1k_chars else self.default_model
        model_costs = self.cost_per_1k_chars.get(cost_model_key, {"prompt_char": 0.0, "completion_char": 0.0})

        prompt_chars = usage.get("prompt_chars", 0)
        completion_chars = usage.get("completion_chars", 0)

        # Cost is per 1000 characters
        prompt_cost = (prompt_chars / 1000) * model_costs.get("prompt_char", 0.0)
        completion_cost = (completion_chars / 1000) * model_costs.get("completion_char", 0.0)

        return prompt_cost + completion_cost

    def list_available_models(self) -> List[str]:
        """
        List available models from the Google Gemini API.
        """
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Filter further? e.g., only include 'models/gemini...'?
            # available_models = [m for m in available_models if m.startswith("models/gemini")]
            return sorted(available_models)
        except google_exceptions.GoogleAPIError as e:
             self.logger.error(f"Failed to list available Gemini models: {e}")
             # Fallback to configured/default models
             fallback_models = list(self.cost_per_1k_chars.keys())
             if self.default_model not in fallback_models:
                  fallback_models.append(self.default_model)
             return sorted(list(set(fallback_models)))
        except Exception as e:
             self.logger.error(f"Unexpected error listing Gemini models: {e}", exc_info=True)
             # Fallback to configured/default models
             fallback_models = list(self.cost_per_1k_chars.keys())
             if self.default_model not in fallback_models:
                  fallback_models.append(self.default_model)
             return sorted(list(set(fallback_models)))