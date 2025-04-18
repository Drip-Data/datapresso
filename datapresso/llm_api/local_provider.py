"""
Local model provider implementation for Datapresso framework.

This module provides an implementation of the LLMProvider interface for local models.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from pathlib import Path

from datapresso.llm_api.llm_provider import LLMProvider

# Import optional dependencies
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LocalProvider(LLMProvider):
    """
    Local model provider implementation.
    
    Provides access to locally deployed models for text generation.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the local model provider.

        Parameters
        ----------
        config : Dict[str, Any]
            Provider configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        super().__init__(config, logger)
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package is required for LocalProvider. Install with 'pip install transformers torch'")
            
        # Model configuration
        self.model_path = config.get("model_path")
        if not self.model_path:
            raise ValueError("model_path is required for LocalProvider")
            
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 1024)
        
        # Device configuration
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = config.get("dtype", "float16" if self.device == "cuda" else "float32")
        
        # Load model and tokenizer
        self.logger.info(f"Loading model from {self.model_path} on {self.device}")
        self._load_model()
        
        self.logger.info(f"Initialized local provider with model: {self.model_path}")

    def _load_model(self) -> None:
        """
        Load the model and tokenizer.
        """
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            model_kwargs = {
                "device_map": self.device,
                "torch_dtype": torch.float16 if self.dtype == "float16" else torch.float32
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            self.logger.info(f"Successfully loaded model from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate text based on a list of messages using a local model.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries, e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        **kwargs : Dict[str, Any]
            Additional parameters for the model, including 'chat_format' (bool, default True).

        Returns
        -------
        Dict[str, Any]
            Response containing generated text and metadata.
        """
        # Prepare generation parameters
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "return_full_text": False
        }
        
        # Add optional parameters
        for param in ["top_p", "top_k", "repetition_penalty", "num_return_sequences"]:
            if param in kwargs:
                generation_kwargs[param] = kwargs[param]
                
        # Format messages into a single prompt string
        apply_chat_template = kwargs.get("chat_format", True) and hasattr(self.tokenizer, "apply_chat_template")
        if apply_chat_template:
            try:
                # Use the tokenizer's chat template for proper formatting
                prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                self.logger.warning(f"Failed to apply chat template: {e}. Falling back to simple concatenation.")
                # Fallback: simple concatenation or use the last user message
                prompt_str = "\n".join([msg["content"] for msg in messages]) # Basic fallback
                # Or consider using only the last user message:
                # last_user_message = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), "")
                # prompt_str = self._format_chat_prompt(last_user_message) # Use old formatting for fallback
        else:
            # If chat format is disabled or template not available, use simple concatenation or last message
            prompt_str = "\n".join([msg["content"] for msg in messages])
            # Or:
            # last_user_message = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), "")
            # prompt_str = last_user_message # No special formatting

        request_details = {"messages": messages, **kwargs} # For history
                
        # Generate text
        start_time = time.time()
        
        try:
            # Count input tokens based on the final prompt string
            input_ids = self.tokenizer.encode(prompt_str)
            input_tokens = len(input_ids)

            # Generate text using the formatted prompt string
            outputs = self.pipe(prompt_str, **generation_kwargs)
            
            # Extract generated text
            text = outputs[0]["generated_text"] if outputs else ""
            
            # Count output tokens
            output_tokens = len(self.tokenizer.encode(text))
            
            # Create response
            response = {
                "text": text,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }
            
            error = None
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            text = ""
            response = {}
            error = {
                "message": f"Generation failed: {str(e)}",
                "type": "generation_error"
            }
            
        latency = time.time() - start_time
        
        # Process response
        processed_response = {
            "text": text,
            "error": error,
            "model": self.model_path,
            "latency": latency,
            "usage": response.get("usage", {}),
            "cost": 0.0,  # Local models have no API cost
            "raw_response": response
        }
        
        # Update metrics and history
        self._update_metrics(processed_response, latency)
        self._add_to_history(request_details, processed_response, latency)
        
        return processed_response

    def generate_with_structured_output(self, messages: List[Dict[str, str]], output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured output based on a list of messages using a local model.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries. The schema instructions will be appended to the content of the last message.
        output_schema : Dict[str, Any]
            Schema defining the expected output structure.
        **kwargs : Dict[str, Any]
            Additional parameters for the model.

        Returns
        -------
        Dict[str, Any]
            Response containing structured output and metadata.
        """
        # Create schema instructions
        schema_str = json.dumps(output_schema, indent=2)
        schema_instructions = (
            f"\n\nPlease provide your response as a valid JSON object that conforms to the following schema:\n\n"
            f"{schema_str}\n\n"
            f"Ensure your response is valid JSON and follows the schema exactly. Only output the JSON object."
        )

        # Append instructions to the last message's content
        # Create a deep copy to avoid modifying the original list
        messages_with_schema = [msg.copy() for msg in messages]
        if messages_with_schema:
            messages_with_schema[-1]["content"] += schema_instructions
        else:
            # Handle empty messages list if necessary, maybe add a user message with instructions
             messages_with_schema.append({"role": "user", "content": schema_instructions})

        # Generate text using the modified messages list
        # Disable chat_format here as we've manually structured the prompt content
        response = self.generate(messages_with_schema, chat_format=False, **kwargs)
        
        # Try to parse JSON from the text
        structured_output = {}
        text = response.get("text", "")
        
        try:
            # Find JSON in the text (look for content between ``` or just try to parse the whole text)
            json_match = None
            if "```json" in text and "```" in text.split("```json", 1)[1]:
                json_match = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text and "```" in text.split("```", 1)[1]:
                json_match = text.split("```", 1)[1].split("```", 1)[0].strip()
                
            if json_match:
                structured_output = json.loads(json_match)
            else:
                # Try to parse the whole text
                structured_output = json.loads(text)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from response: {str(e)}")
            structured_output = {"error": "Failed to parse JSON from response"}
            
        # Create structured response
        structured_response = response.copy()
        structured_response["structured_output"] = structured_output
        
        return structured_response

    def generate_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text for multiple sets of messages using a local model.
        Note: This currently makes sequential calls. Consider batch inference if performance is critical.

        Parameters
        ----------
        messages_list : List[List[Dict[str, str]]]
            List of message lists. Each inner list is a conversation history.
        **kwargs : Dict[str, Any]
            Additional parameters for the model.

        Returns
        -------
        List[Dict[str, Any]]
            List of responses for each message list.
        """
        self.logger.info(f"Generating responses for {len(messages_list)} message lists (sequentially)")
        
        responses = []
        
        # Process each message list
        for i, messages in enumerate(messages_list):
            self.logger.debug(f"Processing message list {i+1}/{len(messages_list)}")

            # Generate response
            try:
                response = self.generate(messages, **kwargs)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to generate response for message list {i+1}: {str(e)}")
                responses.append({
                    "text": "",
                    "error": {"message": f"Generation failed: {str(e)}", "type": "batch_error"},
                    "model": self.model_path,
                    "latency": 0.0,
                    "usage": {},
                    "cost": 0.0
                })
                
        return responses

    def _format_chat_prompt(self, prompt: str) -> str:
        """
        Format a prompt for chat models.

        Parameters
        ----------
        prompt : str
            Input prompt.

        Returns
        -------
        str
            Formatted prompt.
        """
        # Check if the model is a known chat model
        model_name = os.path.basename(self.model_path).lower()
        
        # Llama format
        if "llama" in model_name:
            return f"<s>[INST] {prompt} [/INST]"
            
        # Mistral format
        elif "mistral" in model_name:
            return f"<s>[INST] {prompt} [/INST]"
            
        # Falcon format
        elif "falcon" in model_name:
            return f"User: {prompt}\nAssistant:"
            
        # Default format (works for many models)
        else:
            return f"User: {prompt}\nAssistant:"

    def list_available_models(self) -> List[str]:
        """
        List available models for the LocalProvider.

        Returns
        -------
        List[str]
            A list containing only the configured model path.
        """
        if self.model_path:
            return [self.model_path]
        else:
            return []
