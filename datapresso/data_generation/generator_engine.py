"""
Generator engine for Datapresso framework.

This module handles the generation of data samples using language models.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import os

# from datapresso.utils.base_module import BaseModule
# from datapresso.utils.data_utils import DataUtils
from prompt_manager import GenerationPromptManager,DistillPromptManager,VerificationPromptManager
from initial_filter import InitialFilter
from answer_verifier import AnswerVerifier


from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import DeepSeekConfig
from camel.agents import ChatAgent
from camel.datasets import DataPoint


from camel.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)


# class GeneratorEngine(BaseModule):
class GeneratorEngine:
    """
    Generator engine for Datapresso framework.
    
    Handles the generation of data samples given preference seed data using language models.

    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the generator engine.

        Parameters
        ----------
        config : Dict[str, Any]
            Generator configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        # super().__init__(config, logger)
        
        # Initialize components
        self.task = config.get("task", "generation") # generation , distillation 
        if self.task == "generation":
            self.prompt_manager = GenerationPromptManager(config.get("prompt_templates", {}), logger)
            self.verifier = AnswerVerifier(config.get("verifiers", {}), logger)
        elif self.task == "distillation":
            self.prompt_manager = DistillPromptManager(config.get("prompt_templates", {}), logger)
            self.verifier = None
        else: 
            raise ValueError(f"Unknown task type: {self.task}")
        self.initial_filter = InitialFilter(config.get("initial_filtering", {}), logger)
        
        
        # Configuration
        # self.model = config.get("model", "gpt-4-turbo")
        # TODO: currently fix the model 
        import os 
        self.api_key = "sk-56a20268a62a4c0fab38fa496e5c2d5f"
        os.environ["DEEPSEEK_API_KEY"] = self.api_key
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict=DeepSeekConfig().as_dict(),
        )

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

        self.temperature = config.get("temperature", 0.7)
        self.target_count = config.get("target_count", 5000)
        self.batch_size = config.get("batch_size", 10)
        self.max_retries = config.get("max_retries", 3)
        
        # Output path
        self.output_dir = Path(config.get("output_dir", "data/generated"))
        self.logger = get_logger(__name__)
        self.logger.info(f"Initialized generator engine with model: {self.model}")

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate data samples based on seed data.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Seed data to use as reference.

        Returns
        -------
        List[Dict[str, Any]]
            Generated data samples.
        """
        self.logger.info(f"Starting data generation with {len(data)} seed samples")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate number of batches
        total_batches = (self.target_count + self.batch_size - 1) // self.batch_size
        
        # Initialize results
        generated_data = []
        
        # Generate data in batches
        for batch_idx in range(total_batches):
            self.logger.info(f"Generating batch {batch_idx + 1}/{total_batches}")
            
            # Select seed samples for this batch
            seed_samples = self._select_seed_samples(data, self.batch_size)
            
            # Generate samples for this batch
            batch_results = self._generate_batch(seed_samples)

            # verify the answer given reference answer or generated COT answer
            verified_results = self.verifier.verify_samples(batch_results)
            
            # Apply initial filtering
            filtered_results = self.initial_filter.filter_samples(verified_results)
            
            # Add to results
            generated_data.extend(filtered_results)
            
            # Save intermediate results
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                self._save_intermediate_results(generated_data, batch_idx + 1)
                
            # Update progress
            self._update_status("generating", len(generated_data), self.target_count)
            
            # Check if we've reached the target count
            if len(generated_data) >= self.target_count:
                break
                
        # Truncate to target count if needed
        if len(generated_data) > self.target_count:
            generated_data = generated_data[:self.target_count]
            
        # Save final results
        self._save_final_results(generated_data)
        
        self.logger.info(f"Data generation completed: {len(generated_data)} samples generated")
        
        return generated_data

    def _select_seed_samples(self, data: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """
        Select seed samples for generation.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Available seed samples.
        count : int
            Number of samples to select.

        Returns
        -------
        List[Dict[str, Any]]
            Selected seed samples.
        """
        # For now, just use random selection
        # In a real implementation, this would use more sophisticated selection strategies
        import random
        
        if not data:
            return []
            
        # Ensure we don't try to select more samples than available
        count = min(count, len(data))
        
        return random.sample(data, count)


    def _generate_batch(self, seed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if self.task == "generation":
            return self._generate_reasoning_batch(seed_samples)
        elif self.task == "distillation":
            return self._generate_distill_batch(seed_samples)
        else:
            raise ValueError(f"Unknown task type: {self.task}")
      
    def _generate_reasoning_batch(self, seed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate a batch of samples based on seed samples.

        Parameters
        ----------
        seed_samples : List[Dict[str, Any]]
            Seed samples to use as reference.

        Returns
        -------
        List[Dict[str, Any]]
            Generated samples.
        """
        # This is a placeholder implementation
        # In a real implementation, this would call the LLM API
        
        self.logger.info(f"Generating {len(seed_samples)} samples (placeholder implementation)")
        
        # Placeholder: Create mock generated samples
        generated_samples = []
        
        #TODO modify the procedure to construct the default system message 
        initial_seed = seed_samples[0] if seed_samples else None
        initial_prompt = self.prompt_manager.create_prompt(initial_seed) if initial_seed else None

        self.agent = ChatAgent(system_message=initial_prompt['system_message'], model=self.model)

        for i, seed in tqdm(enumerate(seed_samples)):
            
            # currenly set only one example seed data per generation
            prompt = self.prompt_manager.create_prompt(seed) 
            agent_output = (
                self.agent.step(prompt['user_message'],response_format=DataPoint) # self.agent.step(prompt['user_message'],response_format=DataPoint]
                .msgs[0]
                .parsed
            ) 
            new_sample = {
                "id": f"gen_{i}",
                "instruction": agent_output.question,
                "response": {
                    "rationale": agent_output.rationale ,
                    "final_answer": agent_output.final_answer,
                },
                "metadata": {
                    "domain": seed.get("metadata", {}).get("domain", "unknown"),
                    "difficulty": seed.get("metadata", {}).get("difficulty", "unknown"),
                    "source": f"{self.model}",
                    "seed_id": seed.get("id", "unknown"),
                    "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            }
            
            generated_samples.append(new_sample)
            
        return generated_samples

    def _generate_distill_batch(self, seed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate a batch of samples based on seed questions with think process.

        Parameters
        ----------
        seed_samples : List[Dict[str, Any]]
            Seed samples to use as reference.

        Returns
        -------
        List[Dict[str, Any]]
            Generated samples.
        """

        generated_samples = []

         #TODO modify the procedure to construct the default system message 
        initial_seed = seed_samples[0] if seed_samples else None
        initial_prompt = self.prompt_manager.create_prompt(initial_seed) if initial_seed else None
        

        self.agent = ChatAgent(system_message=initial_prompt['system_message'], model=self.model)


        for i, seed in tqdm(enumerate(seed_samples)):
            
            # currenly set only one example seed data per generation
            prompt = self.prompt_manager.create_prompt(seed) 
            
          
            response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": prompt['system_message']},
                {"role": "user", "content": prompt['user_message']},
            ],
            stream=False
            )
        
            response_content = response.choices[0].message.content
            
            think_process, formal_response = self._parse_structual_format(response_content)

            generated_sample = {
            "id": f"reasoning_{i}",
            "instruction": seed.get("instruction", ""),
            "response": {
                "think_process": think_process,
                "formal_response":formal_response,
                "final_answer": seed['response'].get("final_answer", ""),
            },
            "metadata": {
                "domain": seed.get("metadata", {}).get("domain", "unknown"),
                "difficulty": seed.get("metadata", {}).get("difficulty", "unknown"),
                "source": f"{self.model}",
                "seed_id": seed.get("id", "unknown"),
                "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
            
            generated_samples.append(generated_sample)

        return generated_samples


    def _parse_structual_format(self, response_content):
          # Parse think process and formal response
        think_process = ""
        formal_response = response_content
        
        if "<think>" in response_content and "</think>" in response_content:
            parts = response_content.split("</think>")
            if len(parts) > 1:
                think_process = parts[0].replace("<think>", "").strip()
                formal_response = parts[1].strip()

        return think_process, formal_response

    def _save_intermediate_results(self, data: List[Dict[str, Any]], batch_idx: int) -> None:
        """
        Save intermediate generation results.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Generated data so far.
        batch_idx : int
            Current batch index.
        """
        # Create filename
        timestamp = int(time.time())
        filename = f"generated_batch_{batch_idx}_{timestamp}.jsonl"
        file_path = self.output_dir / filename
        
        # Save data
        DataUtils.write_jsonl(data, file_path)
        
        self.logger.info(f"Saved intermediate results to {file_path}")

    def _save_final_results(self, data: List[Dict[str, Any]]) -> None:
        """
        Save final generation results.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Generated data.
        """
        # Create filename
        timestamp = int(time.time())
        filename = f"generated_final_{timestamp}.jsonl"
        file_path = self.output_dir / filename
        
        # Save data
        DataUtils.write_jsonl(data, file_path)
        
        self.logger.info(f"Saved final results to {file_path}")
        
        # Also save a copy with a fixed name for easier reference
        fixed_path = self.output_dir / "generated_data.jsonl"
        DataUtils.write_jsonl(data, fixed_path)
        
        self.logger.info(f"Saved final results to {fixed_path} (fixed name)")
