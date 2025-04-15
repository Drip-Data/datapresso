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

from datapresso.utils.base_module import BaseModule
from datapresso.utils.data_utils import DataUtils
from datapresso.data_generation.prompt_manager import PromptManager
from datapresso.data_generation.initial_filter import InitialFilter


class GeneratorEngine(BaseModule):
    """
    Generator engine for Datapresso framework.
    
    Handles the generation of data samples using language models.
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
        super().__init__(config, logger)
        
        # Initialize components
        self.prompt_manager = PromptManager(config.get("prompt_templates", {}), logger)
        self.initial_filter = InitialFilter(config.get("initial_filtering", {}), logger)
        
        # Configuration
        self.model = config.get("model", "gpt-4-turbo")
        self.temperature = config.get("temperature", 0.7)
        self.target_count = config.get("target_count", 5000)
        self.batch_size = config.get("batch_size", 10)
        self.max_retries = config.get("max_retries", 3)
        
        # Output path
        self.output_dir = Path(config.get("output_dir", "data/generated"))
        
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
            batch_results = self._generate_reasoning_batch(seed_samples)
            
            # Apply initial filtering
            filtered_results = self.initial_filter.filter_samples(batch_results)
            
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
        
        for i, seed in enumerate(seed_samples):
            # Create a prompt for one sample

            prompt = self.prompt_manager.create_prompt(seed)

            try:
                agent_output = (
                    self.agent.step(prompt['user_message'], response_format=DataPoint)
                    .msgs[0]
                    .parsed
                )

                assert isinstance(agent_output, DataPoint)

                self.agent.reset()

            except (TypeError, KeyError) as e:
                logger.warning(
                    f"Agent output issue: {e}, retrying... "
                    f"({retries + 1}/{max_retries})"
                )
                retries += 1
                continue

            
            # In a real implementation, this would call the LLM API
            # For now, just create a mock response
            generated_sample = {
                "id": f"gen_{int(time.time())}_{i}",
                "question": agent_output.question,
                # f"Generated question based on {seed.get('id', 'unknown')}"
                "response": {
                    "rationale": agent_output.rationale ,
                    "final_answer": verifier_response.result,
                },
                "metadata": {
                    "domain": seed.get("metadata", {}).get("domain", "unknown"),
                    "difficulty": seed.get("metadata", {}).get("difficulty", 0.5),
                    "source": f"{self.model} generated",
                    "seed_id": seed.get("id", "unknown"),
                    "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "initial_quality": 0.8  # Placeholder quality score
                }
            }
            
            generated_samples.append(generated_sample)
            
        return generated_samples

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
