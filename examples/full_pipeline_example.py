#!/usr/bin/env python
"""
Datapresso full pipeline example.

This script demonstrates how to run the complete Datapresso pipeline.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
import time

# Add parent directory to path to import datapresso
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datapresso.user_input.config_manager import ConfigManager
from datapresso.llm_api.llm_api_manager import LLMAPIManager
from datapresso.quality_assessment.multi_dimension_evaluator import MultiDimensionEvaluator
from datapresso.data_filtering.multi_dimension_filter import MultiDimensionFilter
from datapresso.utils.data_utils import DataUtils


def setup_logging(log_dir, level=logging.INFO):
    """
    Set up logging configuration.
    
    Parameters
    ----------
    log_dir : str
        Directory to save log files.
    level : int, optional
        Logging level, by default logging.INFO
    
    Returns
    -------
    logging.Logger
        Configured logger.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("datapresso")
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create file handler
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"datapresso_{timestamp}.log"))
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_seed_data(seed_path, logger):
    """
    Load seed data from a file.
    
    Parameters
    ----------
    seed_path : str
        Path to the seed data file.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    list
        List of seed data samples.
    """
    logger.info(f"Loading seed data from {seed_path}")
    
    try:
        # Load data based on file extension
        if seed_path.endswith(".jsonl"):
            seed_data = DataUtils.read_jsonl(seed_path)
        else:
            raise ValueError(f"Unsupported file format: {seed_path}")
            
        # Ensure response format
        seed_data = DataUtils.ensure_response_format(seed_data)
        
        logger.info(f"Loaded {len(seed_data)} seed data samples")
        return seed_data
        
    except Exception as e:
        logger.error(f"Failed to load seed data: {str(e)}")
        raise


def generate_data(seed_data, llm_manager, config, logger):
    """
    Generate data based on seed data.
    
    Parameters
    ----------
    seed_data : list
        List of seed data samples.
    llm_manager : LLMAPIManager
        LLM API manager instance.
    config : dict
        Generation configuration.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    list
        List of generated data samples.
    """
    logger.info("Starting data generation")
    
    # Get generation parameters
    target_count = config.get("target_count", 100)
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 2048)
    
    # Create prompt template
    prompt_template = """
    You are an expert data generator for AI training. Based on the example below, generate a new, unique example with a similar format but different content.

    EXAMPLE:
    Instruction: {instruction}
    Response: {response}

    Generate a new example with the following format:
    Instruction: [your new instruction]
    Response: [your detailed response]

    Make sure your response includes both a detailed reasoning process and a clear final answer.
    """
    
    generated_data = []
    
    # Generate data
    for i in range(target_count):
        # Select a seed sample
        seed_index = i % len(seed_data)
        seed_sample = seed_data[seed_index]
        
        # Create prompt
        instruction = seed_sample.get("instruction", "")
        response = seed_sample.get("response", {})
        
        if isinstance(response, dict):
            response_text = response.get("origin_text", "")
        else:
            response_text = str(response)
            
        prompt = prompt_template.format(
            instruction=instruction,
            response=response_text
        )
        
        # Generate new sample
        logger.info(f"Generating sample {i+1}/{target_count}")
        
        try:
            # Call LLM API
            result = llm_manager.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse generated text
            generated_text = result.get("text", "")
            
            # Extract instruction and response
            new_instruction = ""
            new_response = ""
            
            if "Instruction:" in generated_text and "Response:" in generated_text:
                parts = generated_text.split("Response:", 1)
                instruction_part = parts[0]
                response_part = parts[1] if len(parts) > 1 else ""
                
                new_instruction = instruction_part.replace("Instruction:", "").strip()
                new_response = response_part.strip()
                
            # Create new sample
            new_sample = {
                "id": f"generated_{i+1}",
                "instruction": new_instruction,
                "response": {
                    "origin_text": new_response,
                    "rationale": new_response,
                    "final_answer": new_response  # This is a simplification; in a real implementation, you would parse the response to extract the final answer
                },
                "metadata": {
                    "source": "generated",
                    "seed_id": seed_sample.get("id", ""),
                    "generation_timestamp": time.time()
                }
            }
            
            generated_data.append(new_sample)
            
        except Exception as e:
            logger.error(f"Failed to generate sample {i+1}: {str(e)}")
            
    logger.info(f"Generated {len(generated_data)} samples")
    return generated_data


def run_pipeline(config_path):
    """
    Run the complete Datapresso pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    # Set up logging
    logger = setup_logging("logs")
    logger.info(f"Starting Datapresso pipeline with config: {config_path}")
    
    try:
        # Load configuration
        config_manager = ConfigManager(logger)
        config = config_manager.load_config(config_path)
        
        # Create output directory
        output_dir = Path(config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM API manager
        llm_config = config.get("llm_api", {})
        llm_manager = LLMAPIManager(llm_config, logger)
        
        # Step 1: Load seed data
        seed_path = config.get("seed_db", {}).get("path")
        if not seed_path:
            raise ValueError("Seed data path not specified in configuration")
            
        seed_data = load_seed_data(seed_path, logger)
        
        # Step 2: Generate data (if enabled)
        generated_data = []
        if config.get("data_generation", {}).get("enabled", True):
            generated_data = generate_data(
                seed_data,
                llm_manager,
                config.get("data_generation", {}),
                logger
            )
            
            # Save generated data
            generated_path = output_dir / "generated_data.jsonl"
            DataUtils.write_jsonl(generated_data, generated_path)
            logger.info(f"Saved generated data to {generated_path}")
            
        # Combine seed and generated data
        combined_data = seed_data + generated_data
        logger.info(f"Combined data: {len(combined_data)} samples")
        
        # Step 3: Quality assessment
        quality_assessment_config = config.get("quality_assessment", {})
        quality_evaluator = MultiDimensionEvaluator(quality_assessment_config, logger)
        
        assessed_data = quality_evaluator.process(combined_data)
        
        # Save assessed data
        assessed_path = output_dir / "assessed_data.jsonl"
        DataUtils.write_jsonl(assessed_data, assessed_path)
        logger.info(f"Saved assessed data to {assessed_path}")
        
        # Step 4: Data filtering
        data_filtering_config = config.get("data_filtering", {})
        data_filter = MultiDimensionFilter(data_filtering_config, logger)
        
        filtered_data = data_filter.process(assessed_data)
        
        # Save filtered data
        filtered_path = output_dir / "filtered_data.jsonl"
        DataUtils.write_jsonl(filtered_data, filtered_path)
        logger.info(f"Saved filtered data to {filtered_path}")
        
        # Step 5: Generate statistics
        stats = DataUtils.get_data_stats(filtered_data)
        
        # Save statistics
        stats_path = output_dir / "data_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(stats, f, indent=2)
            
        logger.info(f"Saved data statistics to {stats_path}")
        
        # Step 6: Save LLM API metrics
        metrics_dir = output_dir / "metrics"
        llm_manager.save_metrics(metrics_dir)
        
        logger.info("Datapresso pipeline completed successfully")
        
        # Print summary
        print("\n" + "="*50)
        print("Datapresso Pipeline Summary")
        print("="*50)
        print(f"Seed data: {len(seed_data)} samples")
        print(f"Generated data: {len(generated_data)} samples")
        print(f"Combined data: {len(combined_data)} samples")
        print(f"Filtered data: {len(filtered_data)} samples")
        print(f"Filtering ratio: {len(filtered_data)/len(combined_data):.2f}")
        print("\nData statistics:")
        print(f"  - Domains: {len(stats['domains'])} domains")
        print(f"  - Difficulty: {stats['difficulty']}")
        print(f"  - Avg. instruction length: {stats['avg_instruction_length']:.1f} chars")
        print(f"  - Avg. response length: {stats['avg_response_length']:.1f} chars")
        print("\nAll outputs saved to: {output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Datapresso pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    run_pipeline(args.config)
