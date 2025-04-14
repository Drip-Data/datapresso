"""
Main pipeline orchestrator for Datapresso framework.

This module provides the main Pipeline class that orchestrates the entire Datapresso data
construction process.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import json

from datapresso.utils.logging_utils import LoggingUtils
from datapresso.utils.data_utils import DataUtils
from datapresso.utils.file_utils import FileUtils
from datapresso.user_input.config_manager import ConfigManager
from datapresso.seed_db.seed_manager import SeedManager
from datapresso.data_generation.generator_engine import GeneratorEngine
from datapresso.quality_assessment.multi_dimension_evaluator import MultiDimensionEvaluator
from datapresso.data_filtering.balanced_selector import BalancedSelector
from datapresso.llamafactory.training_manager import TrainingManager
from datapresso.evaluation.benchmark_tester import BenchmarkTester


class Pipeline:
    """
    Main pipeline orchestrator for Datapresso framework.
    
    This class coordinates the execution of all pipeline stages and manages
    the flow of data between them.
    """

    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Datapresso pipeline.

        Parameters
        ----------
        config_path : Optional[Union[str, Path]], optional
            Path to configuration file, by default None
        config_dict : Optional[Dict[str, Any]], optional
            Configuration dictionary, by default None

        Raises
        ------
        ValueError
            If neither config_path nor config_dict is provided.
        """
        if config_path is None and config_dict is None:
            raise ValueError("Either config_path or config_dict must be provided")
            
        # Initialize configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.validate_config(config_dict)
            
        # Set up logging
        self.logger = LoggingUtils.setup_logging(
            level=self.config.get("logging", {}).get("level", "INFO"),
            log_dir=self.config.get("logging", {}).get("save_path", "logs/"),
            console_output=self.config.get("logging", {}).get("console_output", True),
            file_output=self.config.get("logging", {}).get("file_output", True),
            log_format=self.config.get("logging", {}).get("log_format"),
            project_name=self.config.get("project_name", "datapresso")
        )
        
        # Log configuration
        self.logger.info(f"Initialized Datapresso pipeline for project: {self.config.get('project_name')}")
        LoggingUtils.log_config(self.logger, self.config)
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Initialize results storage
        self.results = {}
        
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components")
        
        # Initialize components with their respective configurations
        self.seed_manager = SeedManager(
            config=self.config.get("seed_db", {}),
            logger=self.logger
        )
        
        # Initialize data generation if enabled
        if self.config.get("data_generation", {}).get("enabled", True):
            self.generator = GeneratorEngine(
                config=self.config.get("data_generation", {}),
                logger=self.logger
            )
        else:
            self.generator = None
            self.logger.info("Data generation disabled in configuration")
            
        # Initialize quality assessment
        self.evaluator = MultiDimensionEvaluator(
            config=self.config.get("quality_assessment", {}),
            logger=self.logger
        )
        
        # Initialize data filtering
        self.selector = BalancedSelector(
            config=self.config.get("data_filtering", {}),
            logger=self.logger
        )
        
        # Initialize advanced assessment if enabled
        if self.config.get("advanced_assessment", {}).get("enabled", False):
            self.advanced_evaluator = TrainingManager(
                config=self.config.get("advanced_assessment", {}),
                logger=self.logger
            )
        else:
            self.advanced_evaluator = None
            self.logger.info("Advanced assessment disabled in configuration")
            
        # Initialize training if enabled
        if self.config.get("training", {}).get("enabled", True):
            self.trainer = TrainingManager(
                config=self.config.get("training", {}),
                logger=self.logger
            )
        else:
            self.trainer = None
            self.logger.info("Training disabled in configuration")
            
        # Initialize evaluation
        self.evaluator = BenchmarkTester(
            config=self.config.get("evaluation", {}),
            logger=self.logger
        )
        
        self.logger.info("All pipeline components initialized")
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete Datapresso pipeline.

        Returns
        -------
        Dict[str, Any]
            Results of the pipeline execution.
        """
        self.logger.info("Starting Datapresso pipeline execution")
        start_time = time.time()
        
        try:
            # Step 1: Load seed data
            self.logger.info("Step 1: Loading seed data")
            seed_data = self.seed_manager.process([])
            self.results["seed_data"] = {
                "count": len(seed_data),
                "path": self.config.get("seed_db", {}).get("path")
            }
            
            # Step 2: Generate data (if enabled)
            if self.generator:
                self.logger.info("Step 2: Generating data")
                generated_data = self.generator.process(seed_data)
                self.results["generated_data"] = {
                    "count": len(generated_data),
                    "path": os.path.join(self.config.get("output_dir", "outputs"), "generated")
                }
            else:
                self.logger.info("Step 2: Skipping data generation (disabled)")
                generated_data = seed_data
                
            # Step 3: Assess data quality
            self.logger.info("Step 3: Assessing data quality")
            assessed_data = self.evaluator.process(generated_data)
            self.results["assessed_data"] = {
                "count": len(assessed_data),
                "path": os.path.join(self.config.get("output_dir", "outputs"), "assessed")
            }
            
            # Step 4: Filter and select data
            self.logger.info("Step 4: Filtering and selecting data")
            filtered_data = self.selector.process(assessed_data)
            self.results["filtered_data"] = {
                "count": len(filtered_data),
                "path": os.path.join(self.config.get("output_dir", "outputs"), "filtered")
            }
            
            # Step 5: Advanced assessment (if enabled)
            if self.advanced_evaluator:
                self.logger.info("Step 5: Performing advanced assessment")
                final_data = self.advanced_evaluator.process(filtered_data)
                self.results["advanced_assessed_data"] = {
                    "count": len(final_data),
                    "path": os.path.join(self.config.get("output_dir", "outputs"), "advanced_assessed")
                }
            else:
                self.logger.info("Step 5: Skipping advanced assessment (disabled)")
                final_data = filtered_data
                
            # Save final LIMO dataset
            final_path = os.path.join(self.config.get("output_dir", "outputs"), "final", "limo_dataset.jsonl")
            FileUtils.ensure_dir(os.path.dirname(final_path))
            DataUtils.write_jsonl(final_data, final_path)
            self.results["final_data"] = {
                "count": len(final_data),
                "path": final_path
            }
            
            # Step 6: Train model (if enabled)
            if self.trainer:
                self.logger.info("Step 6: Training model with LIMO data")
                training_results = self.trainer.process(final_data)
                self.results["training"] = training_results
            else:
                self.logger.info("Step 6: Skipping model training (disabled)")
                
            # Step 7: Evaluate model
            self.logger.info("Step 7: Evaluating model performance")
            evaluation_results = self.evaluator.process([])  # Empty list as data is loaded inside
            self.results["evaluation"] = evaluation_results
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self.results["execution_time"] = execution_time
            
            self.logger.info(f"Datapresso pipeline completed successfully in {execution_time:.2f} seconds")
            
            # Save results summary
            results_path = os.path.join(self.config.get("output_dir", "outputs"), "results_summary.json")
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            self.results["error"] = str(e)
            self.results["status"] = "failed"
            raise
            
    def get_results(self) -> Dict[str, Any]:
        """
        Get the results of the pipeline execution.

        Returns
        -------
        Dict[str, Any]
            Pipeline execution results.
        """
        return self.results
        
    def run_module(self, module_name: str, input_data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Run a specific module of the pipeline.

        Parameters
        ----------
        module_name : str
            Name of the module to run.
        input_data : Optional[List[Dict[str, Any]]], optional
            Input data for the module, by default None

        Returns
        -------
        List[Dict[str, Any]]
            Output data from the module.

        Raises
        ------
        ValueError
            If the module name is invalid.
        """
        self.logger.info(f"Running module: {module_name}")
        
        # Map module names to components
        module_map = {
            "seed_db": self.seed_manager,
            "data_generation": self.generator,
            "quality_assessment": self.evaluator,
            "data_filtering": self.selector,
            "advanced_assessment": self.advanced_evaluator,
            "training": self.trainer,
            "evaluation": self.evaluator
        }
        
        if module_name not in module_map:
            raise ValueError(f"Invalid module name: {module_name}")
            
        module = module_map[module_name]
        
        if module is None:
            raise ValueError(f"Module {module_name} is disabled in configuration")
            
        # Process data
        if input_data is None:
            # For seed_db, empty list is valid input
            if module_name == "seed_db":
                input_data = []
            else:
                raise ValueError(f"Input data required for module: {module_name}")
                
        return module.process(input_data)


@hydra.main(config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Datapresso framework.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    """
    # Convert Hydra config to dictionary
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize and run pipeline
    pipeline = Pipeline(config_dict=config_dict)
    results = pipeline.run()
    
    print(f"Pipeline execution completed. Results saved to: {results.get('final_data', {}).get('path')}")


if __name__ == "__main__":
    main()
