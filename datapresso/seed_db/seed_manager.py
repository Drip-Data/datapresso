"""
Seed database manager for Datapresso framework.

This module manages the seed database of high-quality reference samples.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import jsonlines
import pandas as pd

from datapresso.utils.base_module import BaseModule
from datapresso.utils.data_utils import DataUtils
from datapresso.utils.file_utils import FileUtils
from datapresso.seed_db.data_validator import DataValidator


class SeedManager(BaseModule):
    """
    Seed database manager for Datapresso framework.
    
    Manages loading, validating, and indexing seed data.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the seed database manager.

        Parameters
        ----------
        config : Dict[str, Any]
            Seed database configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        super().__init__(config, logger)
        
        # Initialize data validator
        self.validator = DataValidator(config.get("validation", {}), logger)
        
        # Initialize data storage
        self.seed_data = []
        self.seed_path = Path(config.get("path", "data/seed"))
        self.seed_format = config.get("format", "jsonl")
        
        self.logger.info(f"Initialized seed database manager with path: {self.seed_path}")

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Load and process seed data.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Input data (ignored for seed database).

        Returns
        -------
        List[Dict[str, Any]]
            Loaded and validated seed data.
        """
        self.logger.info("Loading seed data")
        
        # Ensure seed directory exists
        FileUtils.ensure_dir(self.seed_path)
        
        # Find seed files
        seed_files = self._find_seed_files()
        
        if not seed_files:
            self.logger.warning(f"No seed files found in {self.seed_path}")
            return []
            
        self.logger.info(f"Found {len(seed_files)} seed files")
        
        # Load and validate seed data
        all_seed_data = []
        valid_count = 0
        invalid_count = 0
        
        for i, file_path in enumerate(seed_files):
            self.logger.info(f"Processing seed file {i+1}/{len(seed_files)}: {file_path.name}")
            
            try:
                # Load data from file
                file_data = self._load_file(file_path)
                
                # Validate each data item
                valid_data = []
                for item in file_data:
                    if self.validator.validate(item):
                        valid_data.append(item)
                        valid_count += 1
                    else:
                        invalid_count += 1
                        
                all_seed_data.extend(valid_data)
                
                # Update progress
                self._update_status("loading", i + 1, len(seed_files))
                
            except Exception as e:
                self.logger.error(f"Error processing seed file {file_path}: {str(e)}", exc_info=True)
                
        # Log validation results
        self.logger.info(f"Seed data loaded: {valid_count} valid items, {invalid_count} invalid items")
        
        # Store seed data
        self.seed_data = all_seed_data
        
        # Generate statistics
        self._generate_statistics()
        
        return all_seed_data

    def _find_seed_files(self) -> List[Path]:
        """
        Find seed data files in the seed directory.

        Returns
        -------
        List[Path]
            List of seed file paths.
        """
        # Get file extension based on format
        extension_map = {
            "jsonl": ".jsonl",
            "json": ".json",
            "csv": ".csv",
            "tsv": ".tsv"
        }
        extension = extension_map.get(self.seed_format, ".jsonl")
        
        # Find files with the specified extension
        return FileUtils.list_files(self.seed_path, extension, recursive=True)

    def _load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from a file.

        Parameters
        ----------
        file_path : Path
            Path to the data file.

        Returns
        -------
        List[Dict[str, Any]]
            Loaded data.

        Raises
        ------
        ValueError
            If the file format is unsupported.
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.jsonl':
            return DataUtils.read_jsonl(file_path)
        elif suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both list and dictionary formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Convert dictionary to list of records if needed
                    if "data" in data and isinstance(data["data"], list):
                        return data["data"]
                    else:
                        return [data]
                else:
                    raise ValueError(f"Unsupported JSON structure in {file_path}")
        elif suffix in ['.csv', '.tsv']:
            delimiter = '\t' if suffix == '.tsv' else ','
            df = pd.read_csv(file_path, delimiter=delimiter)
            return df.to_dict(orient='records')
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _generate_statistics(self) -> Dict[str, Any]:
        """
        Generate statistics about the seed data.

        Returns
        -------
        Dict[str, Any]
            Statistics dictionary.
        """
        if not self.seed_data:
            stats = {"count": 0}
            self.logger.info("Seed data statistics: empty")
            return stats
            
        # Basic count statistics
        stats = {
            "count": len(self.seed_data),
            "file_count": len(self._find_seed_files())
        }
        
        # Domain distribution if available
        domains = {}
        for item in self.seed_data:
            if "metadata" in item and "domain" in item["metadata"]:
                domain = item["metadata"]["domain"]
                domains[domain] = domains.get(domain, 0) + 1
                
        if domains:
            stats["domain_distribution"] = domains
            
        # Difficulty distribution if available
        difficulties = []
        for item in self.seed_data:
            if "metadata" in item and "difficulty" in item["metadata"]:
                difficulty = item["metadata"]["difficulty"]
                if isinstance(difficulty, (int, float)):
                    difficulties.append(difficulty)
                    
        if difficulties:
            stats["difficulty"] = {
                "mean": sum(difficulties) / len(difficulties),
                "min": min(difficulties),
                "max": max(difficulties)
            }
            
        # Log statistics
        self.logger.info(f"Seed data statistics: {stats['count']} items from {stats['file_count']} files")
        if "domain_distribution" in stats:
            domains_str = ", ".join(f"{k}: {v}" for k, v in stats["domain_distribution"].items())
            self.logger.info(f"Domain distribution: {domains_str}")
            
        return stats

    def get_seed_data(self) -> List[Dict[str, Any]]:
        """
        Get the loaded seed data.

        Returns
        -------
        List[Dict[str, Any]]
            Seed data.
        """
        return self.seed_data.copy()

    def add_seed_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> int:
        """
        Add new data to the seed database.

        Parameters
        ----------
        data : Union[Dict[str, Any], List[Dict[str, Any]]]
            Data to add.

        Returns
        -------
        int
            Number of items added.
        """
        if isinstance(data, dict):
            data = [data]
            
        # Validate data
        valid_data = []
        for item in data:
            if self.validator.validate(item):
                valid_data.append(item)
                
        # Add to seed data
        self.seed_data.extend(valid_data)
        
        # Save to file
        if valid_data:
            self._save_seed_data(valid_data)
            
        return len(valid_data)

    def _save_seed_data(self, data: List[Dict[str, Any]]) -> Path:
        """
        Save seed data to a file.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Data to save.

        Returns
        -------
        Path
            Path to the saved file.
        """
        # Ensure seed directory exists
        FileUtils.ensure_dir(self.seed_path)
        
        # Generate filename
        import time
        timestamp = int(time.time())
        filename = f"seed_data_{timestamp}.jsonl"
        file_path = self.seed_path / filename
        
        # Save data
        DataUtils.write_jsonl(data, file_path)
        
        self.logger.info(f"Saved {len(data)} seed data items to {file_path}")
        
        return file_path
