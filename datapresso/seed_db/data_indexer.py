"""
Data indexer for Datapresso framework.

This module provides indexing and retrieval capabilities for the seed database.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict


class DataIndexer:
    """
    Data indexer for Datapresso framework.
    
    Provides efficient indexing and retrieval of seed data.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the data indexer.

        Parameters
        ----------
        config : Dict[str, Any]
            Indexer configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize index structures
        self.id_index = {}  # Map from ID to data index
        self.domain_index = defaultdict(list)  # Map from domain to list of indices
        self.difficulty_index = defaultdict(list)  # Map from difficulty bucket to list of indices
        
        # Data storage
        self.data = []
        
        # Index configuration
        self.difficulty_buckets = config.get("difficulty_buckets", 5)
        self.index_path = config.get("index_path", None)
        
        self.logger.info("Initialized data indexer")

    def build_index(self, data: List[Dict[str, Any]]) -> None:
        """
        Build indices for the provided data.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Data to index.
        """
        self.logger.info(f"Building index for {len(data)} items")
        
        # Reset indices
        self.id_index = {}
        self.domain_index = defaultdict(list)
        self.difficulty_index = defaultdict(list)
        
        # Store data
        self.data = data
        
        # Build indices
        for i, item in enumerate(data):
            # Index by ID
            item_id = item.get("id")
            if item_id:
                self.id_index[item_id] = i
                
            # Index by metadata if available
            if "metadata" in item:
                metadata = item["metadata"]
                
                # Index by domain
                if "domain" in metadata:
                    domain = metadata["domain"]
                    self.domain_index[domain].append(i)
                    
                # Index by difficulty
                if "difficulty" in metadata:
                    difficulty = metadata["difficulty"]
                    if isinstance(difficulty, (int, float)):
                        # Bucket the difficulty
                        bucket = self._get_difficulty_bucket(difficulty)
                        self.difficulty_index[bucket].append(i)
                        
        # Log indexing results
        self.logger.info(f"Indexed {len(self.id_index)} items by ID")
        self.logger.info(f"Indexed {len(self.domain_index)} domains")
        self.logger.info(f"Indexed {len(self.difficulty_index)} difficulty buckets")
        
        # Save index if path is provided
        if self.index_path:
            self._save_index()

    def _get_difficulty_bucket(self, difficulty: float) -> int:
        """
        Get the difficulty bucket for a difficulty value.

        Parameters
        ----------
        difficulty : float
            Difficulty value (0-1).

        Returns
        -------
        int
            Bucket index.
        """
        # Ensure difficulty is in range [0, 1]
        difficulty = max(0, min(1, difficulty))
        
        # Calculate bucket
        bucket = int(difficulty * self.difficulty_buckets)
        if bucket == self.difficulty_buckets:
            bucket = self.difficulty_buckets - 1
            
        return bucket

    def _save_index(self) -> None:
        """Save the index to disk."""
        if not self.index_path:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Prepare index data
            index_data = {
                "id_index": self.id_index,
                "domain_index": {k: v for k, v in self.domain_index.items()},
                "difficulty_index": {str(k): v for k, v in self.difficulty_index.items()},
                "config": {
                    "difficulty_buckets": self.difficulty_buckets
                }
            }
            
            # Save to file
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f)
                
            self.logger.info(f"Saved index to {self.index_path}")
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")

    def load_index(self) -> bool:
        """
        Load the index from disk.

        Returns
        -------
        bool
            True if the index was loaded successfully, False otherwise.
        """
        if not self.index_path or not os.path.exists(self.index_path):
            self.logger.warning(f"Index file not found: {self.index_path}")
            return False
            
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
                
            # Load indices
            self.id_index = index_data.get("id_index", {})
            self.domain_index = defaultdict(list, index_data.get("domain_index", {}))
            
            # Convert difficulty bucket keys back to integers
            difficulty_index = index_data.get("difficulty_index", {})
            self.difficulty_index = defaultdict(list)
            for k, v in difficulty_index.items():
                self.difficulty_index[int(k)] = v
                
            # Load config
            config = index_data.get("config", {})
            self.difficulty_buckets = config.get("difficulty_buckets", self.difficulty_buckets)
            
            self.logger.info(f"Loaded index from {self.index_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load index: {str(e)}")
            return False

    def get_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a data item by ID.

        Parameters
        ----------
        item_id : str
            Item ID.

        Returns
        -------
        Optional[Dict[str, Any]]
            Data item if found, None otherwise.
        """
        if item_id in self.id_index:
            index = self.id_index[item_id]
            if 0 <= index < len(self.data):
                return self.data[index]
        return None

    def get_by_domain(self, domain: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get data items by domain.

        Parameters
        ----------
        domain : str
            Domain to filter by.
        limit : Optional[int], optional
            Maximum number of items to return, by default None

        Returns
        -------
        List[Dict[str, Any]]
            List of matching data items.
        """
        indices = self.domain_index.get(domain, [])
        
        if limit is not None and limit < len(indices):
            # Randomly sample if limit is specified
            indices = np.random.choice(indices, size=limit, replace=False).tolist()
            
        return [self.data[i] for i in indices if 0 <= i < len(self.data)]

    def get_by_difficulty(
        self, 
        min_difficulty: float = 0.0, 
        max_difficulty: float = 1.0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get data items by difficulty range.

        Parameters
        ----------
        min_difficulty : float, optional
            Minimum difficulty, by default 0.0
        max_difficulty : float, optional
            Maximum difficulty, by default 1.0
        limit : Optional[int], optional
            Maximum number of items to return, by default None

        Returns
        -------
        List[Dict[str, Any]]
            List of matching data items.
        """
        # Get buckets in the difficulty range
        min_bucket = self._get_difficulty_bucket(min_difficulty)
        max_bucket = self._get_difficulty_bucket(max_difficulty)
        
        # Collect indices from all matching buckets
        indices = []
        for bucket in range(min_bucket, max_bucket + 1):
            indices.extend(self.difficulty_index.get(bucket, []))
            
        if limit is not None and limit < len(indices):
            # Randomly sample if limit is specified
            indices = np.random.choice(indices, size=limit, replace=False).tolist()
            
        return [self.data[i] for i in indices if 0 <= i < len(self.data)]

    def get_random_samples(
        self, 
        count: int, 
        domain: Optional[str] = None,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get random samples from the data.

        Parameters
        ----------
        count : int
            Number of samples to return.
        domain : Optional[str], optional
            Domain to filter by, by default None
        min_difficulty : Optional[float], optional
            Minimum difficulty, by default None
        max_difficulty : Optional[float], optional
            Maximum difficulty, by default None

        Returns
        -------
        List[Dict[str, Any]]
            List of random samples.
        """
        # Filter indices based on criteria
        candidate_indices = list(range(len(self.data)))
        
        if domain is not None:
            domain_indices = set(self.domain_index.get(domain, []))
            candidate_indices = [i for i in candidate_indices if i in domain_indices]
            
        if min_difficulty is not None or max_difficulty is not None:
            min_diff = min_difficulty if min_difficulty is not None else 0.0
            max_diff = max_difficulty if max_difficulty is not None else 1.0
            
            min_bucket = self._get_difficulty_bucket(min_diff)
            max_bucket = self._get_difficulty_bucket(max_diff)
            
            difficulty_indices = set()
            for bucket in range(min_bucket, max_bucket + 1):
                difficulty_indices.update(self.difficulty_index.get(bucket, []))
                
            candidate_indices = [i for i in candidate_indices if i in difficulty_indices]
            
        # Sample from candidate indices
        if not candidate_indices:
            return []
            
        sample_count = min(count, len(candidate_indices))
        sampled_indices = np.random.choice(candidate_indices, size=sample_count, replace=False).tolist()
        
        return [self.data[i] for i in sampled_indices]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed data.

        Returns
        -------
        Dict[str, Any]
            Statistics dictionary.
        """
        stats = {
            "total_items": len(self.data),
            "indexed_items": len(self.id_index),
            "domains": {
                "count": len(self.domain_index),
                "distribution": {k: len(v) for k, v in self.domain_index.items()}
            },
            "difficulty": {
                "buckets": self.difficulty_buckets,
                "distribution": {k: len(v) for k, v in self.difficulty_index.items()}
            }
        }
        
        return stats
