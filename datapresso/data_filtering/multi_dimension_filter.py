"""
Multi-dimension filter for Datapresso framework.

This module filters data samples based on multiple dimensions including quality and diversity.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import random

from datapresso.utils.base_module import BaseModule
from datapresso.utils.data_utils import DataUtils
from datapresso.data_filtering.diversity_analyzer import DiversityAnalyzer
from datapresso.data_filtering.quality_filter import QualityFilter


class MultiDimensionFilter(BaseModule):
    """
    Multi-dimension filter for Datapresso framework.
    
    Filters data samples based on multiple dimensions including quality and diversity.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the multi-dimension filter.

        Parameters
        ----------
        config : Dict[str, Any]
            Filter configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        super().__init__(config, logger)
        
        # Initialize components
        self.diversity_analyzer = DiversityAnalyzer(config.get("diversity", {}), logger)
        self.quality_filter = QualityFilter(config.get("quality", {}), logger)
        
        # Configuration
        self.quality_threshold = config.get("quality_threshold", 0.7)
        self.diversity_weight = config.get("diversity_weight", 0.3)
        self.target_size = config.get("target_size", 1000)
        
        # Difficulty distribution
        self.difficulty_distribution = config.get("difficulty_distribution", {
            "easy": 0.3,
            "medium": 0.5,
            "hard": 0.2
        })
        
        # Domain distribution
        self.domain_distribution = config.get("domain_distribution", {})
        
        # Output path
        self.output_dir = Path(config.get("output_dir", "data/filtered"))
        
        self.logger.info(f"Initialized multi-dimension filter with target size: {self.target_size}")

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter data samples based on multiple dimensions.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Data samples to filter.

        Returns
        -------
        List[Dict[str, Any]]
            Filtered data samples.
        """
        self.logger.info(f"Starting multi-dimension filtering for {len(data)} samples")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Step 1: Apply quality threshold
        quality_filtered = self.quality_filter.filter_by_quality(data, self.quality_threshold)
        self.logger.info(f"Quality filtering: {len(data)} -> {len(quality_filtered)} samples")
        
        # Step 2: Analyze diversity
        diversity_scores = self.diversity_analyzer.analyze_diversity(quality_filtered)
        
        # Step 3: Apply balanced selection
        filtered_data = self._balanced_selection(quality_filtered, diversity_scores)
        self.logger.info(f"Balanced selection: {len(quality_filtered)} -> {len(filtered_data)} samples")
        
        # Step 4: Save results
        self._save_results(filtered_data)
        
        self.logger.info(f"Multi-dimension filtering completed: {len(filtered_data)} samples selected")
        
        return filtered_data

    def _balanced_selection(self, data: List[Dict[str, Any]], diversity_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Select samples based on balanced criteria.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Quality-filtered data samples.
        diversity_scores : Dict[str, float]
            Diversity scores for each sample.

        Returns
        -------
        List[Dict[str, Any]]
            Selected data samples.
        """
        # PLACEHOLDER: This method should be implemented by engineers
        # In a real implementation, this would use sophisticated selection algorithms
        
        self.logger.info("Performing balanced selection (placeholder implementation)")
        
        # If we have fewer samples than target size, return all
        if len(data) <= self.target_size:
            return data
            
        # Step 1: Calculate combined scores (quality + diversity)
        combined_scores = {}
        for sample in data:
            sample_id = sample.get("id", "")
            if not sample_id:
                continue
                
            # Get quality score
            quality_score = 0.0
            if "metadata" in sample and "evaluations" in sample["metadata"]:
                quality_score = sample["metadata"]["evaluations"].get("overall_score", 0.0)
                
            # Get diversity score
            diversity_score = diversity_scores.get(sample_id, 0.0)
            
            # Calculate combined score
            combined_score = (quality_score * (1 - self.diversity_weight)) + (diversity_score * self.diversity_weight)
            combined_scores[sample_id] = combined_score
            
        # Step 2: Group samples by difficulty
        difficulty_groups = {"easy": [], "medium": [], "hard": []}
        
        for sample in data:
            sample_id = sample.get("id", "")
            if not sample_id:
                continue
                
            # Get difficulty
            difficulty = "medium"  # Default
            if "metadata" in sample and "evaluations" in sample["metadata"]:
                difficulty_score = sample["metadata"]["evaluations"].get("instruction_complexity", 0.5)
                if difficulty_score < 0.4:
                    difficulty = "easy"
                elif difficulty_score > 0.7:
                    difficulty = "hard"
                    
            # Add to appropriate group
            difficulty_groups[difficulty].append(sample)
            
        # Step 3: Calculate target count for each difficulty level
        target_counts = {}
        for difficulty, percentage in self.difficulty_distribution.items():
            target_counts[difficulty] = int(self.target_size * percentage)
            
        # Ensure we don't exceed target size due to rounding
        total_count = sum(target_counts.values())
        if total_count < self.target_size:
            # Add the remainder to the medium difficulty
            target_counts["medium"] += self.target_size - total_count
            
        # Step 4: Select top samples from each difficulty group
        selected_samples = []
        
        for difficulty, target_count in target_counts.items():
            group = difficulty_groups[difficulty]
            
            # Sort group by combined score
            sorted_group = sorted(group, key=lambda x: combined_scores.get(x.get("id", ""), 0.0), reverse=True)
            
            # Select top samples
            selected = sorted_group[:target_count]
            
            # If we don't have enough samples in this group, take from other groups
            if len(selected) < target_count:
                deficit = target_count - len(selected)
                self.logger.warning(f"Not enough {difficulty} samples, deficit: {deficit}")
                
                # Try to take from other groups
                for other_difficulty in ["medium", "easy", "hard"]:
                    if other_difficulty == difficulty:
                        continue
                        
                    other_group = difficulty_groups[other_difficulty]
                    sorted_other = sorted(other_group, key=lambda x: combined_scores.get(x.get("id", ""), 0.0), reverse=True)
                    
                    # Take samples that haven't been selected yet
                    additional = []
                    for sample in sorted_other:
                        if sample not in selected_samples and sample not in selected and len(additional) < deficit:
                            additional.append(sample)
                            
                    selected.extend(additional)
                    deficit -= len(additional)
                    
                    if deficit <= 0:
                        break
                        
            selected_samples.extend(selected)
            
        # Ensure we don't exceed target size
        if len(selected_samples) > self.target_size:
            selected_samples = selected_samples[:self.target_size]
            
        return selected_samples

    def _save_results(self, data: List[Dict[str, Any]]) -> None:
        """
        Save filtered results.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Filtered data samples.
        """
        # Create filename
        timestamp = int(time.time())
        filename = f"filtered_{timestamp}.jsonl"
        file_path = self.output_dir / filename
        
        # Save data
        DataUtils.write_jsonl(data, file_path)
        
        self.logger.info(f"Saved filtered results to {file_path}")
        
        # Also save a copy with a fixed name for easier reference
        fixed_path = self.output_dir / "filtered_data.jsonl"
        DataUtils.write_jsonl(data, fixed_path)
        
        self.logger.info(f"Saved filtered results to {fixed_path} (fixed name)")
        
    def filter_by_rationale_and_answer(self, data: List[Dict[str, Any]], rationale_threshold: float = 0.7, answer_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Filter samples based on rationale quality and answer accuracy.
        
        Parameters
        ----------
        data : List[Dict[str, Any]]
            Data samples to filter.
        rationale_threshold : float, optional
            Minimum rationale quality score, by default 0.7
        answer_threshold : float, optional
            Minimum answer accuracy score, by default 0.7
            
        Returns
        -------
        List[Dict[str, Any]]
            Filtered samples.
        """
        # PLACEHOLDER: This method should be implemented by engineers
        # In a real implementation, this would use more sophisticated filtering logic
        
        filtered_samples = []
        
        for sample in data:
            # Get evaluation scores
            if "metadata" not in sample or "evaluations" not in sample["metadata"]:
                continue
                
            evaluations = sample["metadata"]["evaluations"]
            
            # Get rationale and answer scores
            rationale_quality = evaluations.get("rationale_quality", 0.0)
            answer_accuracy = evaluations.get("answer_accuracy", 0.0)
            
            # Apply thresholds
            if rationale_quality >= rationale_threshold and answer_accuracy >= answer_threshold:
                filtered_samples.append(sample)
                
        return filtered_samples
