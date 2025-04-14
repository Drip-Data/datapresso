"""
Diversity analyzer for Datapresso framework.

This module analyzes the diversity of data samples across multiple dimensions.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from collections import Counter


class DiversityAnalyzer:
    """
    Diversity analyzer for Datapresso framework.
    
    Analyzes the diversity of data samples across multiple dimensions.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the diversity analyzer.

        Parameters
        ----------
        config : Dict[str, Any]
            Analyzer configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.dimensions = config.get("dimensions", ["domain", "difficulty", "content"])
        self.weights = config.get("weights", {
            "domain": 0.4,
            "difficulty": 0.3,
            "content": 0.3
        })
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
            
        self.logger.info(f"Initialized diversity analyzer with dimensions: {self.dimensions}")

    def analyze_diversity(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze the diversity of data samples.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Data samples to analyze.

        Returns
        -------
        Dict[str, float]
            Diversity scores for each sample.
        """
        self.logger.info(f"Analyzing diversity for {len(data)} samples")
        
        # Step 1: Extract features for each dimension
        features = self._extract_features(data)
        
        # Step 2: Calculate diversity scores for each dimension
        dimension_scores = {}
        for dimension in self.dimensions:
            if dimension in features:
                dimension_scores[dimension] = self._calculate_dimension_diversity(features[dimension], data)
                
        # Step 3: Combine dimension scores into overall diversity scores
        diversity_scores = {}
        for sample in data:
            sample_id = sample.get("id", "")
            if not sample_id:
                continue
                
            # Calculate weighted average of dimension scores
            weighted_sum = 0.0
            total_weight = 0.0
            
            for dimension, weight in self.weights.items():
                if dimension in dimension_scores and sample_id in dimension_scores[dimension]:
                    weighted_sum += dimension_scores[dimension][sample_id] * weight
                    total_weight += weight
                    
            # Normalize by total weight
            if total_weight > 0:
                diversity_scores[sample_id] = weighted_sum / total_weight
            else:
                diversity_scores[sample_id] = 0.0
                
        self.logger.info(f"Diversity analysis completed for {len(data)} samples")
        
        return diversity_scores

    def _extract_features(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract features for each diversity dimension.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Data samples to analyze.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Features for each dimension.
        """
        # PLACEHOLDER: This method should be implemented by engineers
        # In a real implementation, this would extract meaningful features
        
        features = {}
        
        # Domain features
        if "domain" in self.dimensions:
            domain_features = {}
            for sample in data:
                sample_id = sample.get("id", "")
                if not sample_id:
                    continue
                    
                # Extract domain from metadata
                domain = "unknown"
                if "metadata" in sample and "domain" in sample["metadata"]:
                    domain = sample["metadata"]["domain"]
                    
                domain_features[sample_id] = domain
                
            features["domain"] = domain_features
            
        # Difficulty features
        if "difficulty" in self.dimensions:
            difficulty_features = {}
            for sample in data:
                sample_id = sample.get("id", "")
                if not sample_id:
                    continue
                    
                # Extract difficulty from evaluations
                difficulty = 0.5  # Default medium difficulty
                if "metadata" in sample and "evaluations" in sample["metadata"]:
                    difficulty = sample["metadata"]["evaluations"].get("instruction_complexity", 0.5)
                    
                difficulty_features[sample_id] = difficulty
                
            features["difficulty"] = difficulty_features
            
        # Content features
        if "content" in self.dimensions:
            content_features = {}
            for sample in data:
                sample_id = sample.get("id", "")
                if not sample_id:
                    continue
                    
                # PLACEHOLDER: Extract content features
                # In a real implementation, this would use NLP techniques
                
                # Simple placeholder: use instruction length as a feature
                instruction = sample.get("instruction", "")
                content_features[sample_id] = len(instruction)
                
            features["content"] = content_features
            
        return features

    def _calculate_dimension_diversity(self, dimension_features: Dict[str, Any], data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate diversity scores for a specific dimension.

        Parameters
        ----------
        dimension_features : Dict[str, Any]
            Features for the dimension.
        data : List[Dict[str, Any]]
            Data samples.

        Returns
        -------
        Dict[str, float]
            Diversity scores for each sample.
        """
        # PLACEHOLDER: This method should be implemented by engineers
        # In a real implementation, this would use sophisticated diversity metrics
        
        diversity_scores = {}
        
        # Check feature type
        first_feature = next(iter(dimension_features.values()), None)
        
        if isinstance(first_feature, str):
            # Categorical features (e.g., domain)
            return self._calculate_categorical_diversity(dimension_features)
        elif isinstance(first_feature, (int, float)):
            # Numerical features (e.g., difficulty)
            return self._calculate_numerical_diversity(dimension_features)
        else:
            # Default: treat as categorical
            return self._calculate_categorical_diversity(dimension_features)

    def _calculate_categorical_diversity(self, features: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate diversity scores for categorical features.

        Parameters
        ----------
        features : Dict[str, str]
            Categorical features.

        Returns
        -------
        Dict[str, float]
            Diversity scores for each sample.
        """
        # Count occurrences of each category
        category_counts = Counter(features.values())
        total_samples = len(features)
        
        # Calculate inverse frequency (rarer categories get higher scores)
        inverse_frequency = {}
        for category, count in category_counts.items():
            inverse_frequency[category] = 1.0 - (count / total_samples)
            
        # Assign diversity scores based on inverse frequency
        diversity_scores = {}
        for sample_id, category in features.items():
            diversity_scores[sample_id] = inverse_frequency[category]
            
        return diversity_scores

    def _calculate_numerical_diversity(self, features: Dict[str, Union[int, float]]) -> Dict[str, float]:
        """
        Calculate diversity scores for numerical features.

        Parameters
        ----------
        features : Dict[str, Union[int, float]]
            Numerical features.

        Returns
        -------
        Dict[str, float]
            Diversity scores for each sample.
        """
        # Convert to numpy array for easier processing
        values = np.array(list(features.values()))
        
        # Calculate density estimate (more unique values get higher scores)
        from scipy.stats import gaussian_kde
        try:
            # Add small noise to prevent singular matrix
            values_with_noise = values + np.random.normal(0, 1e-6, size=values.shape)
            kde = gaussian_kde(values_with_noise)
            densities = kde(values)
            
            # Invert densities (lower density = higher diversity)
            max_density = np.max(densities)
            if max_density > 0:
                inverse_densities = 1.0 - (densities / max_density)
            else:
                inverse_densities = np.ones_like(densities)
                
            # Assign diversity scores
            diversity_scores = {}
            for i, (sample_id, _) in enumerate(features.items()):
                diversity_scores[sample_id] = float(inverse_densities[i])
                
            return diversity_scores
            
        except Exception as e:
            # Fallback if KDE fails
            self.logger.warning(f"KDE failed: {str(e)}. Using uniform diversity scores.")
            return {sample_id: 0.5 for sample_id in features.keys()}
