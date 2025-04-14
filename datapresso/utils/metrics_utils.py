"""
Metrics utility functions for Datapresso framework.

This module provides utilities for calculating and tracking metrics.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
from collections import defaultdict


class MetricsUtils:
    """Utility class for metrics calculations in Datapresso framework."""

    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of values.

        Parameters
        ----------
        values : List[float]
            List of numeric values.

        Returns
        -------
        Dict[str, float]
            Dictionary containing statistics (mean, median, min, max, std).
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0
            }
            
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "count": len(values)
        }

    @staticmethod
    def calculate_distribution(
        values: List[float], 
        bins: int = 10
    ) -> Dict[str, List[float]]:
        """
        Calculate histogram distribution of values.

        Parameters
        ----------
        values : List[float]
            List of numeric values.
        bins : int, optional
            Number of histogram bins, by default 10

        Returns
        -------
        Dict[str, List[float]]
            Dictionary with 'bins' and 'counts' lists.
        """
        if not values:
            return {"bins": [], "counts": []}
            
        hist, bin_edges = np.histogram(values, bins=bins)
        
        return {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist()
        }

    @staticmethod
    def group_by_category(
        data: List[Dict[str, Any]], 
        category_key: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group data by a category field.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of data records.
        category_key : str
            Key to group by (can be a nested key using dot notation).

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary mapping category values to lists of records.
        """
        result = defaultdict(list)
        
        for item in data:
            # Handle nested keys (e.g., "metadata.domain")
            if "." in category_key:
                parts = category_key.split(".")
                value = item
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
            else:
                value = item.get(category_key)
                
            # Convert to string for consistent keys
            category = str(value) if value is not None else "unknown"
            result[category].append(item)
            
        return dict(result)

    @staticmethod
    def calculate_metric_by_group(
        data: List[Dict[str, Any]],
        group_key: str,
        metric_fn: Callable[[List[Dict[str, Any]]], float]
    ) -> Dict[str, float]:
        """
        Calculate a metric for each group of data.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of data records.
        group_key : str
            Key to group by.
        metric_fn : Callable[[List[Dict[str, Any]]], float]
            Function that calculates a metric for a list of records.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping group values to metric values.
        """
        grouped = MetricsUtils.group_by_category(data, group_key)
        return {group: metric_fn(items) for group, items in grouped.items()}

    @staticmethod
    def calculate_improvement(
        before: float, 
        after: float
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate improvement metrics between before and after values.

        Parameters
        ----------
        before : float
            Value before change.
        after : float
            Value after change.

        Returns
        -------
        Dict[str, Union[float, str]]
            Dictionary with absolute and percentage improvements.
        """
        absolute = after - before
        
        if before == 0:
            percentage = float('inf') if after > 0 else 0.0
        else:
            percentage = (absolute / before) * 100
            
        return {
            "absolute": absolute,
            "percentage": percentage,
            "formatted": f"{absolute:.4f} ({percentage:+.2f}%)"
        }
