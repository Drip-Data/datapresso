"""
Data utility functions for Datapresso framework.

This module provides utilities for data handling, conversion, and validation.
"""

import json
import jsonlines
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from pathlib import Path


class DataUtils:
    """Utility class for data operations in Datapresso framework."""

    @staticmethod
    def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Read data from a JSONL file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the JSONL file.

        Returns
        -------
        List[Dict[str, Any]]
            List of data records.
        """
        data = []
        with jsonlines.open(file_path, mode='r') as reader:
            for item in reader:
                data.append(item)
        return data

    @staticmethod
    def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
        """
        Write data to a JSONL file.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of data records to write.
        file_path : Union[str, Path]
            Path to the output JSONL file.
        """
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(file_path, mode='w') as writer:
            for item in data:
                writer.write(item)

    @staticmethod
    def to_pandas(data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert list of dictionaries to pandas DataFrame.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of data records.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame containing the data.
        """
        return pd.DataFrame(data)

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert pandas DataFrame to list of dictionaries.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame to convert.

        Returns
        -------
        List[Dict[str, Any]]
            List of data records.
        """
        return df.to_dict(orient='records')

    @staticmethod
    def validate_data_format(data: Dict[str, Any]) -> bool:
        """
        Validate if a data record follows the required format.

        Parameters
        ----------
        data : Dict[str, Any]
            Data record to validate.

        Returns
        -------
        bool
            True if the data format is valid, False otherwise.
        """
        # Check required fields
        required_fields = ['id', 'instruction', 'response']
        for field in required_fields:
            if field not in data:
                return False
            
        # Check metadata if present
        if 'metadata' in data and not isinstance(data['metadata'], dict):
            return False
            
        return True

    @staticmethod
    def merge_metadata(original: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge new metadata with original data, following the extension strategy.

        Parameters
        ----------
        original : Dict[str, Any]
            Original data record.
        new_data : Dict[str, Any]
            New metadata to merge.

        Returns
        -------
        Dict[str, Any]
            Updated data record with merged metadata.
        """
        result = original.copy()
        
        # Initialize metadata if not present
        if 'metadata' not in result:
            result['metadata'] = {}
            
        # Update metadata with new data
        if 'metadata' in new_data:
            for key, value in new_data['metadata'].items():
                result['metadata'][key] = value
        
        return result

    @staticmethod
    def generate_id(prefix: str = "sample") -> str:
        """
        Generate a unique ID for a data sample.

        Parameters
        ----------
        prefix : str, optional
            Prefix for the ID, by default "sample"

        Returns
        -------
        str
            Unique ID string.
        """
        import uuid
        import time
        
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        
        return f"{prefix}_{timestamp}_{unique_id}"
