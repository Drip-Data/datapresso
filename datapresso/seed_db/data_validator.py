"""
Data validator for Datapresso framework.

This module provides validation for data records in the seed database.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import json
import jsonschema


class DataValidator:
    """
    Data validator for Datapresso framework.
    
    Validates data records against schema and quality requirements.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the data validator.

        Parameters
        ----------
        config : Dict[str, Any]
            Validation configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Default schema for data validation
        self.schema = {
            "type": "object",
            "required": ["id", "instruction", "response"],
            "properties": {
                "id": {"type": "string"},
                "instruction": {"type": "string"},
                "response": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "difficulty": {"type": ["number", "string"]},
                        "source": {"type": "string"},
                        "creation_timestamp": {"type": "string"}
                    }
                }
            }
        }
        
        # Load custom schema if provided
        if "schema_path" in config:
            try:
                with open(config["schema_path"], 'r', encoding='utf-8') as f:
                    self.schema = json.load(f)
                self.logger.info(f"Loaded custom schema from {config['schema_path']}")
            except Exception as e:
                self.logger.error(f"Failed to load custom schema: {str(e)}")
                
        self.schema_check = config.get("schema_check", True)
        self.content_check = config.get("content_check", True)
        
        self.logger.info("Initialized data validator")

    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate a data record.

        Parameters
        ----------
        data : Dict[str, Any]
            Data record to validate.

        Returns
        -------
        bool
            True if the data is valid, False otherwise.
        """
        # Schema validation
        if self.schema_check:
            try:
                jsonschema.validate(instance=data, schema=self.schema)
            except jsonschema.exceptions.ValidationError as e:
                self.logger.debug(f"Schema validation failed: {str(e)}")
                return False
                
        # Content validation
        if self.content_check:
            if not self._validate_content(data):
                return False
                
        return True

    def _validate_content(self, data: Dict[str, Any]) -> bool:
        """
        Validate the content of a data record.

        Parameters
        ----------
        data : Dict[str, Any]
            Data record to validate.

        Returns
        -------
        bool
            True if the content is valid, False otherwise.
        """
        # Check for empty or very short instruction/response
        min_instruction_length = self.config.get("min_instruction_length", 10)
        min_response_length = self.config.get("min_response_length", 10)
        
        if len(data.get("instruction", "").strip()) < min_instruction_length:
            self.logger.debug(f"Instruction too short: {data.get('id', 'unknown')}")
            return False
            
        if len(data.get("response", "").strip()) < min_response_length:
            self.logger.debug(f"Response too short: {data.get('id', 'unknown')}")
            return False
            
        # Check for duplicate instruction/response
        if data.get("instruction", "").strip() == data.get("response", "").strip():
            self.logger.debug(f"Instruction and response are identical: {data.get('id', 'unknown')}")
            return False
            
        # Check for valid metadata if present
        if "metadata" in data:
            metadata = data["metadata"]
            
            # Check domain if required
            if self.config.get("require_domain", False) and "domain" not in metadata:
                self.logger.debug(f"Missing required domain in metadata: {data.get('id', 'unknown')}")
                return False
                
            # Check difficulty if present
            if "difficulty" in metadata:
                difficulty = metadata["difficulty"]
                if isinstance(difficulty, (int, float)) and (difficulty < 0 or difficulty > 1):
                    self.logger.debug(f"Invalid difficulty value: {difficulty}")
                    return False
                    
        return True

    def batch_validate(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate a batch of data records.

        Parameters
        ----------
        data_list : List[Dict[str, Any]]
            List of data records to validate.

        Returns
        -------
        List[Dict[str, Any]]
            List of valid data records.
        """
        valid_data = []
        invalid_count = 0
        
        for item in data_list:
            if self.validate(item):
                valid_data.append(item)
            else:
                invalid_count += 1
                
        self.logger.info(f"Batch validation: {len(valid_data)} valid, {invalid_count} invalid")
        
        return valid_data
