"""
Logging utilities for Datapresso framework.

This module provides logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import os
from datetime import datetime


class LoggingUtils:
    """Utility class for logging in Datapresso framework."""

    @staticmethod
    def setup_logging(
        level: str = "INFO",
        log_dir: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        file_output: bool = True,
        log_format: Optional[str] = None,
        project_name: str = "datapresso"
    ) -> logging.Logger:
        """
        Set up logging configuration.

        Parameters
        ----------
        level : str, optional
            Logging level, by default "INFO"
        log_dir : Optional[Union[str, Path]], optional
            Directory to save log files, by default None
        console_output : bool, optional
            Whether to output logs to console, by default True
        file_output : bool, optional
            Whether to output logs to file, by default True
        log_format : Optional[str], optional
            Log message format, by default None
        project_name : str, optional
            Project name for logger, by default "datapresso"

        Returns
        -------
        logging.Logger
            Configured logger instance.
        """
        # Convert level string to logging level
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")

        # Create logger
        logger = logging.getLogger(project_name)
        logger.setLevel(numeric_level)
        logger.handlers = []  # Clear existing handlers

        # Set log format
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Add file handler if requested
        if file_output and log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{project_name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
        """
        Log configuration parameters.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance.
        config : Dict[str, Any]
            Configuration dictionary to log.
        """
        logger.info("Configuration parameters:")
        for key, value in config.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"  {sub_key}: {sub_value}")
            else:
                logger.info(f"{key}: {value}")

    @staticmethod
    def log_progress(
        logger: logging.Logger, 
        current: int, 
        total: int, 
        stage: str, 
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log progress information.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance.
        current : int
            Current progress count.
        total : int
            Total items to process.
        stage : str
            Current processing stage.
        additional_info : Optional[Dict[str, Any]], optional
            Additional information to log, by default None
        """
        percentage = (current / total) * 100 if total > 0 else 0
        progress_msg = f"Progress [{stage}]: {current}/{total} ({percentage:.2f}%)"
        
        if additional_info:
            info_str = ", ".join(f"{k}={v}" for k, v in additional_info.items())
            progress_msg += f" - {info_str}"
            
        logger.info(progress_msg)

    @staticmethod
    def log_error(
        logger: logging.Logger, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log error information with context.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance.
        error : Exception
            Exception to log.
        context : Optional[Dict[str, Any]], optional
            Context information, by default None
        """
        error_msg = f"Error: {str(error)}"
        
        if context:
            context_str = json.dumps(context, default=str)
            error_msg += f" - Context: {context_str}"
            
        logger.error(error_msg, exc_info=True)
