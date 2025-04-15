"""
Base module interface for Datapresso framework.

This module defines the base interface that all pipeline modules must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging


class BaseModule(ABC):
    """
    Base abstract class for all Datapresso framework modules.
    
    All pipeline modules should inherit from this class and implement
    its abstract methods.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the module with configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Module configuration.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._status = {"state": "initialized", "progress": 0, "total": 0}

    @abstractmethod
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the input data.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Input data to process.

        Returns
        -------
        List[Dict[str, Any]]
            Processed data.
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the module.

        Returns
        -------
        Dict[str, Any]
            Status information.
        """
        return self._status.copy()

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the module configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            New configuration to merge with existing.
        """
        self.config.update(config)
        self.logger.info(f"Configuration updated for {self.__class__.__name__}")

    def _update_status(self, state: str, progress: int, total: int) -> None:
        """
        Update the module status.

        Parameters
        ----------
        state : str
            Current state description.
        progress : int
            Current progress count.
        total : int
            Total items to process.
        """
        self._status = {
            "state": state,
            "progress": progress,
            "total": total,
            "percentage": (progress / total) * 100 if total > 0 else 0
        }
        
        # Log progress
        if total > 0:
            self.logger.info(
                f"Progress [{self.__class__.__name__}]: "
                f"{progress}/{total} ({self._status['percentage']:.2f}%)"
            )
