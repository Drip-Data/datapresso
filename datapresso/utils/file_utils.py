"""
File utility functions for Datapresso framework.

This module provides utilities for file operations.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Set


class FileUtils:
    """Utility class for file operations in Datapresso framework."""

    @staticmethod
    def ensure_dir(directory: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Parameters
        ----------
        directory : Union[str, Path]
            Directory path.

        Returns
        -------
        Path
            Path object for the directory.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def list_files(
        directory: Union[str, Path], 
        extension: Optional[str] = None,
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in a directory, optionally filtering by extension.

        Parameters
        ----------
        directory : Union[str, Path]
            Directory path.
        extension : Optional[str], optional
            File extension to filter by (e.g., ".jsonl"), by default None
        recursive : bool, optional
            Whether to search recursively, by default False

        Returns
        -------
        List[Path]
            List of file paths.
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
            
        if recursive:
            if extension:
                return list(directory.glob(f"**/*{extension}"))
            else:
                return [p for p in directory.glob("**/*") if p.is_file()]
        else:
            if extension:
                return list(directory.glob(f"*{extension}"))
            else:
                return [p for p in directory.iterdir() if p.is_file()]

    @staticmethod
    def safe_move(
        src: Union[str, Path], 
        dst: Union[str, Path],
        overwrite: bool = False
    ) -> Path:
        """
        Safely move a file with error handling.

        Parameters
        ----------
        src : Union[str, Path]
            Source file path.
        dst : Union[str, Path]
            Destination file path.
        overwrite : bool, optional
            Whether to overwrite existing destination, by default False

        Returns
        -------
        Path
            Destination path.

        Raises
        ------
        FileExistsError
            If destination exists and overwrite is False.
        FileNotFoundError
            If source file does not exist.
        """
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
            
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Destination file already exists: {dst}")
            
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Use a temporary file to ensure atomic move
        with tempfile.NamedTemporaryFile(delete=False, dir=dst.parent) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Copy to temporary file
            shutil.copy2(src, tmp_path)
            
            # Rename temporary file to destination
            if dst.exists():
                dst.unlink()
            tmp_path.rename(dst)
            
            # Remove source file
            src.unlink()
            
            return dst
        except Exception as e:
            # Clean up temporary file on error
            if tmp_path.exists():
                tmp_path.unlink()
            raise e

    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes.

        Parameters
        ----------
        file_path : Union[str, Path]
            File path.

        Returns
        -------
        int
            File size in bytes.
        """
        return Path(file_path).stat().st_size

    @staticmethod
    def get_latest_file(
        directory: Union[str, Path], 
        extension: Optional[str] = None
    ) -> Optional[Path]:
        """
        Get the most recently modified file in a directory.

        Parameters
        ----------
        directory : Union[str, Path]
            Directory path.
        extension : Optional[str], optional
            File extension to filter by, by default None

        Returns
        -------
        Optional[Path]
            Path to the latest file, or None if no files found.
        """
        files = FileUtils.list_files(directory, extension)
        
        if not files:
            return None
            
        return max(files, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def clean_directory(
        directory: Union[str, Path],
        exclude: Optional[Set[str]] = None
    ) -> None:
        """
        Remove all files and subdirectories in a directory.

        Parameters
        ----------
        directory : Union[str, Path]
            Directory to clean.
        exclude : Optional[Set[str]], optional
            Set of file/directory names to exclude from cleaning, by default None
        """
        directory = Path(directory)
        exclude = exclude or set()
        
        if not directory.exists():
            return
            
        for item in directory.iterdir():
            if item.name in exclude:
                continue
                
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
