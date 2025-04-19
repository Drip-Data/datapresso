


def _save_intermediate_results(self, data: List[Dict[str, Any]], batch_idx: int) -> None:
    """
    Save intermediate generation results.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        Generated data so far.
    batch_idx : int
        Current batch index.
    """
    # Create filename
    timestamp = int(time.time())
    filename = f"generated_batch_{batch_idx}_{timestamp}.jsonl"
    file_path = self.output_dir / filename
    
    # Save data
    DataUtils.write_jsonl(data, file_path)
    
    self.logger.info(f"Saved intermediate results to {file_path}")

def _save_final_results(self, data: List[Dict[str, Any]]) -> None:
    """
    Save final generation results.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        Generated data.
    """
    # Create filename
    timestamp = int(time.time())
    filename = f"generated_final_{timestamp}.jsonl"
    file_path = self.output_dir / filename
    
    # Save data
    DataUtils.write_jsonl(data, file_path)
    
    self.logger.info(f"Saved final results to {file_path}")
    
    # Also save a copy with a fixed name for easier reference
    fixed_path = self.output_dir / "generated_data.jsonl"
    DataUtils.write_jsonl(data, fixed_path)
    
    self.logger.info(f"Saved final results to {fixed_path} (fixed name)")
