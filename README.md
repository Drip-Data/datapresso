# Datapresso Data Construction Framework

A systematic framework for generating, evaluating, and filtering high-quality small-sample datasets (LIMO data) for efficient model fine-tuning.

## Overview

Datapresso aims to solve the data bottleneck and computational cost challenges in traditional large-scale data fine-tuning. By focusing on data quality, diversity, and automated evaluation, the framework enables the creation of small but highly effective datasets for model fine-tuning.

## Key Features

- End-to-end pipeline for LIMO data construction
- Quality-first approach with multi-dimensional assessment
- Diversity-driven data selection
- Automated generation and evaluation
- Flexible configuration system (via Hydra/YAML)
- Integration with training and evaluation workflows (e.g., LlamaFactory)
- Support for multiple LLM providers (OpenAI, Anthropic, Local)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Datapresso.git
cd Datapresso

# 2. Create a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# 3. Install the package and core dependencies
pip install -e .

# 4. Install development dependencies (Optional, for testing and linting)
pip install -e ".[dev]"
```

## Getting Started: A Quick Example

1.  **Configure your LLM API Keys:**
    *   For providers like OpenAI or Anthropic, set the corresponding environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or place them securely in your configuration file (not recommended for version control). Refer to `datapresso/llm_api/` documentation for details.
2.  **Prepare Seed Data:** Place your initial seed data file(s) (e.g., `seed_tasks.jsonl`) in the `data/seed/` directory. Ensure it follows the expected format (see `docs/data_format/`).
3.  **Adapt an Example Configuration:**
    *   Copy the example configuration: `cp config/example_config.yaml config/user_configs/my_first_run.yaml`
    *   Edit `config/user_configs/my_first_run.yaml`:
        *   Adjust `llm_api` settings (provider, model name).
        *   Configure paths if necessary (defaults usually work).
        *   Review generation, assessment, and filtering parameters.
4.  **Run the Pipeline:**
    ```python
    # Option 1: Using the Python script (create a run.py if needed)
    # In run.py:
    # from datapresso import Pipeline
    # pipeline = Pipeline(config_path="config/user_configs/my_first_run.yaml")
    # pipeline.run()
    # results = pipeline.get_results()
    # print(results)
    #
    # Then execute: python run.py

    # Option 2: Directly in Python interpreter
    from datapresso import Pipeline
    pipeline = Pipeline(config_path="config/user_configs/my_first_run.yaml")
    pipeline.run()
    results = pipeline.get_results() # Check what this returns
    print("Pipeline finished. Check logs in logs/ and output data in data/")
    ```
5.  **Check Outputs:**
    *   Monitor the console output and logs in the `logs/` directory.
    *   Examine the generated/assessed/filtered data in the corresponding `data/` subdirectories.

## Configuration

Datapresso uses [Hydra](https://hydra.cc/) for flexible configuration management via YAML files located in the `config/` directory.

*   `config/default.yaml`: Defines the default structure and values for all configuration parameters. **Do not edit directly.**
*   `config/example_config.yaml`: A practical example demonstrating how to set up a pipeline run. Use this as a template.
*   `config/user_configs/`: Place your custom configuration files here.
*   The pipeline is run using a specific configuration file (e.g., `pipeline = Pipeline(config_path="config/user_configs/my_config.yaml")`).
*   Refer to `docs/architecture/01_user_input_layer.md` (or similar) for detailed configuration options.

## Documentation

For detailed architecture, module descriptions, and advanced usage, please refer to the main documentation entry point:

*   [**Project Documentation Index**](docs/index.md) (You might need to create this index file)
*   [Pipeline Orchestration](docs/architecture/00a_pipeline_orchestration.md)
*   [Error Handling & Logging](docs/architecture/09_error_handling_logging.md)

## Troubleshooting

*   **`ModuleNotFoundError`**: Ensure you have installed the package correctly (`pip install -e .`) and activated your virtual environment. Install optional dependencies if needed (`pip install -e ".[dev]"` or specific extras).
*   **API Key Errors (401 Unauthorized, etc.)**: Double-check that your API keys are correctly set as environment variables or in your configuration (ensure the config file is not tracked by Git if it contains secrets). Verify the key has the necessary permissions.
*   **`FileNotFoundError`**: Verify that the input data paths specified in your configuration file (e.g., seed data path) are correct and point to existing files relative to the project root. Check that the `data/` subdirectories exist or that the framework creates them.
*   **Rate Limit Errors (429 Too Many Requests)**: You may be hitting API rate limits. Check the specific provider's limits. Consider adding delays or reducing concurrency if applicable (check configuration options). The framework might have built-in retry mechanisms (see Error Handling docs).
*   **Configuration Errors**: The pipeline should fail early with messages indicating issues in your YAML file. Check the syntax, indentation, and parameter names against `config/default.yaml` or documentation. Use a YAML validator if needed.
*   **Unexpected Output/Behavior**:
    *   Check the logs in the `logs/` directory. Increase the log level in your configuration for more details (e.g., set `logging.level: DEBUG`).
    *   Review the configuration file carefully to ensure parameters match your intent.
    *   Isolate the problematic stage by disabling other stages in the config and running them one by one.
*   **Dependency Conflicts**: If you encounter issues after installing other packages, try creating a fresh virtual environment and installing Datapresso first. Check `setup.py` for version constraints.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. (Assuming a LICENSE file exists or will be created).
