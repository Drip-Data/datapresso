#!/bin/bash
# Run Datapresso pipeline with example configuration

# Set environment variables for API keys (replace with your actual keys)
# export OPENAI_API_KEY="your-openai-api-key"
# export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Create necessary directories
mkdir -p data/seed
mkdir -p logs
mkdir -p outputs

# Check if seed data exists
if [ ! -f "data/seed/seed_data.jsonl" ]; then
    echo "Seed data not found. Please create a seed data file at data/seed/seed_data.jsonl"
    echo "Example format:"
    echo '[{"id":"seed_1","instruction":"Explain quantum computing","response":{"origin_text":"Quantum computing...","rationale":"Quantum computing...","final_answer":"Quantum computing is..."}}]'
    exit 1
fi

# Run the pipeline
echo "Running Datapresso pipeline..."
python examples/full_pipeline_example.py --config config/example_config.yaml

# Check if the pipeline was successful
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully!"
    echo "Results are available in the outputs directory."
else
    echo "Pipeline failed. Check logs for details."
    exit 1
fi
