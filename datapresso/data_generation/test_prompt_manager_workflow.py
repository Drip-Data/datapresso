import pytest
import json
from pathlib import Path
from datetime import datetime
from prompt_manager import GenerationPromptManager, DistillPromptManager
from generator_engine import GeneratorEngine

def save_test_results(results: dict, test_name: str, timestamp: str) -> Path:
    """Save test results to JSON file in test_output directory"""
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{test_name}_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    return output_file

def test_generation_task():
    """Test generation task with reasoning templates"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare seed samples for generation
    seed_samples = [
        {
            "id": "seed_001",
            "instruction": "计算1+1的值",
            "response": {
                "origin_text": "让我们一步步计算：\n1. 取第一个数 1\n2. 加上第二个数 1\n3. 得到 2\n因此，1+1=2",
                "rationale": "让我们一步步计算：\n1. 取第一个数 1\n2. 加上第二个数 1\n3. 得到 2",
                "final_answer": "2"
            },
            "metadata": {
                "domain": "数学",
            }
        },
        {
            "id": "seed_002",
            "instruction": "计算2+2的值",
            "response": {
                "origin_text": "让我们计算：\n1. 取第一个数 2\n2. 加上第二个数 2\n3. 得到 4\n因此，2+2=4",
                "rationale": "让我们计算：\n1. 取第一个数 2\n2. 加上第二个数 2\n3. 得到 4",
                "final_answer": "4"
            },
            "metadata": {
                "domain": "math",
            }
        }
    ]

    # Initialize generator config based on generator_engine requirements
    generator_config = {
        "task": "generation",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "target_count": 5000,
        "batch_size": 10,
        "max_retries": 3,
        "output_dir": "data/generated",
        "prompt_templates": {
            "template_type": "reasoning",
            "template_name": "math"
        },
        "initial_filtering": {
            "enabled": False,
            "min_length": 50,
            "max_length": 2000
        },
        "verifiers": {
            "type": "math",
            "math_validation": {
                "verified_model": "xverify"
            },
            "code_execution": {
                "required_packages": ["numpy", "pandas"],
                "timeout": 30.0,
                "max_retries": 3
            }
        }
    }

    # Create generator engine with complete config
    generator = GeneratorEngine(config=generator_config)
    
    # Generate samples for this batch
    math_batch_results = generator._generate_batch(seed_samples)

    # Save results
    save_test_results(math_batch_results, "math_generation_task", timestamp)
    return math_batch_results

def test_distill_task():
    """Test distillation task with math and code templates"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize distillation config
    distill_config = {
        "task": "distillation",
        "model": "deepseek-reasoner",  # 使用与generator_engine一致的模型
        "temperature": 0.7,
        "target_count": 5000,
        "batch_size": 10,
        "max_retries": 3,
        "output_dir": "data/distilled",
        "prompt_templates": {
            "template_type": "reasoning",
            "template_name": "math_reasoning",  # 与generator_engine保持一致的模板格式
            "include_think_process": True  # 添加思考过程配置
        },
        "initial_filtering": {
            "enabled": True,
            "min_length": 50,
            "max_length": 2000
        },
        "verifiers": {
            "type": "math",  # 与generator_engine保持一致的验证器配置
            "math_validation": {
                "verified_model": "xverify"
            },
            "code_execution": {
                "required_packages": ["numpy", "pandas"],
                "timeout": 30.0,
                "max_retries": 3
            }
        }
    }
    
    # Test samples for distillation
    seed_samples = [
        {
            "id": "math_qa_001",
            "instruction": "计算1+1的值",
            "response": {
                "final_answer": "2"
            },
            "metadata": {
                "domain": "math",
                "level": "Level 1"
            }
        },
        {
            "id": "math_qa_002",
            "instruction": "求解方程 2x + 3 = 7",
            "response": {
                "final_answer": "x = 2"
            },
            "metadata": {
                "domain": "math",
                "level": "Level 2"
            }
        }
    ]

    # Create generator engine with complete config
    generator = GeneratorEngine(config=distill_config)
    
    # Generate samples for this batch
    distill_batch_results = generator._generate_batch(seed_samples)

    # Save results
    save_test_results(distill_batch_results, "math_distill_think_task", timestamp)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize distillation config
    distill_config = {
        "task": "distillation",
        "model": "deepseek-reasoner",  # 使用与generator_engine一致的模型
        "temperature": 0.7,
        "target_count": 5000,
        "batch_size": 10,
        "max_retries": 3,
        "output_dir": "data/distilled",
        "prompt_templates": {
            "template_type": "reasoning",
            "template_name": "math_reasoning",  # 与generator_engine保持一致的模板格式
            "include_think_process": False  # 添加思考过程配置
        },
        "initial_filtering": {
            "enabled": True,
            "min_length": 50,
            "max_length": 2000
        },
        "verifiers": {
            "type": "math",  # 与generator_engine保持一致的验证器配置
            "math_validation": {
                "verified_model": "xverify"
            },
            "code_execution": {
                "required_packages": ["numpy", "pandas"],
                "timeout": 30.0,
                "max_retries": 3
            }
        }
    }

    # Create generator engine with complete config
    generator = GeneratorEngine(config=distill_config)
    
    # Generate samples for this batch
    distill_batch_results = generator._generate_batch(seed_samples)

    # Save results
    save_test_results(distill_batch_results, "math_distill_no_think_task", timestamp)


    return distill_batch_results

if __name__ == "__main__":
    # Run all tests
    test_generation_task()
    test_distill_task()

