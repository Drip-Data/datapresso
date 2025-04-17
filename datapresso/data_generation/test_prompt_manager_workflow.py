import pytest
import json
from pathlib import Path
from datetime import datetime
from prompt_manager import GenerationPromptManager,DistillPromptManager
from generator_engine import GeneratorEngine

def test_generate_reasoning_batch(task):
    """Test batch generation with reasoning templates"""
    

    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"generated_samples_{timestamp}.json"

    # Only support one template once
    if task == "reasoning":
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
                },
            }
            
        ]

        config = {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "batch_size": 2,
            "prompt_templates": {
                "template_type": "reasoning",
                "template_name": "math"
            },
            "output_dir": "test_output"
        }

    else:

        seed_samples = [
            {
                "id": "seed_001",
                "question": "什么是Python?",
                "response": {
                "origin_text": "Python是一种高级编程语言",
                },
                "metadata": {
                    "domain": "programming"
                }
            }
                     ]
        
        config = {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "batch_size": 2,
            "prompt_templates": {
                "template_type": "standard",
                "template_name": "domain"
            },
            "output_dir": "test_output"
        }



    # 初始化生成器引擎
    generator = GeneratorEngine(config)

    # 执行批量生成
    generated_samples = generator._generate_reasoning_batch(seed_samples)

    # # 验证生成结果
    # assert len(generated_samples) == len(seed_samples), "生成样本数量应与种子样本相同"
    
    # for sample in generated_samples:
    #     # 验证生成样本的结构
    #     assert "question" in sample, "生成的样本应包含问题"
    #     assert "response" in sample, "生成的样本应包含响应"
    #     assert "rationale" in sample["response"], "响应应包含推理过程"
    #     assert "metadata" in sample, "生成的样本应包含元数据"
        
    #     # 验证元数据字段
    #     metadata = sample["metadata"]
    #     assert "domain" in metadata, "元数据应包含领域信息"
    #     assert "source" in metadata, "元数据应包含来源信息"
    #     assert "creation_timestamp" in metadata, "元数据应包含创建时间戳"


    
    # 创建完整的输出数据结构
    output_data = {
        "samples": generated_samples
    }
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Generated samples saved to: {output_file}")

def test_prompt_generation():
    """Test and display prompt generation for both standard and reasoning templates"""
    reasoning_seed = {
        "instruction": "计算1+1的值",
        "response": {
            "origin_text": "让我们一步步计算：\n1. 取第一个数 1\n2. 加上第二个数 1\n3. 得到 2\n因此，1+1=2",
            "rationale": "让我们一步步计算：\n1. 取第一个数 1\n2. 加上第二个数 1\n3. 得到 2",
            "final_answer": "2"
        },
        "metadata": {
            "domain": "math"
        }
    }

    # Test data without rationale (for standard template)
    standard_seed = {
        "instruction": "什么是Python?",
        "response": {
            "origin_text": "Python是一种高级编程语言",
        },
        "metadata": {
            "domain": "programming"
        }
    }

    # Test standard template
    print("\n=== Standard Template Prompt ===")
    standard_config = {"template_type": "standard","template_name":"domain"}
    standard_manager = GenerationPromptManager(standard_config)
    standard_prompt = standard_manager.create_prompt(standard_seed)
    print("System:", standard_prompt["system_message"])
    print("User:", standard_prompt["user_message"])

    # Test math reasoning template
    print("\n=== Math Reasoning Template Prompt ===")
    math_config = {"template_type": "reasoning", "template_name": "math"}
    math_manager = GenerationPromptManager(math_config)
    math_prompt = math_manager.create_prompt(reasoning_seed)
    print("System:", math_prompt["system_message"])
    print("User:", math_prompt["user_message"])


def test_prompt_distillation():
    """Test distillation prompt generation workflow with different configurations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"test_outputs/distillation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # test_code_samples = [
    #     {
    #         "question": "编写一个函数计算斐波那契数列的第n项",
    #         "response": {
    #                     "final_answer": """def fibonacci(n):
    #         if n <= 1:
    #             return n
    #         return fibonacci(n-1) + fibonacci(n-2)"""
    #         },
    #         "metadata": {
    #             "domain": "programming",
    #             "level": "Level 5"
    #         }
    #     },
    #     {
    #         "question": "实现一个简单的冒泡排序算法",
    #         "response": {
    #             "final_answer": """def bubble_sort(arr):
    # n = len(arr)
    # for i in range(n):
    #     for j in range(0, n-i-1):
    #         if arr[j] > arr[j+1]:
    #             arr[j], arr[j+1] = arr[j+1], arr[j]
    # return arr"""
    #         },
    #         "metadata": {
    #             "domain": "programming",
    #             "level": "Level 5"
    #         }
    #     }
    # ]

    #     # 测试代码推理蒸馏
    # code_config = {
    #     "template_name": "code_reasoning"
    # }
    # code_manager = DistillationPromptManager(code_config)

    test_math_samples = [
        {
            "question": "计算1+1的值",
            "response": {
                "final_answer": "2"
            },
            "metadata": {
                "domain": "math",
                "level": "Level 4"
            }
        },
        {
            "question": "求解方程 2x + 3 = 7",
            "response": {
                "final_answer": "x = 2"
            },
            "metadata": {
                "domain": "math",
                "level": "Level 4"
            }
        },
    ]

    results = {
        "math_reasoning": [],
        "code_reasoning": []
    }

    # 测试数学推理蒸馏
    math_config = {
        "template_name": "math_reasoning",
        "include_think_process": True
    }
    math_manager = DistillPromptManager(math_config)

    from openai import OpenAI

    client = OpenAI(api_key="sk-56a20268a62a4c0fab38fa496e5c2d5f", base_url="https://api.deepseek.com")

    # 生成并保存提示
    for sample in test_math_samples:
    
        prompt = math_manager.create_prompt(sample)


        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": prompt['system_message']},
                {"role": "user", "content": prompt['user_message']},
            ],
            stream=False
        )
        
        response_content = response.choices[0].message.content
        
        # Parse think process and formal response
        think_process = ""
        formal_response = response_content
        
        if "<think>" in response_content and "</think>" in response_content:
            parts = response_content.split("</think>")
            if len(parts) > 1:
                think_process = parts[0].replace("<think>", "").strip()
                formal_response = parts[1].strip()
        
        results["math_reasoning"].append({
            "sample": sample,
            "prompt": prompt,
            "response": {
                "think_process": think_process,
                "formal_response": formal_response,
                "full_response": response_content
            }
        })


    # 保存结果
    for template_type, template_results in results.items():
        output_file = output_dir / f"{template_type}_prompts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template_results, f, ensure_ascii=False, indent=2)
        print(f"Saved {template_type} results to {output_file}")

    # # 生成汇总报告
    # report = {
    #     "timestamp": timestamp,
    #     "total_samples": len(test_math_samples),
    #     "math_samples": len(results["math_reasoning"]),
    #     "code_samples": len(results["code_reasoning"]),
    #     "test_status": "passed"
    # }

    # report_file = output_dir / "test_report.json"
    # with open(report_file, 'w', encoding='utf-8') as f:
    #     json.dump(report, f, ensure_ascii=False, indent=2)
    # print(f"Saved test report to {report_file}")

    return results

if __name__ == "__main__":
    task = 'reasoning'
    test_prompt_generation()
    # test_generate_reasoning_batch(task)  # 保持原有的测试
    test_prompt_distillation()     # 添加新的蒸馏测试


