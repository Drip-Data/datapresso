import pytest
from prompt_manager import GenerationPromptManager

def test_prompt_generation():
    """Test and display prompt generation for both standard and reasoning templates"""
    
    # Test data with rationale (for reasoning templates)
    reasoning_seed = {
        "question": "计算1+1的值",
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
        "question": "什么是Python?",
        "response": {
            "origin_text": "Python是一种高级编程语言",
        },
        "metadata": {
            "domain": "programming"
        }
    }

    # Test standard template
    print("\n=== Standard Template Prompt ===")
    standard_config = {"template_type": "standard","template_name":"domain_specific"}
    standard_manager = GenerationPromptManager(standard_config)
    standard_prompt = standard_manager.create_prompt(standard_seed)
    print("System:", standard_prompt["system_message"])
    print("User:", standard_prompt["user_message"])

    # Test math reasoning template
    print("\n=== Math Reasoning Template Prompt ===")
    math_config = {"template_type": "reasoning", "template_name": "math_generation"}
    math_manager = GenerationPromptManager(math_config)
    math_prompt = math_manager.create_prompt(reasoning_seed)
    print("System:", math_prompt["system_message"])
    print("User:", math_prompt["user_message"])


if __name__ == "__main__":
    test_prompt_generation()