# 高级评估与筛选层规范

## 1. 层的目标

高级评估与筛选层通过模型反馈和学习效果评估，进一步筛选和优化数据集，确保数据对模型训练的有效性。

## 2. 主要职责

- 实现高级评估方法（如IFD、LIMR、LESS等）
- 通过小规模模型训练评估数据样本的学习效果
- 基于评估结果进行二次筛选
- 优化最终数据集的组成
- 记录评估和筛选过程
- 生成评估统计报告

## 3. 输入规范

### 3.1 配置参数

从用户配置输入层接收以下配置参数：

```json
{
  "advanced_assessment": {
    "enabled": true,
    "methods": [
      "ifd",
      "limr",
      "less"
    ],
    "model": "llama3-7b",
    "batch_size": 4,
    "learning_samples": 100,
    "evaluation_steps": 50,
    "learning_rate": 2e-5,
    "max_samples_per_method": 2000,
    "final_selection": {
      "strategy": "ensemble",
      "weights": {
        "ifd": 0.4,
        "limr": 0.3,
        "less": 0.3
      },
      "target_size": 800
    }
  }
}
```

### 3.2 数据格式

从数据多维度筛选层接收的数据格式：

```json
{
  "id": "gen_001",
  "instruction": "计算123乘以456的结果",
  "response": {
    "origin_text": "123乘以456等于56088",
    "rationale": "计算123乘以456，我们可以分步进行。首先，3×6=18，写下8，进1；2×6=12，加上进位的1得13，写下3，进1；1×6=6，加上进位的1得7，写下7。然后，3×5=15，写下5，进1；2×5=10，加上进位的1得11，写下1，进1；1×5=5，加上进位的1得6，写下6。最后，3×4=12，写下2，进1；2×4=8，加上进位的1得9，写下9；1×4=4，加上进位的1得5，写下5。所以结果是56088。",
    "final_answer": "56088"
  },
  "metadata": {
    "domain": "数学",
    "difficulty": 0.35,
    "source": "gpt-4-turbo生成",
    "seed_id": "seed_001",
    "creation_timestamp": "2024-05-01T11:20:45Z",
    "generation": {
      "model": "gpt-4-turbo",
      "temperature": 0.7,
      "prompt_template": "basic_generation",
      "generation_time": 2.5,
      "initial_quality": 0.85
    },
    "evaluations": {
      "instruction_complexity": 0.75,
      "response_quality": 0.92,
      "reasoning_depth": 0.65,
      "safety_score": 0.98,
      "overall_score": 0.83,
      "verification": {
        "verified": true,
        "method": "math_validation",
        "results": {
          "expected_answer": 56088,
          "actual_answer": 56088,
          "is_correct": true
        }
      },
      "evaluation_timestamp": "2024-05-01T12:30:15Z",
      "evaluator_model": "gpt-4-turbo"
    },
    "filtering": {
      "selected": true,
      "reason": "高质量且增加数学领域多样性",
      "selection_score": 0.88,
      "quality_contribution": 0.83,
      "diversity_contribution": 0.65,
      "filtering_timestamp": "2024-05-01T13:45:20Z"
    }
  }
}
```

## 4. 输出规范

### 4.1 高级评估后数据格式

在原数据基础上添加高级评估结果：

```json
{
  "id": "gen_001",
  "instruction": "计算123乘以456的结果",
  "response": {
    "origin_text": "123乘以456等于56088",
    "rationale": "计算123乘以456，我们可以分步进行。首先，3×6=18，写下8，进1；2×6=12，加上进位的1得13，写下3，进1；1×6=6，加上进位的1得7，写下7。然后，3×5=15，写下5，进1；2×5=10，加上进位的1得11，写下1，进1；1×5=5，加上进位的1得6，写下6。最后，3×4=12，写下2，进1；2×4=8，加上进位的1得9，写下9；1×4=4，加上进位的1得5，写下5。所以结果是56088。",
    "final_answer": "56088"
  },
  "metadata": {
    "domain": "数学",
    "difficulty": 0.35,
    "source": "gpt-4-turbo生成",
    "seed_id": "seed_001",
    "creation_timestamp": "2024-05-01T11:20:45Z",
    "generation": {
      "model": "gpt-4-turbo",
      "temperature": 0.7,
      "prompt_template": "basic_generation",
      "generation_time": 2.5,
      "initial_quality": 0.85
    },
    "evaluations": {
      "instruction_complexity": 0.75,
      "response_quality": 0.92,
      "reasoning_depth": 0.65,
      "safety_score": 0.98,
      "overall_score": 0.83,
      "verification": {
        "verified": true,
        "method": "math_validation",
        "results": {
          "expected_answer": 56088,
          "actual_answer": 56088,
          "is_correct": true
        }
      },
      "evaluation_timestamp": "2024-05-01T12:30:15Z",
      "evaluator_model": "gpt-4-turbo"
    },
    "filtering": {
      "selected": true,
      "reason": "高质量且增加数学领域多样性",
      "selection_score": 0.88,
      "quality_contribution": 0.83,
      "diversity_contribution": 0.65,
      "filtering_timestamp": "2024-05-01T13:45:20Z"
    },
    "advanced_assessment": {
      "ifd_score": 0.76,
      "limr_score": 0.82,
      "less_score": 0.79,
      "ensemble_score": 0.78,
      "final_selected": true,
      "rationale_quality": 0.85,
      "answer_quality": 0.92,
      "learning_effect": {
        "loss_reduction": 0.23,
        "accuracy_improvement": 0.18
      },
      "assessment_timestamp": "2024-05-01T15:20:30Z",
      "assessment_model": "llama3-7b"
    }
  }
}
```

### 4.2 高级评估统计报告

高级评估过程的统计报告：

```json
{
  "advanced_assessment_stats": {
    "input_samples": 1000,
    "assessed_samples": 1000,
    "final_selected": 800,
    "method_scores": {
      "ifd": {
        "mean": 0.72,
        "median": 0.75,
        "std": 0.15
      },
      "limr": {
        "mean": 0.78,
        "median": 0.80,
        "std": 0.12
      },
      "less": {
        "mean": 0.76,
        "median": 0.77,
        "std": 0.14
      },
      "ensemble": {
        "mean": 0.75,
        "median": 0.76,
        "std": 0.13
      }
    },
    "learning_effect": {
      "mean_loss_reduction": 0.21,
      "mean_accuracy_improvement": 0.15
    },
    "domain_performance": {
      "数学": {
        "mean_score": 0.82,
        "selected_ratio": 0.85
      },
      "物理": {
        "mean_score": 0.78,
        "selected_ratio": 0.80
      },
      "计算机科学": {
        "mean_score": 0.76,
        "selected_ratio": 0.75
      }
    },
    "rationale_vs_answer_quality": {
      "rationale_mean": 0.81,
      "answer_mean": 0.88,
      "correlation": 0.72
    }
  }
}
```

## 5. 接口定义

### 5.1 IFD计算器接口

```python
def calculate_ifd(samples: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """计算样本的IFD（Instruction-Following Difficulty）分数"""
    pass

def _train_and_evaluate(samples: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """训练模型并评估学习效果"""
    pass
```

### 5.2 学习评估器接口

```python
def evaluate_learning_effect(samples: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """评估样本的学习效果"""
    pass

def _calculate_loss_reduction(sample: Dict[str, Any], model_before: Any, model_after: Any) -> float:
    """计算样本的损失减少量"""
    pass
```

### 5.3 二次筛选器接口

```python
def filter_samples(samples: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
    """基于高级评估结果筛选样本"""
    pass

def _ensemble_scores(sample: Dict[str, Any], weights: Dict[str, float]) -> float:
    """综合多种评估方法的分数"""
    pass
```

## 6. 实现注意事项

- 高级评估方法应考虑计算资源需求，支持分批处理
- 应实现模型训练和评估的自动化流程
- 应支持多种评估方法的组合使用
- 应分别评估推理过程（rationale）和最终答案（final_answer）的质量
- 应记录详细的评估过程和结果，便于分析和优化
- 应支持断点续传，避免长时间评估过程中断导致的数据丢失
- 应实现结果可视化，便于理解和分析
- 应考虑评估方法的可扩展性，支持添加新的评估方法
