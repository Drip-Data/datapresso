# 数据质量评估层规范

## 1. 层的目标

数据质量评估层对数据池中的每个样本进行多维度、系统化的质量评估，为筛选决策提供量化依据。

## 2. 主要职责

- 对数据样本进行多维度质量评估
- 执行技术验证（如代码执行、数学验证等）
- 聚合各维度评分，生成综合质量指标
- 记录评估过程和结果
- 生成评估统计报告

## 3. 输入规范

### 3.1 配置参数

从用户配置输入层接收以下配置参数：

```json
{
  "quality_assessment": {
    "metrics": [
      "instruction_complexity",
      "response_quality",
      "reasoning_depth",
      "safety_score"
    ],
    "verification_methods": [
      "code_execution",
      "math_validation"
    ],
    "llm_evaluator": {
      "model": "gpt-4-turbo",
      "batch_size": 10,
      "max_retries": 3
    },
    "thresholds": {
      "min_overall_score": 0.7,
      "min_safety_score": 0.9
    }
  }
}
```

### 3.2 数据格式

从数据生成扩充层接收的数据格式：

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
    }
  }
}
```

## 4. 输出规范

### 4.1 评估后数据格式

在原数据基础上添加评估结果：

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
    }
  }
}
```

### 4.2 评估统计报告

评估过程的统计报告：

```json
{
  "assessment_stats": {
    "total_assessed": 5120,
    "passed_verification": 4850,
    "failed_verification": 270,
    "score_distribution": {
      "instruction_complexity": {
        "mean": 0.72,
        "median": 0.75,
        "std": 0.15
      },
      "response_quality": {
        "mean": 0.85,
        "median": 0.88,
        "std": 0.12
      },
      "reasoning_depth": {
        "mean": 0.68,
        "median": 0.70,
        "std": 0.18
      },
      "safety_score": {
        "mean": 0.95,
        "median": 0.98,
        "std": 0.05
      },
      "overall_score": {
        "mean": 0.80,
        "median": 0.82,
        "std": 0.14
      }
    },
    "domain_performance": {
      "数学": {
        "mean_score": 0.85,
        "verification_success_rate": 0.96
      },
      "物理": {
        "mean_score": 0.82,
        "verification_success_rate": 0.92
      },
      "计算机科学": {
        "mean_score": 0.78,
        "verification_success_rate": 0.88
      }
    }
  }
}
```

## 5. 接口定义

### 5.1 多维度评估器接口

```python
def process(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """评估数据样本的质量"""
    pass

def _evaluate_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """评估一批样本"""
    pass

def _save_intermediate_results(data: List[Dict[str, Any]], batch_idx: int) -> None:
    """保存中间评估结果"""
    pass
```

### 5.2 技术验证器接口

```python
def verify_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """验证样本的技术正确性"""
    pass

def _verify_code(instruction: str, response: Dict[str, str]) -> Dict[str, Any]:
    """验证代码执行结果"""
    pass

def _verify_math(instruction: str, response: Dict[str, str]) -> Dict[str, Any]:
    """验证数学计算结果"""
    pass
```

### 5.3 评分聚合器接口

```python
def aggregate_scores(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """聚合各维度评分"""
    pass

def _calculate_overall_score(scores: Dict[str, float]) -> float:
    """计算综合评分"""
    pass
```

## 6. 实现注意事项

- 评估过程应支持批量处理，提高效率
- 应实现请求限速和错误重试机制，应对API限制
- 技术验证应在安全沙箱中执行，防止恶意代码
- 评估指标应具有可解释性，便于理解和分析
- 应保存中间结果，支持断点续传
- 评估结果应包含时间戳和评估模型信息，便于追踪
- 应记录详细的评估统计信息，便于分析和优化
- 评估过程应处理各种异常情况，确保稳定性
