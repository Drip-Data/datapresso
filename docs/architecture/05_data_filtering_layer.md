# 数据多维度筛选层规范

## 1. 层的目标

数据多维度筛选层基于质量评估结果，应用多层次筛选策略，构建满足质量与多样性平衡的精选数据集。

## 2. 主要职责

- 基于质量阈值筛选高质量样本
- 分析数据集多样性，确保领域和难度的均衡覆盖
- 平衡质量与多样性需求，优化数据集构成
- 控制最终数据集规模
- 记录筛选过程和结果
- 生成筛选统计报告

## 3. 输入规范

### 3.1 配置参数

从用户配置输入层接收以下配置参数：

```json
{
  "data_filtering": {
    "quality_threshold": 0.8,
    "diversity_weight": 0.3,
    "target_size": 1000,
    "domain_balance": true,
    "difficulty_distribution": {
      "easy": 0.2,
      "medium": 0.5,
      "hard": 0.3
    },
    "min_domain_samples": 50,
    "deduplication": {
      "enabled": true,
      "similarity_threshold": 0.85
    }
  }
}
```

### 3.2 数据格式

从数据质量评估层接收的数据格式：

```json
{
  "id": "gen_001",
  "instruction": "计算123乘以456的结果",
  "response": "123乘以456等于56088",
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

## 4. 输出规范

### 4.1 筛选后数据格式

在原数据基础上添加筛选结果：

```json
{
  "id": "gen_001",
  "instruction": "计算123乘以456的结果",
  "response": "123乘以456等于56088",
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

### 4.2 筛选统计报告

筛选过程的统计报告：

```json
{
  "filtering_stats": {
    "input_samples": 4850,
    "selected_samples": 1000,
    "rejected_samples": 3850,
    "quality_distribution": {
      "mean": 0.86,
      "median": 0.88,
      "min": 0.80,
      "max": 0.98
    },
    "domain_distribution": {
      "数学": 250,
      "物理": 200,
      "计算机科学": 300,
      "生物学": 150,
      "其他": 100
    },
    "difficulty_distribution": {
      "easy (0.0-0.3)": 200,
      "medium (0.3-0.7)": 500,
      "hard (0.7-1.0)": 300
    },
    "deduplication": {
      "duplicate_clusters_found": 120,
      "duplicates_removed": 180
    },
    "selection_criteria": {
      "quality_threshold": 0.8,
      "diversity_weight": 0.3,
      "target_size": 1000
    }
  }
}
```

## 5. 接口定义

### 5.1 平衡选择器接口

```python
def process(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """筛选数据样本"""
    pass

def _calculate_selection_score(sample: Dict[str, Any], dataset_stats: Dict[str, Any]) -> float:
    """计算样本的选择分数"""
    pass

def _save_results(data: List[Dict[str, Any]]) -> None:
    """保存筛选结果"""
    pass
```

### 5.2 质量过滤器接口

```python
def filter_by_quality(samples: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """基于质量阈值筛选样本"""
    pass

def filter_by_verification(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于验证结果筛选样本"""
    pass
```

### 5.3 多样性分析器接口

```python
def analyze_diversity(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析数据集多样性"""
    pass

def calculate_diversity_contribution(sample: Dict[str, Any], dataset_stats: Dict[str, Any]) -> float:
    """计算样本对多样性的贡献"""
    pass
```

## 6. 实现注意事项

- 筛选过程应平衡质量与多样性需求
- 应支持基于领域和难度的平衡采样
- 应实现重复检测和去重机制
- 筛选算法应可配置，支持不同的筛选策略
- 应记录详细的筛选理由，便于理解和分析
- 筛选结果应包含时间戳，便于追踪
- 应记录详细的筛选统计信息，便于分析和优化
- 筛选过程应处理各种异常情况，确保稳定性
