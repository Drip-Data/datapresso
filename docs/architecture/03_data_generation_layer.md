# 数据生成扩充层规范

## 1. 层的目标

数据生成扩充层基于种子数据，利用先进大模型蒸馏生成更多样本，扩充数据池的规模和多样性。

## 2. 主要职责

- 基于种子数据生成新的指令-响应对
- 确保生成数据的多样性和质量
- 应用初步质量筛选，剔除低质量生成结果
- 跟踪生成过程和统计信息
- 保存生成的数据供后续处理

## 3. 输入规范

### 3.1 配置参数

从用户配置输入层接收以下配置参数：

```json
{
  "data_generation": {
    "enabled": true,
    "target_count": 5000,
    "model": "gpt-4-turbo",
    "temperature": 0.7,
    "batch_size": 10,
    "max_retries": 3,
    "prompt_templates": {
      "path": "config/prompts",
      "default_template": "basic_generation"
    },
    "initial_filtering": {
      "enabled": true,
      "min_length": 50,
      "max_length": 2000,
      "banned_patterns": [
        "I'm sorry, I cannot",
        "As an AI language model"
      ]
    }
  }
}
```

### 3.2 种子数据格式

从种子数据库层接收标准化的数据格式：

```json
{
  "id": "seed_001",
  "instruction": "计算45乘以67的结果",
  "response": {
    "origin_text": "45乘以67等于3015",
    "rationale": "计算45乘以67，首先5×7=35，写下5，进3；4×7=28，加上进位的3得31，写下1，进3；5×6=30，加上进位的3得33，写下3，进3；4×6=24，加上进位的3得27，所以结果是3015",
    "final_answer": "3015"
  },
  "metadata": {
    "domain": "数学",
    "difficulty": 0.3,
    "source": "用户提供",
    "creation_timestamp": "2024-05-01T10:15:30Z"
  }
}
```

## 4. 输出规范

### 4.1 生成数据格式

生成的数据应遵循以下格式，在种子数据的基础上添加生成相关的元数据：

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

### 4.2 生成统计报告

生成过程的统计报告：

```json
{
  "generation_stats": {
    "total_generated": 5500,
    "passed_initial_filter": 5120,
    "rejected": 380,
    "generation_time": "45分钟",
    "average_time_per_sample": 0.5,
    "domain_distribution": {
      "数学": 1450,
      "物理": 1280,
      "计算机科学": 1620,
      "其他": 770
    },
    "difficulty_distribution": {
      "easy (0.0-0.3)": 1200,
      "medium (0.3-0.7)": 2800,
      "hard (0.7-1.0)": 1120
    }
  }
}
```

## 5. 接口定义

### 5.1 生成器引擎接口

```python
def process(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于种子数据生成新样本"""
    pass

def _generate_batch(seed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成一批样本"""
    pass

def _save_intermediate_results(data: List[Dict[str, Any]], batch_idx: int) -> None:
    """保存中间生成结果"""
    pass
```

### 5.2 提示模板管理器接口

```python
def create_prompt(seed_sample: Dict[str, Any]) -> str:
    """基于种子样本创建生成提示"""
    pass

def get_template(name: str) -> Optional[str]:
    """获取指定名称的模板"""
    pass

def get_template_names() -> List[str]:
    """获取所有可用模板名称"""
    pass
```

### 5.3 初步过滤器接口

```python
def filter_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤生成的样本"""
    pass

def _is_valid_sample(sample: Dict[str, Any]) -> bool:
    """检查样本是否有效"""
    pass
```

## 6. 实现注意事项

- 生成过程应支持批量处理，提高效率
- 应实现请求限速和错误重试机制，应对API限制
- 提示模板应支持多样化，针对不同领域和难度
- 初步过滤应快速剔除明显低质量的生成结果
- 应保存中间结果，支持断点续传
- 生成数据应保留与种子数据的关联，便于追踪
- 应记录详细的生成统计信息，便于分析和优化
- 生成数据量应为目标数据集规模的3-5倍，为后续质量筛选提供足够空间
