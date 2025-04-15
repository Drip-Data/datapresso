# 用户配置输入层规范

## 1. 层的目标

用户配置输入层作为整个Pipeline的起点和控制中心，负责接收并管理用户的关键配置参数，指导后续流程的执行路径和策略。

## 2. 主要职责

- 接收用户配置（YAML/JSON文件或命令行参数）
- 验证配置参数的有效性和一致性
- 生成默认配置（当用户未提供时）
- 合并用户配置与默认配置
- 将配置转换为标准格式供后续层使用

## 3. 输入规范

### 3.1 配置文件格式

支持YAML或JSON格式的配置文件，包含以下核心参数：

```yaml
# 项目基本信息
project_name: "datapresso_project"
output_dir: "outputs/datapresso_project"

# 种子数据库配置
seed_db:
  path: "data/seed"
  format: "jsonl"
  validation:
    enabled: true
    schema_check: true

# 数据生成配置
data_generation:
  enabled: true
  target_count: 5000
  model: "gpt-4-turbo"
  temperature: 0.7
  batch_size: 10

# 质量评估配置
quality_assessment:
  metrics:
    - "instruction_complexity"
    - "response_quality"
    - "reasoning_depth"
    - "safety_score"
  verification_methods:
    - "code_execution"
    - "math_validation"

# 数据筛选配置
data_filtering:
  quality_threshold: 0.8
  diversity_weight: 0.3
  target_size: 1000

# 高级评估配置
advanced_assessment:
  enabled: false
  methods:
    - "ifd"
    - "limr"
    - "less"

# 训练配置
training:
  enabled: true
  model: "llama3"
  batch_size: 8
  learning_rate: 2e-5
  epochs: 3

# 评估配置
evaluation:
  benchmark_datasets:
    - "mmlu"
    - "bbh"
  metrics:
    - "accuracy"
    - "f1"

# 日志配置
logging:
  level: "INFO"
  save_path: "logs/"
  console_output: true
  file_output: true
```

### 3.2 命令行参数

支持通过命令行参数覆盖配置文件中的设置：

```bash
python -m datapresso.main --config config/my_config.yaml --project_name custom_project --data_generation.enabled false
```

## 4. 输出规范

### 4.1 标准化配置字典

将用户配置转换为标准化的Python字典，供后续层使用：

```python
{
  "project_name": "datapresso_project",
  "output_dir": "outputs/datapresso_project",
  "seed_db": {
    "path": "data/seed",
    "format": "jsonl",
    "validation": {
      "enabled": True,
      "schema_check": True
    }
  },
  # 其他配置项...
}
```

### 4.2 配置日志

生成配置处理日志，记录配置加载、验证和合并过程：

```
INFO: Loading configuration from: config/my_config.yaml
INFO: Validating configuration
INFO: Merging with default configuration
INFO: Configuration validated successfully
INFO: Saved configuration copy to: outputs/datapresso_project/configs/config_20240501_123045.json
```

## 5. 接口定义

### 5.1 配置管理器接口

```python
def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件并返回标准化配置字典"""
    pass

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证配置并设置默认值"""
    pass

def generate_default_config(output_path: str) -> None:
    """生成默认配置文件"""
    pass
```

## 6. 实现注意事项

- 配置验证应严格检查必要参数的存在性和类型正确性
- 应提供合理的默认值，减少用户配置负担
- 配置变更应记录日志，便于追踪
- 应保存配置副本，确保实验可重现性
- 应支持配置的层次化覆盖（命令行 > 配置文件 > 默认配置）
