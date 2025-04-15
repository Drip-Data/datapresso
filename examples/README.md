# Datapresso 示例

本目录包含Datapresso框架的示例代码和使用说明。

## 完整流程示例

`full_pipeline_example.py`展示了如何运行完整的Datapresso数据构建流程，包括：

1. 加载种子数据
2. 生成新数据
3. 评估数据质量
4. 筛选高质量数据
5. 生成数据统计信息

### 运行示例

1. 首先确保已安装所有依赖：

```bash
pip install -r requirements.txt
```

2. 准备配置文件（可以使用`config/example_config.yaml`作为模板）

3. 准备种子数据（放在配置文件中指定的路径）

4. 运行示例脚本：

```bash
# 使用默认配置文件
python examples/full_pipeline_example.py

# 或指定配置文件
python examples/full_pipeline_example.py --config config/my_config.yaml
```

### 配置说明

示例使用`config/example_config.yaml`作为配置模板，主要包括以下部分：

- **项目设置**：项目名称和输出目录
- **种子数据库**：种子数据路径和格式
- **LLM API**：LLM提供商配置（OpenAI、Anthropic、本地模型等）
- **数据生成**：生成参数和提示模板
- **质量评估**：评估指标和阈值
- **数据筛选**：筛选策略和目标分布
- **日志设置**：日志级别和输出位置

### 输出说明

运行示例后，将在配置的输出目录中生成以下文件：

- `generated_data.jsonl`：生成的新数据
- `assessed_data.jsonl`：评估后的数据（包含质量评分）
- `filtered_data.jsonl`：筛选后的高质量数据
- `data_stats.json`：数据统计信息
- `metrics/`：LLM API使用指标

## 其他示例

### 单独使用LLM API

```python
from datapresso.llm_api import LLMAPIManager

# 加载配置
with open("config/example_config.yaml", "r") as f:
    import yaml
    config = yaml.safe_load(f)

# 初始化LLM API管理器
llm_manager = LLMAPIManager(config["llm_api"])

# 生成文本
response = llm_manager.generate(
    prompt="解释量子计算的基本原理",
    temperature=0.7
)

print(response["text"])
```

### 单独使用质量评估

```python
from datapresso.quality_assessment.multi_dimension_evaluator import MultiDimensionEvaluator
from datapresso.utils.data_utils import DataUtils

# 加载配置
with open("config/example_config.yaml", "r") as f:
    import yaml
    config = yaml.safe_load(f)

# 加载数据
data = DataUtils.read_jsonl("data/my_data.jsonl")

# 初始化评估器
evaluator = MultiDimensionEvaluator(config["quality_assessment"])

# 评估数据
assessed_data = evaluator.process(data)

# 保存评估结果
DataUtils.write_jsonl(assessed_data, "outputs/assessed_data.jsonl")
```

### 单独使用数据筛选

```python
from datapresso.data_filtering.multi_dimension_filter import MultiDimensionFilter
from datapresso.utils.data_utils import DataUtils

# 加载配置
with open("config/example_config.yaml", "r") as f:
    import yaml
    config = yaml.safe_load(f)

# 加载已评估的数据
assessed_data = DataUtils.read_jsonl("outputs/assessed_data.jsonl")

# 初始化筛选器
filter = MultiDimensionFilter(config["data_filtering"])

# 筛选数据
filtered_data = filter.process(assessed_data)

# 保存筛选结果
DataUtils.write_jsonl(filtered_data, "outputs/filtered_data.jsonl")
```

## 自定义扩展示例

Datapresso框架设计为高度可扩展，您可以通过以下方式自定义框架行为：

### 自定义评估指标

```python
from datapresso.quality_assessment.multi_dimension_evaluator import MultiDimensionEvaluator

class CustomEvaluator(MultiDimensionEvaluator):
    def _evaluate_batch(self, batch):
        # 实现自定义评估逻辑
        # ...
        return evaluated_samples
        
    def evaluate_rationale_and_answer(self, sample):
        # 实现自定义推理过程和答案评估逻辑
        # ...
        return {"rationale_quality": score1, "answer_accuracy": score2}
```

### 自定义筛选策略

```python
from datapresso.data_filtering.multi_dimension_filter import MultiDimensionFilter

class CustomFilter(MultiDimensionFilter):
    def _balanced_selection(self, data, diversity_scores):
        # 实现自定义平衡选择算法
        # ...
        return selected_samples
```

### 自定义LLM提供商

```python
from datapresso.llm_api import LLMProvider

class CustomProvider(LLMProvider):
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        # 初始化自定义提供商
        
    def generate(self, prompt, **kwargs):
        # 实现自定义生成逻辑
        # ...
        return response
```
