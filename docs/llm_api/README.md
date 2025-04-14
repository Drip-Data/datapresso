# Datapresso LLM API 接口层

Datapresso LLM API 接口层提供了一个统一的接口，用于与各种大型语言模型(LLM)服务提供商进行交互。这个接口层是Datapresso框架的基础组件，主要用于数据生成、评估和筛选等核心任务。

## 主要功能

- **统一调用格式**：标准化API请求和响应格式，简化与不同LLM提供商的交互
- **多供应商支持**：支持主流LLM API（如OpenAI、Anthropic）及本地部署模型
- **结构化输出**：支持生成符合指定JSON模式的结构化输出
- **批量处理**：支持批量处理多个提示
- **指标跟踪**：记录使用情况、成本和性能指标
- **错误处理**：提供统一的错误处理和重试机制

## 架构设计

LLM API 接口层采用了模块化的设计，主要包含以下组件：

1. **LLMProvider**：抽象基类，定义了所有提供商必须实现的接口
2. **具体提供商实现**：如OpenAIProvider、AnthropicProvider和LocalProvider
3. **LLMAPIManager**：统一管理多个提供商的接口，提供简单的访问方式

## 使用方法

### 基本配置

首先，在配置文件中设置LLM API的配置：

```yaml
llm_api:
  default_provider: "openai"
  providers:
    openai:
      api_key: "your-openai-api-key"  # 或使用环境变量OPENAI_API_KEY
      model: "gpt-4-turbo"
      temperature: 0.7
      max_tokens: 1024
    anthropic:
      api_key: "your-anthropic-api-key"  # 或使用环境变量ANTHROPIC_API_KEY
      model: "claude-3-opus-20240229"
    local:
      model_path: "models/llama3-8b"
      device: "cuda"  # 或"cpu"
```

### 初始化API管理器

```python
from datapresso.llm_api import LLMAPIManager

# 从配置初始化
llm_manager = LLMAPIManager(config["llm_api"])
```

### 生成文本

```python
# 使用默认提供商生成文本
response = llm_manager.generate(
    prompt="解释量子计算的基本原理",
    temperature=0.5
)

# 获取生成的文本
generated_text = response["text"]

# 查看使用情况
usage = response["usage"]
print(f"使用了 {usage['total_tokens']} 个token")
```

### 使用特定提供商

```python
# 使用Anthropic的Claude模型
claude_response = llm_manager.generate(
    prompt="解释量子计算的基本原理",
    provider_name="anthropic",
    temperature=0.5
)
```

### 生成结构化输出

```python
# 定义输出模式
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        },
        "difficulty_level": {
            "type": "string",
            "enum": ["beginner", "intermediate", "advanced"]
        }
    },
    "required": ["summary", "key_points", "difficulty_level"]
}

# 生成结构化输出
structured_response = llm_manager.generate_with_structured_output(
    prompt="解释量子计算的基本原理",
    output_schema=schema
)

# 获取结构化输出
structured_data = structured_response["structured_output"]
print(f"难度级别: {structured_data['difficulty_level']}")
print(f"要点数量: {len(structured_data['key_points'])}")
```

### 批量处理

```python
prompts = [
    "解释量子计算的基本原理",
    "描述机器学习中的过拟合问题",
    "简述区块链技术的工作原理"
]

# 批量生成响应
batch_responses = llm_manager.generate_batch(prompts)

# 处理每个响应
for i, response in enumerate(batch_responses):
    print(f"提示 {i+1} 的响应: {response['text'][:100]}...")
```

### 获取使用指标

```python
# 获取默认提供商的指标
metrics = llm_manager.get_metrics()
print(f"总请求数: {metrics['total_requests']}")
print(f"总token数: {metrics['total_tokens']}")
print(f"总成本: ${metrics['total_cost']:.4f}")

# 获取所有提供商的指标
all_metrics = llm_manager.get_all_metrics()
for provider, metrics in all_metrics.items():
    print(f"{provider} 总成本: ${metrics['total_cost']:.4f}")

# 保存指标到文件
llm_manager.save_metrics("outputs/metrics")
```

## 在Datapresso框架中的应用

LLM API接口层被框架中的多个组件使用：

### 数据生成扩充层

```python
from datapresso.llm_api import LLMAPIManager

class DataGenerator:
    def __init__(self, config):
        self.llm_manager = LLMAPIManager(config["llm_api"])
        
    def generate_sample(self, seed_data):
        prompt = self._create_prompt(seed_data)
        response = self.llm_manager.generate(prompt)
        return self._process_response(response)
```

### 数据质量评估层

```python
class QualityEvaluator:
    def __init__(self, config):
        self.llm_manager = LLMAPIManager(config["llm_api"])
        
    def evaluate_sample(self, sample):
        prompt = self._create_evaluation_prompt(sample)
        schema = {
            "type": "object",
            "properties": {
                "reasoning_depth": {"type": "number", "minimum": 0, "maximum": 1},
                "response_quality": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"}
            }
        }
        response = self.llm_manager.generate_with_structured_output(prompt, schema)
        return response["structured_output"]
```

## 扩展支持新的LLM提供商

要添加新的LLM提供商支持，只需继承`LLMProvider`基类并实现必要的方法：

```python
from datapresso.llm_api import LLMProvider

class NewProvider(LLMProvider):
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        # 初始化提供商特定的配置
        
    def generate(self, prompt, **kwargs):
        # 实现生成方法
        
    def generate_with_structured_output(self, prompt, output_schema, **kwargs):
        # 实现结构化输出生成方法
        
    def generate_batch(self, prompts, **kwargs):
        # 实现批量生成方法
```

然后在`LLMAPIManager`中注册新的提供商：

```python
# 在llm_api_manager.py的_initialize_providers方法中添加
if "new_provider" in provider_configs:
    try:
        self.providers["new_provider"] = NewProvider(provider_configs["new_provider"], self.logger)
        self.logger.info("Initialized new provider")
    except Exception as e:
        self.logger.error(f"Failed to initialize new provider: {str(e)}")
```

## 最佳实践

1. **API密钥管理**：优先使用环境变量存储API密钥，而不是硬编码在配置文件中
2. **错误处理**：始终检查响应中的错误字段，处理可能的API错误
3. **成本控制**：监控使用指标，设置合理的token限制，避免意外的高成本
4. **缓存策略**：对于重复的请求，考虑实现缓存机制
5. **降级策略**：当首选提供商不可用时，实现自动降级到备用提供商
6. **批量处理**：尽可能使用批量处理来减少API调用次数
7. **结构化输出**：对于需要结构化数据的场景，使用`generate_with_structured_output`而不是手动解析文本
