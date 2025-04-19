# Datapresso LLM API Layer

## 概述 (Overview)

`datapresso.llm_api` 模块提供了一个统一、可配置的接口，用于与各种大型语言模型 (LLM) 服务进行交互。其核心目标是简化在 Datapresso 工作流（如数据生成、质量评估）中调用不同 LLM 的过程，同时提供灵活性以支持多种模型提供商和部署方式。

主要特性：

*   **统一接口**: 通过 `LLMAPIManager` 提供一致的调用方法 (`generate`, `generate_with_structured_output`, `generate_batch`)。
*   **多 Provider 支持**: 内置支持 OpenAI, Anthropic, Google Gemini, DeepSeek，以及本地 Hugging Face Transformers 模型。
*   **可扩展性**: 轻松添加新的 Provider 实现。
*   **配置驱动**: 通过项目目录下的 `llm_api_config.yaml` 文件集中管理 Provider 配置、API 密钥和默认参数。
*   **自定义端点**: 支持连接到任何 OpenAI 兼容的 API 端点（如 OpenRouter, vLLM, TGI）。
*   **本地模型**: 支持直接加载和运行本地 Hugging Face Transformers 模型。
*   **标准化输入**: 使用 `messages` 列表（包含 `system`, `user`, `assistant` 角色）作为 LLM 输入，便于控制模型行为。
*   **成本与用量追踪**: 内建基础的 API 调用指标（请求数、Token 数、成本、延迟）追踪。

## 架构 (Architecture)

1.  **`LLMProvider` (抽象基类)**: 定义了所有 LLM Provider 必须实现的通用接口，包括 `generate`, `generate_with_structured_output`, `generate_batch`, `list_available_models` 等方法。位于 `datapresso.llm_api.llm_provider`.
2.  **具体 Provider 实现**:
    *   `OpenAIProvider`: 对接 OpenAI API。
    *   `AnthropicProvider`: 对接 Anthropic API。
    *   `GeminiProvider`: 对接 Google Gemini API (需要 `pip install google-generativeai`)。
    *   `DeepSeekProvider`: 对接 DeepSeek API。
    *   `LocalProvider`: 加载并运行本地 Hugging Face Transformers 模型 (需要 `pip install transformers torch`)。
    *   `GenericOpenAIProvider`: 对接任何 OpenAI 兼容的 API 端点。
    *   *(未来可能添加 `LocalAPIProvider` 等)*
3.  **`LLMAPIManager`**: 作为用户的主要入口点。它负责：
    *   读取 `llm_api_config.yaml` 配置。
    *   根据配置初始化和管理所有可用的 Provider 实例。
    *   将用户的 API 调用路由到指定的或默认的 Provider。
    *   聚合来自不同 Provider 的指标。
    位于 `datapresso.llm_api.llm_api_manager`.
4.  **`llm_api_config.yaml` (用户配置文件)**: 用户在自己的工作目录中创建此文件，用于定义要使用的 Provider、API 密钥、模型名称、默认参数等。

## 配置 (`llm_api_config.yaml`)

这是使用 LLM API 层的关键。你需要在使用 Datapresso 相关功能的项目（或子模块）的根目录下创建一个名为 `llm_api_config.yaml` 的文件。

**模板和说明:**

请参考 `datapresso/config/templates/llm_api_config.yaml.template` 文件获取详细的模板和配置说明。以下是关键部分的摘要：

```yaml
# Default provider to use if not specified in the API call.
default_provider: openai # 必须是下面 providers 中的一个 key

# Global settings (optional defaults)
global_settings:
  max_retries: 3
  retry_delay: 1.0
  timeout: 120
  # temperature: 0.7
  # max_tokens: 2048

# Provider configurations
providers:
  # --- Official Cloud Providers ---
  openai:
    provider_type: openai          # 指定 Provider 类型 (可选, 如果 key 与类型名相同)
    api_key: env(OPENAI_API_KEY) # 强烈建议使用环境变量!
    model: gpt-4-turbo           # 此 Provider 的默认模型
    temperature: 0.7             # 此 Provider 的默认温度
    max_tokens: 4096             # 此 Provider 的默认最大 Token 数

  anthropic:
    provider_type: anthropic
    api_key: env(ANTHROPIC_API_KEY)
    model: claude-3-opus-20240229
    # ... 其他参数

  gemini:
    provider_type: gemini
    api_key: env(GOOGLE_API_KEY)
    model: gemini-1.5-pro-latest
    # ... 其他参数

  deepseek:
    provider_type: deepseek
    api_key: env(DEEPSEEK_API_KEY)
    model: deepseek-chat
    # ... 其他参数

  # --- Custom / Local Providers ---
  my_custom_openai: # 自定义名称
    provider_type: generic_openai    # 使用通用 OpenAI Provider
    api_base: http://localhost:8000/v1 # *必需*: 你的 API 端点 URL
    # api_key: sk-xxxx               # 可选: 如果需要 Key
    model: MyCustomModel-v1        # *必需*: 你的端点期望的模型名称
    temperature: 0.5

  local_model_hf: # 自定义名称
    provider_type: local             # 使用本地 Provider
    model_path: /path/to/your/model  # *必需*: 本地模型路径
    device: cuda                     # 可选: cuda 或 cpu
    # ... 其他参数
```

**关键点:**

*   **`default_provider`**: 指定未明确选择 Provider 时使用的默认 Provider 名称。
*   **`providers`**: 一个字典，其中每个键是你为 Provider 选择的唯一名称（例如 `openai`, `my_custom_openai`），值是该 Provider 的具体配置。
*   **`provider_type`**: （可选，但推荐）明确指定使用哪个 Provider 类 (`openai`, `anthropic`, `gemini`, `deepseek`, `local`, `generic_openai`)。如果省略，管理器会尝试根据键名推断。
*   **`api_key`**: **强烈建议使用 `env(YOUR_ENV_VAR_NAME)` 语法从环境变量加载 API 密钥**，以避免将敏感信息硬编码到配置文件中。
*   **`api_base`**: 对于 `generic_openai` 类型是必需的，指定你的 API 端点 URL。
*   **`model`**: 指定该 Provider 默认使用的模型名称。
*   **`model_path`**: 对于 `local` 类型是必需的，指定本地模型的路径。
*   其他参数（`temperature`, `max_tokens`, `timeout` 等）可以在 `global_settings` 或特定 Provider 配置中设置，特定配置会覆盖全局配置。

## 使用方法 (Usage)

在你的 Python 代码中（例如 Datapresso 的某个处理节点），你需要：

1.  **加载配置**: 使用 `PyYAML` 或其他方式加载工作目录下的 `llm_api_config.yaml` 文件。
2.  **处理环境变量**: 如果使用了 `env()` 语法，需要解析并替换为实际的环境变量值。
3.  **初始化 `LLMAPIManager`**: 将加载并处理后的配置字典传递给 `LLMAPIManager`。
4.  **调用 API**: 使用 `LLMAPIManager` 实例的 `generate`, `generate_with_structured_output`, 或 `generate_batch` 方法。

**示例代码:**

```python
import logging
import os
import yaml # 需要 pip install PyYAML
from pathlib import Path
from datapresso.llm_api import LLMAPIManager # 假设 __init__.py 已更新

# --- 配置加载和处理 ---
def load_llm_config(config_path: Path = Path('llm_api_config.yaml')) -> Dict[str, Any]:
    """Loads LLM API config, resolving environment variables."""
    if not config_path.is_file():
        raise FileNotFoundError(f"LLM API config file not found at: {config_path.resolve()}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve env() variables recursively
    def resolve_env_vars(cfg):
        if isinstance(cfg, dict):
            return {k: resolve_env_vars(v) for k, v in cfg.items()}
        elif isinstance(cfg, list):
            return [resolve_env_vars(i) for i in cfg]
        elif isinstance(cfg, str) and cfg.startswith('env(') and cfg.endswith(')'):
            var_name = cfg[4:-1]
            value = os.environ.get(var_name)
            if value is None:
                # Decide behavior: raise error, return None, or return the env() string?
                # Raising error is safer for required keys like api_key.
                # Returning None might be ok for optional keys.
                # For simplicity here, we might warn and return None or empty string.
                print(f"Warning: Environment variable '{var_name}' not found.")
                # Or raise ValueError(f"Required environment variable '{var_name}' not set.")
                return None # Or ""
            return value
        return cfg

    return resolve_env_vars(config)

# --- 初始化和使用 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LLM_API_Usage_Example")

    try:
        # 1. 加载配置 (假设 llm_api_config.yaml 在当前目录)
        llm_config = load_llm_config()

        # 2. 初始化 Manager
        llm_manager = LLMAPIManager(config=llm_config, logger=logger)

        # --- 3. 调用 API ---

        # 示例 1: 基本生成 (使用默认 Provider)
        messages_simple = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        response_simple = llm_manager.generate(messages=messages_simple)
        if response_simple.get("error"):
            logger.error(f"Simple generation failed: {response_simple['error']}")
        else:
            logger.info(f"Simple Response ({response_simple.get('model')}): {response_simple.get('text')}")
            logger.info(f"Usage: {response_simple.get('usage')}, Cost: ${response_simple.get('cost'):.6f}")

        # 示例 2: 指定 Provider 和参数
        messages_specific = [
            {"role": "user", "content": "Write a short poem about clouds."}
        ]
        try:
            # 假设配置中有 'anthropic' provider
            response_specific = llm_manager.generate(
                messages=messages_specific,
                provider_name="anthropic", # 指定 Provider
                temperature=0.8,          # 覆盖默认温度
                max_tokens=100            # 覆盖默认 max_tokens
            )
            if response_specific.get("error"):
                 logger.error(f"Specific generation failed: {response_specific['error']}")
            else:
                 logger.info(f"Specific Response ({response_specific.get('model')}): {response_specific.get('text')}")
        except ValueError as e:
             logger.error(f"Could not use specified provider: {e}")


        # 示例 3: 结构化输出 (使用默认 Provider)
        messages_structured = [
            {"role": "system", "content": "Extract the name and city from the text."},
            {"role": "user", "content": "John Doe lives in New York."}
        ]
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The person's full name."},
                "city": {"type": "string", "description": "The city where the person lives."}
            },
            "required": ["name", "city"]
        }
        response_structured = llm_manager.generate_with_structured_output(
            messages=messages_structured,
            output_schema=schema
        )
        if response_structured.get("error"):
            logger.error(f"Structured generation failed: {response_structured['error']}")
        else:
            logger.info(f"Structured Output ({response_structured.get('model')}): {response_structured.get('structured_output')}")


        # 示例 4: 获取可用模型
        try:
            openai_models = llm_manager.get_available_models(provider_name="openai")
            logger.info(f"Available OpenAI models: {openai_models[:5]}...") # Print first 5

            all_provider_models = llm_manager.get_available_models()
            logger.info(f"All available models by provider: {all_provider_models}")
        except ValueError as e:
             logger.error(f"Error getting models: {e}")


        # 示例 5: 获取指标
        all_metrics = llm_manager.get_all_metrics()
        logger.info(f"All Provider Metrics: {json.dumps(all_metrics, indent=2)}")

        # 保存指标
        # llm_manager.save_metrics(output_dir=Path("./llm_metrics"))


    except FileNotFoundError as e:
        logger.error(e)
    except ValueError as e:
        logger.error(f"Configuration or Initialization Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

```

**处理响应:**

`generate` 和 `generate_with_structured_output` 方法返回一个包含以下键的字典：

*   `text` (str): 生成的文本内容 (对于 `generate`)。
*   `structured_output` (dict): 解析后的结构化数据 (对于 `generate_with_structured_output`)。
*   `error` (dict | None): 如果发生错误，则包含错误信息（`message`, `type`, `details`）；否则为 `None`。
*   `model` (str): 实际使用的模型名称。
*   `latency` (float): API 调用的延迟（秒）。
*   `usage` (dict): Token 或字符使用情况（`prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_chars`, `completion_chars`, `total_chars`）。**注意**: 并非所有 Provider 都返回所有指标（特别是 Token 数）。
*   `cost` (float): 根据配置估算的 API 调用成本（美元）。
*   `raw_response` (dict): 包含来自 Provider API 的原始响应的关键信息（用于调试，内容可能因 Provider 而异）。

请务必检查 `error` 字段以确定调用是否成功。

## 添加新 Provider

要添加对新 LLM 服务（例如 `SomeNewAPIProvider`）的支持：

1.  在 `datapresso/llm_api/` 目录下创建一个新文件，例如 `some_new_api_provider.py`。
2.  在该文件中创建一个继承自 `LLMProvider` 的新类 `SomeNewAPIProvider`。
3.  实现 `__init__` 方法以处理其特定配置（API Key, URL, 模型等）。
4.  实现 `generate`, `generate_with_structured_output`, `generate_batch`, 和 `list_available_models` 方法，与目标 API 进行交互。
5.  (可选) 更新 `datapresso/llm_api/__init__.py` 以导出新类。
6.  更新 `datapresso/llm_api/llm_api_manager.py` 中的 `provider_class_map` 字典，将新的 `provider_type` 字符串映射到你的新类。
7.  更新 `llm_api_config.yaml.template` 以包含新 Provider 的配置示例。

现在，用户可以在他们的 `llm_api_config.yaml` 中配置和使用这个新的 Provider。
