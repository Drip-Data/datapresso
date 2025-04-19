# Datapresso LLM API Layer

## 概述 (Overview)

本模块 (`datapresso.llm_api`) 提供了一个统一且高度封装的接口，用于在 Datapresso 的不同工作流阶段（如数据生成、质量评估）中调用大型语言模型 (LLM)。

**核心设计理念:**

*   **阶段特定 API:** 为每个 Datapresso 阶段（如 `data_generator`, `quality_assessor`）提供预配置好的、独立的 API 对象（如 `data_generator_api`, `quality_assessor_api`），用户可以直接导入并使用。
*   **中心化配置:** 所有阶段的 LLM 配置（Provider、API Keys、模型、模板等）都集中存储在 `datapresso/datapresso/llm_api/configs/` 目录下的 YAML 文件中，便于管理。
*   **模板化:** 支持在配置文件中预定义系统提示（System Prompt）和结构化输出模式（Output Schema）模板，调用时通过名称引用，简化代码。
*   **易用性:** 用户只需导入对应阶段的 API 对象，调用封装好的 `generate` 或 `generate_with_structured_output` 方法，无需关心底层的 `LLMAPIManager` 或配置加载细节。
*   **可扩展性:** 提供简单的机制来注册新的 Datapresso 阶段，并支持添加新的 LLM Provider 实现。
*   **当前支持的 Provider 类型:** `openai`, `anthropic`, `gemini`, `deepseek`, `local` (Hugging Face Transformers), `generic_openai` (用于兼容 OpenAI 的自定义端点)。

## 架构 (Architecture)

1.  **`configs/` 目录:** 存放每个阶段的 YAML 配置文件（如 `data_generator.yaml`）。
2.  **`templates/` 目录:** 存放默认的配置文件模板 (`llm_api_config.yaml.template`)。
3.  **`config_loader.py`:** 提供 `load_llm_config` 函数，用于加载 YAML 并解析环境变量。
4.  **`llm_provider.py` (`LLMProvider`):** 定义所有 Provider 的抽象基类。
5.  **Provider 实现:** 对接具体 LLM 服务的类。
    *   `openai_provider.py` (类型: `openai`)
    *   `anthropic_provider.py` (类型: `anthropic`)
    *   `gemini_provider.py` (类型: `gemini`, 需要 `google-generativeai`)
    *   `deepseek_provider.py` (类型: `deepseek`)
    *   `local_provider.py` (类型: `local`, 需要 `transformers`, `torch`)
    *   `generic_openai_provider.py` (类型: `generic_openai`)
6.  **`llm_api_manager.py` (`LLMAPIManager`):** 负责管理单个配置对应的 Provider 实例和路由 API 调用。
7.  **`stage_api.py` (`StageLLMApi`):** 核心封装类。每个实例对应一个 Datapresso 阶段，加载该阶段的配置，持有专属的 `LLMAPIManager`，缓存模板，并提供简化的 `generate` 和 `generate_with_structured_output` 方法。
8.  **`__init__.py`:** 模块入口。负责：
    *   定义已知阶段 (`KNOWN_STAGES`)。
    *   检查并（如果需要）创建默认配置文件。
    *   为已知阶段预先实例化 `StageLLMApi` 对象。
    *   将这些实例暴露为模块级变量（如 `data_generator_api`）。
    *   提供 `register_llm_stage` 函数。

## 配置管理

所有 LLM API 的配置都集中在 `datapresso/datapresso/llm_api/configs/` 目录下。

1.  **阶段配置文件:**
    *   每个需要使用 LLM API 的 Datapresso 阶段都应在此目录下有一个对应的 YAML 配置文件。
    *   文件名应与阶段的标识符（通常是小写类名或模块名）匹配，例如 `data_generator.yaml`, `quality_assessor.yaml`。
    *   **首次使用或注册新阶段时，如果对应的 YAML 文件不存在，系统会自动从 `templates/llm_api_config.yaml.template` 复制一份默认配置。**
2.  **编辑配置:**
    *   **用户需要手动编辑这些生成的 `{stage_name}.yaml` 文件**，以填入正确的 API 密钥（强烈建议使用环境变量）、选择合适的模型、调整默认参数，并定义该阶段所需的系统提示和输出模式模板。
    *   配置文件的结构与 `templates/llm_api_config.yaml.template` 一致。
3.  **环境变量:**
    *   在 YAML 文件中，使用 `env(YOUR_ENV_VAR_NAME)` 语法来引用环境变量，例如 `api_key: env(OPENAI_API_KEY)`。这是存放 API 密钥等敏感信息的推荐方式。确保运行 Datapresso 的环境中设置了相应的环境变量。
4.  **模板 (`system_prompt_templates` & `output_schema_templates`):**
    *   在每个阶段的 YAML 文件中，可以定义该阶段常用的系统提示和 JSON Schema。
    *   **`system_prompt_templates`**: 一个字典，键是模板名称，值是提示字符串。
    *   **`output_schema_templates`**: 一个字典，键是模板名称，值是 JSON Schema 字典。
    *   在调用阶段 API 的 `generate` 或 `generate_with_structured_output` 方法时，可以通过模板名称来引用这些预定义的模板。

**示例 `configs/data_generator.yaml`:**

```yaml
# datapresso/datapresso/llm_api/configs/data_generator.yaml
# (用户需要编辑此文件, 特别是 API Key)

default_provider: openai

providers:
  openai:
    provider_type: openai
    api_key: env(OPENAI_API_KEY) # !!! 需要设置环境变量 !!!
    model: gpt-4-turbo
    temperature: 0.8 # Data Gen 可能需要更高温度

  # 可以添加其他 provider...

system_prompt_templates:
  default: "You are a creative data generation assistant."
  instruction_follower: "Generate data strictly following the provided instructions and format."
  persona_generator: "Create a detailed character profile based on the given keywords."

output_schema_templates:
  person_details:
    type: object
    properties:
      # ... (schema 定义) ...
    required: [name]
  # ... 其他 data_generator 阶段需要的 schema ...
```

## 使用方法

用户可以直接从 `datapresso.llm_api` 导入为特定阶段预配置好的 API 对象进行调用。

```python
# 简化示例，仅展示核心调用，省略错误检查和日志记录以求清晰
# 实际使用中，务必检查 is_available() 和返回结果中的 'error' 字段！

from datapresso.llm_api import data_generator_api, quality_assessor_api

# --- 使用 Data Generator API ---

# 1. 简单生成 (使用 data_generator.yaml 配置的默认 provider 和模板)
# !! 实际代码需要检查 data_generator_api 是否成功加载且可用 !!
response_gen_simple = data_generator_api.generate(
    user_prompt="Generate a list of 3 fantasy character names.",
    system_prompt_template="instruction_follower", # 引用模板
    # provider_name="openai", # 可选：覆盖默认 provider
    # temperature=0.9         # 可选：覆盖默认 temperature
)
# !! 实际代码需要检查 response_gen_simple['error'] !!
print("Generated Names:", response_gen_simple.get('text'))
print(f"(Model: {response_gen_simple.get('model')}, Cost: ${response_gen_simple.get('cost', 0.0):.6f})")


# 2. 结构化输出 (使用 data_generator.yaml 配置的模板)
# !! 实际代码需要检查 data_generator_api 是否成功加载且可用 !!
response_gen_structured = data_generator_api.generate_with_structured_output(
    user_prompt="Generate profile for a wizard named Elara.",
    output_schema_template="person_details",    # 引用 schema 模板
    system_prompt_template="persona_generator" # 引用 system prompt 模板
)
# !! 实际代码需要检查 response_gen_structured['error'] !!
print("Generated Profile:", response_gen_structured.get('structured_output'))
print(f"(Model: {response_gen_structured.get('model')}, Cost: ${response_gen_structured.get('cost', 0.0):.6f})")


# --- 使用 Quality Assessor API ---

# 3. 结构化输出，并覆盖 provider 和 temperature
# !! 实际代码需要检查 quality_assessor_api 是否成功加载且可用 !!
text_to_assess = "The quick brown fox jumps over the lazy dog."
response_qa_structured = quality_assessor_api.generate_with_structured_output(
    user_prompt=f"Assess the quality of this text: '{text_to_assess}'",
    output_schema_template="quality_score",     # 引用 schema 模板
    system_prompt_template="quality_assessor",  # 引用 system prompt 模板
    provider_name="gemini",                     # 覆盖默认 provider (需在 quality_assessor.yaml 中配置 gemini)
    temperature=0.1                             # 覆盖默认 temperature
)
# !! 实际代码需要检查 response_qa_structured['error'] !!
print("Assessment Score:", response_qa_structured.get('structured_output'))
print(f"(Model: {response_qa_structured.get('model')}, Cost: ${response_qa_structured.get('cost', 0.0):.6f})")

```

**调用参数说明:**

*   `generate(user_prompt, system_prompt_template=None, system_prompt_override=None, provider_name=None, **kwargs)`
*   `generate_with_structured_output(user_prompt, output_schema_template, output_schema_override=None, system_prompt_template=None, system_prompt_override=None, provider_name=None, **kwargs)`

    *   `user_prompt` (str): 必需，用户的主要输入。
    *   `system_prompt_template` (str, optional): 在阶段配置文件中定义的系统提示模板名称。
    *   `system_prompt_override` (str, optional): 直接提供系统提示字符串，覆盖模板。
    *   `output_schema_template` (str): (仅限结构化输出) 必需，在阶段配置文件中定义的输出 Schema 模板名称。
    *   `output_schema_override` (dict, optional): (仅限结构化输出) 直接提供 JSON Schema 字典，覆盖模板。
    *   `provider_name` (str, optional): 指定使用哪个 Provider（必须在阶段配置文件 `{stage_name}.yaml` 中定义）。
        *   **默认行为:** 如果省略，则使用该阶段配置文件中指定的 `default_provider`。
    *   `**kwargs`: 其他传递给底层 LLM Provider 的参数（如 `temperature`, `max_tokens` 等）。
        *   **优先级:** 调用时在 `**kwargs` 中指定的参数 > 该阶段配置文件中为该 Provider 设置的参数 > Provider 类内部的默认值。
        *   **示例:** 如果调用时指定 `temperature=0.9`，则使用 0.9；如果未指定，则查找 `{stage_name}.yaml` 中 `providers.<provider_name>.temperature`；如果仍未找到，则使用 Provider（如 `OpenAIProvider`）内部设定的默认温度。
        *   常用 `**kwargs` 参数包括：
        *   `temperature` (float): 控制生成文本的随机性。
        *   `max_tokens` (int): 控制生成文本的最大长度 (或 `max_output_tokens` for Gemini)。
        *   `top_p` (float): 控制核心采样的概率阈值。
        *   `top_k` (int): 控制 Top-K 采样的数量。
        *   `stop_sequences` (List[str]): 指定停止生成的字符串序列。
        *   `presence_penalty` (float, OpenAI/Compatible): 控制新主题的引入。
        *   `frequency_penalty` (float, OpenAI/Compatible): 控制重复内容的生成。
        *   `function_description` (str, for structured output): 函数调用的描述 (如果 Provider 支持)。
        *   `stream` (bool, OpenAI/Compatible): 是否流式返回结果 (当前封装不支持流式处理)。
        *   *(具体可用参数取决于所选的 Provider)*

**返回值:**

两个 `generate` 方法都返回一个字典，包含 `text` 或 `structured_output`, `error`, `model`, `latency`, `usage`, `cost`, `raw_response` 等键，详见上文示例或 `StageLLMApi` 代码。**务必检查 `error` 字段！**

## 注册新阶段

如果你创建了一个新的 Datapresso 阶段（例如 `MyNewAnalyzer`）并且需要使用 LLM API：

1.  **选择一个阶段标识符:** 通常是小写的类名或模块名，例如 `"my_new_analyzer"`。
2.  **调用注册函数:** 在你的新阶段模块加载时（例如，在模块的顶层或 `__init__.py` 中），调用 `register_llm_stage`：
    ```python
    from datapresso.llm_api import register_llm_stage

    STAGE_NAME = "my_new_analyzer"
    registration_successful = register_llm_stage(STAGE_NAME)

    if not registration_successful:
        # 处理注册失败的情况，可能 LLM 功能无法使用
        print(f"Warning: Failed to register LLM stage '{STAGE_NAME}'. Check logs.")
    ```
    这将确保 `datapresso/datapresso/llm_api/configs/my_new_analyzer.yaml` 文件存在（如果不存在则从模板创建），并且 `my_new_analyzer_api` 对象会被创建并可供导入。
3.  **配置:** 提醒用户去编辑新生成的 `configs/my_new_analyzer.yaml` 文件。
4.  **使用:** 在你的代码中，可以直接导入并使用新的 API 对象：
    ```python
    from datapresso.llm_api import my_new_analyzer_api

    if my_new_analyzer_api and my_new_analyzer_api.is_available():
        response = my_new_analyzer_api.generate(...)
    ```

## 添加新 Provider

请参考 `llm_provider.py` 中的 `LLMProvider` 抽象类和现有 Provider 的实现（如 `openai_provider.py`）。主要步骤：

1.  创建新的 Provider 类，继承 `LLMProvider`。
2.  实现 `__init__`, `generate`, `generate_with_structured_output`, `generate_batch`, `list_available_models` 方法。
3.  在 `llm_api_manager.py` 的 `_initialize_providers` 方法中的 `provider_class_map` 字典里添加新的 `provider_type` 映射。
4.  更新 `templates/llm_api_config.yaml.template` 和本文档，包含新 Provider 的配置示例。