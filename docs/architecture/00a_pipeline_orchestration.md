# Pipeline 编排 (Pipeline Orchestration)

Datapresso 框架的核心是一个可配置的 Pipeline，负责按顺序执行数据构建的各个阶段。本篇文档旨在阐明 Pipeline 的设计、执行流程以及配置方式。

## 核心类: `datapresso.Pipeline`

框架的主要入口点通常是 `datapresso.Pipeline` 类 (具体实现可能位于 `datapresso/main.py` 或 `datapresso/pipeline.py` 等核心模块)。

```python
from datapresso import Pipeline

# 通过配置文件初始化 Pipeline
# 配置决定了 Pipeline 将执行哪些阶段以及如何执行
pipeline = Pipeline(config_path="config/user_configs/my_config.yaml")

# 运行完整的 Pipeline
pipeline.run()
```

## 执行流程 (`pipeline.run()`)

`pipeline.run()` 方法负责按照预定义的顺序（通常由配置驱动）调用框架中的各个核心模块。一个典型的执行流程可能如下：

1.  **加载和验证配置:** 读取指定的 YAML 配置文件 (`config/*.yaml`)，并使用 `datapresso.user_input.ConfigValidator` (或类似模块) 进行结构和内容的校验。确保配置的有效性是 Pipeline 运行的第一步。
2.  **初始化环境:**
    *   设置日志记录器 (`datapresso.utils.LoggingUtils` 或类似模块)，根据配置确定日志级别和输出目标（控制台、文件）。
    *   准备必要的目录结构（如 `data/` 下的子目录）。
3.  **种子数据处理 (Seed Data Processing):**
    *   调用 `datapresso.seed_db.SeedManager` (或类似模块) 加载、验证和（可选地）索引种子数据。输入通常来自 `data/seed/` 目录。
4.  **数据生成 (Data Generation):**
    *   如果配置启用，则调用 `datapresso.data_generation.GeneratorEngine` (或类似模块)。
    *   该引擎会利用 `PromptManager` 构造提示，并通过 `LLMApiManager` 调用配置好的 LLM (OpenAI, Anthropic, Local等) 生成初始数据。
    *   可能包含初步的过滤逻辑 (`InitialFilter`)。
    *   生成的数据通常输出到 `data/generated/` 目录。
5.  **质量评估 (Quality Assessment):**
    *   如果配置启用，则调用 `datapresso.quality_assessment` 下的模块 (如 `MultiDimensionEvaluator`, `TechnicalVerifier`)。
    *   对 `data/generated/` 中的数据进行多维度打分（如相关性、流畅性、安全性、技术准确性等）。
    *   评估结果通常会附加到原始数据上，或保存到 `data/assessed/` 目录。
6.  **数据过滤 (Data Filtering):**
    *   如果配置启用，则调用 `datapresso.data_filtering` 下的模块 (如 `QualityFilter`, `DiversityAnalyzer`, `MultiDimensionFilter`)。
    *   根据质量评估分数、多样性指标或其他规则，从 `data/assessed/` (或 `data/generated/`) 中筛选出最终的数据集。
    *   筛选后的数据保存到 `data/filtered/` 或 `data/final/` 目录。
7.  **(可选) 高级评估/训练集成 (Advanced Assessment / Training Integration):**
    *   根据配置，可能触发 `datapresso.evaluation` 中的高级评估流程（例如，使用筛选出的数据微调模型并进行评测），或与 `datapresso.llamafactory` 集成。
8.  **结果汇总与报告:** Pipeline 结束时，可能会生成运行总结、关键指标统计等报告。

## 配置驱动

Pipeline 的灵活性主要来源于其配置驱动的设计。`config/*.yaml` 文件控制着：

*   **阶段启用/禁用:** 可以通过布尔标志控制是否执行某个阶段 (e.g., `pipeline.run_generation: true`, `pipeline.run_filtering: false`)。
*   **模块参数:** 每个阶段使用的具体参数，如 LLM 模型名称、API密钥、评估指标和阈值、过滤策略、采样数量等。
*   **路径配置:** 指定输入种子数据、输出中间数据和最终数据的路径。
*   **资源配置:** 如并行处理的 worker 数量等。

查阅 `config/default.yaml` 和示例配置可以了解具体的配置项。

## 数据流转

各阶段之间的数据传递主要依赖于文件系统中的 `data/` 子目录：

*   `data/seed/` -> **Seed Processing** -> (内部表示)
*   (内部表示) -> **Data Generation** -> `data/generated/` (e.g., `generated_data.jsonl`)
*   `data/generated/` -> **Quality Assessment** -> `data/assessed/` (e.g., `assessed_data.jsonl`, 可能包含评分)
*   `data/assessed/` -> **Data Filtering** -> `data/filtered/` / `data/final/` (e.g., `final_limo_data.jsonl`)

这种方式使得各阶段相对独立，便于单独调试和扩展，但也需要关注文件读写的效率。

## 错误处理与日志 (初步设计)

*   **日志:** Pipeline 在关键步骤（阶段开始/结束、处理记录数、遇到的问题等）都会输出日志。日志级别（DEBUG, INFO, WARNING, ERROR）可通过配置调整。详细的日志是追踪流程和排查问题的关键。
*   **错误处理:**
    *   **配置错误:** 无效或缺失的配置项会导致 Pipeline 在初始化阶段失败并报错退出。
    *   **阶段错误:** 默认情况下，如果某个核心阶段（如数据生成、评估）发生严重错误（如 LLM API 调用失败且无法重试、关键文件无法读取），Pipeline 会记录错误日志并终止执行（Fail Fast），避免后续阶段基于错误数据运行。
    *   **可恢复错误:** 对于某些非致命错误（如单条数据处理失败），Pipeline 可能会记录错误并继续处理剩余数据，具体行为可配置。
    *   **退出码:** Pipeline 结束时会返回一个退出码（0表示成功，非0表示失败），便于脚本调用和自动化。