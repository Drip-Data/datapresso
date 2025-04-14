# Datapresso 文档中心

欢迎来到 Datapresso 数据构建框架的文档中心。

Datapresso 是一个用于高效生成、评估和过滤高质量小样本数据集（LIMO 数据）的系统化框架，旨在加速和优化模型微调过程。

## 快速导航

*   **项目首页:** [README.md](../README.md) (包含安装、快速入门和故障排除)
*   **核心架构:**
    *   [架构概览](architecture/00_overview.md)
    *   [Pipeline 编排](architecture/00a_pipeline_orchestration.md)
    *   [用户输入层 (配置)](architecture/01_user_input_layer.md)
    *   [种子数据库层](architecture/02_seed_db_layer.md)
    *   [数据生成层](architecture/03_data_generation_layer.md)
    *   [质量评估层](architecture/04_quality_assessment_layer.md)
    *   [数据过滤层](architecture/05_data_filtering_layer.md)
    *   [高级评估层](architecture/06_advanced_assessment_layer.md) (下游任务评估，详见 `evaluation/README.md`)
    *   [训练集成层](architecture/07_training_integration_layer.md) (例如 LlamaFactory)
    *   [模型评估层](architecture/08_model_evaluation_layer.md) (与高级评估层相关)
    *   [错误处理与日志记录](architecture/09_error_handling_logging.md)
*   **模块说明:**
    *   [模型管理 (`models/`)](../models/README.md)
    *   [评估模块 (`evaluation/`)](../evaluation/README.md) (下游任务评估)
    *   [LLM API 接口 (`llm_api/`)](llm_api/README.md)
*   **数据格式:**
    *   [响应格式](data_format/response_format.md) (示例)
*   **使用示例:**
    *   [示例说明](../examples/README.md)

## 如何贡献

(待补充 - 可以添加关于如何报告问题、提交代码、改进文档的指南)