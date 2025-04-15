# Datapresso 框架架构设计文档

本文档定义了Datapresso数据构建框架的整体架构、各层职责以及层间数据流转规范。

## 1. 整体架构

Datapresso框架采用分层架构设计，从用户配置输入到模型评估形成完整闭环。各层独立运行同时保持紧密衔接，通过标准化的JSON数据格式进行交互。

框架包含以下核心层：

1. **用户配置输入层**：接收和验证用户配置
2. **种子数据库层**：管理高质量基础数据集
3. **数据生成扩充层**：基于种子数据生成更多样本
4. **数据质量评估层**：对数据进行多维度质量评估
5. **数据多维度筛选层**：基于评估结果筛选数据
6. **高级评估与筛选层**：通过模型反馈进一步评估数据
7. **集成训练层**：使用筛选后的数据进行模型训练
8. **模型评估层**：评估训练效果

## 2. 数据流转规范

各层之间通过标准化的JSON Lines格式进行数据交换。随着数据在Pipeline中流转，元数据字段会逐步丰富，各层负责添加和更新相关信息。

### 2.1 基础数据格式

所有层间传递的数据必须遵循以下基础格式：

```json
{
  "id": "唯一标识符",
  "instruction": "指令文本",
  "response": {
    "origin_text": "原始响应文本",
    "rationale": "推理过程",
    "final_answer": "最终答案"
  },
  "metadata": {
    // 元数据字段，随着数据流转逐步丰富
  }
}
```

### 2.2 层间数据传递机制

1. **文件级传递**：各层处理的中间结果保存为JSONL文件，支持断点续传和并行处理
2. **内存级传递**：小规模数据处理时，采用Python对象直接传递，提高处理效率
3. **分布式传递**：大规模处理时，通过消息队列和分布式存储实现高效数据流转

## 3. 各层职责与数据规范

每个层都有明确的职责和输入/输出数据格式规范，详见各层的具体文档：

- [00_overview.md](00_overview.md) - 框架架构概览
- [01_user_input_layer.md](01_user_input_layer.md) - 用户配置输入层规范
- [02_seed_db_layer.md](02_seed_db_layer.md) - 种子数据库层规范
- [03_data_generation_layer.md](03_data_generation_layer.md) - 数据生成扩充层规范
- [04_quality_assessment_layer.md](04_quality_assessment_layer.md) - 数据质量评估层规范
- [05_data_filtering_layer.md](05_data_filtering_layer.md) - 数据多维度筛选层规范
- [06_advanced_assessment_layer.md](06_advanced_assessment_layer.md) - 高级评估与筛选层规范
- [07_training_integration_layer.md](07_training_integration_layer.md) - 集成训练层规范
- [08_model_evaluation_layer.md](08_model_evaluation_layer.md) - 模型评估层规范

## 4. 响应数据三部分结构

Datapresso框架的一个核心创新是将响应数据分为三个部分：

```json
"response": {
  "origin_text": "原始响应文本",
  "rationale": "推理过程",
  "final_answer": "最终答案"
}
```

这种分离设计允许分别评估推理质量和答案准确性，详细规范请参见[响应格式规范文档](../data_format/response_format.md)。

## 5. 实现注意事项

- 所有层必须保持响应字段的三部分结构
- 评估指标应分别针对推理过程和最终答案
- 数据生成时应同时生成高质量的推理过程和准确的最终答案
- 数据筛选应考虑推理质量和答案准确性的平衡
- 训练时可以尝试不同的策略，如仅使用推理过程、仅使用最终答案或同时使用两者
- 评估时应分析推理质量与答案准确性之间的关系
