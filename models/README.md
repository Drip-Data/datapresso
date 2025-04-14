# 模型管理 (Model Management)

本目录 (`models/`) 用于存放与 Datapresso 框架运行相关的各类模型文件。清晰的模型管理有助于代码的复用和实验的可复现性。

## 目录结构建议

建议采用以下子目录结构来组织模型文件：

```
models/
├── base_llms/             # 存放用于数据生成或评估的基础大语言模型 (LLMs)
│   ├── model_provider_1/  # 例如: openai, anthropic, local_hf
│   │   └── model_name_1/  # 具体的模型名称或路径
│   │       └── ... (模型文件或配置文件)
│   └── ...
├── evaluation_models/     # 存放用于特定质量评估任务的模型 (如 NLI 模型, 毒性检测模型等)
│   ├── task_name_1/
│   │   └── model_name_1/
│   │       └── ...
│   └── ...
├── finetuned_models/      # (可选) 存放使用 Datapresso 生成的数据进行微调后得到的模型 Checkpoint
│   ├── experiment_id_1/
│   │   └── checkpoint_step/
│   │       └── ...
│   └── ...
└── README.md              # 本说明文件
```

## 使用说明

1.  **基础 LLMs (`base_llms/`)**:
    *   存放从外部下载或获取的基础 LLM 文件。
    *   对于通过 API 访问的模型（如 OpenAI, Anthropic），此目录可能只包含配置文件或标识符。
    *   对于本地模型（如 Hugging Face Hub 下载的模型），这里可以存放完整的模型权重和配置文件。
    *   代码中（例如 `datapresso/llm_api/` 或 `datapresso/data_generation/`）应通过配置指向此处的模型路径或标识符。

2.  **评估模型 (`evaluation_models/`)**:
    *   存放用于特定评估任务（如文本相似度、内容分类、事实校验等）的模型。
    *   `datapresso/quality_assessment/` 或 `datapresso/evaluation/` 模块应配置为使用这些模型。

3.  **微调模型 (`finetuned_models/`)**:
    *   如果使用 Datapresso 生成的数据进行下游模型微调（例如使用 `datapresso/llamafactory/` 或外部训练脚本），可以将训练产生的模型 Checkpoint 保存在此目录下，便于管理和后续评估。

## 注意事项

*   对于体积较大的模型文件，考虑使用 Git LFS (Large File Storage) 进行管理，或者在 `.gitignore` 文件中忽略它们，仅通过文档说明如何获取。
*   模型的配置文件（如果适用）应与模型文件放在一起，或者集中管理在 `config/` 目录下并在配置中引用。