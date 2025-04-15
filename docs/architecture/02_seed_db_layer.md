# 种子数据库层规范

## 1. 层的目标

种子数据库层负责构建并维护高质量的基础数据集，为后续数据生成和评估提供参考标准和基准样本。

## 2. 主要职责

- 加载用户提供的原始数据
- 验证数据格式和质量
- 标准化数据结构
- 构建数据索引，支持高效检索
- 生成数据统计报告

## 3. 输入规范

### 3.1 配置参数

从用户配置输入层接收以下配置参数：

```json
{
  "seed_db": {
    "path": "data/seed",
    "format": "jsonl",
    "validation": {
      "enabled": true,
      "schema_check": true,
      "schema_path": "config/schemas/seed_schema.json",
      "min_instruction_length": 10,
      "min_response_length": 10,
      "require_domain": false
    },
    "indexing": {
      "enabled": true,
      "difficulty_buckets": 5,
      "index_path": "data/seed/index.json"
    }
  }
}
```

### 3.2 原始数据格式

支持多种格式的原始数据输入：

#### JSONL 格式（推荐）

```jsonl
{"instruction": "计算45乘以67的结果", "response": {"origin_text": "45乘以67等于3015", "rationale": "计算45乘以67，首先5×7=35，写下5，进3；4×7=28，加上进位的3得31，写下1，进3；5×6=30，加上进位的3得33，写下3，进3；4×6=24，加上进位的3得27，所以结果是3015", "final_answer": "3015"}, "metadata": {"domain": "数学", "difficulty": 0.3}}
{"instruction": "解释量子纠缠的概念", "response": {"origin_text": "量子纠缠是量子力学中的一种现象...", "rationale": "量子纠缠是量子力学中的一种现象，当两个或多个粒子以某种方式相互作用或共同产生，使得它们的量子状态无法独立描述...", "final_answer": "量子纠缠是指两个或多个粒子之间存在的一种特殊关联，使得一个粒子的状态改变会立即影响到另一个粒子，无论它们相距多远"}, "metadata": {"domain": "物理", "difficulty": 0.8}}
```

#### JSON 格式

```json
[
  {
    "instruction": "计算45乘以67的结果",
    "response": {
      "origin_text": "45乘以67等于3015",
      "rationale": "计算45乘以67，首先5×7=35，写下5，进3；4×7=28，加上进位的3得31，写下1，进3；5×6=30，加上进位的3得33，写下3，进3；4×6=24，加上进位的3得27，所以结果是3015",
      "final_answer": "3015"
    },
    "metadata": {"domain": "数学", "difficulty": 0.3}
  },
  {
    "instruction": "解释量子纠缠的概念",
    "response": {
      "origin_text": "量子纠缠是量子力学中的一种现象...",
      "rationale": "量子纠缠是量子力学中的一种现象，当两个或多个粒子以某种方式相互作用或共同产生，使得它们的量子状态无法独立描述...",
      "final_answer": "量子纠缠是指两个或多个粒子之间存在的一种特殊关联，使得一个粒子的状态改变会立即影响到另一个粒子，无论它们相距多远"
    },
    "metadata": {"domain": "物理", "difficulty": 0.8}
  }
]
```

#### CSV/TSV 格式

```csv
instruction,response,domain,difficulty
"计算45乘以67的结果","{\"origin_text\":\"45乘以67等于3015\",\"rationale\":\"计算过程...\",\"final_answer\":\"3015\"}","数学",0.3
"解释量子纠缠的概念","{\"origin_text\":\"量子纠缠是量子力学中的一种现象...\",\"rationale\":\"详细解释...\",\"final_answer\":\"量子纠缠是指两个或多个粒子之间存在的特殊关联\"}","物理",0.8
```

## 4. 输出规范

### 4.1 标准化数据格式

将原始数据转换为标准化的JSON格式：

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

### 4.2 数据统计报告

生成种子数据的统计报告：

```json
{
  "count": 1200,
  "file_count": 5,
  "domain_distribution": {
    "数学": 350,
    "物理": 280,
    "计算机科学": 420,
    "其他": 150
  },
  "difficulty": {
    "mean": 0.65,
    "min": 0.2,
    "max": 0.95
  }
}
```

## 5. 接口定义

### 5.1 种子管理器接口

```python
def process(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """加载和处理种子数据"""
    pass

def get_seed_data() -> List[Dict[str, Any]]:
    """获取加载的种子数据"""
    pass

def add_seed_data(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> int:
    """添加新数据到种子数据库"""
    pass
```

### 5.2 数据验证器接口

```python
def validate(data: Dict[str, Any]) -> bool:
    """验证单条数据记录"""
    pass

def batch_validate(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量验证数据记录"""
    pass
```

### 5.3 数据索引器接口

```python
def build_index(data: List[Dict[str, Any]]) -> None:
    """为数据构建索引"""
    pass

def get_by_domain(domain: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """按领域获取数据"""
    pass

def get_by_difficulty(min_difficulty: float = 0.0, max_difficulty: float = 1.0) -> List[Dict[str, Any]]:
    """按难度获取数据"""
    pass

def get_random_samples(count: int, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取随机样本"""
    pass
```

## 6. 实现注意事项

- 数据验证应严格检查必要字段的存在性和格式正确性
- 应为每条数据生成唯一ID（如果原始数据中没有）
- 应规范化元数据字段，确保一致性
- 索引应支持多维度检索（ID、领域、难度等）
- 应处理各种边缘情况（空文件、格式错误等）
- 应提供详细的数据统计，帮助用户了解数据分布
