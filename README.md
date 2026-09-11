# GraphRAG-Neo4j Demo

一个基于 **LangChain + Neo4j + DeepSeek** 的 GraphRAG（图检索增强生成）学习示例。

流程演示了从「数据获取」到「知识图谱构建」再到「图数据库问答」的完整链路：

```
中文维基百科 ──> 文档(JSON) ──> LLM 抽取实体/关系 ──> 写入 Neo4j 图数据库 ──> GraphCypherQAChain 问答
```

## ✨ 功能特性

- 📚 **数据获取**：从中文维基百科抓取「阿里巴巴」相关文档，网络不可达时自动回退本地缓存
- 🕸️ **图谱构建**：使用 DeepSeek LLM 从文档中抽取实体（Person / Company / Organization 等）与关系（`FOUNDER_OF`、`CEO_OF` 等），并统一节点 ID 防止拆点
- 🗄️ **幂等写入**：`MERGE` + 唯一约束确保重复执行不产生重复数据，支持文档溯源（`Document -[:MENTIONS]-> 实体`）
- 💬 **图数据库问答**：`GraphCypherQAChain` 让 LLM 自动写 Cypher 查询并基于结果回答
- 🧠 **双模型分工**：Cypher 生成使用低成本快模型（deepseek-v4-flash），结果理解使用高能力模型（deepseek-v4-pro），并支持 ToolMessage 输出接入 Agent 工作流

## 📁 目录结构

```
graphrag-neo4j-demo/
├── demo/                            # 演示脚本（按执行顺序）
│   ├── get_wikipedia_data.py        # ① 获取中文维基"阿里巴巴"文档（含本地缓存回退）
│   ├── load_wikipedia_datas.py      # ② 从缓存 JSON 加载为 Document
│   ├── write_graph_database.py      # ③ LLM 抽取实体关系 + 幂等写入 Neo4j
│   └── search_graph_database.py     # ④ 图数据库 Cypher 问答
├── models/
│   └── init_chat_model_llm.py       # 共享的 DeepSeek / GLM 模型实例
├── data/
│   └── alibaba_docs.json            # 维基百科文档缓存（需提交仓库，离线可用）
├── env_utils.py                     # .env 环境变量加载
├── .env.example                     # 环境变量模板（提交）
├── requirements.txt                 # 依赖清单（提交）
├── venv.txt                         # 虚拟环境创建命令（提交）
└── .gitignore                       # Git 忽略规则（提交）
```

## 🚀 快速开始

### 1. 环境准备

使用 Python 3.13 创建虚拟环境并安装依赖：

```bash
conda create -n graphrag-neo4j python=3.13 -y
conda activate graphrag-neo4j
pip install -r requirements.txt
```

### 2. 配置环境变量

复制模板并填入你的密钥：

```bash
cp .env.example .env
```

编辑 `.env`，填入 DeepSeek / 智谱 GLM 的 API 信息：

```ini
# ---- DeepSeek ----
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com

# ---- 智谱 GLM（走 OpenAI 兼容接口）----
GLM_API_KEY=你的密钥
GLM_BASE_URL=你的地址
```

> ⚠️ `.env` 已加入 `.gitignore`，**不要提交**。

### 3. 启动 Neo4j

示例使用本地 Neo4j（`bolt://127.0.0.1:7687`，默认库 `neo4j`）。请确保 Neo4j 已启动且能通过 `neo4j/密码` 认证访问。

### 4. 按顺序运行

```bash
# ① 获取文档（网络可达则在线抓取，否则用本地缓存 data/alibaba_docs.json）
python demo/get_wikipedia_data.py

# ② 从 JSON 加载文档
python demo/load_wikipedia_datas.py

# ③ LLM 抽取实体关系并写入 Neo4j（自动建唯一约束，可重复执行）
python demo/write_graph_database.py

# ④ 图数据库问答（如：马化腾是谁？）
python demo/search_graph_database.py
```

## 💡 核心实现说明

### 实体 / 关系抽取

`write_graph_database.py` 使用 `LLMGraphTransformer` 限定可抽取的节点与关系类型，保证图谱质量：

- **允许的节点**：`Person`、`Company`、`Organization`、`Product`、`Event`、`City` 等
- **允许的关系**：`FOUNDER_OF`、`CEO_OF`、`HEADQUARTERED_IN`、`ACQUIRED`、`OWNS` 等（建议大写蛇形）
- **ID 归一化**：对实体名做 NFKC 全角→半角、空白压缩等规整，避免「阿里 巴巴」与「阿里巴巴集团」被拆成两个节点

### 幂等写入

```cypher
-- 基于实体 id 的唯一约束（LLMGraphTransformer 预留的 __Entity__ 标签）
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:`__Entity__`) REQUIRE e.id IS UNIQUE
```

配合 `baseEntityLabel=True` 与 `MERGE`，重复运行不会产生重复节点。

### 三种问答模式（见 `search_graph_database.py`）

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| 单模型标准问答 | 一个模型两次调用（写 Cypher + 理解结果），最简 | 普通对话，控制成本 |
| 单模型工具调用 | 结果封装为 ToolMessage 供下游使用 | LangGraph / Agent 链条 |
| 双模型 + 工具调用 | Cypher 用快模型、QA 用强模型，输出 ToolMessage | 生产级 / 复杂查询（当前启用） |

## ⚠️ 注意事项

- **Neo4j 凭据是硬编码**：示例脚本中直接写入了 `bolt://127.0.0.1:7687`、用户名 `neo4j` 与密码。仅限本地学习使用；如果要分享或部署，**务必**改为读取 `.env` 环境变量。
- **`.env` 严禁提交**：内含真实 API 密钥。
- **`data/` 目录需要提交**：它保存了维基百科文档缓存，网络不可达时也能离线演示完整流程。
- **网络不可达时自动回退**：`get_wikipedia_data.py` 会先探测 `zh.wikipedia.org` 连通性，失败则从 `data/alibaba_docs.json` 读取。
- **`use_function_response` 参数**：较新版本的 `langchain-neo4j` 可能已移除该参数，若报错可删除。

## 🔗 技术栈

- [LangChain](https://www.langchain.com/) / `langchain-neo4j`
- [Neo4j](https://neo4j.com/) 图数据库
- DeepSeek / 智谱 GLM LLM
- [wikipedia](https://pypi.org/project/wikipedia/) 中文词条抓取