# Memory Agent

自带向量记忆系统的 CLI Agent。每次对话从干净上下文开始，通过 LLM 决策自动检索相关历史记忆，注入 System Prompt 后执行任务；对话结束后自动提取关键信息，经用户审核后存入向量数据库。

## 快速开始

### 1. 安装

```bash
pip install -e .
```

依赖：Python 3.10+，其余由 pip 自动安装（chromadb、httpx、pyyaml、openai）。

### 2. 配置环境变量

```bash
export DEEPSEEK_API_KEY=sk-your-deepseek-key    # LLM（对话 + 决策 + 提取）
export SF_API_KEY=your-siliconflow-key           # Embedding（向量检索）
```

### 3. 首次运行

```bash
memory-agent "帮我写一个 Python hello world 脚本"
```

首次运行时会在当前目录自动创建 `.agent-memory/` 目录，包含默认配置文件和数据存储。

### 4. 继续之前的工作

```bash
memory-agent "继续上次的脚本优化工作"
```

Agent 会自动检索相关历史记忆并注入上下文。

## 工作流程

每次调用是一个独立对话，三步走：

```
┌─────────────────────────────────────────────────────┐
│ Step 1: 记忆检索（Retriever）                         │
│   LLM 判断是否需要检索 → 语义匹配 + 时间序匹配 → 注入  │
├─────────────────────────────────────────────────────┤
│ Step 2: Agent Loop                                 │
│   带记忆上下文的 tool-calling 迭代，直到任务完成       │
├─────────────────────────────────────────────────────┤
│ Step 3: 记忆提取（Extractor）                        │
│   LLM 提取摘要/标签/实体 → 用户审核 → 存入向量库       │
└─────────────────────────────────────────────────────┘
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `memory-agent "your task"` | 执行任务 |
| `memory-agent -p /path/to/project "task"` | 指定项目目录 |
| `memory-agent --no-memory "task"` | 跳过记忆检索 |
| `memory-agent --no-extract "task"` | 跳过记忆提取（不保存本次对话） |
| `echo "task" \| memory-agent` | 从管道读入任务 |

## 对话内命令

在对话 query 中使用 `/memory` 前缀：

| 命令 | 说明 |
|------|------|
| `/memory` | 查看本次注入的记忆列表 |
| `/memory recent [N]` | 查看最近 N 条记忆摘要（默认 10） |
| `/memory search <query>` | 语义搜索记忆库 |
| `/memory show <id>` | 查看指定记忆详情 |
| `/memory delete <id>` | 删除指定记忆 |
| `/memory status` | 记忆库统计信息 |

## 配置

配置文件位于 `<项目根>/.agent-memory/config.yaml`，首次运行自动生成默认配置：

```yaml
llm:
  api_base: https://api.deepseek.com/v1
  api_key: ${DEEPSEEK_API_KEY}
  model: deepseek-chat

embedding:
  api_base: https://api.siliconflow.cn/v1
  api_key: ${SF_API_KEY}
  model: BAAI/bge-m3

retrieval:
  top_k: 10
  similarity_threshold: 0.5

extractor:
  auto_confirm: false          # true = 跳过审核，自动存储
  keep_full_transcript: true   # 是否保留完整对话原文
```

`${VAR}` 格式的值会自动从环境变量替换。

## 数据存储

```
.agent-memory/
├── config.yaml       # 配置文件
├── memories.db       # SQLite：记忆元数据、标签、实体
└── chroma/           # ChromaDB：向量嵌入
```

## 运行测试

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## 架构

```
src/memory_agent/
├── cli.py          # 入口：三步走编排
├── config.py       # 配置加载、环境变量替换
├── storage.py      # SQLite + ChromaDB 存储层
├── retriever.py    # LLM 决策 + 双通道检索
├── agent_loop.py   # OpenAI 兼容 API + tool calling
├── extractor.py    # 对话后记忆提取 + 用户审核
├── tools.py        # 内置工具（read_file / write_file / run_bash）
├── commands.py     # /memory 系列命令
└── prompts.py      # Prompt 模板
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 向量数据库 | ChromaDB（embedded，零外部依赖） |
| 元数据存储 | SQLite |
| LLM | 兼容 OpenAI API 的任意后端（默认 deepseek-chat） |
| Embedding | 兼容 OpenAI API（默认 SiliconFlow BAAI/bge-m3） |
| 语言 | Python 3.10+ |
