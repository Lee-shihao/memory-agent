# Memory Agent: 自带记忆系统的 CLI Agent

## 概述

构建一个带记忆系统的 CLI Agent。每次对话开始时上下文完全干净，通过 LLM 决策是否检索历史记忆、用何种方式检索，将相关记忆注入 System Prompt 后开始 Agent Loop。对话结束后，LLM 自动提取关键信息生成记忆摘要，用户审核确认后存入向量数据库。

## 核心机制

| 环节 | 行为 |
|------|------|
| 对话开始 | 干净上下文 → LLM 决策是否需要检索 → 语义匹配 + 时间序匹配 → 注入记忆到 System Prompt |
| 对话进行 | 正常 Agent Loop，`/memory` 可查看已注入的记忆或浏览历史记忆 |
| 对话结束 | LLM 自动提取摘要/标签/实体/决策 → 用户审核（编辑/保存/丢弃）→ Embedding 写入 ChromaDB + 元数据写入 SQLite |

## 架构

```
┌──────────────────────────────────────────────────┐
│                   CLI Agent                       │
│  ┌────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ /memory    │  │ /forget  │  │ /memory-graph│  │
│  │ (查看记忆)  │  │ (删除记忆) │  │ (预留)        │  │
│  └────────────┘  └──────────┘  └──────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │              Agent Loop                      │ │
│  │  System Prompt (base)                       │ │
│  │  + Memory Context (injected, if retrieved)  │ │
│  │  + Native Tool List                         │ │
│  │  ↓ OpenAI-compatible API                    │ │
│  └─────────────────────────────────────────────┘ │
│                      │                            │
│  ┌───────────────────┼─────────────────────────┐ │
│  │         Memory Engine                        │ │
│  │                                              │ │
│  │  ┌──────────┐ ┌──────────┐                  │ │
│  │  │Retriever │ │Extractor │                  │ │
│  │  │(记忆检索) │ │(记忆提取) │                  │ │
│  │  └────┬─────┘ └────┬─────┘                  │ │
│  │       │             │                        │ │
│  │  ┌────┴─────────────┴────────────────────┐  │ │
│  │  │           Storage Layer               │  │ │
│  │  │  ┌──────────┐  ┌───────────────────┐  │  │ │
│  │  │  │  SQLite  │  │    ChromaDB       │  │  │ │
│  │  │  │ 元数据    │  │  embedding+文本   │  │  │ │
│  │  │  └──────────┘  └───────────────────┘  │  │ │
│  │  └───────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

四个核心组件：

| 组件 | 职责 | 触发时机 |
|------|------|----------|
| Retriever | LLM 决策是否检索 + 生成检索词/时间范围 → 检索 ChromaDB + SQLite → 注入 System Prompt | 每次对话开始 |
| Extractor | 对话结束后，LLM 分析完整对话 → 提取摘要/关键结论/标签/实体/决策 → 展示给用户审核 | 每次对话结束 |
| Storage Layer | SQLite 存结构化元数据，ChromaDB 存 embedding 向量 | 持续 |
| CLI Commands | `/memory` 系列命令，查看/搜索/删除记忆 | 对话期间 |

数据目录（per-project）：

```
<project-root>/.agent-memory/
├── memories.db         # SQLite
├── chroma/             # ChromaDB 持久化目录
└── config.yaml         # 配置文件
```

---

## Retriever：LLM 决策式双通道检索

### 流程

```
User Query 进入（干净上下文）
      │
      ▼
┌──────────────────────────────────────┐
│ 1. LLM 决策（轻量调用，非 agent loop） │
│                                      │
│  System Prompt:                      │
│  "你有一个历史记忆库。                 │
│   分析用户的问题，判断是否需要查询      │
│   记忆库。如果需要，生成最合适的        │
│   语义检索词（1-3个）和/或指定取最近    │
│   的 N~M 条记忆。如果不需要，返回空。"   │
│                                      │
│  → 返回:                             │
│    need_retrieve: true|false         │
│    semantic_queries: ["query1",...]  │
│    recent_range: {start:N, end:M}|null│
└──────────────┬───────────────────────┘
               │
          need_retrieve?
          ┌────┴────┐
          │ false   │ true
          ▼         ▼
      跳过检索  ┌───────────────────────┐
               │ 2. 双通道并行检索       │
               │                       │
               │ 通道 A: 语义检索        │
               │  每个 query → embedding │
               │  → ChromaDB.query()    │
               │  → 按相似度排列         │
               │                       │
               │ 通道 B: 时间序检索      │
               │  SQLite:               │
               │  SELECT * FROM memories│
               │  ORDER BY created_at   │
               │  DESC LIMIT M          │
               │  OFFSET N-1            │
               │                       │
               │ 两通道结果合并          │
               │ 按 memory_id 去重       │
               │ 按时间排序              │
               └───────────┬───────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ 3. 注入上下文  │
                    │              │
                    │ Memory 块拼入 │
                    │ System Prompt│
                    └──────────────┘
```

### 设计决策

| 决策点 | 做法 | 理由 |
|--------|------|------|
| 谁决定是否检索 | LLM 轻量调用（1次API，不进入 agent loop） | 让模型判断何时需要记忆，而非默认触发 |
| 语义通道 | 每个 query 独立 embedding 查询 | 多角度覆盖，提高召回 |
| 时间序通道 | SQLite 按时间倒序取 N~M 条 | 覆盖"刚才那个"、"上次的"、"上一轮"等场景 |
| 去重 | 按 memory_id 去重，再按时间排序 | 同一记忆可能同时被语义通道和时间序通道命中，避免重复注入 |
| 检索结果注入 | 追加到 System Prompt 末尾，以结构化块呈现 | 清晰分隔，不影响 base prompt 语义 |

### 典型场景

| 用户说法 | LLM 决策 |
|---------|---------|
| "刚才那个 bug 怎么修的来着" | `semantic_queries: ["bug 修复"]`, `recent_range: {1, 5}` |
| "继续上次的工作" | `recent_range: {1, 1}`, `semantic_queries: []` |
| "之前讨论过的 Python async 方案" | `semantic_queries: ["Python async 方案"]`, `recent_range: null` |
| "跟上上轮那个架构有关的问题" | `recent_range: {3, 5}`, `semantic_queries: []` |
| "你好" / "帮我写个 hello world" | `need_retrieve: false` |

### 注入格式

```markdown
## Relevant Memories (from past conversations)

### [2026-07-28] Python async patterns
- Summary: 讨论了 asyncio.create_task vs asyncio.gather 的选择
- Key Points:
  - 对于独立的协程，优先使用 asyncio.create_task
  - worker 数量建议设置为 CPU 核数 × 2
- Tags: python, async, optimization

### [2026-07-25] Database connection pooling
- ...
```

---

## Extractor：对话结束后的记忆提取

### 流程

```
对话完整内容（user ↔ assistant 完整轮次 transcript）
      │
      ▼
┌─────────────────────────────────────────┐
│ 1. LLM 提取（Extractor Prompt）          │
│                                          │
│  输入: 完整对话 transcript                │
│                                          │
│  生成结构化输出:                          │
│  ┌──────────────────────────────────┐    │
│  │ summary:    ≤200字对话摘要         │    │
│  │ key_points: ["关键结论1", ...]    │    │
│  │ tags:       ["python","async"]   │    │
│  │ entities:   [{name, type, desc}] │    │
│  │ decisions:  ["做的决定1", ...]    │    │
│  └──────────────────────────────────┘    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. 展示给用户审核                         │
│                                          │
│  ┌──────────────────────────────────┐    │
│  │ 📝 Memory Preview                │    │
│  │                                  │    │
│  │ Summary: 讨论了 Python asyncio   │    │
│  │ worker 线程模型的选择...          │    │
│  │                                  │    │
│  │ Tags: python, async, worker      │    │
│  │                                  │    │
│  │ Key Points:                      │    │
│  │  • create_task 优于 gather       │    │
│  │  • worker 数量 = CPU * 2         │    │
│  │                                  │    │
│  │ Entities:                        │    │
│  │  • src/worker.py (file)          │    │
│  │  • WorkerPool (class)            │    │
│  │                                  │    │
│  │ Decisions:                       │    │
│  │  • 采用 create_task 作为默认策略  │    │
│  │                                  │    │
│  │ [Save] [Edit] [Discard]          │    │
│  └──────────────────────────────────┘    │
└──────────────┬──────────────────────────┘
               │
          ┌────┴────┐
          │ 用户操作  │
          ├─────────┤
          │ Save    → 进入存储流程
          │ Edit    → 打开编辑器修改字段后保存
          │ Discard → 不存储此对话
          └─────────┘
```

### 提取字段定义

| 字段 | 类型 | 说明 |
|------|------|------|
| `summary` | string | 自然语言摘要，≤200字。**这是向量检索时匹配的主要文本** |
| `key_points` | string[] | 3-8 条关键结论/知识点，每条一句话 |
| `tags` | string[] | 3-6 个标签。Extractor 从已有标签表中选择，也可创建新标签 |
| `entities` | `{name, type, description}[]` | 对话涉及的关键实体。type: `file`, `function`, `class`, `concept`, `dependency`, `config` |
| `decisions` | string[] | 对话中明确做出的决策/选择 |

### Embedding 策略

- 将 `summary` + 所有 `key_points` 拼接为一个文本块 → 调用 embedding 模型 → 存入 ChromaDB
- `tags`、`entities`、`decisions` 只存入 SQLite，不参与 embedding

---

## Storage Layer

### SQLite Schema

```sql
-- 记忆主表
CREATE TABLE memories (
    id TEXT PRIMARY KEY,              -- UUID
    summary TEXT NOT NULL,            -- 摘要（≤200字）
    conversation_at TIMESTAMP,        -- 对话发生时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversation_json TEXT,           -- 完整对话 transcript（可选保留）
    chroma_doc_id TEXT                -- 对应 ChromaDB 中的文档 ID
);

-- 关键结论
CREATE TABLE key_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    sort_order INTEGER DEFAULT 0
);

-- 标签
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE memory_tags (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, tag_id)
);

-- 实体
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT NOT NULL,               -- file, function, class, concept, dependency, config
    description TEXT
);

-- 决策记录
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    content TEXT NOT NULL
);

-- 索引
CREATE INDEX idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX idx_memories_conversation_at ON memories(conversation_at DESC);
CREATE INDEX idx_entities_type ON entities(type);
CREATE INDEX idx_memory_tags_tag_id ON memory_tags(tag_id);
```

### ChromaDB

- **Collection**: `memories`
- **Document**: `summary + key_points 拼接文本`
- **Metadata**: `{ memory_id, tags_json, conversation_at }`
- **Embedding 模型**: 可配置，默认 `text-embedding-3-small`

### 配置文件 `.agent-memory/config.yaml`

```yaml
embedding:
  model: text-embedding-3-small
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}

retrieval:
  top_k: 10
  similarity_threshold: 0.5

llm:
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o

extractor:
  auto_confirm: false              # 是否跳过审核直接存储
  keep_full_transcript: true       # 是否保留完整对话原文
```

---

## CLI Commands

| 命令 | 功能 |
|------|------|
| `/memory` | 显示**本次对话开始时**注入的记忆列表（如果检索触发了的话），让用户知道 agent 带入了什么上下文 |
| `/memory recent [N]` | 查看最近 N 条记忆摘要列表，默认 N=10 |
| `/memory search <query>` | 手动语义搜索记忆库，按相似度排列结果 |
| `/memory show <id>` | 查看指定记忆的详情（摘要、关键结论、标签、实体、决策） |
| `/memory delete <id>` | 删除指定记忆（从 SQLite + ChromaDB 中同时移除） |
| `/memory status` | 数据库统计：总记忆数、标签分布、最后入库时间、占用空间 |

---

## 待定 / 后续考虑

- **图谱功能**: 话题相似度边、工作演进链、实体引用关系。当前不实现，后续作为记忆匹配的补充方式
- **自动确认模式**: `extractor.auto_confirm: true` 跳过审核，对话结束自动入库
- **记忆压缩/合并**: 当相似记忆过多时的去重策略
- **记忆过期**: 按时间或按重要性自动清理旧记忆
- **多项目支持**: 当前 per-project，后续可扩展全局记忆层

---

## 技术选型

| 组件 | 选择 | 说明 |
|------|------|------|
| 向量数据库 | ChromaDB (embedded) | 零外部依赖，验证阶段最佳 |
| 元数据存储 | SQLite | 轻量，单文件，易于备份 |
| Embedding 模型 | 可配置，默认 text-embedding-3-small | 通过 OpenAI 兼容 API 调用 |
| LLM | OpenAI 兼容 API（任何后端） | 不绑定特定 SDK |
| 语言 | Python | ChromaDB + SQLite 生态成熟 |
