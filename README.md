# Memory Agent (Rust)

自带向量记忆系统的 CLI Agent，Rust 实现。每次对话从干净上下文开始，通过 LLM 决策自动检索相关历史记忆，注入 System Prompt 后执行任务；对话结束后自动提取关键信息，经用户审核后存入向量数据库。

## 快速开始

### 1. 安装 Rust

需要 Rust 1.75+：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. 编译

```bash
cd rust-refactor
cargo build --release
```

二进制文件位于 `target/release/memory-agent`。

### 3. 配置环境变量

```bash
export DEEPSEEK_API_KEY=sk-your-deepseek-key    # LLM（对话 + 决策 + 提取）
export SF_API_KEY=your-siliconflow-key           # Embedding（向量检索）
```

### 4. 首次运行

```bash
./target/release/memory-agent "帮我写一个 Python hello world 脚本"
```

首次运行时会在当前目录自动创建 `.agent-memory/` 目录，包含默认配置文件和数据存储。

### 5. 继续之前的工作

```bash
./target/release/memory-agent "继续上次的脚本优化工作"
```

Agent 会自动检索相关历史记忆并注入上下文。

## 工作流程

每次调用是一个独立对话，三步走：

```
┌─────────────────────────────────────────────────────┐
│ Step 1: 会话初始化（Session Init）                    │
│   去重状态重置 → skill 索引预热（不注入 prompt）       │
├─────────────────────────────────────────────────────┤
│ Step 2: Agent Loop（工具驱动检索）                    │
│   Round 1: LLM 分析任务 → 按需调用 search_memory /    │
│            search_skills 获取上下文 → 执行任务        │
│   Round 2+: 聚焦任务，已有上下文不够时继续搜索          │
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
| `memory-agent --manual-extract "task"` | 提取记忆前逐条确认 |
| `memory-agent --debug "task"` | 启用 HTTP 调试日志 |
| `echo "task" \| memory-agent` | 从管道读入任务 |
| `memory-agent` | 进入交互模式（无参数） |

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

## 技能管理

| 命令 | 说明 |
|------|------|
| `memory-agent --skill-list` | 列出已安装的技能 |
| `memory-agent --skill-install <path\|url>` | 从本地目录或 git URL 安装技能 |
| `memory-agent --skill-dir <DIR>` | 添加额外的技能搜索目录 |

## 配置

配置文件位于 `<项目根>/.agent-memory/config.yaml`，首次运行自动生成：

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
  auto_confirm: true
  keep_full_transcript: true
```

`${VAR}` 格式的值会自动从环境变量替换。

## 数据存储

```
.agent-memory/
├── config.yaml       # 配置文件
└── memories.db       # SQLite：记忆元数据 + sqlite-vec 向量（单库存储）
```

## 运行测试

```bash
cargo test
```

## 架构

```
src/
├── main.rs          # 入口：CLI + 会话初始化 + Agent Loop + 记忆提取
├── config.rs        # 配置加载、环境变量替换
├── storage/         # 存储层：schema.rs / relation.rs / vector.rs
│   └── mod.rs       # MemoryStore：SQLite 关系表 + sqlite-vec 向量（单库）
├── retriever.rs     # LLM 决策 + 双通道检索
├── agent_loop.rs    # OpenAI 兼容 API + 动态 system prompt + tool calling
├── extractor.rs     # 对话后记忆提取 + 用户审核
├── tools.rs         # 内置工具（含 search_memory / search_skills + 去重）
├── skills.rs        # Skill 发现、sqlite-vec 路由
├── commands.rs      # /memory 系列命令
├── prompts.rs       # Prompt 模板
└── debug.rs         # HTTP 调试日志 + token 统计
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 向量存储 | sqlite-vec（vec0 虚拟表，与元数据同库） |
| 元数据存储 | SQLite（rusqlite） |
| LLM | 兼容 OpenAI API 的任意后端（默认 deepseek-chat） |
| Embedding | 兼容 OpenAI API（默认 SiliconFlow BAAI/bge-m3） |
| 异步运行时 | Tokio |
| HTTP 客户端 | reqwest |
| CLI | clap derive |
| 语言 | Rust 1.75+ |

## 与 Python 版的区别

- 向量存储从 ChromaDB/LanceDB/HNSW 演进为 **sqlite-vec**（vec0 虚拟表，向量与元数据同库，增量写入）
- 异步运行时基于 **Tokio**，支持真正的并发请求
- 编译为单个静态二进制文件，无外部运行时依赖
- 性能更优，内存占用更低
