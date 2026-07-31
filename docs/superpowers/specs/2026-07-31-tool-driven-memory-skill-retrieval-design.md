# Tool-Driven Memory & Skill Retrieval — Design Spec

**Date:** 2026-07-31
**Status:** Design approved, pending implementation plan

## Overview

将 memory 和 skill 的检索从 pre-loop 一次性注入 system prompt 的模式，改为 LLM 通过内置工具主动获取的模式。System prompt 只做行为引导，数据通过工具返回流入对话。

### Motivation

当前问题：
- Pre-loop 检索是一次性的 — LLM 在 Agent Loop 中无法主动发起新的检索
- 检索结果注入 system prompt — prompt 臃肿，且检索逻辑与 agent 行为分离
- 缺少 `search_skills` embedding 匹配工具 — LLM 只能按名称加载 skill
- System prompt 静态 — 不随迭代轮次变化
- 无跨工具调用的去重机制

### Design Decisions

| 决策 | 选择 |
|------|------|
| Pre-loop 检索 | 完全移除，改为工具驱动 |
| Skill 工具拆分 | 三个独立工具：`search_skills` / `load_skill` / `list_skills` |
| System prompt 变化 | 每轮动态更新（Round 1 vs Round 2+） |
| 去重层面 | 工具层透明去重（session 级 ID set） |
| 简单任务 | LLM 自行判断，可跳过检索直接执行 |

---

## Architecture & Data Flow

### 新流程

```
User Query
  │
  └─ Agent Loop（system prompt 轻量，只做行为引导）
       │
       ├─ Round 1:
       │    System Prompt: "先分析任务 → 需要记忆？需要skill？→ 调用工具获取"
       │    LLM 分析 → 调用 search_memory / search_skills / 直接执行
       │
       ├─ Round 2+:
       │    System Prompt: "聚焦任务，需要更多上下文时可继续搜索"
       │    LLM 基于已获取的记忆/skill（在对话历史 tool 消息中）+ 任务执行
       │
       └─ ... 迭代直到完成
```

### Data Flow Changes

| 内容 | 当前流向 | 新流向 |
|------|---------|--------|
| 记忆内容 | System Prompt 注入 | `search_memory` tool response → conversation |
| Skill 匹配 | System Prompt 注入 | `search_skills` tool response → conversation |
| Skill 详情 | `load_skill` tool response | 不变 |
| 行为引导 | 静态 BASE_AGENT_SYSTEM_PROMPT | 动态：Round 1 vs Round 2+ |

### Session 定义

一次 `run_pipeline()` 调用 = 一个完整 session。去重状态在 session 开始时重置，跟随整个 Agent Loop 生命周期。

---

## Tool Design

### Tool Inventory (10 tools total)

| 工具 | 状态 | 用途 |
|------|------|------|
| `read_file` | 不变 | 读取文件 |
| `write_file` | 不变 | 写入文件 |
| `edit_file` | 不变 | 精确字符串替换 |
| `grep_files` | 不变 | 正则搜索文件内容 |
| `git_ops` | 不变 | 安全 git 操作 |
| `run_bash` | 不变 | 执行 shell 命令 |
| `search_memory` | **增强** | embedding 检索历史记忆，带去重 |
| `search_skills` | **新增** | embedding 语义匹配 skill，带去重 |
| `load_skill` | 不变 | 按名称加载完整 SKILL.md |
| `list_skills` | **新增** | 列出所有 skill 摘要 |

### New Tool: `search_skills`

```
Description: Search for relevant skills using semantic matching.
             Returns skill names, descriptions, and relevance scores.
             Use load_skill(name) afterwards to get full instructions.

Parameters:
  query (string, required): What kind of skill or capability you need
  top_k (integer, optional): Number of results (default: 3)

Returns: [{name, description, source, score}], sorted by relevance.
         Already-returned skills are filtered out (dedup).
```

### New Tool: `list_skills`

```
Description: List all installed skills with their summaries.
             Prefer search_skills for targeted matching when many skills are installed.

Parameters: none

Returns: List of all installed skills with name, source, and description.
```

### Enhanced Tool: `search_memory`

Interface unchanged. Internal enhancement: session-level dedup.

### Dedup Mechanism (Tool-Layer, Transparent)

```
Session-level state (module-level, reset per pipeline invocation):
  _returned_memory_ids: set[str]
  _returned_skill_names: set[str]

search_memory execution:
  1. Embedding search → raw results
  2. Filter: remove results where memory_id ∈ _returned_memory_ids
  3. Add new IDs to _returned_memory_ids
  4. Return filtered results
  5. If all filtered: "No new memories found. Try a different query."

search_skills execution:
  1. Embedding search → raw results
  2. Filter: remove results where name ∈ _returned_skill_names
  3. Add new names to _returned_skill_names
  4. Return filtered results
  5. If all filtered: prompt to try different query terms

load_skill: also adds loaded skill name to _returned_skill_names
list_skills: does NOT add to dedup set (informational listing)
```

Dedup is transparent to the LLM — it doesn't need to track IDs. The tools silently filter duplicates.

---

## Dynamic System Prompts

### Round 1 Prompt (first iteration)

```
You are a helpful AI assistant with access to tools. You can read files,
write files, and execute shell commands to help the user accomplish tasks.

## Before You Start

Before diving into the task, analyze what the user is asking:

1. **Need past conversation context?**
   If the user references previous work, past discussions, or prior decisions,
   call search_memory(query) with specific search terms to find relevant memories.

2. **Need specialized skills or workflows?**
   Call search_skills(query) to find matching skills, or list_skills()
   to see all available skills. Then use load_skill(name) for full instructions.

3. **Simple, self-contained tasks** (e.g., "write hello world", "what is 2+2")
   can be executed directly — skip retrieval.

Work step by step. When done, provide a clear summary of what was accomplished.
```

### Round 2+ Prompt (iteration 2 onwards, appended)

```
Continue working on the user's task. Use the context you've already retrieved
from tool calls earlier in this conversation.

If you discover gaps and need more:
- search_memory(query) — search past conversations
- search_skills(query) — find additional skills
- load_skill(name) — load full skill instructions
```

### Switching Mechanism

In `agent_loop.py`:
```python
# iteration == 0: messages[0]["content"] = ROUND_1_SYSTEM_PROMPT
# iteration >= 1: messages[0]["content"] = ROUND_1_SYSTEM_PROMPT + "\n\n" + ROUND_2_PLUS_PROMPT
```

The prompt changes by modifying `messages[0]["content"]` each iteration. The Round 1 prompt is always present as the base; Round 2+ prompt is appended starting from the second iteration.

---

## Implementation Changes

### Files Changed

| File | Change |
|------|--------|
| `prompts.py` | Add `ROUND_1_SYSTEM_PROMPT`, `ROUND_2_PLUS_PROMPT`; keep existing templates for retriever/extractor |
| `tools.py` | Add `search_skills` tool + executor; add `list_skills` tool + executor; add dedup to `search_memory`; add session state management (`reset_session_state`) |
| `agent_loop.py` | Remove `memory_context` parameter; add dynamic prompt switching per iteration |
| `cli.py` | Remove pre-loop memory retrieval + skill routing injection; add `reset_session_state()` + `pre_index_skills()` at pipeline start |
| `skills.py` | Minor — expose `SkillRouter` for tool use if needed |

### Files NOT Changed

| File | Reason |
|------|--------|
| `retriever.py` | `Retriever` class kept — still used by `/memory search` command; just no longer called in pre-loop |
| `storage.py` | No changes needed |
| `extractor.py` | Extraction logic unchanged |
| `commands.py` | `/memory` commands unchanged |
| `config.py` | No config changes needed |

### Backward Compatibility

| Feature | Impact |
|------|--------|
| `--no-memory` flag | Retained — skips `pre_index_skills()`; tools still work via lazy init |
| `/memory` slash commands | Unchanged — processed before agent loop |
| `--no-extract` flag | Unchanged |
| Bash confirmation (`run_bash`) | Unchanged |
| Interactive REPL | Unchanged |

---

## Edge Cases

1. **LLM skips retrieval on a task that needs it** → LLM will likely hit a dead end, then Round 2+ prompt guides it to try `search_memory`/`search_skills`
2. **All memories/skills filtered by dedup** → Tool returns "No new results. Try a different query." — LLM can rephrase and try again
3. **No skills installed** → `search_skills` returns empty, `list_skills` returns "No skills installed"
4. **Embedding API down** → Tool returns error message, LLM continues without memory/skill context
5. **Very large skill list** → `list_skills` description warns to prefer `search_skills`; if LLM calls it anyway, result is truncated at a reasonable limit

---

## Spec Self-Review

- **Placeholders:** None — all decisions are concrete.
- **Internal consistency:** Architecture, tool design, prompts, and implementation changes are all aligned. Round 1 prompt's "analyze then search" matches tool-driven data flow. Dedup is tool-layer and transparent, matching the prompt's simplicity.
- **Scope:** Single cohesive change — convert pre-loop injection to tool-driven retrieval. No unrelated refactoring.
- **Ambiguity:** Prompt wording will be refined during implementation; exact function signatures are specified.
