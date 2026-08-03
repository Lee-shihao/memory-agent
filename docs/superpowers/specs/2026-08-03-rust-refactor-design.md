# Rust Refactor Design

## Overview

Rewrite the `memory-agent` Python CLI in Rust, preserving exact functionality and semantics.
Target directory: `rust-refactor/` under the existing project root.

## Tech Stack

| Component    | Python Original      | Rust Replacement                  |
| ------------ | -------------------- | --------------------------------- |
| Async        | sync (httpx)         | Tokio + reqwest                   |
| CLI          | argparse             | clap derive                       |
| Config       | PyYAML               | serde_yaml                        |
| Vector DB    | ChromaDB (embedded)  | LanceDB (embedded, Rust-native)   |
| SQLite       | sqlite3              | rusqlite                          |
| HTTP         | httpx                | reqwest                           |
| Crate layout | single package       | library crate + binary            |

LanceDB stores both memory embeddings (`memories` table) and skill embeddings (`skills` table).

## Crate Structure

```
rust-refactor/
├── Cargo.toml
└── src/
    ├── main.rs           # Binary entry point, clap CLI
    ├── lib.rs            # Library root, re-exports
    ├── config.rs         # YAML loading + env var substitution
    ├── storage.rs        # SQLite metadata + LanceDB vectors
    ├── agent_loop.rs     # OpenAI API tool-calling loop
    ├── tools.rs          # 10 built-in tools + dispatch
    ├── extractor.rs      # Post-conversation memory extraction
    ├── retriever.rs      # LLM-driven dual-channel search
    ├── skills.rs         # Skill discovery + LanceDB routing
    ├── commands.rs       # /memory slash commands
    ├── prompts.rs        # Prompt template constants
    └── debug.rs          # HTTP debug logging + token tracking
```

## Module Design

### config.rs
- `Config` struct with `#[derive(Deserialize)]`, all fields as in Python
- `load_config(project_root) -> Config`: reads or creates `.agent-memory/config.yaml`
- `${VAR}` env substitution with `regex` crate before YAML parse
- `DEFAULT_CONFIG_YAML` const for bootstrap

### storage.rs
- `MemoryStore` struct holding `rusqlite::Connection` + LanceDB handle
- SQLite schema identical to Python: `memories`, `key_points`, `tags`, `memory_tags`, `entities`, `decisions`
- `init_schema()`, `init_lancedb(persist_dir, api_base, api_key, model)`
- `get_embedding(text) -> Vec<f32>`: direct HTTP call to embedding API
- `add_to_vectordb(memory_id, text, metadata)` / `query_vectordb(text, top_k)` for memories table
- `add_skill_to_vectordb(name, text, metadata)` / `search_skills_vectordb(query, top_k)` for skills table
- Standard CRUD: `insert_memory`, `get_memory`, `get_recent_memories`, `delete_memory`, `search_by_tag`, `get_status`, `get_all_tags`
- `_hydrate_memory(row) -> Memory` joins across all related tables

### agent_loop.rs
- `run_agent_loop(config, user_query, tools, max_iters=50, confirm_callback) -> String`
- OpenAI-compatible `/chat/completions` with tool calling via reqwest
- Round 1 system prompt then Round 2+ prompts (same as Python)
- `ConfirmCallback` type: `Fn(&str, &HashMap<String, Value>) -> (bool, String)`
- Returns full transcript; handles HTTP errors gracefully

### tools.rs
- `TOOL_DEFINITIONS: Vec<Value>` — OpenAI function-calling JSON schemas
- `execute_tool(name, args) -> String` — dispatcher with match
- 10 tool implementations: `tool_read_file`, `tool_write_file`, `tool_edit_file`, `tool_grep_files`, `tool_git_ops`, `tool_run_bash`, `tool_ask_user`, `tool_load_skill`, `tool_search_memory`, `tool_search_skills`
- Session state: `AtomicRefCell<HashSet<String>>` for memory/skill dedup
- `classify_bash_command(cmd) -> Tier` (Safe/Dangerous/Unknown)
- `set_workspace_root(path)`, path resolution with fallback logic

### skills.rs
- `Skill` struct: name, path, description, source
- `discover_skills(project_root) -> Vec<Skill>`: scans search paths
- `SkillRouter` struct wrapping LanceDB skills table
- `index_skills(skills)`: embed + upsert, remove deleted
- `search(query, top_k) -> Vec<SkillMatch>`: cosine similarity search
- `install_skill(source, project_root) -> Result<String>`: dir copy or git clone
- `get_skill(name)`, `get_skill_list_text()`, `format_skills_for_injection()`

### extractor.rs
- `ExtractionResult` struct: summary, key_points, tags, entities, decisions
- `extract_and_store(transcript, config, store, auto_confirm) -> bool`
- LLM extraction call, auto-confirm path or interactive save/edit/discard loop
- `_store_result()`: embed summary+key_points → LanceDB, insert metadata → SQLite

### retriever.rs
- `Retriever::new(config, store)`
- `retrieve(user_query) -> (Vec<Memory>, String)`: LLM decision → semantic search + time range → dedup by ID → hydrate → format for injection
- `_llm_decision(query) -> RetrievalDecision`: JSON response parse
- `_semantic_search(query)`, `_time_range_search(limit, offset)`

### commands.rs
- `handle_slash_command(message, store, injected_memories) -> Option<String>`
- Subcommand dispatch: show injected, recent N, search, show ID, delete ID, status
- Returns `None` if not a `/memory` command

### prompts.rs
- All prompt constants as `&str` statics
- `format_memory_for_injection()`, `format_memories_for_injection()`
- `ROUND_1_SYSTEM_PROMPT`, `ROUND_2_PLUS_PROMPT`
- `EXTRACTOR_SYSTEM_PROMPT`, `EXTRACTOR_USER_TEMPLATE`
- `RETRIEVAL_DECISION_SYSTEM_PROMPT`, `RETRIEVAL_DECISION_USER_TEMPLATE`

### debug.rs
- `enable(memory_dir)`: open debug log, truncate
- `is_enabled() -> bool`
- `log_request(module, method, url, headers, body) -> request_id`
- `log_response(request_id, status_code, body)`
- `accumulate_usage(usage)`, `get_session_stats()`, `reset_session_stats()`
- Thread-safe with `Mutex`, auth header redaction

### main.rs
- clap derive: `Cli` struct with flags matching Python argparse
- `--version`, `--project`, `--no-memory`, `--no-extract`, `--manual-extract`, `--debug`
- Skill management: `--skill-list`, `--skill-install`, `--skill-dir`
- Subcommand or positional `query` args
- Single-shot mode (pipe/args) vs interactive REPL loop
- REPL: rustyline for history, tab completion for `/memory` subcommands
- `_tool_confirm()` callback: ask_user interaction, bash tier confirmation
- Token stats printing in debug mode

## Pipeline (same 3-step flow)

```
1. Session Init: reset dedup state → pre-index skills
2. Agent Loop: R1 prompt → tool calls (search_memory/search_skills) → R2+ prompt → final response
3. Memory Extraction: LLM extraction → user review (or auto-confirm) → store
```

## Dependencies (Cargo.toml)

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde_yaml = "0.9"
rusqlite = { version = "0.31", features = ["bundled"] }
lancedb = "0.9"
clap = { version = "4", features = ["derive"] }
rustyline = "14"
regex = "1"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4"] }
```

## Error Handling

- `anyhow::Result` for application-level error propagation
- `thiserror` for library error types where structured errors are needed
- Tool execution errors returned as strings (same as Python)

## Testing Strategy

- Unit tests inline in each module (`#[cfg(test)] mod tests`)
- Integration tests in `tests/` directory mirroring Python test coverage
- Test vectors: mock LLM responses, in-memory SQLite, temp LanceDB directories
