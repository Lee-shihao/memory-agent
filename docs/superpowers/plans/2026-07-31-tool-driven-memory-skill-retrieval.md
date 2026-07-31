# Tool-Driven Memory & Skill Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert memory/skill retrieval from pre-loop system-prompt injection to LLM-driven tool-based retrieval with dynamic system prompts and session-level dedup.

**Architecture:** Remove pre-loop injection from `cli.py`; add `ROUND_1_SYSTEM_PROMPT`/`ROUND_2_PLUS_PROMPT` to `prompts.py`; add `search_skills` tool + session-state management to `tools.py`; update `agent_loop.py` for dynamic prompt switching and simplified signature.

**Tech Stack:** Python 3.10+, httpx, ChromaDB (existing stack, no new dependencies)

## Global Constraints

- `load_skill` and `list_skills` are implemented as internal functions but NOT registered in TOOL_DEFINITIONS
- `search_skills` returns full skill content (not just name+description)
- Dedup is tool-layer transparent — LLM doesn't track IDs
- Session = one `run_pipeline()` call; state resets at pipeline start
- `--no-memory` flag retained; `/memory` slash commands unchanged
- `Retriever` class kept — still used by `/memory search` command

---

### Task 1: Add dynamic system prompt constants to prompts.py

**Files:**
- Modify: `src/memory_agent/prompts.py` (append new constants)

**Interfaces:**
- Produces: `ROUND_1_SYSTEM_PROMPT: str`, `ROUND_2_PLUS_PROMPT: str`

`ROUND_1_SYSTEM_PROMPT` replaces `BASE_AGENT_SYSTEM_PROMPT` as the initial system message. `ROUND_2_PLUS_PROMPT` is appended starting from iteration 2.

- [ ] **Step 1: Add ROUND_1_SYSTEM_PROMPT and ROUND_2_PLUS_PROMPT to prompts.py**

Append after the existing `BASE_AGENT_SYSTEM_PROMPT` constant (do not remove it — it may be referenced elsewhere). Add at end of file:

```python
ROUND_1_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools. You can read files,
write files, and execute shell commands to help the user accomplish tasks.

## Before You Start

Before diving into the task, analyze what the user is asking:

1. **Need past conversation context?**
   If the user references previous work, past discussions, or prior decisions,
   call search_memory(query) with specific search terms to find relevant memories.

2. **Need specialized skills or workflows?**
   Call search_skills(query) to find matching skills with their full instructions.

3. **Simple, self-contained tasks** (e.g., "write hello world", "what is 2+2")
   can be executed directly — skip retrieval.

Work step by step. When done, provide a clear summary of what was accomplished.
"""

ROUND_2_PLUS_PROMPT = """\
Continue working on the user's task. Use the context you've already retrieved
from tool calls earlier in this conversation.

If you discover gaps and need more:
- search_memory(query) — search past conversations
- search_skills(query) — find additional skills with full instructions
"""
```

- [ ] **Step 2: Verify constants import cleanly**

Run: `python -c "from memory_agent.prompts import ROUND_1_SYSTEM_PROMPT, ROUND_2_PLUS_PROMPT; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/memory_agent/prompts.py
git commit -m "feat: add ROUND_1 and ROUND_2_PLUS system prompt constants

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add search_skills tool, session state, and dedup to tools.py

**Files:**
- Modify: `src/memory_agent/tools.py` (add tool + state management)
- Create: `tests/test_tools.py` (new test file)

**Interfaces:**
- Consumes: `SkillRouter` from `memory_agent.skills` (already exists), `Config` from `memory_agent.config`
- Produces:
  - `reset_session_state() -> None`
  - `pre_index_skills(config: Config) -> None`
  - `tool_search_skills(query: str, top_k: int = 3) -> str`
  - `_list_skills() -> str` (internal, not in TOOL_DEFINITIONS)
  - Updated `tool_search_memory` with dedup
  - New `search_skills` entry in `TOOL_DEFINITIONS`

- [ ] **Step 1: Write failing tests for search_skills and dedup**

Create `tests/test_tools.py`:

```python
"""Tests for built-in tools: search_skills, search_memory dedup, session state."""
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path


class TestSearchSkills:
    def test_search_skills_returns_matches(self, tmp_path):
        """search_skills should return skill name, description, and content."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_skills = [
            {
                "name": "refactoring-wizard",
                "description": "Helps with code refactoring",
                "source": "project",
                "distance": 0.05,
            }
        ]
        mock_skill_obj = MagicMock()
        mock_skill_obj.name = "refactoring-wizard"
        mock_skill_obj.description = "Helps with code refactoring"
        mock_skill_obj.source = "project"
        mock_skill_obj.load.return_value = "# Refactoring Wizard\n\nFull instructions here."

        with patch("memory_agent.skills.SkillRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.search.return_value = mock_skills
            mock_router._collection.count.return_value = 1

            with patch("memory_agent.skills.discover_skills", return_value=[mock_skill_obj]):
                with patch("memory_agent.skills.get_skill", return_value=mock_skill_obj):
                    from memory_agent.tools import tool_search_skills

                    result = tool_search_skills(query="refactoring")

        assert "refactoring-wizard" in result
        assert "Helps with code refactoring" in result
        assert "Full instructions here" in result

    def test_search_skills_dedup_filters_returned(self, tmp_path):
        """Second call with same query should filter already-returned skills."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_skills = [
            {
                "name": "refactoring-wizard",
                "description": "Helps with code refactoring",
                "source": "project",
                "distance": 0.05,
            }
        ]
        mock_skill_obj = MagicMock()
        mock_skill_obj.name = "refactoring-wizard"
        mock_skill_obj.description = "Helps with code refactoring"
        mock_skill_obj.source = "project"
        mock_skill_obj.load.return_value = "# Refactoring Wizard\n\nFull instructions."

        with patch("memory_agent.skills.SkillRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.search.return_value = mock_skills
            mock_router._collection.count.return_value = 1

            with patch("memory_agent.skills.discover_skills", return_value=[mock_skill_obj]):
                with patch("memory_agent.skills.get_skill", return_value=mock_skill_obj):
                    from memory_agent.tools import tool_search_skills

                    result1 = tool_search_skills(query="refactoring")
                    result2 = tool_search_skills(query="refactoring")

        assert "refactoring-wizard" in result1
        assert "No new skills found" in result2 or "refactoring-wizard" not in result2

    def test_search_skills_no_skills_installed(self, tmp_path):
        """When no skills exist, return appropriate message."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        with patch("memory_agent.skills.SkillRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.search.return_value = []
            mock_router._collection.count.return_value = 0

            from memory_agent.tools import tool_search_skills

            result = tool_search_skills(query="anything")

        assert "No skills" in result or "not found" in result.lower()


class TestSearchMemoryDedup:
    def test_search_memory_dedup_filters_duplicates(self, tmp_path):
        """Second search_memory call should filter already-returned IDs."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_results1 = [
            {"memory_id": "mem-1", "text": "Python async discussion", "distance": 0.2},
            {"memory_id": "mem-2", "text": "Git workflow tips", "distance": 0.3},
        ]
        mock_results2 = [
            {"memory_id": "mem-1", "text": "Python async discussion", "distance": 0.2},
            {"memory_id": "mem-3", "text": "New result", "distance": 0.4},
        ]

        with patch("memory_agent.tools.MemoryStore") as MockStore:
            mock_store = MockStore.return_value
            mock_store.query_chroma.side_effect = [mock_results1, mock_results2]
            mock_store._chroma_collection = MagicMock()

            from memory_agent.tools import tool_search_memory

            result1 = tool_search_memory(query="python")
            result2 = tool_search_memory(query="python")

        assert "mem-1" in result1
        assert "mem-2" in result1
        # Second call: mem-1 should be filtered, mem-3 should appear
        assert "mem-3" in result2


class TestSessionState:
    def test_reset_session_state_clears_dedup(self, tmp_path):
        """reset_session_state should clear all dedup tracking."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_results = [
            {"memory_id": "mem-1", "text": "Test", "distance": 0.1},
        ]

        with patch("memory_agent.tools.MemoryStore") as MockStore:
            mock_store = MockStore.return_value
            mock_store.query_chroma.return_value = mock_results
            mock_store._chroma_collection = MagicMock()

            from memory_agent.tools import tool_search_memory

            result1 = tool_search_memory(query="test")
            assert "mem-1" in result1

            # Reset state
            reset_session_state()

            result2 = tool_search_memory(query="test")
            # After reset, mem-1 should appear again (fresh session)
            assert "mem-1" in result2
```

- [ ] **Step 2: Run tests to verify they FAIL**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `tool_search_skills` not defined, `reset_session_state` not defined

- [ ] **Step 3: Implement session state management in tools.py**

Add near the top of `tools.py`, after existing imports and before `_workspace_root`:

```python
# ── session state (dedup tracking) ────────────────────────────────────────────

_returned_memory_ids: set[str] = set()
_returned_skill_names: set[str] = set()


def reset_session_state() -> None:
    """Reset dedup state at the beginning of each pipeline invocation."""
    global _returned_memory_ids, _returned_skill_names
    _returned_memory_ids.clear()
    _returned_skill_names.clear()


def pre_index_skills(config) -> None:
    """Ensure all skills are indexed in ChromaDB before the agent loop.
    
    Does NOT inject skills into the system prompt — only indexes them so
    search_skills can find them via embedding lookup.
    """
    from memory_agent.skills import SkillRouter, discover_skills

    skills = discover_skills(config.memory_dir.parent)
    if not skills:
        return

    router = SkillRouter(
        chroma_dir=config.memory_dir / "chroma",
        embedding_api_base=config.embedding_api_base,
        embedding_api_key=config.embedding_api_key,
        embedding_model=config.embedding_model,
    )
    router.index_skills(skills)
```

- [ ] **Step 4: Implement tool_search_skills + _list_skills**

Add after `tool_load_skill`, before the `search_memory` section:

```python
# ── search_skills ─────────────────────────────────────────────────────────────

def tool_search_skills(query: str, top_k: int = 3) -> str:
    """Search for relevant skills using embedding-based semantic matching.
    
    Returns full skill content for matched skills, filtered by dedup state.
    """
    from memory_agent.skills import SkillRouter, discover_skills, get_skill
    from memory_agent.config import load_config

    config = load_config(_workspace_root)

    # Discover and index skills
    skills = discover_skills(config.memory_dir.parent)
    if not skills:
        return "No skills installed. Use the skill management commands to install skills."

    router = SkillRouter(
        chroma_dir=config.memory_dir / "chroma",
        embedding_api_base=config.embedding_api_base,
        embedding_api_key=config.embedding_api_key,
        embedding_model=config.embedding_model,
    )
    router.index_skills(skills)

    if router._collection.count() == 0:
        return "No skills indexed."

    try:
        raw_results = router.search(query, top_k=top_k)
    except Exception as e:
        return f"Skill search failed: {e}"

    # Filter by dedup state
    new_results = []
    for r in raw_results:
        if r["name"] not in _returned_skill_names:
            new_results.append(r)

    if not new_results:
        return (
            "No new skills found for this query. "
            "Previously matched skills have already been returned. "
            "Try a different query to find additional skills."
        )

    # Track returned names
    for r in new_results:
        _returned_skill_names.add(r["name"])

    # Load full content for each matched skill
    lines = [f"Skill search results for '{query}':\n"]
    for i, r in enumerate(new_results):
        dist = r.get("distance")
        score = f" (score: {1 - dist:.2f})" if dist is not None else ""
        lines.append(f"## {r['name']}{score}")
        lines.append(f"**Description:** {r['description']}")
        lines.append(f"**Source:** {r['source']}")

        # Load full SKILL.md content
        skill = get_skill(r["name"])
        if skill:
            lines.append(f"\n{skill.load()}")
        else:
            lines.append("\n(Full instructions not available)")

        if i < len(new_results) - 1:
            lines.append("\n---\n")

    return "\n".join(lines)


def _list_skills() -> str:
    """List all installed skills with summaries (internal, not exposed as tool)."""
    from memory_agent.skills import discover_skills

    skills = discover_skills(_workspace_root)
    if not skills:
        return "No skills installed."

    lines = [f"Installed skills ({len(skills)}):\n"]
    for s in skills:
        lines.append(f"  - **{s.name}** ({s.source}): {s.description}")
    return "\n".join(lines)
```

- [ ] **Step 5: Add dedup to existing tool_search_memory**

Replace the existing `tool_search_memory` function (lines 261-294) with this version that adds dedup:

```python
# ── search_memory ─────────────────────────────────────────────────────────────

def tool_search_memory(query: str, top_k: int = 5) -> str:
    """Search the memory vector database for relevant past conversations."""
    from memory_agent.storage import MemoryStore
    from memory_agent.config import load_config

    config = load_config(_workspace_root)
    db_path = config.memory_dir / "memories.db"
    store = MemoryStore(db_path)
    store.init_schema()

    if not hasattr(store, "_chroma_collection") or store._chroma_collection is None:
        store.init_chroma(
            persist_dir=config.memory_dir / "chroma",
            embedding_api_base=config.embedding_api_base,
            embedding_api_key=config.embedding_api_key,
            embedding_model=config.embedding_model,
        )

    try:
        results = store.query_chroma(query_text=query, top_k=top_k)
    except Exception as e:
        return f"Memory search failed: {e}"

    if not results:
        return f"No memories found matching: {query}"

    # Filter by dedup state
    new_results = []
    for r in results:
        mid = r.get("memory_id", "?")
        if mid not in _returned_memory_ids:
            new_results.append(r)

    if not new_results:
        return (
            "No new memories found for this query. "
            "Previously matched memories have already been returned. "
            "Try a different query or check /memory recent for time-based retrieval."
        )

    # Track returned IDs
    for r in new_results:
        mid = r.get("memory_id", "?")
        if mid != "?":
            _returned_memory_ids.add(mid)

    lines = [f"Memory search results for '{query}':\n"]
    for r in new_results:
        mid = r.get("memory_id", "?")
        text = r.get("text", "")[:200]
        dist = r.get("distance")
        score = f" (score: {1 - dist:.2f})" if dist is not None else ""
        lines.append(f"[{mid}]{score} {text}")
    return "\n".join(lines)
```

- [ ] **Step 6: Add search_skills to TOOL_DEFINITIONS**

After the `search_memory` entry in `TOOL_DEFINITIONS` (before the closing `]`), add:

```python
    {
        "type": "function",
        "function": {
            "name": "search_skills",
            "description": (
                "Search for relevant skills using semantic matching. "
                "Returns the full content of matched skills (name, description, "
                "full instructions), sorted by relevance score. "
                "Use this to discover and immediately apply specialized workflows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What kind of skill or capability you need",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 3)",
                    },
                },
                "required": ["query"],
            },
        },
    },
```

- [ ] **Step 7: Add tool_search_skills to TOOL_EXECUTORS**

In `TOOL_EXECUTORS` dict, after `"search_memory": tool_search_memory,`, add:

```python
    "search_skills": tool_search_skills,
```

- [ ] **Step 8: Run tests to verify they PASS**

Run: `pytest tests/test_tools.py -v`
Expected: All tests PASS

- [ ] **Step 9: Run existing tests to verify nothing broke**

Run: `pytest tests/test_agent_loop.py tests/test_retriever.py tests/test_commands.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add src/memory_agent/tools.py tests/test_tools.py
git commit -m "feat: add search_skills tool, session dedup, state management

- Add search_skills tool with embedding-based skill matching
- Add _list_skills internal function (not exposed as tool)
- Add session-level dedup for search_memory and search_skills
- Add reset_session_state() and pre_index_skills()

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Update agent_loop.py for dynamic prompts and simplified signature

**Files:**
- Modify: `src/memory_agent/agent_loop.py`
- Modify: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `ROUND_1_SYSTEM_PROMPT`, `ROUND_2_PLUS_PROMPT` from `memory_agent.prompts`
- Produces: `run_agent_loop(config, user_query, tools=None, max_iterations=50, confirm_callback=None) -> str`
  - `memory_context` parameter removed

- [ ] **Step 1: Update test_agent_loop.py for new signature and dynamic prompts**

The tests need to work with the new `run_agent_loop` signature (no `memory_context` parameter) and verify dynamic prompt switching.

Replace `tests/test_agent_loop.py`:

```python
from unittest.mock import patch, MagicMock
from memory_agent.config import Config
from memory_agent.agent_loop import run_agent_loop


def make_config(**kwargs):
    defaults = {
        "llm_api_base": "https://test.com/v1", "llm_api_key": "sk-test", "llm_model": "test-model",
        "embedding_api_base": "https://test.com/v1", "embedding_api_key": "sk-test", "embedding_model": "test-embed",
        "retrieval_top_k": 10, "retrieval_similarity_threshold": 0.5,
    }
    defaults.update(kwargs)
    return Config(**defaults)


class TestAgentLoop:
    def test_simple_text_response(self):
        cfg = make_config()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Hello!", "tool_calls": None}}]
            }
            transcript = run_agent_loop(config=cfg, user_query="Hi")
            assert "Hello" in transcript

    def test_tool_call_loop(self):
        cfg = make_config()
        call_count = [0]

        def fake_json():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path":"t.txt"}'}}
                ]}}]}
            return {"choices": [{"message": {"content": "File contents: hello", "tool_calls": None}}]}

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.side_effect = fake_json
            with patch("memory_agent.tools.tool_read_file", return_value="hello"):
                transcript = run_agent_loop(config=cfg, user_query="Read t.txt")
            assert call_count[0] == 2

    def test_respects_max_iterations(self):
        cfg = make_config()
        def fake_json():
            return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path":"t.txt"}'}}
            ]}}]}
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.side_effect = fake_json
            with patch("memory_agent.tools.tool_read_file", return_value="contents"):
                transcript = run_agent_loop(config=cfg, user_query="Read", max_iterations=5)
            assert "Max tool call iterations" in transcript

    def test_round1_prompt_used_on_first_iteration(self):
        """Verify ROUND_1_SYSTEM_PROMPT is sent on the first API call."""
        from memory_agent.prompts import ROUND_1_SYSTEM_PROMPT

        cfg = make_config()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Done", "tool_calls": None}}]
            }
            run_agent_loop(config=cfg, user_query="test")

            # Verify first call used ROUND_1 prompt
            call_args = mock_post.call_args
            req_body = call_args[1]["json"]
            messages = req_body["messages"]
            system_msg = messages[0]["content"]
            assert "Before diving into the task" in system_msg

    def test_round2_prompt_appended_after_first_iteration(self):
        """Verify ROUND_2_PLUS_PROMPT is appended from the second iteration."""
        from memory_agent.prompts import ROUND_2_PLUS_PROMPT

        cfg = make_config()
        call_count = [0]
        captured_system = []

        def fake_json():
            call_count[0] += 1
            return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path":"t.txt"}'}}
            ]}}]}

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.side_effect = fake_json

            with patch("memory_agent.tools.tool_read_file", return_value="content"):
                run_agent_loop(config=cfg, user_query="Read", max_iterations=3)

            # Check second call used ROUND_2 prompt
            assert mock_post.call_count >= 2
            second_call = mock_post.call_args_list[1]
            req_body = second_call[1]["json"]
            messages = req_body["messages"]
            system_msg = messages[0]["content"]
            assert "Continue working on the user's task" in system_msg
```

- [ ] **Step 2: Run tests to verify they FAIL**

Run: `pytest tests/test_agent_loop.py::TestAgentLoop::test_round1_prompt_used_on_first_iteration tests/test_agent_loop.py::TestAgentLoop::test_round2_prompt_appended_after_first_iteration -v`
Expected: FAIL — tests reference new prompt content that isn't used yet

- [ ] **Step 3: Update run_agent_loop in agent_loop.py**

Apply these changes to `src/memory_agent/agent_loop.py`:

**Change 1 — Update imports (line 11):**
```python
# Replace:
from memory_agent.prompts import BASE_AGENT_SYSTEM_PROMPT
# With:
from memory_agent.prompts import BASE_AGENT_SYSTEM_PROMPT, ROUND_1_SYSTEM_PROMPT, ROUND_2_PLUS_PROMPT
```

**Change 2 — Update function signature (lines 20-27):**
```python
# Replace:
def run_agent_loop(
    config: Config,
    user_query: str,
    memory_context: str,
    tools: list[dict] | None = None,
    max_iterations: int = 50,
    confirm_callback: ConfirmCallback | None = None,
) -> str:
    if tools is None:
        tools = TOOL_DEFINITIONS

    system_content = BASE_AGENT_SYSTEM_PROMPT
    if memory_context:
        system_content += "\n\n" + memory_context

# With:
def run_agent_loop(
    config: Config,
    user_query: str,
    tools: list[dict] | None = None,
    max_iterations: int = 50,
    confirm_callback: ConfirmCallback | None = None,
) -> str:
    if tools is None:
        tools = TOOL_DEFINITIONS

    # Start with Round 1 prompt; it will be updated to Round 2+ after first iteration
    system_content = ROUND_1_SYSTEM_PROMPT
```

**Change 3 — Add dynamic prompt switching inside the loop (after `for _ in range(max_iterations):`, line 42):**
```python
    for iteration in range(max_iterations):
        # --- dynamic prompt switching ---
        if iteration >= 1:
            messages[0]["content"] = ROUND_1_SYSTEM_PROMPT + "\n\n" + ROUND_2_PLUS_PROMPT

        url = f"{config.llm_api_base}/chat/completions"
        # ... rest of loop unchanged
```

The rest of the loop body (API call, tool dispatch) remains unchanged.

- [ ] **Step 4: Run agent loop tests to verify PASS**

Run: `pytest tests/test_agent_loop.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/agent_loop.py tests/test_agent_loop.py
git commit -m "feat: dynamic system prompt switching + simplified agent loop signature

- Remove memory_context parameter (no more pre-loop injection)
- ROUND_1_SYSTEM_PROMPT on first iteration
- ROUND_2_PLUS_PROMPT appended from iteration 2+

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Update cli.py pipeline — remove pre-loop injection, wire session state

**Files:**
- Modify: `src/memory_agent/cli.py`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: `reset_session_state`, `pre_index_skills` from `memory_agent.tools`
- Consumes: `run_agent_loop` with new signature (no `memory_context`) from `memory_agent.agent_loop`
- Produces: Updated `run_pipeline()` that resets state, pre-indexes skills, and passes no memory_context to agent loop

- [ ] **Step 1: Update test_integration.py for new flow**

Replace `tests/test_integration.py`:

```python
"""Smoke test: full pipeline with tool-driven retrieval — no pre-loop injection."""
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone
from memory_agent.config import load_config, Config
from memory_agent.storage import MemoryStore
from memory_agent.agent_loop import run_agent_loop
from memory_agent.extractor import extract_and_store
from memory_agent.tools import reset_session_state


def setup_config(temp_project: Path) -> Config:
    config_dir = temp_project / ".agent-memory"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("""
llm:
  api_base: https://api.deepseek.com/v1
  api_key: sk-test
  model: deepseek-chat
embedding:
  api_base: https://api.openai.com/v1
  api_key: sk-test
  model: text-embedding-3-small
retrieval:
  top_k: 5
  similarity_threshold: 0.5
extractor:
  auto_confirm: true
  keep_full_transcript: false
""")
    return load_config(temp_project)


class TestAgentLoopToolDriven:
    """Agent loop now receives no pre-injected memory_context."""

    def test_agent_loop_no_memory_context_parameter(self, temp_project):
        cfg = setup_config(temp_project)

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Task completed.", "tool_calls": None}}]
            }
            # New signature: no memory_context parameter
            transcript = run_agent_loop(config=cfg, user_query="Simple task")
            assert "Task completed" in transcript

    def test_agent_loop_uses_round1_prompt(self, temp_project):
        cfg = setup_config(temp_project)

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "OK", "tool_calls": None}}]
            }
            run_agent_loop(config=cfg, user_query="test")

            call_args = mock_post.call_args
            req_body = call_args[1]["json"]
            system_msg = req_body["messages"][0]["content"]
            assert "Before diving into the task" in system_msg
            assert "search_memory" in system_msg
            assert "search_skills" in system_msg

    def test_session_state_reset(self, temp_project):
        """reset_session_state clears dedup sets."""
        from memory_agent.tools import _returned_memory_ids, _returned_skill_names

        _returned_memory_ids.add("test-1")
        _returned_skill_names.add("test-skill")

        reset_session_state()

        assert len(_returned_memory_ids) == 0
        assert len(_returned_skill_names) == 0


class TestExtractionStillWorks:
    """Memory extraction after agent loop is unchanged."""

    def test_extract_after_agent_loop(self, temp_project):
        cfg = setup_config(temp_project)
        store = MemoryStore(cfg.memory_dir / "memories.db")
        store.init_schema()

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()

            call_count = [0]
            def fake_json():
                call_count[0] += 1
                if call_count[0] == 1:
                    return {"choices": [{"message": {"content": "Done.", "tool_calls": None}}]}
                else:
                    return {"choices": [{"message": {"content": '{"summary":"Task done","key_points":["Done"],"tags":["test"],"entities":[],"decisions":[]}'}}]}

            mock_post.return_value.json.side_effect = fake_json

            with patch.object(store, "init_chroma"):
                store._chroma_collection = MagicMock()
                store._chroma_collection.count.return_value = 0
                store._get_embedding = MagicMock(return_value=[0.0] * 1536)

                transcript = run_agent_loop(config=cfg, user_query="Do something")

                result = extract_and_store(transcript=transcript, config=cfg, store=store)
                assert result is True
```

- [ ] **Step 2: Run integration tests to verify FAIL**

Run: `pytest tests/test_integration.py -v`
Expected: Some tests PASS (the new ones), `test_first_conversation_no_prior_memories` and `test_second_conversation_finds_previous` FAIL — they still call `run_agent_loop` with `memory_context=` kwarg

- [ ] **Step 3: Update cli.py run_pipeline**

In `src/memory_agent/cli.py`, make these changes:

**Change 1 — Add import (after existing imports, around line 20):**
```python
from memory_agent.tools import set_workspace_root, reset_session_state, pre_index_skills
```

Replace the existing `from memory_agent.tools import set_workspace_root` line.

**Change 2 — Replace the pre-loop retrieval block (lines 127-166):**

Replace:
```python
    # Step 1: Memory Retrieval + Skill Routing
    memory_context = ""
    skill_context = ""
    injected_memories: list[dict] = []

    if not skip_memory:
        # Memory retrieval
        print("Checking memory...", file=sys.stderr)
        try:
            retriever = Retriever(config, store)
            injected_memories, memory_context = retriever.retrieve(user_query)
            if memory_context:
                print(f"  Injected {len(injected_memories)} memory/memories.", file=sys.stderr)
        except Exception as e:
            print(f"  Memory retrieval failed: {e}", file=sys.stderr)

        # Skill routing (embedding-based)
        try:
            from memory_agent.skills import SkillRouter, discover_skills, format_skills_for_injection

            skills = discover_skills(config.memory_dir.parent)
            if skills:
                router = SkillRouter(
                    chroma_dir=config.memory_dir / "chroma",
                    embedding_api_base=config.embedding_api_base,
                    embedding_api_key=config.embedding_api_key,
                    embedding_model=config.embedding_model,
                )
                router.index_skills(skills)
                matched = router.search(user_query, top_k=3)
                if matched:
                    skill_context = format_skills_for_injection(matched)
                    print(f"  Matched {len(matched)} skill(s).", file=sys.stderr)
        except Exception as e:
            print(f"  Skill routing failed: {e}", file=sys.stderr)

    # Combine contexts for system prompt
    combined_context = memory_context
    if skill_context:
        combined_context = skill_context + ("\n\n" + memory_context if memory_context else "")
```

With:
```python
    # Step 1: Initialize session state + pre-index skills (no prompt injection)
    injected_memories: list[dict] = []
    combined_context = ""

    reset_session_state()

    if not skip_memory:
        try:
            pre_index_skills(config)
        except Exception as e:
            print(f"  Skill indexing failed: {e}", file=sys.stderr)
```

**Change 3 — Update agent loop call (line 177):**

Replace:
```python
    transcript = run_agent_loop(
        config=config,
        user_query=user_query,
        memory_context=combined_context,
        confirm_callback=_bash_confirm if interactive else None,
    )
```

With:
```python
    transcript = run_agent_loop(
        config=config,
        user_query=user_query,
        confirm_callback=_bash_confirm if interactive else None,
    )
```

**Change 4 — Remove unused import (line 17):**

Remove the `Retriever` import if it's no longer used in `run_pipeline`:
```python
# Remove this line:
from memory_agent.retriever import Retriever
```

But check: `Retriever` is still imported at line 17. It's no longer used in `run_pipeline`. Remove it from the imports.

- [ ] **Step 4: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/memory_agent/cli.py tests/test_integration.py
git commit -m "feat: remove pre-loop injection, wire session state into pipeline

- Remove memory/skill retrieval injection from run_pipeline
- Add reset_session_state() at pipeline start
- Add pre_index_skills() to warm skill index (no prompt injection)
- Update run_agent_loop calls for new signature (no memory_context)
- Remove unused Retriever import from cli.py

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Execution Order

```
Task 1 (prompts.py) ─────────────────────────────────────────┐
                                                              │
Task 2 (tools.py + tests) ────────────────────────────────────┤
                                                              │
                              ┌─ Task 3 (agent_loop + tests) ─┤
                              │                               │
                              └─ Task 4 (cli.py + integration)┘
```

- Task 1: No dependencies, safe to start immediately
- Task 2: No dependencies on other tasks, safe to start in parallel with Task 1
- Task 3: Depends on Task 1 (needs new prompt constants)
- Task 4: Depends on Task 2 (needs `reset_session_state`, `pre_index_skills`) and Task 3 (new `run_agent_loop` signature)

Recommended execution: Task 1 → (Task 2 || Task 3) → Task 4

## Verification Checklist

After all tasks are complete:

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `python -c "from memory_agent.tools import TOOL_DEFINITIONS; names = [t['function']['name'] for t in TOOL_DEFINITIONS]; assert 'search_skills' in names; assert 'load_skill' not in names; assert 'list_skills' not in names; print('OK')"` — only 8 tools exposed
- [ ] `python -c "from memory_agent.agent_loop import run_agent_loop; import inspect; sig = inspect.signature(run_agent_loop); assert 'memory_context' not in sig.parameters; print('OK')"` — memory_context removed
- [ ] `python -c "from memory_agent.prompts import ROUND_1_SYSTEM_PROMPT, ROUND_2_PLUS_PROMPT; assert 'search_memory' in ROUND_1_SYSTEM_PROMPT; assert 'search_skills' in ROUND_1_SYSTEM_PROMPT; print('OK')"` — new prompts reference tools
- [ ] `python -c "from memory_agent.tools import reset_session_state, pre_index_skills; print('OK')"` — public API available

---

## Plan Self-Review

**1. Spec coverage:**
- Pre-loop injection removal → Task 4 ✅
- `search_skills` tool with full content → Task 2 ✅
- `load_skill` / `list_skills` internal only → Task 2 (not in TOOL_DEFINITIONS) ✅
- Session-level dedup → Task 2 ✅
- ROUND_1 prompt with analysis guidance → Task 1 ✅
- ROUND_2+ prompt with light reminder → Task 1 ✅
- Dynamic prompt switching → Task 3 ✅
- Agent loop signature simplified → Task 3 ✅
- `pre_index_skills` → Task 2 ✅
- `reset_session_state` → Task 2 ✅
- `--no-memory` retained → Task 4 (skip pre_index on --no-memory) ✅
- Edge case: no skills installed → tested in Task 2 ✅
- Edge case: all filtered by dedup → tested in Task 2 ✅

**2. Placeholder scan:** No TBD, TODO, or vague references. All code is concrete.

**3. Type consistency:**
- `reset_session_state() -> None` — consistent across Task 2 and Task 4
- `pre_index_skills(config: Config) -> None` — consistent across Task 2 and Task 4
- `run_agent_loop(config, user_query, tools=None, max_iterations=50, confirm_callback=None)` — consistent across Task 3 and Task 4
- `tool_search_skills(query: str, top_k: int = 3) -> str` — defined in Task 2, registered in TOOL_DEFINITIONS and TOOL_EXECUTORS
