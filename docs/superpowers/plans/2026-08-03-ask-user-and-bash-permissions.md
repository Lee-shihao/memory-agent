# ask_user Tool & Bash Permission Tiering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `ask_user` tool for agent-to-user questions and implement three-tier bash permission system (safe/dangerous/unknown).

**Architecture:** Extend the existing `ConfirmCallback` pattern in `agent_loop.py` — the callback in `cli.py` handles both `ask_user` (user interaction) and `run_bash` (command classification + confirmation). New constants and classification logic go in `tools.py`. Single-shot mode now also uses the callback.

**Tech Stack:** Python 3.10+, existing `select`/`subprocess`/`readline` patterns.

## Global Constraints

- Follow existing code patterns in the project (ConfirmCallback, subprocess.run, select.select for stdin)
- No new dependencies
- No config-driven command lists (hardcoded constants)
- No changes to `write_file`/`edit_file` permissions
- No changes to `git_ops` allowlist
- All terminal I/O stays in `cli.py`

---

### Task 1: Add Bash Command Classification in tools.py

**Files:**
- Modify: `src/memory_agent/tools.py` (add after `_workspace_root` block, before `_resolve_path`)

**Interfaces:**
- Produces: `SAFE_BASH_COMMANDS: set[str]`, `DANGEROUS_BASH_COMMANDS: set[str]`, `classify_bash_command(command: str) -> str`

- [ ] **Step 1: Add SAFE_BASH_COMMANDS and DANGEROUS_BASH_COMMANDS constants**

Add after line 46 (`_workspace_root = Path.cwd()`):

```python
# ── bash command classification ────────────────────────────────────────────────

SAFE_BASH_COMMANDS: set[str] = {
    # Read / view
    "ls", "cat", "file", "head", "tail", "less", "more",
    # Search / count
    "find", "grep", "wc", "stat", "du", "df", "sort", "uniq",
    # Info
    "pwd", "which", "type", "env", "printenv", "uname", "whoami",
    "date", "id", "hostname", "tree",
    # Text processing (read-only usage assumed)
    "awk", "sed", "cut", "tr", "tee",
    # No-side-effect ops
    "echo", "true", "false", "diff", "cmp", "dirname", "basename",
    "realpath", "readlink", "xargs",
    # Lightweight create ops (non-destructive)
    "mkdir", "touch",
}

DANGEROUS_BASH_COMMANDS: set[str] = {
    # Delete / overwrite
    "rm", "rmdir", "dd",
    # Permissions / privilege
    "chmod", "chown", "chgrp", "sudo", "su",
    # Process signals
    "kill", "killall", "pkill",
    # System management
    "shutdown", "reboot", "halt", "systemctl", "service",
    "mount", "umount", "mkfs", "fdisk",
    # Package managers
    "apt", "apt-get", "yum", "dnf", "pacman",
    "pip", "pip3", "npm", "yarn", "npx", "cargo", "go",
    # Network download (often piped to shell)
    "curl", "wget",
    # Remote access / sync
    "ssh", "scp", "rsync",
    # Eval / exec
    "eval", "exec", "source",
}
```

- [ ] **Step 2: Add classify_bash_command function**

Add after the constants:

```python
def classify_bash_command(command: str) -> str:
    """Classify a bash command as 'safe', 'dangerous', or 'unknown'.

    Checks the base command and special patterns:
      - 'git push', 'git fetch', 'git pull' → dangerous
      - 'curl ... | sh', 'wget ... | bash' → dangerous
      - Other git subcommands → safe (git has its own tool)
    """
    stripped = command.strip()
    if not stripped:
        return "safe"  # empty command is harmless

    # Check for dangerous pipe patterns: curl/wget piped to sh/bash
    if re.search(r"(curl|wget)\s+.*\|\s*(sh|bash)", stripped):
        return "dangerous"

    # Extract base command
    parts = stripped.split()
    base = parts[0]

    # Check multi-word prefixes (e.g. "git push")
    if base == "git" and len(parts) > 1:
        sub = parts[1]
        if sub in ("push", "fetch", "pull"):
            return "dangerous"
        return "safe"  # git is handled by git_ops tool, allow here for scripting

    if base in DANGEROUS_BASH_COMMANDS:
        return "dangerous"
    if base in SAFE_BASH_COMMANDS:
        return "safe"
    return "unknown"
```

- [ ] **Step 3: Run existing tests to verify no regressions**

```bash
python -m pytest tests/test_tools.py -v
```

Expected: all existing tests pass (new constants/functions are not imported by existing tests).

- [ ] **Step 4: Commit**

```bash
git add src/memory_agent/tools.py
git commit -m "feat: add bash command classification (safe/dangerous/unknown)

Add SAFE_BASH_COMMANDS and DANGEROUS_BASH_COMMANDS constants and
classify_bash_command() function for three-tier bash permission system.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add ask_user Tool Definition and Executor in tools.py

**Files:**
- Modify: `src/memory_agent/tools.py` (add `tool_ask_user` executor; add tool definition to `TOOL_DEFINITIONS`; add executor to `TOOL_EXECUTORS`)

**Interfaces:**
- Produces: `tool_ask_user(question: str, header: str, options: list[dict] | None = None, multi_select: bool = False) -> str`
- Modifies: `TOOL_DEFINITIONS` list (append entry), `TOOL_EXECUTORS` dict (add key)

- [ ] **Step 1: Add tool_ask_user executor function**

Add after `_list_skills` (around line 375), before the tool registry section:

```python
# ── ask_user ──────────────────────────────────────────────────────────────────

def tool_ask_user(
    question: str,
    header: str,
    options: list[dict] | None = None,
    multi_select: bool = False,
) -> str:
    """Ask the user a question during the agent loop.

    This executor is a pass-through — actual user interaction happens
    in the confirm callback (cli.py), which intercepts ask_user calls
    before this runs. If this executor is reached (no callback / non-interactive),
    it returns a timeout/default response.
    """
    if options:
        # Return first option as default when no interactive handler
        selected = [options[0]["label"]] if not multi_select else [o["label"] for o in options[:1]]
        return f"[auto-selected] {', '.join(selected)}"
    return ""
```

- [ ] **Step 2: Add ask_user to TOOL_DEFINITIONS list**

Add after the `run_bash` entry (after line 556), before `search_memory`:

```python
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user for input when you lack critical information "
                "or need to choose between approaches. Use for clarifying "
                "requirements, requesting feedback, or selecting from options. "
                "Supports multiple choice (2-4 options, single or multi-select) "
                "and open-ended questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The complete question to ask the user",
                    },
                    "header": {
                        "type": "string",
                        "description": "Short category label (max 12 chars), e.g. 'Approach', 'Library'",
                    },
                    "options": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "Short label for this option (1-5 words)",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "What this option means or what will happen if chosen",
                                },
                            },
                            "required": ["label", "description"],
                        },
                        "description": "2-4 predefined choices. Omit for open-ended questions.",
                    },
                    "multi_select": {
                        "type": "boolean",
                        "description": "Allow multiple selections (default false). Only valid when options is provided.",
                    },
                },
                "required": ["question", "header"],
            },
        },
    },
```

- [ ] **Step 3: Add tool_ask_user to TOOL_EXECUTORS**

Add after `"search_skills": tool_search_skills,` (around line 613):

```python
    "ask_user": tool_ask_user,
```

- [ ] **Step 4: Run existing tests**

```bash
python -m pytest tests/test_tools.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/tools.py
git commit -m "feat: add ask_user tool definition and executor

Add tool_ask_user with support for multiple choice (single/multi-select)
and open-ended questions. Actual user interaction handled in cli.py
confirm callback.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Refactor Confirm Callback and Add Interactive Handlers in cli.py

**Files:**
- Modify: `src/memory_agent/cli.py`

**Interfaces:**
- Consumes: `classify_bash_command` from `tools.py`, `tool_ask_user` signature
- Produces: `_tool_confirm(tool_name: str, arguments: dict) -> tuple[bool, str]`
- Modifies: `run_pipeline` (use new callback name), `_interactive_loop` (use new callback name)

- [ ] **Step 1: Rename _bash_confirm to _tool_confirm and add run_bash classification**

Replace the existing `_bash_confirm` function (lines 69-114) with:

```python
# ── tool confirmation ────────────────────────────────────────────────────────

_ASK_USER_TIMEOUT = 60      # seconds — timeout for ask_user
_DANGEROUS_TIMEOUT = None   # no timeout for dangerous commands
_UNKNOWN_TIMEOUT = 30       # seconds — timeout for unknown commands


def _present_ask_user(arguments: dict) -> str:
    """Present an ask_user question to the user and collect their response.

    Returns the user's response as a string (to be used as tool result).
    """
    question = arguments.get("question", "")
    header = arguments.get("header", "Question")
    options = arguments.get("options")
    multi_select = arguments.get("multi_select", False)

    # Build display
    print(f"\n  ❓ {header}", file=sys.stderr)
    print(f"  {question}", file=sys.stderr)

    if options:
        for i, opt in enumerate(options, 1):
            label = opt.get("label", "?")
            desc = opt.get("description", "")
            print(f"  [{i}] {label}: {desc}", file=sys.stderr)
        if multi_select:
            print(f"  Enter numbers (e.g. 1,3) or type custom", file=sys.stderr, end="")
        else:
            print(f"  Enter number or type custom", file=sys.stderr, end="")
    else:
        print(f"  Type your response", file=sys.stderr, end="")

    print(f"  ({_ASK_USER_TIMEOUT}s timeout)", file=sys.stderr, flush=True)

    # Wait for input
    ready, _, _ = select.select([sys.stdin], [], [], _ASK_USER_TIMEOUT)

    if not ready:
        print("  ⏰ (timeout, using default)", file=sys.stderr)
        if options:
            if multi_select:
                return f"[Selected] {options[0]['label']}"
            return f"[Selected] {options[0]['label']}"
        return ""

    try:
        user_input = sys.stdin.readline().strip()
    except (EOFError, OSError):
        user_input = ""

    print(file=sys.stderr)

    if not user_input:
        if options:
            return f"[Selected] {options[0]['label']}"
        return ""

    # Try to parse as option numbers
    if options:
        parts = user_input.replace(",", " ").split()
        numbers = []
        for p in parts:
            try:
                n = int(p)
                if 1 <= n <= len(options):
                    numbers.append(n)
            except ValueError:
                pass
        if numbers:
            selected_labels = [options[n - 1]["label"] for n in numbers]
            if multi_select:
                return f"[Selected] {', '.join(selected_labels)}"
            return f"[Selected] {selected_labels[0]}"

    # Free text response
    return user_input


def _tool_confirm(tool_name: str, arguments: dict) -> tuple[bool, str]:
    """Confirm tool execution. Handles ask_user and run_bash specially.

    Returns (allowed, feedback) where:
      - allowed=True  → execute tool, append feedback (if any) to result
      - allowed=False → skip tool, use feedback as tool result
    """

    # --- ask_user: handle entirely here, block execution ---
    if tool_name == "ask_user":
        result = _present_ask_user(arguments)
        return False, result  # block execution, use result as tool response

    # --- run_bash: classify and confirm ---
    if tool_name == "run_bash":
        from memory_agent.tools import classify_bash_command

        command = arguments.get("command", "")
        tier = classify_bash_command(command)

        if tier == "safe":
            # Silent auto-allow
            return True, ""

        # Display prompt for dangerous/unknown
        tool_timeout = arguments.get("timeout", 120)
        display_cmd = command if len(command) <= 200 else command[:197] + "..."

        if tier == "dangerous":
            print(f"\n  ⚠️  run_bash [DANGEROUS]", file=sys.stderr)
        else:
            print(f"\n  🔧 run_bash  (timeout: {tool_timeout}s)", file=sys.stderr)

        print(f"  {display_cmd}", file=sys.stderr)

        if tier == "dangerous":
            print(
                f"  [y] allow  [n] deny  or type feedback (e.g. 'n use mv instead')",
                file=sys.stderr, end="", flush=True,
            )
            timeout = _DANGEROUS_TIMEOUT
        else:
            print(
                f"  [y] allow  [n] deny  or type feedback  "
                f"({_UNKNOWN_TIMEOUT}s timeout → auto-allow)",
                file=sys.stderr, end="", flush=True,
            )
            timeout = _UNKNOWN_TIMEOUT

        # Wait for input
        ready, _, _ = select.select([sys.stdin], [], [], timeout)

        if not ready:
            if tier == "dangerous":
                print("  ⛔ (timeout — dangerous command skipped)", file=sys.stderr)
                return False, "Blocked: dangerous command timed out without confirmation"
            else:
                print("  ⏰ (timeout, auto-allowed)", file=sys.stderr)
                return True, ""

        try:
            user_input = sys.stdin.readline().strip()
        except (EOFError, OSError):
            user_input = ""

        print(file=sys.stderr)

        if not user_input:
            return True, ""

        lowered = user_input.lower()
        if lowered in ("y", "yes"):
            return True, ""
        elif lowered.startswith("n") and (len(lowered) == 1 or lowered[1:].startswith("o") or lowered[1] == " "):
            # "n", "no", or "n <feedback>"
            if lowered in ("n", "no"):
                return False, ""
            # "n <feedback>" — deny with guidance
            feedback = user_input[1:].strip()
            return False, f"[Denied] {feedback}"
        else:
            return True, user_input  # free text → allow with feedback

    # All other tools: allow
    return True, ""
```

- [ ] **Step 2: Update references from _bash_confirm to _tool_confirm**

In `run_pipeline` (line 181), change:
```python
confirm_callback=_bash_confirm if interactive else None,
```
to:
```python
confirm_callback=_tool_confirm,
```

This enables the confirm callback in both interactive and single-shot mode.

- [ ] **Step 3: Update _interactive_loop to pass confirm callback**

In `_interactive_loop` (line 263), the call to `run_pipeline` already passes `interactive=True`. After step 2, confirm_callback is always `_tool_confirm`, so no change needed here.

- [ ] **Step 4: Run existing tests**

```bash
python -m pytest tests/ -v --ignore=tests/test_integration.py
```

Expected: all unit tests pass (confirm callback change doesn't affect test path).

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/cli.py
git commit -m "feat: refactor confirm callback with ask_user and bash tiering

Rename _bash_confirm to _tool_confirm with three behaviors:
- ask_user: present question UI, collect user response
- run_bash: classify as safe/dangerous/unknown, confirm accordingly
  - safe: auto-allow, no prompt
  - dangerous: mandatory confirm, no timeout, supports 'n <feedback>'
  - unknown: confirm with 30s timeout
- All other tools: auto-allow

Enable confirm callback in single-shot mode (was previously disabled).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Update System Prompts

**Files:**
- Modify: `src/memory_agent/prompts.py`

**Interfaces:**
- Consumes: `ask_user` tool name and behavior
- Modifies: `ROUND_1_SYSTEM_PROMPT`, `ROUND_2_PLUS_PROMPT`

- [ ] **Step 1: Update ROUND_1_SYSTEM_PROMPT to mention ask_user**

Replace the existing `ROUND_1_SYSTEM_PROMPT` (lines 87-106) with:

```python
ROUND_1_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools. You can read files,
write files, execute shell commands, and ask the user questions to help
accomplish tasks.

## Before You Start

Before diving into the task, analyze what the user is asking:

1. **Need past conversation context?**
   If the user references previous work, past discussions, or prior decisions,
   call search_memory(query) with specific search terms to find relevant memories.

2. **Need specialized skills or workflows?**
   Call search_skills(query) to find matching skills with their full instructions.

3. **Need user input or clarification?**
   Call ask_user(question, header, [options]) when you:
   - Lack critical information to proceed
   - Need to choose between multiple valid approaches
   - Are unsure about the user's requirements or preferences
   - Need feedback on a decision before continuing

4. **Simple, self-contained tasks** (e.g., "write hello world", "what is 2+2")
   can be executed directly — skip retrieval and questions.

Work step by step. When done, provide a clear summary of what was accomplished.
"""
```

- [ ] **Step 2: Update ROUND_2_PLUS_PROMPT to mention ask_user**

Replace existing `ROUND_2_PLUS_PROMPT` (lines 108-115) with:

```python
ROUND_2_PLUS_PROMPT = """\
Continue working on the user's task. Use the context you've already retrieved
from tool calls earlier in this conversation.

If you discover gaps and need more:
- search_memory(query) — search past conversations
- search_skills(query) — find additional skills with full instructions
- ask_user(question, header, [options]) — ask the user for input when
  you need clarification, choices, or feedback
"""
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -v --ignore=tests/test_integration.py
```

Expected: all pass (prompts are imported but tests don't assert on their content).

- [ ] **Step 4: Commit**

```bash
git add src/memory_agent/prompts.py
git commit -m "feat: add ask_user guidance to system prompts

Update Round 1 and Round 2+ prompts to mention ask_user as an
available tool for requesting user input during the agent loop.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Write Tests

**Files:**
- Modify: `tests/test_tools.py` (add test classes)

**Interfaces:**
- Consumes: `classify_bash_command`, `SAFE_BASH_COMMANDS`, `DANGEROUS_BASH_COMMANDS`, `tool_ask_user`, `TOOL_DEFINITIONS`
- Produces: test coverage for new functionality

- [ ] **Step 1: Add tests for classify_bash_command**

Add to end of `tests/test_tools.py`:

```python
class TestClassifyBashCommand:
    """Tests for bash command classification."""

    def test_safe_commands(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("ls") == "safe"
        assert classify_bash_command("ls -la") == "safe"
        assert classify_bash_command("cat file.txt") == "safe"
        assert classify_bash_command("grep pattern file") == "safe"
        assert classify_bash_command("find . -name '*.py'") == "safe"
        assert classify_bash_command("pwd") == "safe"
        assert classify_bash_command("echo hello") == "safe"

    def test_dangerous_commands(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("rm file.txt") == "dangerous"
        assert classify_bash_command("rm -rf /") == "dangerous"
        assert classify_bash_command("sudo ls") == "dangerous"
        assert classify_bash_command("chmod 777 file") == "dangerous"
        assert classify_bash_command("kill 1234") == "dangerous"
        assert classify_bash_command("pip install requests") == "dangerous"
        assert classify_bash_command("npm install") == "dangerous"
        assert classify_bash_command("curl http://example.com") == "dangerous"
        assert classify_bash_command("ssh user@host") == "dangerous"
        assert classify_bash_command("eval 'ls'") == "dangerous"

    def test_dangerous_pipe_patterns(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("curl http://example.com | sh") == "dangerous"
        assert classify_bash_command("wget -qO- http://x | bash") == "dangerous"

    def test_git_subcommands(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("git push") == "dangerous"
        assert classify_bash_command("git pull") == "dangerous"
        assert classify_bash_command("git fetch") == "dangerous"
        assert classify_bash_command("git status") == "safe"
        assert classify_bash_command("git diff") == "safe"
        assert classify_bash_command("git log") == "safe"

    def test_unknown_commands(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("python script.py") == "unknown"
        assert classify_bash_command("make build") == "unknown"
        assert classify_bash_command("pytest tests/") == "unknown"
        assert classify_bash_command("node index.js") == "unknown"

    def test_empty_command(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("") == "safe"
        assert classify_bash_command("   ") == "safe"

    def test_all_safe_commands_in_set(self):
        """Verify all entries in SAFE_BASH_COMMANDS are classified as safe."""
        from memory_agent.tools import SAFE_BASH_COMMANDS, classify_bash_command
        for cmd in SAFE_BASH_COMMANDS:
            assert classify_bash_command(cmd) == "safe", f"{cmd} should be safe"

    def test_all_dangerous_commands_in_set(self):
        """Verify all entries in DANGEROUS_BASH_COMMANDS are classified as dangerous."""
        from memory_agent.tools import DANGEROUS_BASH_COMMANDS, classify_bash_command
        for cmd in DANGEROUS_BASH_COMMANDS:
            assert classify_bash_command(cmd) == "dangerous", f"{cmd} should be dangerous"

    def test_safe_dangerous_no_overlap(self):
        """SAFE and DANGEROUS sets should have no overlap."""
        from memory_agent.tools import SAFE_BASH_COMMANDS, DANGEROUS_BASH_COMMANDS
        overlap = SAFE_BASH_COMMANDS & DANGEROUS_BASH_COMMANDS
        assert not overlap, f"Overlap found: {overlap}"
```

- [ ] **Step 2: Add tests for ask_user tool schema and executor**

Add after the above class:

```python
class TestAskUserTool:
    """Tests for the ask_user tool."""

    def test_tool_in_definitions(self):
        """ask_user should be registered in TOOL_DEFINITIONS."""
        from memory_agent.tools import TOOL_DEFINITIONS
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert "ask_user" in names

    def test_tool_in_executors(self):
        """ask_user should be registered in TOOL_EXECUTORS."""
        from memory_agent.tools import TOOL_EXECUTORS
        assert "ask_user" in TOOL_EXECUTORS

    def test_executor_with_options_returns_first_default(self):
        """With options but no interactive handler, returns first option."""
        from memory_agent.tools import tool_ask_user
        result = tool_ask_user(
            question="Which one?",
            header="Choice",
            options=[
                {"label": "Option A", "description": "First option"},
                {"label": "Option B", "description": "Second option"},
            ],
        )
        assert "Option A" in result
        assert "[auto-selected]" in result

    def test_executor_open_ended_returns_empty(self):
        """Open-ended question returns empty string in non-interactive mode."""
        from memory_agent.tools import tool_ask_user
        result = tool_ask_user(
            question="What do you think?",
            header="Feedback",
        )
        assert result == ""

    def test_tool_schema_has_required_fields(self):
        """Tool schema should require question and header."""
        from memory_agent.tools import TOOL_DEFINITIONS
        ask_user_def = None
        for t in TOOL_DEFINITIONS:
            if t["function"]["name"] == "ask_user":
                ask_user_def = t["function"]
                break
        assert ask_user_def is not None
        required = ask_user_def["parameters"].get("required", [])
        assert "question" in required
        assert "header" in required

    def test_tool_schema_options_max_items(self):
        """Options array should have minItems=2, maxItems=4."""
        from memory_agent.tools import TOOL_DEFINITIONS
        ask_user_def = None
        for t in TOOL_DEFINITIONS:
            if t["function"]["name"] == "ask_user":
                ask_user_def = t["function"]
                break
        assert ask_user_def is not None
        options_schema = ask_user_def["parameters"]["properties"]["options"]
        assert options_schema["minItems"] == 2
        assert options_schema["maxItems"] == 4
```

- [ ] **Step 3: Run the new tests**

```bash
python -m pytest tests/test_tools.py -v
```

Expected: all tests pass (new and existing).

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_integration.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools.py
git commit -m "test: add tests for bash classification and ask_user tool

Test classify_bash_command for safe/dangerous/unknown/git/pipe patterns.
Test ask_user tool schema (required fields, min/max items) and executor
default behavior in non-interactive mode.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Integration Verification

**Files:**
- Verify: `src/memory_agent/tools.py`, `src/memory_agent/cli.py`, `src/memory_agent/prompts.py`

- [ ] **Step 1: Verify imports work correctly**

```bash
python -c "
from memory_agent.tools import (
    SAFE_BASH_COMMANDS, DANGEROUS_BASH_COMMANDS, classify_bash_command,
    tool_ask_user, TOOL_DEFINITIONS, TOOL_EXECUTORS, execute_tool
)
print('SAFE commands:', len(SAFE_BASH_COMMANDS))
print('DANGEROUS commands:', len(DANGEROUS_BASH_COMMANDS))
print('classify ls:', classify_bash_command('ls'))
print('classify rm:', classify_bash_command('rm'))
print('classify python:', classify_bash_command('python'))
print('ask_user in definitions:', any(t['function']['name'] == 'ask_user' for t in TOOL_DEFINITIONS))
print('ask_user in executors:', 'ask_user' in TOOL_EXECUTORS)
result = execute_tool('ask_user', {'question': 'Test?', 'header': 'Test'})
print('execute ask_user:', repr(result))
print('All imports OK')
"
```

- [ ] **Step 2: Verify prompts reference ask_user**

```bash
python -c "
from memory_agent.prompts import ROUND_1_SYSTEM_PROMPT, ROUND_2_PLUS_PROMPT
assert 'ask_user' in ROUND_1_SYSTEM_PROMPT, 'ROUND_1 missing ask_user'
assert 'ask_user' in ROUND_2_PLUS_PROMPT, 'ROUND_2_PLUS missing ask_user'
print('Prompts OK')
"
```

- [ ] **Step 3: Quick manual test in interactive mode**

Run `python -m memory_agent` and verify:
1. Type a simple query like "hello" — should work as before
2. When agent would need to run bash, safe commands auto-allow

- [ ] **Step 4: Final commit (if any fixes needed)**

Only if integration verification found issues.
