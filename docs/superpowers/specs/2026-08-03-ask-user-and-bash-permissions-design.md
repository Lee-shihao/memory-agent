# Design: ask_user Tool & Bash Permission Tiering

**Date**: 2026-08-03
**Status**: Approved

## Overview

Two additions to the Memory Agent:

1. **`ask_user` tool** — Allows the agent to ask the user for input during the agent loop when it lacks critical information. Supports structured multiple choice (single/multi-select) and open-ended text input.

2. **Bash permission tiering** — Replaces the current all-or-nothing `run_bash` confirmation with a three-tier system: safe commands auto-allow, dangerous commands require mandatory confirmation, unknown commands require confirmation with timeout.

## Design

### 1. `ask_user` Tool

#### Schema

Added to `TOOL_DEFINITIONS` in `tools.py`:

- **name**: `ask_user`
- **parameters**:
  - `question` (required, string): The question to present to the user
  - `header` (required, string): Short category label, max 12 chars
  - `options` (optional, array of {label, description}): 2-4 predefined choices. Omit for open-ended question.
  - `multi_select` (optional, boolean, default false): Allow multiple selections. Only valid with `options`.

#### Execution Flow

1. Agent calls `ask_user(question, header, [options], [multi_select])`
2. Terminal displays the question with header
3. If `options` provided: show numbered list, user types number(s)
4. If no `options`: show prompt line, user types free text
5. `select.select()` on stdin with 60s timeout
6. On timeout: returns first option (for choices) or empty string (for free text)
7. On user response: returns selected option(s) or input text as tool result

#### Implementation

- New function `tool_ask_user` in `tools.py`
- The tool executor needs access to stdin — passed via a module-level callable (similar to the `ConfirmCallback` pattern) or handled directly in `cli.py`'s confirm callback, which intercepts `ask_user` calls before reaching `execute_tool`
- Actually, simpler approach: the `ConfirmCallback` in `agent_loop.py` already intercepts *before* execution. For `ask_user`, we handle user interaction in the callback (in `cli.py`) and block the tool execution, returning the user's response directly.

**Revised approach**: `ask_user` is handled entirely in the confirm callback, not in `execute_tool`. The callback:

1. Detects `tool_name == "ask_user"`
2. Presents the question to the user via terminal
3. Collects user response (or times out)
4. Returns `(False, user_response)` — blocks actual tool execution, uses feedback string as the tool result

This keeps all terminal I/O in `cli.py` and avoids threading/readline conflicts.

#### Timeout Behavior

| Question type | Timeout behavior |
|---|---|
| Multiple choice (single) | Returns first option |
| Multiple choice (multi) | Returns first option as selected |
| Open-ended | Returns empty string |

Timeout duration: 60 seconds.

### 2. Bash Permission Tiering

#### Command Classification

Three tiers defined as module-level sets in `tools.py`:

**SAFE_BASH_COMMANDS** — Read-only / informational, auto-allow:
`ls`, `cat`, `file`, `head`, `tail`, `less`, `more`, `find`, `grep`, `wc`, `stat`, `du`, `df`, `sort`, `uniq`, `pwd`, `which`, `type`, `env`, `printenv`, `uname`, `whoami`, `date`, `id`, `hostname`, `tree`, `awk`, `sed`, `cut`, `tr`, `tee`, `echo`, `true`, `false`, `diff`, `cmp`, `dirname`, `basename`, `realpath`, `readlink`, `xargs`, `mkdir`, `touch`

**DANGEROUS_BASH_COMMANDS** — Destructive / system-level, mandatory confirmation (no timeout):
`rm`, `rmdir`, `dd`, `chmod`, `chown`, `chgrp`, `sudo`, `su`, `kill`, `killall`, `pkill`, `shutdown`, `reboot`, `halt`, `systemctl`, `service`, `mount`, `umount`, `mkfs`, `fdisk`, `apt`, `apt-get`, `yum`, `dnf`, `pacman`, `pip`, `pip3`, `npm`, `yarn`, `npx`, `cargo`, `go`, `curl`, `wget`, `ssh`, `scp`, `rsync`, `eval`, `exec`, `source`

All other commands: **unknown** — confirmation with 30s timeout (auto-allow on timeout).

#### Classification Logic

```python
def classify_bash_command(command: str) -> str:
    """Classify a bash command as 'safe', 'dangerous', or 'unknown'."""
    # Extract base command (first word after stripping)
    # Check multi-word prefixes: "git push", "curl ... | sh"
    # Returns one of: "safe", "dangerous", "unknown"
```

Special patterns to detect:
- `git push`, `git fetch`, `git pull` — dangerous (modify remote)
- `curl ... | sh`, `wget ... | bash` — dangerous (remote execution)
- Pipes/redirects with dangerous commands

#### Confirmation Behavior by Tier

| Tier | Interactive mode | Single-shot mode | Timeout |
|---|---|---|---|
| Safe | Silent auto-allow | Silent auto-allow | N/A |
| Dangerous | Prompt [y/n/s], no timeout | Prompt [y/n/s], no timeout | No timeout |
| Unknown | Prompt [y/n], 30s timeout | Prompt [y/n], 30s timeout | Auto-allow |

#### Single-shot Mode Change

Previously: single-shot mode had `confirm_callback=None` — all bash commands executed without confirmation.

After change: single-shot mode also uses the confirm callback. Safe commands auto-allow without blocking, dangerous/unknown commands still prompt the user.

### 3. Confirm Callback Refactoring

Current `_bash_confirm` in `cli.py` only handles `run_bash`. It is renamed to `_tool_confirm` and handles:

1. `run_bash` → classify command, apply tier-appropriate confirmation
2. `ask_user` → present question, collect user response, return as tool result
3. All other tools → allow (return `True, ""`)

### Files Changed

| File | Changes |
|---|---|
| `src/memory_agent/tools.py` | Add `SAFE_BASH_COMMANDS`, `DANGEROUS_BASH_COMMANDS` constants; add `classify_bash_command()`; add `ask_user` tool definition; add `tool_ask_user` executor (simple pass-through, actual I/O in cli.py) |
| `src/memory_agent/cli.py` | Rename `_bash_confirm` → `_tool_confirm`; add `ask_user` handling with terminal UI; add bash command classification check; enable confirm callback in single-shot mode |
| `src/memory_agent/prompts.py` | Update `ROUND_1_SYSTEM_PROMPT` to mention `ask_user` as an available tool |

### Non-Goals

- No changes to `write_file` / `edit_file` permissions (stay as-is)
- No persistent "remember my choice" for permissions
- No sandboxing / containerization
- No config-driven command lists (hardcoded constants for simplicity)
- No changes to `git_ops` (already has its own allowlist)
