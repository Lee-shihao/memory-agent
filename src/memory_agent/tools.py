"""Built-in tools: file ops, search, git, skills."""
import fnmatch
import os
import re
import subprocess
from pathlib import Path

# ── session state (dedup tracking) ────────────────────────────────────────────

_returned_memory_ids: set[str] = set()
_returned_skill_names: set[str] = set()


def reset_session_state() -> None:
    """Reset dedup state at the beginning of each pipeline invocation."""
    global _returned_memory_ids, _returned_skill_names
    _returned_memory_ids.clear()
    _returned_skill_names.clear()


def pre_index_skills(config: "Config") -> None:
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

_workspace_root = Path.cwd()


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


def set_workspace_root(path: Path) -> None:
    global _workspace_root
    _workspace_root = path.resolve()


def _resolve_path(file_path: str) -> Path:
    """Resolve a file path to an absolute path.

    Tries multiple interpretations to handle LLM path confusion:
      1. Path as given (absolute or relative to workspace)
      2. If it starts with '/', try treating as relative (strip leading '/')
      3. Try './' + basename
    """
    p = Path(file_path)
    if p.is_absolute():
        if p.exists():
            return p
        # The LLM might have used '/' as root for a relative path.
        # Try stripping the leading '/' and resolving relative to workspace.
        rel = Path(str(p).lstrip("/"))
        candidate = _workspace_root / rel
        if candidate.exists():
            return candidate
        # Still return the original — error message will guide the LLM
        return p

    return _workspace_root / p


def _read_error(path: Path) -> str:
    """Generate a helpful error message for file-not-found."""
    rel = None
    try:
        rel = path.relative_to(_workspace_root)
    except ValueError:
        rel = path
    return (
        f"Error: File not found: {rel}\n"
        f"  Workspace root: {_workspace_root}\n"
        f"  Tip: use relative paths like 'src/main.py' not '/src/main.py'\n"
        f"  Tip: run 'ls' or 'find' first to discover the correct path"
    )


# ── read_file ─────────────────────────────────────────────────────────────────

def tool_read_file(file_path: str, offset: int = 0, limit: int | None = None) -> str:
    path = _resolve_path(file_path)
    if not path.exists():
        return _read_error(path)
    try:
        with open(path) as f:
            lines = f.readlines()
        total = len(lines)
        if limit is None:
            limit = total
        selected = lines[offset : offset + limit]
        result = "".join(selected)
        return f"File: {path} (lines {offset+1}-{min(offset+limit, total)} of {total})\n\n{result}"
    except Exception as e:
        return f"Error reading file: {e}"


# ── write_file ────────────────────────────────────────────────────────────────

def tool_write_file(file_path: str, content: str) -> str:
    path = _resolve_path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"File written: {path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error writing file: {e}"


# ── edit_file ─────────────────────────────────────────────────────────────────

def tool_edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Exact string replacement in a file. Fails if old_string is not unique."""
    path = _resolve_path(file_path)
    if not path.exists():
        return _read_error(path)
    try:
        content = path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"

    count = content.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path}"
    if count > 1 and not replace_all:
        return (
            f"Error: old_string appears {count} times in {path}. "
            f"Use replace_all=true to replace all occurrences, "
            f"or make old_string more specific."
        )

    new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)

    try:
        path.write_text(new_content)
    except Exception as e:
        return f"Error writing file: {e}"

    replaced = count if replace_all else 1
    return f"File edited: {path} ({replaced} replacement(s))"


# ── grep_files ────────────────────────────────────────────────────────────────

def tool_grep_files(
    pattern: str,
    path: str = ".",
    include: str = "*",
    recursive: bool = True,
    ignore_case: bool = False,
    max_results: int = 50,
) -> str:
    """Search file contents for a regex pattern."""
    search_root = _resolve_path(path)
    if not search_root.exists():
        return f"Error: Path not found: {search_root}"

    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results: list[str] = []
    files = search_root.rglob(include) if recursive else search_root.glob(include)

    for file_path in files:
        if not file_path.is_file():
            continue
        # Skip binary and hidden dirs
        if any(p.startswith(".") for p in file_path.parts if p != "."):
            continue
        if file_path.suffix in (".pyc", ".pyo", ".so", ".o", ".a"):
            continue

        try:
            content = file_path.read_text()
        except (UnicodeDecodeError, OSError):
            continue

        for lineno, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                rel_path = file_path.relative_to(_workspace_root)
                results.append(f"{rel_path}:{lineno}: {line.strip()[:200]}")
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            results.append(f"... (truncated at {max_results} results)")
            break

    if not results:
        return f"No matches for '{pattern}' in {search_root}"
    return "\n".join(results)


# ── git_ops ───────────────────────────────────────────────────────────────────

def tool_git_ops(operation: str, args: str = "") -> str:
    """Safe git operations: status, diff, log, add, commit, branch, show."""
    safe_ops = {
        "status", "diff", "log", "add", "commit",
        "branch", "show", "checkout", "restore", "stash",
    }
    op = operation.strip().split()[0] if operation.strip() else ""
    if op not in safe_ops:
        return (
            f"Error: Unsupported git operation '{op}'. "
            f"Allowed: {', '.join(sorted(safe_ops))}"
        )

    cmd = ["git"] + operation.strip().split() + args.strip().split()
    cmd = [c for c in cmd if c]  # remove empties

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=60,
            cwd=str(_workspace_root),
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Git operation timed out after 60s"
    except Exception as e:
        return f"Error executing git: {e}"


# ── run_bash ──────────────────────────────────────────────────────────────────

def tool_run_bash(command: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(_workspace_root),
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing command: {e}"


# ── load_skill ────────────────────────────────────────────────────────────────

def tool_load_skill(name: str = "") -> str:
    """Load a skill's content, or list available skills if no name given."""
    from memory_agent.skills import discover_skills, get_skill, get_skill_list_text

    if not name:
        return get_skill_list_text()

    skill = get_skill(name)
    if skill is None:
        skills = discover_skills()
        names = [s.name for s in skills]
        return f"Skill '{name}' not found. Available: {', '.join(names)}" if names else "No skills installed."

    content = skill.load()
    return (
        f"--- SKILL: {skill.name} ({skill.source}) ---\n"
        f"Description: {skill.description}\n"
        f"{'-' * 40}\n"
        f"{content}\n"
        f"--- END SKILL: {skill.name} ---"
    )


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


# ── tool registry ─────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents. Use offset and limit for line ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "offset": {"type": "integer", "description": "Start line (0-indexed)"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file. Creates parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit a file by exact string replacement. "
                "old_string must match exactly (including whitespace) and be unique in the file "
                "unless replace_all=true. Prefer this over write_file for targeted changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file to edit"},
                    "old_string": {"type": "string", "description": "Exact text to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false, requires uniqueness)",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": (
                "Search file contents for a regex pattern. "
                "Returns matching lines with file:line:content. "
                "Use this to find code, function definitions, error messages, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search (default: '.')"},
                    "include": {"type": "string", "description": "File glob pattern (default: '*')"},
                    "recursive": {"type": "boolean", "description": "Search recursively (default: true)"},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                    "max_results": {"type": "integer", "description": "Max results (default: 50)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_ops",
            "description": (
                "Safe git operations. Allowed: status, diff, log, add, commit, "
                "branch, show, checkout, restore, stash. "
                "Use 'operation' for the git subcommand with flags, 'args' for additional arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Git subcommand, e.g. 'status', 'diff', 'log --oneline -10', 'add file.py'",
                    },
                    "args": {"type": "string", "description": "Additional arguments (e.g. commit message)"},
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Execute a shell command in workspace root. "
                "⚠️ Requires user confirmation before execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)"},
                },
                "required": ["command"],
            },
        },
    },
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
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search past conversation memories for relevant context. "
                "Use this mid-conversation when you need to recall what was discussed "
                "in previous conversations — e.g., past decisions, bug fixes, or design discussions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for finding relevant memories"},
                    "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
                },
                "required": ["query"],
            },
        },
    },
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
]

TOOL_EXECUTORS = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "grep_files": tool_grep_files,
    "git_ops": tool_git_ops,
    "run_bash": tool_run_bash,
    "load_skill": tool_load_skill,
    "search_memory": tool_search_memory,
    "search_skills": tool_search_skills,
    "ask_user": tool_ask_user,
}


def execute_tool(name: str, arguments: dict) -> str:
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return f"Error: Unknown tool '{name}'"
    return executor(**arguments)
