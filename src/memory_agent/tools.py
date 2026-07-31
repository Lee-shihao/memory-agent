"""Built-in tools: file ops, search, git, skills."""
import fnmatch
import os
import re
import subprocess
from pathlib import Path

_workspace_root = Path.cwd()


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
            "name": "load_skill",
            "description": (
                "Load a skill's instructions. Skills extend your capabilities with "
                "domain-specific workflows, tools, or knowledge. "
                "Call with no arguments to list available skills. "
                "Call with a skill name to load its full instructions into context. "
                "Use this when: you need specialized knowledge, the user asks for a "
                "specific skill, or you're unsure how to proceed in a domain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name to load, or empty to list all skills",
                    },
                },
                "required": [],
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
}


def execute_tool(name: str, arguments: dict) -> str:
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return f"Error: Unknown tool '{name}'"
    return executor(**arguments)
