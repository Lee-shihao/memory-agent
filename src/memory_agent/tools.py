"""Built-in tools: read_file, write_file, run_bash."""
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path.cwd()


def _resolve_path(file_path: str) -> Path:
    p = Path(file_path)
    return p if p.is_absolute() else WORKSPACE_ROOT / p


def tool_read_file(file_path: str, offset: int = 0, limit: int | None = None) -> str:
    path = _resolve_path(file_path)
    if not path.exists():
        return f"Error: File not found: {path}"
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


def tool_write_file(file_path: str, content: str) -> str:
    path = _resolve_path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"File written: {path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_run_bash(command: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(WORKSPACE_ROOT))
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
            "description": "Write content to a file. Creates parent directories.",
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
            "name": "run_bash",
            "description": "Execute a shell command in workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
                },
                "required": ["command"],
            },
        },
    },
]

TOOL_EXECUTORS = {"read_file": tool_read_file, "write_file": tool_write_file, "run_bash": tool_run_bash}


def execute_tool(name: str, arguments: dict) -> str:
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return f"Error: Unknown tool '{name}'"
    return executor(**arguments)
