#!/usr/bin/env python3
"""Memory Agent CLI — 3-step pipeline: Retrieve → Agent Loop → Extract.

Run without arguments to enter interactive mode.
"""
import argparse
import atexit
import os
import readline
import select
import sys
from pathlib import Path

from memory_agent.config import load_config, Config
from memory_agent.debug import enable as _debug_enable
from memory_agent.debug import is_enabled as _debug_is_enabled
from memory_agent.debug import reset_session_stats as _reset_token_stats
from memory_agent.debug import get_session_stats as _get_token_stats
from memory_agent.storage import MemoryStore
from memory_agent.agent_loop import run_agent_loop
from memory_agent.extractor import extract_and_store
from memory_agent.tools import set_workspace_root, reset_session_state, pre_index_skills
from memory_agent import __version__
from memory_agent.commands import handle_slash_command


# ── readline setup ───────────────────────────────────────────────────────────

_HISTORY_FILE = os.path.expanduser("~/.memory_agent_history")
_MEMORY_SUBCOMMANDS = ["recent", "search", "show", "delete", "status"]


def _setup_readline():
    """Enable line editing, persistent history, and tab completion."""
    try:
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass
    try:
        readline.read_history_file(_HISTORY_FILE)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, _HISTORY_FILE)


def _completer(text: str, state: int) -> str | None:
    """Tab-completion for /memory subcommands."""
    line = readline.get_line_buffer()
    stripped = line.lstrip()
    if stripped.startswith("/memory"):
        parts = stripped.split(maxsplit=1)
        if len(parts) == 1 and not stripped.endswith(" "):
            options = ["/memory "]
            if state < len(options):
                return options[state]
        elif len(parts) == 2:
            prefix = parts[1] if not stripped.endswith(" ") else ""
            matches = [cmd for cmd in _MEMORY_SUBCOMMANDS if cmd.startswith(prefix)]
            if state < len(matches):
                return parts[0] + " " + matches[state]
    return None


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


# ── pipeline ─────────────────────────────────────────────────────────────────

def _print_token_stats() -> None:
    """Print accumulated per-conversation token usage to stderr."""
    import sys
    stats = _get_token_stats()
    if stats["llm_call_count"] == 0:
        return
    cache_rate = ""
    if stats["prompt_tokens"] > 0 and stats["cached_tokens"] > 0:
        rate = stats["cached_tokens"] / stats["prompt_tokens"] * 100
        cache_rate = f"\n  Cache hit rate:    {rate:.1f}%"
    print(
        f"\n{'='*50}",
        f"📊 Token usage this conversation:",
        f"  LLM calls:         {stats['llm_call_count']}",
        f"  Prompt tokens:     {stats['prompt_tokens']:,}",
        f"  Completion tokens: {stats['completion_tokens']:,}",
        f"  Total tokens:      {stats['total_tokens']:,}",
        f"  Cached tokens:     {stats['cached_tokens']:,}{cache_rate}",
        f"{'='*50}\n",
        sep="\n", file=sys.stderr,
    )


def run_pipeline(
    user_query: str,
    config: Config,
    store: MemoryStore,
    *,
    skip_memory: bool = False,
    skip_extract: bool = False,
    manual_extract: bool = False,
    interactive: bool = False,
) -> None:
    """Execute the full 3-step pipeline for a single conversation turn."""

    # Reset per-conversation token counters (debug mode)
    if _debug_is_enabled():
        _reset_token_stats()

    # Step 1: Initialize session state + pre-index skills (no prompt injection)
    injected_memories: list[dict] = []

    reset_session_state()

    if not skip_memory:
        try:
            pre_index_skills(config)
        except Exception as e:
            print(f"  Skill indexing failed: {e}", file=sys.stderr)

    # Handle /memory slash commands
    if user_query.startswith("/memory"):
        is_cmd, response = handle_slash_command(user_query, store, injected_memories)
        if is_cmd:
            print(response)
            return

    # Step 2: Agent Loop
    print(file=sys.stderr)
    transcript = run_agent_loop(
        config=config,
        user_query=user_query,
        confirm_callback=_tool_confirm,
    )
    print(transcript)

    # Step 3: Memory Extraction
    if not skip_extract:
        print(file=sys.stderr)
        try:
            extract_and_store(
                transcript=transcript, config=config, store=store,
                auto_confirm=not manual_extract,
            )
        except Exception as e:
            print(f"Memory extraction failed: {e}", file=sys.stderr)

    # Print per-conversation token stats (debug mode)
    if _debug_is_enabled():
        _print_token_stats()


# ── interactive mode ─────────────────────────────────────────────────────────

_BANNER = r"""
╔══════════════════════════════════════════════╗
║            🧠  Memory Agent                  ║
║                                              ║
║  自带向量记忆的 AI 助手                        ║
║  输入问题开始对话，/memory 查看和管理记忆       ║
║  Ctrl+D 或 /exit 退出                        ║
╚══════════════════════════════════════════════╝
"""


def _interactive_loop(
    config: Config, store: MemoryStore,
    *,
    skip_memory: bool = False,
    skip_extract: bool = False,
    manual_extract: bool = False,
):
    """Run the interactive REPL."""
    _setup_readline()
    readline.set_completer(_completer)

    print(_BANNER, file=sys.stderr)

    while True:
        try:
            prompt = "> "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.", file=sys.stderr)
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit", "/q"):
            print("Goodbye.", file=sys.stderr)
            break

        if user_input == "/help":
            print(
                "  Enter a question or task to start a conversation.",
                "  /memory              Show injected memories",
                "  /memory recent [N]   Show recent N memories",
                "  /memory search <q>   Semantic search",
                "  /memory show <id>    Show memory details",
                "  /memory delete <id>  Delete a memory",
                "  /memory status       Database statistics",
                "  /exit, /quit, /q     Exit",
                "  /help                Show this help",
                "  Ctrl+D               Exit",
                sep="\n",
            )
            continue

        run_pipeline(
            user_input, config, store,
            skip_memory=skip_memory,
            skip_extract=skip_extract,
            manual_extract=manual_extract,
            interactive=True,
        )


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Memory Agent — AI assistant with persistent memory"
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"memory-agent {__version__}",
    )
    parser.add_argument(
        "query", nargs="*",
        help="Your query or task for the agent. Omit to enter interactive mode.",
    )
    parser.add_argument(
        "-p", "--project", type=Path, default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Skip memory retrieval for this invocation",
    )
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Skip memory extraction after the conversation",
    )
    parser.add_argument(
        "--manual-extract", action="store_true",
        help="Prompt for save/edit/discard on each extracted memory "
             "(default: auto-save without asking)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Log all HTTP API calls to .agent-memory/debug.log",
    )

    # Skill management
    skill_group = parser.add_argument_group("Skill management")
    skill_group.add_argument(
        "--skill-list", action="store_true",
        help="List installed skills and exit",
    )
    skill_group.add_argument(
        "--skill-install", type=str, metavar="SOURCE",
        help="Install a skill from a local directory or git URL",
    )
    skill_group.add_argument(
        "--skill-dir", type=str, metavar="DIR",
        help="Additional skill directory to search",
    )
    args = parser.parse_args()

    # Handle --skill-dir (register before any skill operations)
    if args.skill_dir:
        from memory_agent.skills import add_search_path
        for d in args.skill_dir.split(":"):
            add_search_path(d)

    # Handle skill management commands (exit after processing)
    if args.skill_list:
        from memory_agent.skills import list_installed_skills
        print(list_installed_skills())
        return
    if args.skill_install:
        from memory_agent.skills import install_skill
        project_root = args.project.resolve()
        print(install_skill(args.skill_install, project_root))
        return

    if args.query:
        user_query = " ".join(args.query)
    elif not sys.stdin.isatty():
        user_query = sys.stdin.read().strip()
    else:
        user_query = None  # interactive mode

    project_root = args.project.resolve()
    set_workspace_root(project_root)
    config = load_config(project_root)

    if args.debug:
        _debug_enable(config.memory_dir)
        print(f"Debug logging enabled → {config.memory_dir / 'debug.log'}", file=sys.stderr)

    db_path = config.memory_dir / "memories.db"
    store = MemoryStore(db_path)
    store.init_schema()
    store.init_chroma(
        persist_dir=config.memory_dir / "chroma",
        embedding_api_base=config.embedding_api_base,
        embedding_api_key=config.embedding_api_key,
        embedding_model=config.embedding_model,
    )

    if user_query:
        # Single-shot mode — no bash confirmation (batch run)
        run_pipeline(
            user_query, config, store,
            skip_memory=args.no_memory,
            skip_extract=args.no_extract,
            manual_extract=args.manual_extract,
            interactive=False,
        )
    else:
        # Interactive REPL — bash confirmation enabled
        _interactive_loop(
            config, store,
            skip_memory=args.no_memory,
            skip_extract=args.no_extract,
            manual_extract=args.manual_extract,
        )


if __name__ == "__main__":
    main()
