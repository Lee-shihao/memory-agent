#!/usr/bin/env python3
"""Memory Agent CLI — 3-step pipeline: Retrieve → Agent Loop → Extract.

Run without arguments to enter interactive mode.
"""
import argparse
import atexit
import os
import readline
import sys
from pathlib import Path

from memory_agent.config import load_config, Config
from memory_agent.storage import MemoryStore
from memory_agent.retriever import Retriever
from memory_agent.agent_loop import run_agent_loop
from memory_agent.extractor import extract_and_store
from memory_agent.tools import set_workspace_root
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

    # Load history
    try:
        readline.read_history_file(_HISTORY_FILE)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, _HISTORY_FILE)


def _completer(text: str, state: int) -> str | None:
    """Tab-completion for /memory subcommands."""
    line = readline.get_line_buffer()
    stripped = line.lstrip()

    # Complete /memory subcommands
    if stripped.startswith("/memory"):
        parts = stripped.split(maxsplit=1)
        if len(parts) == 1 and not stripped.endswith(" "):
            # Completing "/memory" itself
            options = ["/memory "]
            if state < len(options):
                return options[state]
        elif len(parts) == 2:
            # Completing subcommand
            prefix = parts[1] if not stripped.endswith(" ") else ""
            matches = [cmd for cmd in _MEMORY_SUBCOMMANDS if cmd.startswith(prefix)]
            if state < len(matches):
                return parts[0] + " " + matches[state]
    return None


# ── pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    config: Config,
    store: MemoryStore,
    *,
    skip_memory: bool = False,
    skip_extract: bool = False,
) -> None:
    """Execute the full 3-step pipeline for a single conversation turn."""

    # Step 1: Memory Retrieval
    memory_context = ""
    injected_memories: list[dict] = []
    if not skip_memory:
        print("Checking memory...", file=sys.stderr)
        try:
            retriever = Retriever(config, store)
            injected_memories, memory_context = retriever.retrieve(user_query)
            if memory_context:
                print(f"  Injected {len(injected_memories)} memory/memories.", file=sys.stderr)
        except Exception as e:
            print(f"  Memory retrieval failed: {e}", file=sys.stderr)

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
        memory_context=memory_context,
    )
    print(transcript)

    # Step 3: Memory Extraction
    if not skip_extract:
        print(file=sys.stderr)
        try:
            extract_and_store(transcript=transcript, config=config, store=store)
        except Exception as e:
            print(f"Memory extraction failed: {e}", file=sys.stderr)


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


def _interactive_loop(config: Config, store: MemoryStore):
    """Run the interactive REPL."""
    _setup_readline()
    readline.set_completer(_completer)

    print(_BANNER, file=sys.stderr)

    # Per-session flags
    skip_memory = False
    skip_extract = False

    while True:
        try:
            prompt = "> "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.", file=sys.stderr)
            break

        if not user_input:
            continue

        # Meta commands
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
            user_input,
            config,
            store,
            skip_memory=skip_memory,
            skip_extract=skip_extract,
        )


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Memory Agent — AI assistant with persistent memory"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Your query or task for the agent. Omit to enter interactive mode.",
    )
    parser.add_argument(
        "-p", "--project",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Skip memory retrieval for this invocation",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip memory extraction after the conversation",
    )
    args = parser.parse_args()

    # Collect query from args or stdin
    if args.query:
        user_query = " ".join(args.query)
    elif not sys.stdin.isatty():
        user_query = sys.stdin.read().strip()
    else:
        user_query = None  # interactive mode

    project_root = args.project.resolve()
    set_workspace_root(project_root)
    config = load_config(project_root)

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
        # Single-shot mode
        run_pipeline(
            user_query,
            config,
            store,
            skip_memory=args.no_memory,
            skip_extract=args.no_extract,
        )
    else:
        # Interactive REPL mode
        _interactive_loop(config, store)


if __name__ == "__main__":
    main()
