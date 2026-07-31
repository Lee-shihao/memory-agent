#!/usr/bin/env python3
"""Memory Agent CLI — 3-step pipeline: Retrieve → Agent Loop → Extract."""
import argparse
import sys
from pathlib import Path
from memory_agent.config import load_config
from memory_agent.storage import MemoryStore
from memory_agent.retriever import Retriever
from memory_agent.agent_loop import run_agent_loop
from memory_agent.extractor import extract_and_store


def main():
    parser = argparse.ArgumentParser(description="Memory Agent — AI assistant with persistent memory")
    parser.add_argument("query", nargs="*", help="Your query or task for the agent")
    parser.add_argument("-p", "--project", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--no-memory", action="store_true", help="Skip memory retrieval")
    parser.add_argument("--no-extract", action="store_true", help="Skip memory extraction")
    args = parser.parse_args()

    if args.query:
        user_query = " ".join(args.query)
    elif not sys.stdin.isatty():
        user_query = sys.stdin.read().strip()
    else:
        parser.print_help(); sys.exit(1)

    project_root = args.project.resolve()
    config = load_config(project_root)

    db_path = config.memory_dir / "memories.db"
    store = MemoryStore(db_path)
    store.init_schema()

    # Step 1: Memory Retrieval
    memory_context = ""
    injected_memories = []
    if not args.no_memory:
        print("Checking memory...", file=sys.stderr)
        try:
            retriever = Retriever(config, store)
            injected_memories, memory_context = retriever.retrieve(user_query)
            if memory_context:
                print(f"  Injected {len(injected_memories)} memory/memories.", file=sys.stderr)
        except Exception as e:
            print(f"  Memory retrieval failed: {e}", file=sys.stderr)

    # Step 2: Agent Loop
    print(file=sys.stderr)
    transcript = run_agent_loop(config=config, user_query=user_query, memory_context=memory_context)
    print(transcript)

    # Step 3: Memory Extraction
    if not args.no_extract:
        print(file=sys.stderr)
        try:
            extract_and_store(transcript=transcript, config=config, store=store)
        except Exception as e:
            print(f"Memory extraction failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
