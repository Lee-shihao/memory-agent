"""Slash command handlers for /memory operations."""
from memory_agent.storage import MemoryStore


def handle_slash_command(message: str, store: MemoryStore, injected_memories: list[dict]) -> tuple[bool, str]:
    stripped = message.strip()
    if not stripped.startswith("/memory"):
        return False, ""

    parts = stripped.split(maxsplit=2)
    subcommand = parts[1] if len(parts) > 1 else ""
    args = parts[2] if len(parts) > 2 else ""

    if not subcommand:
        return True, _cmd_show_injected(injected_memories)
    elif subcommand == "recent":
        n = int(args) if args.isdigit() else 10
        return True, _cmd_recent(store, n)
    elif subcommand == "search":
        if not args: return True, "Usage: /memory search <query>"
        return True, _cmd_search(store, args)
    elif subcommand == "show":
        if not args: return True, "Usage: /memory show <id>"
        return True, _cmd_show(store, args)
    elif subcommand == "delete":
        if not args: return True, "Usage: /memory delete <id>"
        return True, _cmd_delete(store, args)
    elif subcommand == "status":
        return True, _cmd_status(store)
    else:
        return True, _cmd_usage()


def _cmd_show_injected(injected):
    if not injected:
        return "No memories were injected for this conversation."
    lines = ["Memories injected for this conversation:"]
    for i, mem in enumerate(injected, 1):
        lines.append(f"  {i}. [{mem.get('memory_id') or mem.get('id', '?')}] {mem.get('summary', '(no summary)')}")
    return "\n".join(lines)


def _cmd_recent(store, n):
    memories = store.get_recent_memories(limit=n)
    if not memories: return "No memories in database."
    lines = [f"Recent {len(memories)} memories:"]
    for mem in memories:
        lines.append(f"  [{mem['id']}] {mem['summary'][:80]}")
    return "\n".join(lines)


def _cmd_search(store, query):
    if not hasattr(store, "_chroma_collection") or store._chroma_collection is None:
        return "Vector search not available (ChromaDB not initialized)."
    try:
        results = store.query_chroma(query, top_k=5)
    except Exception as e:
        return f"Search failed: {e}"
    if not results: return f"No memories found matching: {query}"
    lines = [f"Search results for '{query}':"]
    for r in results:
        lines.append(f"  [{r.get('memory_id','?')}] (distance:{r.get('distance',0):.3f}) {r.get('text','')[:100]}")
    return "\n".join(lines)


def _cmd_show(store, memory_id):
    mem = store.get_memory(memory_id)
    if mem is None: return f"Memory not found: {memory_id}"
    lines = [f"=== Memory: {memory_id} ===", f"Summary: {mem['summary']}", f"Tags: {', '.join(mem.get('tags',[])) or '(none)'}", f"Conversation at: {mem.get('conversation_at','unknown')}", f"Created at: {mem.get('created_at','unknown')}", "", "Key Points:"]
    for kp in mem.get("key_points", []): lines.append(f"  • {kp}")
    lines.append(""); lines.append("Entities:")
    for ent in mem.get("entities", []): lines.append(f"  • {ent['name']} ({ent['type']}): {ent.get('description','')}")
    if not mem.get("entities"): lines.append("  (none)")
    lines.append(""); lines.append("Decisions:")
    for dec in mem.get("decisions", []): lines.append(f"  • {dec}")
    if not mem.get("decisions"): lines.append("  (none)")
    return "\n".join(lines)


def _cmd_delete(store, memory_id):
    mem = store.get_memory(memory_id)
    if mem is None: return f"Memory not found: {memory_id}"
    chroma_doc_id = mem.get("chroma_doc_id")
    if chroma_doc_id and hasattr(store, "_chroma_collection") and store._chroma_collection:
        try: store.delete_from_chroma(chroma_doc_id)
        except Exception: pass
    store.delete_memory(memory_id)
    return f"Memory deleted: {memory_id}"


def _cmd_status(store):
    status = store.get_status()
    lines = ["=== Memory Database Status ===", f"Total memories: {status['total_memories']}", f"Total tags: {status['total_tags']}", f"Last insert: {status['last_insert_at'] or 'never'}", f"DB path: {status['db_path']}", f"DB size: {status['db_size_bytes']} bytes"]
    tags = store.get_all_tags()
    if tags: lines.append(f"\nTags: {', '.join(tags)}")
    return "\n".join(lines)


def _cmd_usage():
    return """Usage:
  /memory                  Show injected memories
  /memory recent [N]       Show recent N memories (default 10)
  /memory search <query>   Semantic search
  /memory show <id>        Show memory details
  /memory delete <id>      Delete a memory
  /memory status           Database statistics"""
