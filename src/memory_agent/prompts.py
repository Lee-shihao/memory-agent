"""Prompt templates for the memory agent."""

RETRIEVAL_DECISION_SYSTEM_PROMPT = """\
You are a memory retrieval decision engine. You have access to a history memory database
containing past conversations between a user and an AI assistant.

Given the user's query, decide whether to retrieve relevant past memories.
If needed, generate 1-3 semantic search queries AND/OR specify a recent range (N through M)
of the most recent memories.

Rules:
- If the user's query references past work, previous discussions, or prior context,
  retrieve relevant memories.
- For phrases like "just now", "last time", "previous", "a moment ago", use recent_range.
- For topical references like "Python async we discussed", use semantic_queries.
- For simple, self-contained questions (e.g., "hello", "write a hello world"),
  return need_retrieve: false.
- You can use both semantic_queries and recent_range together.

Output ONLY a JSON object, no other text:
{"need_retrieve": true/false, "semantic_queries": ["q1","q2"], "recent_range": {"start":N,"end":M} or null}
"""

RETRIEVAL_DECISION_USER_TEMPLATE = "User query: {user_query}"


EXTRACTOR_SYSTEM_PROMPT = """\
You are a conversation memory extractor. Given a complete transcript of a conversation
between a user and an AI assistant, extract the key information as structured data.

Output ONLY a JSON object:
{
  "summary": "Concise summary in <=200 characters, in the conversation's language",
  "key_points": ["Key conclusion 1", "Key conclusion 2", ...],
  "tags": ["tag1", "tag2", ...],
  "entities": [{"name":"...", "type":"file|function|class|concept|dependency|config", "description":"..."}],
  "decisions": ["Decision 1", "Decision 2", ...]
}

Guidelines:
- summary: <=200 chars, captures the essence of the conversation
- key_points: 3-8 items, each a single sentence
- tags: 3-6 lowercase tags for categorization
- entities: type must be one of file/function/class/concept/dependency/config
- decisions: explicit choices made. Can be empty array.
"""

EXTRACTOR_USER_TEMPLATE = "Conversation transcript:\n\n{transcript}"


BASE_AGENT_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools. You can read files, write files,
and execute shell commands to help the user accomplish their tasks.

Work step by step. Use tools when needed. When you have completed the user's request,
provide a clear summary of what was done.
"""


MEMORY_CONTEXT_HEADER = "## Relevant Memories (from past conversations)\n"

_MEMORY_ENTRY_TEMPLATE = """\
### [{date}] {summary}
- Key Points:
{key_points}
- Tags: {tags}
"""


def format_memory_for_injection(memory: dict) -> str:
    date = memory.get("conversation_at", "unknown")
    if isinstance(date, str) and len(date) >= 10:
        date = date[:10]
    key_points = memory.get("key_points", [])
    kp_lines = "\n".join(f"  - {kp}" for kp in key_points) if key_points else "  (none)"
    tags = ", ".join(memory.get("tags", [])) or "none"
    return _MEMORY_ENTRY_TEMPLATE.format(date=date, summary=memory.get("summary", ""), key_points=kp_lines, tags=tags)


def format_memories_for_injection(memories: list[dict]) -> str:
    if not memories:
        return ""
    entries = [format_memory_for_injection(m) for m in memories]
    return MEMORY_CONTEXT_HEADER + "\n".join(entries)


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
