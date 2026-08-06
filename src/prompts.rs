/// Prompt templates for the memory agent.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct Memory {
    pub id: String,
    pub summary: String,
    pub conversation_at: Option<String>,
    pub created_at: Option<String>,
    pub key_points: Vec<String>,
    pub tags: Vec<String>,
    pub entities: Vec<Entity>,
    pub decisions: Vec<String>,
    pub chroma_doc_id: Option<String>,
    pub conversation_json: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    pub description: Option<String>,
}

// -- Retrieval Decision prompts --

pub const RETRIEVAL_DECISION_SYSTEM_PROMPT: &str = r#"You are a memory retrieval decision engine. You have access to a history memory database
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
"#;

pub const RETRIEVAL_DECISION_USER_TEMPLATE: &str = "User query: {user_query}";

// -- Extractor prompts --

pub const EXTRACTOR_SYSTEM_PROMPT: &str = r#"You are a conversation memory extractor. Given a complete transcript of a conversation
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
"#;

pub const EXTRACTOR_USER_TEMPLATE: &str = "Conversation transcript:\n\n{transcript}";

// -- Agent Loop prompts --

pub fn build_system_prompt(workspace_root: &std::path::Path) -> String {
    let ws = workspace_root.display();
    let os = std::env::consts::OS;
    let home = dirs::home_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "~".to_string());

    format!(
        r#"You are a helpful AI assistant with access to tools.

## Environment
- Workspace: {ws}
- OS: {os}
- Shell: bash
- Home: {home}
- All file paths are relative to workspace root.

## Before You Start

Before diving into the task:
- Call **search_memory(query)** if referencing past work or prior decisions.
- Call **search_skills(query)** if the task may benefit from specialized workflows.
- Call **ask_user** when you lack critical info or need to choose between approaches.

Work step by step. When done, provide a clear summary.
"#
    )
}

pub const ROUND_2_PLUS_PROMPT: &str = r#"Continue working on the user's task. Use the context you've already retrieved
from tool calls earlier in this conversation.

If you discover gaps and need more:
- search_memory(query) — search past conversations
- search_skills(query) — find additional skills with full instructions
- ask_user(question, header, [options]) — ask the user for input when
  you need clarification, choices, or feedback
"#;

// -- Memory formatting --

const MEMORY_CONTEXT_HEADER: &str = "## Relevant Memories (from past conversations)\n";

pub fn format_memory_for_injection(memory: &Memory) -> String {
    let date = memory.conversation_at.as_deref().unwrap_or("unknown");
    let date = crate::tools::safe_truncate(date, 10);
    let key_points = if memory.key_points.is_empty() {
        "  (none)".to_string()
    } else {
        memory
            .key_points
            .iter()
            .map(|kp| format!("  - {kp}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let tags = if memory.tags.is_empty() {
        "none".to_string()
    } else {
        memory.tags.join(", ")
    };

    format!(
        "### [{date}] {summary}\n- Key Points:\n{key_points}\n- Tags: {tags}\n",
        summary = memory.summary,
    )
}

pub fn format_memories_for_injection(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return String::new();
    }
    let entries: Vec<String> = memories.iter().map(format_memory_for_injection).collect();
    format!("{MEMORY_CONTEXT_HEADER}{}", entries.join("\n"))
}
