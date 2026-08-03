use crate::config::Config;
use crate::prompts::{
    format_memories_for_injection, Memory, RETRIEVAL_DECISION_SYSTEM_PROMPT,
    RETRIEVAL_DECISION_USER_TEMPLATE,
};
use crate::storage::MemoryStore;
use anyhow::Result;
use serde::Deserialize;
/// Retriever: LLM decision → dual-channel search → format for injection.
use std::collections::HashSet;

#[derive(Debug, Deserialize)]
pub struct RetrievalDecision {
    #[serde(default)]
    pub need_retrieve: bool,
    #[serde(default)]
    pub semantic_queries: Vec<String>,
    #[serde(default)]
    pub recent_range: Option<RecentRange>,
}

#[derive(Debug, Deserialize)]
pub struct RecentRange {
    pub start: usize,
    pub end: usize,
}

pub struct Retriever<'a> {
    pub config: &'a Config,
    pub store: &'a MemoryStore,
}

impl<'a> Retriever<'a> {
    pub fn new(config: &'a Config, store: &'a MemoryStore) -> Self {
        Retriever { config, store }
    }

    pub async fn retrieve(&self, user_query: &str) -> Result<(Vec<Memory>, String)> {
        let decision = self.llm_decision(user_query).await?;

        if !decision.need_retrieve {
            return Ok((vec![], String::new()));
        }

        let mut raw_results: Vec<Memory> = Vec::new();

        // Semantic search
        for query in &decision.semantic_queries {
            if let Ok(results) = self.semantic_search(query).await {
                raw_results.extend(results);
            }
        }

        // Recent range search
        if let Some(ref range) = decision.recent_range {
            let limit = range.end.saturating_sub(range.start) + 1;
            let offset = range.start.saturating_sub(1);
            if let Ok(results) = self.time_range_search(limit, offset) {
                raw_results.extend(results);
            }
        }

        // Dedup by memory ID
        let mut seen = HashSet::new();
        let mut deduped = Vec::new();
        for r in raw_results {
            if seen.contains(&r.id) {
                continue;
            }
            seen.insert(r.id.clone());
            deduped.push(r);
        }

        // Hydrate — ensure full data for each memory
        let mut hydrated = Vec::new();
        for r in deduped {
            if r.summary.is_empty() {
                if let Ok(Some(full)) = self.store.get_memory(&r.id) {
                    hydrated.push(full);
                } else {
                    hydrated.push(r);
                }
            } else {
                hydrated.push(r);
            }
        }

        let context = format_memories_for_injection(&hydrated);
        Ok((hydrated, context))
    }

    async fn llm_decision(&self, user_query: &str) -> Result<RetrievalDecision> {
        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", self.config.llm_api_base);
        let resp = client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.config.llm_api_key),
            )
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content": RETRIEVAL_DECISION_SYSTEM_PROMPT},
                    {"role": "user", "content":
                        RETRIEVAL_DECISION_USER_TEMPLATE.replace("{user_query}", user_query)},
                ],
                "temperature": 0,
                "max_tokens": 200,
            }))
            .send()
            .await?
            .error_for_status()?;

        let data: serde_json::Value = resp.json().await?;
        let content = data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{}");

        Ok(serde_json::from_str(content).unwrap_or(RetrievalDecision {
            need_retrieve: false,
            semantic_queries: vec![],
            recent_range: None,
        }))
    }

    async fn semantic_search(&self, query: &str) -> Result<Vec<Memory>> {
        let results = self
            .store
            .query_lancedb(query, self.config.retrieval_top_k)
            .await?;

        let mut memories = Vec::new();
        for r in results {
            let mid = r
                .get("memory_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let text = r
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Build a Memory from LanceDB result
            // Try to hydrate from SQLite first
            if let Ok(Some(full)) = self.store.get_memory(&mid) {
                memories.push(full);
            } else {
                memories.push(Memory {
                    id: mid,
                    summary: if text.len() > 200 {
                        text[..200].to_string()
                    } else {
                        text
                    },
                    ..Default::default()
                });
            }
        }
        Ok(memories)
    }

    fn time_range_search(&self, limit: usize, offset: usize) -> Result<Vec<Memory>> {
        self.store.get_recent_memories(limit, offset)
    }
}
