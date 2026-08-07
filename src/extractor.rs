use crate::config::Config;
use crate::prompts::{EXTRACTOR_SYSTEM_PROMPT, EXTRACTOR_USER_TEMPLATE};
use crate::storage::MemoryStore;
use anyhow::Result;
use serde::{Deserialize, Serialize};
/// Extractor: post-conversation memory extraction with user review.
use std::io::{self, Write};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub summary: String,
    #[serde(default)]
    pub key_points: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub entities: Vec<serde_json::Value>,
    #[serde(default)]
    pub decisions: Vec<String>,
}

async fn call_extraction_llm(config: &Config, transcript: &str) -> Result<ExtractionResult> {
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", config.llm_api_base);

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", config.llm_api_key))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "model": config.llm_model,
            "messages": [
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content":
                    EXTRACTOR_USER_TEMPLATE.replace("{transcript}", transcript)},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }))
        .send()
        .await?
        .error_for_status()?;

    let data: serde_json::Value = resp.json().await?;
    let content = data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("{}");

    Ok(serde_json::from_str(content)?)
}

fn display_preview(result: &ExtractionResult) {
    eprintln!("\n==================================================");
    eprintln!("\u{1F4DD} Memory Preview");
    eprintln!("==================================================");
    eprintln!("\nSummary: {}", result.summary);
    eprintln!(
        "\nTags: {}",
        if result.tags.is_empty() {
            "(none)".to_string()
        } else {
            result.tags.join(", ")
        }
    );
    eprintln!("\nKey Points:");
    if result.key_points.is_empty() {
        eprintln!("  (none)");
    } else {
        for kp in &result.key_points {
            eprintln!("  \u{2022} {kp}");
        }
    }
    eprintln!("\nEntities:");
    if result.entities.is_empty() {
        eprintln!("  (none)");
    } else {
        for ent in &result.entities {
            eprintln!(
                "  \u{2022} {} ({}): {}",
                ent["name"].as_str().unwrap_or(""),
                ent["type"].as_str().unwrap_or(""),
                ent["description"].as_str().unwrap_or(""),
            );
        }
    }
    eprintln!("\nDecisions:");
    if result.decisions.is_empty() {
        eprintln!("  (none)");
    } else {
        for dec in &result.decisions {
            eprintln!("  \u{2022} {dec}");
        }
    }
    eprintln!();
}

fn get_user_choice() -> String {
    loop {
        eprint!("[S]ave  [E]dit  [D]iscard: ");
        let _ = io::stderr().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return "discard".to_string();
        }
        match input.trim().to_lowercase().as_str() {
            "s" | "save" | "y" | "yes" => return "save".to_string(),
            "d" | "discard" | "n" | "no" => return "discard".to_string(),
            "e" | "edit" => return "edit".to_string(),
            _ => eprintln!("Please enter S, E, or D"),
        }
    }
}

fn open_editor(result: &ExtractionResult) -> ExtractionResult {
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vim".to_string());
    let json_str = serde_json::to_string_pretty(result).unwrap_or_default();
    let tmp_path =
        std::env::temp_dir().join(format!("memory_extract_{}.json", uuid::Uuid::new_v4()));

    if std::fs::write(&tmp_path, &json_str).is_err() {
        return result.clone();
    }

    // Launch editor
    let _ = std::process::Command::new(&editor).arg(&tmp_path).status();

    // Read back
    if let Ok(edited) = std::fs::read_to_string(&tmp_path) {
        if let Ok(r) = serde_json::from_str(&edited) {
            let _ = std::fs::remove_file(&tmp_path);
            return r;
        }
    }
    let _ = std::fs::remove_file(&tmp_path);
    result.clone()
}

pub async fn extract_and_store(
    transcript: &str,
    config: &Config,
    store: &MemoryStore,
    auto_confirm: Option<bool>,
) -> Result<bool> {
    eprintln!("\nExtracting memory from conversation...");

    let result = match call_extraction_llm(config, transcript).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Extraction failed: {e}");
            return Ok(false);
        }
    };

    let effective_auto = auto_confirm.unwrap_or(config.extractor_auto_confirm);

    if effective_auto {
        store_result(&result, transcript, config, store).await?;
        eprintln!("Memory saved (auto-confirm).");
        return Ok(true);
    }

    let mut current = result;
    loop {
        display_preview(&current);
        match get_user_choice().as_str() {
            "save" => {
                store_result(&current, transcript, config, store).await?;
                eprintln!("Memory saved.");
                return Ok(true);
            }
            "edit" => {
                current = open_editor(&current);
            }
            "discard" => {
                eprintln!("Memory discarded.");
                return Ok(false);
            }
            _ => unreachable!(),
        }
    }
}

async fn store_result(
    result: &ExtractionResult,
    transcript: &str,
    config: &Config,
    store: &MemoryStore,
) -> Result<()> {
    let now = chrono::Utc::now();

    // Build embedding text from summary + key points
    let mut embedding_text = result.summary.clone();
    if !result.key_points.is_empty() {
        embedding_text.push('\n');
        embedding_text.push_str(&result.key_points.join("\n"));
    }

    let memory_id: String = uuid::Uuid::new_v4().to_string().chars().take(12).collect();

    // Build conversation JSON
    let conversation_json = if config.extractor_keep_full_transcript {
        Some(serde_json::json!({"transcript": transcript}).to_string())
    } else {
        None
    };

    // Insert metadata first, then the vector (which sets memories.vec_rowid).
    store.insert_memory(
        &result.summary,
        &now,
        conversation_json.as_deref(),
        &result.key_points,
        &result.tags,
        &result.entities,
        &result.decisions,
        Some(&memory_id),
    )?;
    store.add_memory_vector(&memory_id, &embedding_text).await?;

    eprintln!("  Memory ID: {memory_id}");
    Ok(())
}
