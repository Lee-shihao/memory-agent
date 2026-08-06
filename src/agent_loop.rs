use crate::config::Config;
use crate::debug;
use crate::prompts::{build_system_prompt, ROUND_2_PLUS_PROMPT};
use crate::tools::{execute_tool_async, workspace_root, TOOL_DEFINITIONS};
use anyhow::Result;
use serde_json::Value as JsonValue;
/// Agent loop: OpenAI-compatible API with tool calling iteration.
use std::collections::HashMap;
use std::sync::Arc;

pub type ConfirmCallback =
    Arc<dyn Fn(&str, &HashMap<String, JsonValue>) -> (bool, String) + Send + Sync>;

pub async fn run_agent_loop(
    config: &Config,
    user_query: &str,
    tools: Option<&[JsonValue]>,
    max_iterations: usize,
    confirm_callback: Option<ConfirmCallback>,
    skill_context: Option<&str>,
) -> Result<String> {
    let tools = tools.unwrap_or(&TOOL_DEFINITIONS);
    let client = reqwest::Client::new();

    let mut base_prompt = build_system_prompt(&workspace_root());
    if let Some(skill) = skill_context {
        base_prompt.push_str(&format!(
            "\n\n--- Skill Instructions (user requested this skill, follow it) ---\n{skill}\n--- End Skill ---"
        ));
    }

    let mut messages: Vec<JsonValue> = vec![
        serde_json::json!({"role": "system", "content": &base_prompt}),
        serde_json::json!({"role": "user", "content": user_query}),
    ];

    let mut transcript_parts = vec![format!("User: {user_query}")];

    for iteration in 0..max_iterations {
        // Dynamic prompt switching after first iteration
        if iteration >= 1 {
            messages[0]["content"] =
                serde_json::json!(format!("{base_prompt}\n\n{ROUND_2_PLUS_PROMPT}"));
        }

        let url = format!("{}/chat/completions", config.llm_api_base);
        let req_body = serde_json::json!({
            "model": config.llm_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        });

        // Debug logging
        let rid = if debug::is_enabled() {
            let headers_json = serde_json::json!({
                "Authorization": format!("Bearer ...{}",
                    &config.llm_api_key.chars().rev().take(8).collect::<String>().chars().rev().collect::<String>()),
                "Content-Type": "application/json",
            });
            debug::log_request(
                "agent_loop",
                "POST",
                &url,
                Some(&headers_json),
                Some(&req_body),
            )
        } else {
            String::new()
        };

        // Make API call
        let response = match client
            .post(&url)
            .header("Authorization", format!("Bearer {}", config.llm_api_key))
            .header("Content-Type", "application/json")
            .json(&req_body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                transcript_parts.push(format!(
                    "Assistant: [Connection error] Cannot reach {} — {e}",
                    config.llm_api_base
                ));
                return Ok(transcript_parts.join("\n\n"));
            }
        };

        let status = response.status().as_u16();

        if !response.status().is_success() {
            transcript_parts.push(format!(
                "Assistant: [API error {status}] LLM returned an error — \
                 please check your API key and network, then retry."
            ));
            if debug::is_enabled() {
                let body_text = response.text().await.unwrap_or_default();
                let body_json: JsonValue =
                    serde_json::from_str(&body_text).unwrap_or(JsonValue::String(body_text));
                debug::log_response(&rid, status, Some(&body_json), None);
            }
            return Ok(transcript_parts.join("\n\n"));
        }

        let data: JsonValue = match response.json().await {
            Ok(d) => d,
            Err(e) => {
                transcript_parts.push(format!(
                    "Assistant: [Parse error] Failed to parse LLM response — {e}"
                ));
                return Ok(transcript_parts.join("\n\n"));
            }
        };

        if debug::is_enabled() {
            debug::log_response(&rid, status, Some(&data), None);
            if let Some(usage) = data.get("usage") {
                debug::accumulate_usage(usage);
            }
        }

        let choice = match data["choices"].get(0) {
            Some(c) => c,
            None => {
                transcript_parts.push("Assistant: [Error] LLM returned no choices.".to_string());
                return Ok(transcript_parts.join("\n\n"));
            }
        };

        let message = &choice["message"];
        let tool_calls = message.get("tool_calls").and_then(|tc| tc.as_array());

        if let Some(tool_calls) = tool_calls {
            if !tool_calls.is_empty() {
                // Record assistant message with tool calls
                let tc_array: Vec<JsonValue> = tool_calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            }
                        })
                    })
                    .collect();

                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tc_array,
                }));

                for tc in tool_calls {
                    let tool_name = tc["function"]["name"].as_str().unwrap_or("");
                    let args: HashMap<String, JsonValue> = tc["function"]["arguments"]
                        .as_str()
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or_default();

                    // Confirmation hook
                    let (allowed, feedback) = if let Some(ref cb) = confirm_callback {
                        cb(tool_name, &args)
                    } else {
                        (true, String::new())
                    };

                    let tool_result = if allowed {
                        let mut result = execute_tool_async(tool_name, &args).await;
                        if !feedback.is_empty() {
                            result.push_str(&format!("\n\n[User note: {feedback}]"));
                        }
                        result
                    } else {
                        let mut blocked = "[Blocked by user]".to_string();
                        if !feedback.is_empty() {
                            blocked.push_str(&format!(" {feedback}"));
                        }
                        blocked
                    };

                    let result_preview = if tool_result.len() > 500 {
                        format!("{}...", crate::tools::safe_truncate(&tool_result, 500))
                    } else {
                        tool_result.clone()
                    };

                    transcript_parts.push(format!(
                        "Tool [{}]: {}\nResult: {}",
                        tool_name, tc["function"]["arguments"], result_preview,
                    ));

                    messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }));
                }
                continue;
            }
        }

        // No tool calls — final response
        let assistant_content = message["content"].as_str().unwrap_or("");
        transcript_parts.push(format!("Assistant: {assistant_content}"));
        return Ok(transcript_parts.join("\n\n"));
    }

    transcript_parts.push("[Max tool call iterations reached]".to_string());
    Ok(transcript_parts.join("\n\n"))
}
