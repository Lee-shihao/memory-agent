use anyhow::{Context, Result};
use regex::Regex;
use std::fs;
/// Configuration loading with env var substitution.
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_YAML: &str = r#"# Memory Agent Configuration
llm:
  api_base: https://api.deepseek.com/v1
  api_key: ${DEEPSEEK_API_KEY}
  model: deepseek-chat

embedding:
  api_base: https://api.siliconflow.cn/v1
  api_key: ${SF_API_KEY}
  model: BAAI/bge-m3

retrieval:
  top_k: 10
  similarity_threshold: 0.5

extractor:
  auto_confirm: true
  keep_full_transcript: true
"#;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub llm_api_base: String,
    pub llm_api_key: String,
    pub llm_model: String,

    pub embedding_api_base: String,
    pub embedding_api_key: String,
    pub embedding_model: String,

    #[serde(default = "default_top_k")]
    pub retrieval_top_k: usize,
    #[serde(default = "default_threshold")]
    pub retrieval_similarity_threshold: f64,

    #[serde(default = "default_true")]
    pub extractor_auto_confirm: bool,
    #[serde(default = "default_true")]
    pub extractor_keep_full_transcript: bool,

    #[serde(skip)]
    pub memory_dir: PathBuf,
}

fn default_top_k() -> usize {
    10
}
fn default_threshold() -> f64 {
    0.5
}
fn default_true() -> bool {
    true
}

fn resolve_env_vars(raw: &str) -> String {
    let re = Regex::new(r"\$\{(\w+)\}").unwrap();
    re.replace_all(raw, |caps: &regex::Captures| {
        std::env::var(&caps[1]).unwrap_or_else(|_| caps[0].to_string())
    })
    .to_string()
}

#[derive(Debug, serde::Deserialize)]
struct RawConfig {
    llm: Option<RawLlm>,
    embedding: Option<RawEmbedding>,
    retrieval: Option<RawRetrieval>,
    extractor: Option<RawExtractor>,
}

#[derive(Debug, serde::Deserialize)]
struct RawLlm {
    api_base: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct RawEmbedding {
    api_base: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct RawRetrieval {
    top_k: Option<usize>,
    similarity_threshold: Option<f64>,
}

#[derive(Debug, serde::Deserialize)]
struct RawExtractor {
    auto_confirm: Option<bool>,
    keep_full_transcript: Option<bool>,
}

impl RawConfig {
    fn into_config(self, memory_dir: PathBuf) -> Config {
        let llm = self.llm.unwrap_or(RawLlm {
            api_base: None,
            api_key: None,
            model: None,
        });
        let embedding = self.embedding.unwrap_or(RawEmbedding {
            api_base: None,
            api_key: None,
            model: None,
        });
        let retrieval = self.retrieval.unwrap_or(RawRetrieval {
            top_k: None,
            similarity_threshold: None,
        });
        let extractor = self.extractor.unwrap_or(RawExtractor {
            auto_confirm: None,
            keep_full_transcript: None,
        });

        Config {
            llm_api_base: llm.api_base.unwrap_or_default(),
            llm_api_key: llm.api_key.unwrap_or_default(),
            llm_model: llm.model.unwrap_or_default(),
            embedding_api_base: embedding.api_base.unwrap_or_default(),
            embedding_api_key: embedding.api_key.unwrap_or_default(),
            embedding_model: embedding.model.unwrap_or_default(),
            retrieval_top_k: retrieval.top_k.unwrap_or(10),
            retrieval_similarity_threshold: retrieval.similarity_threshold.unwrap_or(0.5),
            extractor_auto_confirm: extractor.auto_confirm.unwrap_or(true),
            extractor_keep_full_transcript: extractor.keep_full_transcript.unwrap_or(true),
            memory_dir,
        }
    }
}

pub fn load_config(project_root: &Path) -> Result<Config> {
    let config_dir = project_root.join(".agent-memory");
    fs::create_dir_all(&config_dir).context("Failed to create .agent-memory directory")?;

    let config_file = config_dir.join("config.yaml");
    if !config_file.exists() {
        fs::write(&config_file, DEFAULT_CONFIG_YAML)
            .context("Failed to write default config.yaml")?;
    }

    let raw_yaml = fs::read_to_string(&config_file).context("Failed to read config.yaml")?;
    let resolved = resolve_env_vars(&raw_yaml);
    let raw: RawConfig = serde_yaml::from_str(&resolved).context("Failed to parse config.yaml")?;

    Ok(raw.into_config(config_dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_env_vars_replaces_known_var() {
        std::env::set_var("TEST_CFG_VAR", "resolved_value");
        let result = resolve_env_vars("hello ${TEST_CFG_VAR} world");
        assert_eq!(result, "hello resolved_value world");
    }

    #[test]
    fn test_resolve_env_vars_keeps_unknown() {
        let result = resolve_env_vars("hello ${UNKNOWN_VAR_XYZ} world");
        assert_eq!(result, "hello ${UNKNOWN_VAR_XYZ} world");
    }

    #[test]
    fn test_load_config_creates_default() {
        let tmp = std::env::temp_dir().join("test_config_default_rs");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = load_config(&tmp).unwrap();
        assert_eq!(config.retrieval_top_k, 10);
        assert!(config.extractor_auto_confirm);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
