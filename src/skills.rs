/// Skill discovery, loading, installation, and LanceDB-based embedding routing.
///
/// Skills are cached in memory after initial scan. Subsequent lookups return
/// from cache without disk I/O. File content is loaded lazily only when needed.
use anyhow::Result;
use serde_json::Value as JsonValue;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

// -- Global skill cache -------------------------------------------------------

static SKILLS_CACHE: RwLock<Vec<Skill>> = RwLock::new(Vec::new());
static CACHE_INITIALIZED: RwLock<bool> = RwLock::new(false);
static EXTRA_SEARCH_PATHS: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());

pub fn add_search_path(path: &str) {
    EXTRA_SEARCH_PATHS
        .lock()
        .unwrap()
        .push(Path::new(path).to_path_buf());
}

fn search_paths(project_root: Option<&Path>) -> Vec<PathBuf> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let proj = project_root.unwrap_or(&cwd);
    let mut paths = vec![
        proj.join(".agent-memory").join("skills"),
        dirs::home_dir()
            .unwrap_or_default()
            .join(".memory_agent")
            .join("skills"),
    ];
    paths.extend(EXTRA_SEARCH_PATHS.lock().unwrap().clone());
    paths
}

// -- Skill struct -------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub path: PathBuf,
    pub description: String,
    pub source: String, // "project" or "user"
}

impl Skill {
    /// Load SKILL.md content from disk. Called lazily only when full content is needed.
    pub fn load(&self) -> String {
        let skill_file = self.path.join("SKILL.md");
        fs::read_to_string(&skill_file).unwrap_or_else(|_| format!("# {}\n\n(empty skill)", self.name))
    }

    pub fn index_text(&self) -> String {
        format!("{}: {}", self.name, self.description)
    }
}

// -- Description extraction ---------------------------------------------------

fn extract_description(content: &str) -> String {
    let mut in_frontmatter = false;
    for line in content.lines() {
        let stripped = line.trim();
        if stripped == "---" {
            in_frontmatter = !in_frontmatter;
            continue;
        }
        if in_frontmatter {
            continue;
        }
        if stripped.starts_with('#') {
            let desc = stripped.trim_start_matches('#').trim();
            return if desc.is_empty() {
                "No description".to_string()
            } else {
                desc.to_string()
            };
        }
        if !stripped.is_empty() {
            return stripped.chars().take(120).collect();
        }
    }
    "No description".to_string()
}

// -- Disk scan (called only during init and install) --------------------------

fn scan_skills(project_root: Option<&Path>) -> Vec<Skill> {
    let mut skills = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let cwd = std::env::current_dir().unwrap_or_default();

    for search_dir in search_paths(project_root) {
        if !search_dir.exists() {
            continue;
        }
        let entries = match fs::read_dir(&search_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();

        for entry in &dirs {
            let name = entry
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if name.starts_with('.') || seen.contains(&name) {
                continue;
            }

            let skill_md = entry.join("SKILL.md");
            if !skill_md.exists() {
                continue;
            }

            seen.insert(name.clone());

            let source = if search_dir.starts_with(project_root.unwrap_or(&cwd)) {
                "project"
            } else {
                "user"
            };

            let desc = extract_description(&fs::read_to_string(&skill_md).unwrap_or_default());

            skills.push(Skill {
                name,
                path: entry.clone(),
                description: desc,
                source: source.to_string(),
            });
        }
    }
    skills
}

// -- Public API (cache-based) -------------------------------------------------

/// Initialize the skill cache. Call once at startup.
pub fn init_skills(project_root: &Path) {
    let skills = scan_skills(Some(project_root));
    *SKILLS_CACHE.write().unwrap() = skills;
    *CACHE_INITIALIZED.write().unwrap() = true;
}

pub fn is_cache_initialized() -> bool {
    *CACHE_INITIALIZED.read().unwrap()
}

/// Look up a skill by name from the in-memory cache.
pub fn get_skill(name: &str) -> Option<Skill> {
    SKILLS_CACHE
        .read()
        .unwrap()
        .iter()
        .find(|s| s.name == name)
        .cloned()
}

/// Get all cached skills (no disk I/O).
pub fn cached_skills() -> Vec<Skill> {
    SKILLS_CACHE.read().unwrap().clone()
}

/// Load full skill content. Only reads the SKILL.md file from disk — no directory scan.
pub fn load_skill_content(name: &str) -> String {
    if name.is_empty() {
        return get_skill_list_text();
    }
    match get_skill(name) {
        Some(skill) => {
            let content = skill.load(); // lazy file read
            format!(
                "--- SKILL: {name} ({source}) ---\nDescription: {desc}\n{sep}\n{content}\n--- END SKILL: {name} ---",
                name = skill.name,
                source = skill.source,
                desc = skill.description,
                sep = "-".repeat(40),
            )
        }
        None => {
            let skills = cached_skills();
            let names: Vec<String> = skills.iter().map(|s| s.name.clone()).collect();
            if names.is_empty() {
                "No skills installed.".to_string()
            } else {
                format!("Skill '{name}' not found. Available: {}", names.join(", "))
            }
        }
    }
}

/// List available skills from cache (no disk I/O).
pub fn get_skill_list_text() -> String {
    let skills = cached_skills();
    if skills.is_empty() {
        return "No skills installed.".to_string();
    }
    let mut lines = vec!["Available skills:\n".to_string()];
    for s in &skills {
        lines.push(format!("  {} ({}) — {}", s.name, s.source, s.description));
    }
    lines.join("\n")
}

/// List installed skills grouped by search path (uses disk scan for path grouping display).
pub fn list_installed_skills(project_root: Option<&Path>) -> String {
    let skills = if is_cache_initialized() {
        cached_skills()
    } else {
        scan_skills(project_root)
    };
    if skills.is_empty() {
        return "No skills installed.".to_string();
    }
    // Group by search path for display
    let mut lines = vec!["Installed skills:".to_string()];
    for search_dir in search_paths(project_root) {
        if !search_dir.exists() {
            continue;
        }
        let names: Vec<&str> = skills
            .iter()
            .filter(|s| s.path.parent().map_or(false, |p| p == search_dir))
            .map(|s| s.name.as_str())
            .collect();
        if !names.is_empty() {
            lines.push(format!("\n  [{}]", search_dir.display()));
            for name in &names {
                lines.push(format!("    {name}"));
            }
        }
    }
    lines.join("\n")
}

// -- SkillRouter (LanceDB-based embedding search) ----------------------------

pub struct SkillRouter {
    pub embedding_api_base: String,
    pub embedding_api_key: String,
    pub embedding_model: String,
    pub chroma_dir: PathBuf,
    pub indexed: HashSet<String>,
    initialized: bool,
}

impl SkillRouter {
    pub fn new(
        chroma_dir: &Path,
        embedding_api_base: &str,
        embedding_api_key: &str,
        embedding_model: &str,
    ) -> Self {
        SkillRouter {
            embedding_api_base: embedding_api_base.to_string(),
            embedding_api_key: embedding_api_key.to_string(),
            embedding_model: embedding_model.to_string(),
            chroma_dir: chroma_dir.to_path_buf(),
            indexed: HashSet::new(),
            initialized: false,
        }
    }

    async fn ensure_initialized(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        self.initialized = true;
        Ok(())
    }

    pub async fn index_skills(&mut self, skills: &[Skill]) -> Result<()> {
        self.ensure_initialized().await?;

        let current_names: HashSet<String> = skills.iter().map(|s| s.name.clone()).collect();

        // Remove deleted skills from indexed set
        let to_remove: Vec<String> = self
            .indexed
            .iter()
            .filter(|n| !current_names.contains(*n))
            .cloned()
            .collect();
        for name in &to_remove {
            self.indexed.remove(name);
        }

        // Add new skills
        let new_skills: Vec<&Skill> = skills
            .iter()
            .filter(|s| !self.indexed.contains(&s.name))
            .collect();

        if new_skills.is_empty() {
            return Ok(());
        }

        let db_path = self
            .chroma_dir
            .parent()
            .unwrap_or(&self.chroma_dir)
            .join("memories.db");
        let mut store = crate::storage::MemoryStore::new(&db_path)?;
        store.init_schema()?;
        if !store.is_lancedb_initialized() {
            store
                .init_lancedb(
                    &self.chroma_dir,
                    &self.embedding_api_base,
                    &self.embedding_api_key,
                    &self.embedding_model,
                )
                .await?;
        }

        for s in &new_skills {
            store
                .add_skill_to_lancedb(&s.name, &s.index_text(), &s.description, &s.source)
                .await?;
            self.indexed.insert(s.name.clone());
        }

        Ok(())
    }

    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<JsonValue>> {
        let db_path = self
            .chroma_dir
            .parent()
            .unwrap_or(&self.chroma_dir)
            .join("memories.db");
        let mut store = crate::storage::MemoryStore::new(&db_path)?;
        store.init_schema()?;
        if !store.is_lancedb_initialized() {
            store
                .init_lancedb(
                    &self.chroma_dir,
                    &self.embedding_api_base,
                    &self.embedding_api_key,
                    &self.embedding_model,
                )
                .await?;
        }
        store.search_skills_lancedb(query, top_k).await
    }
}

// -- Installation -------------------------------------------------------------

pub fn install_skill(source: &str, project_root: Option<&Path>) -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let proj = project_root.unwrap_or(&cwd);
    let target_dir = proj.join(".agent-memory").join("skills");
    fs::create_dir_all(&target_dir).ok();

    let source_path = Path::new(source);
    let (installed_name, result_msg) = if source_path.is_dir() {
        let name = source_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        let dest = target_dir.join(name.as_ref());
        if dest.exists() {
            let _ = fs::remove_dir_all(&dest);
        }
        let msg = match copy_dir_recursive(source_path, &dest) {
            Ok(_) => format!("Skill '{name}' installed from {}", source_path.display()),
            Err(e) => return format!("Error copying directory: {e}"),
        };
        (name.to_string(), msg)
    } else {
        let name = source
            .trim_end_matches('/')
            .split('/')
            .next_back()
            .unwrap_or("unknown")
            .trim_end_matches(".git")
            .to_string();
        let dest = target_dir.join(&name);
        if dest.exists() {
            let _ = fs::remove_dir_all(&dest);
        }
        let msg = match std::process::Command::new("git")
            .args(["clone", "--depth", "1", source])
            .arg(&dest)
            .output()
        {
            Ok(output) if output.status.success() => {
                format!("Skill '{name}' installed from {source}")
            }
            Ok(output) => {
                return format!(
                    "Failed to clone: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(e) => {
                return format!("Error: git not available for remote skill installation: {e}");
            }
        };
        (name, msg)
    };

    // Update in-memory cache
    if is_cache_initialized() {
        let skill_dir = target_dir.join(&installed_name);
        let skill_md = skill_dir.join("SKILL.md");
        if skill_md.exists() {
            let desc = extract_description(&fs::read_to_string(&skill_md).unwrap_or_default());
            let source_label = if skill_dir.starts_with(proj) {
                "project"
            } else {
                "user"
            };
            let new_skill = Skill {
                name: installed_name,
                path: skill_dir,
                description: desc,
                source: source_label.to_string(),
            };
            let mut cache = SKILLS_CACHE.write().unwrap();
            cache.retain(|s| s.name != new_skill.name);
            cache.push(new_skill);
        }
    }

    result_msg
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
}
