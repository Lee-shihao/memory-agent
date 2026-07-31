"""Skill discovery, loading, installation, and embedding-based routing.

Skills are markdown files (.md) in skill directories. Each skill directory
contains a SKILL.md with the skill instructions.

Search paths (in order):
  1. <project>/.agent-memory/skills/
  2. ~/.memory_agent/skills/
  3. Any extra paths added via --skill-dir

Embedding routing:
  Skills are indexed into a ChromaDB 'skills' collection at install/discovery time.
  At conversation start, the user query is embedded and matched against skills
  to auto-inject the top-K most relevant skill descriptions into the system prompt.
"""
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class Skill:
    name: str
    path: Path
    description: str
    source: str  # "project", "user", or "installed"

    def load(self) -> str:
        """Read the full SKILL.md content."""
        skill_file = self.path / "SKILL.md"
        if not skill_file.exists():
            for f in self.path.glob("*.md"):
                skill_file = f
                break
        if skill_file.exists():
            return skill_file.read_text()
        return f"# {self.name}\n\n(empty skill)"

    @property
    def index_text(self) -> str:
        """Text used for embedding — name + description."""
        return f"{self.name}: {self.description}"


_extra_search_paths: list[Path] = []


def add_search_path(path: str) -> None:
    """Add an additional skill search directory."""
    _extra_search_paths.append(Path(path).expanduser().resolve())


def _project_root() -> Path:
    return Path.cwd()


def _search_paths(project_root: Path | None = None) -> list[Path]:
    if project_root is None:
        project_root = _project_root()
    paths = [
        project_root / ".agent-memory" / "skills",
        Path.home() / ".memory_agent" / "skills",
    ]
    paths.extend(_extra_search_paths)
    return paths


def _extract_description(content: str) -> str:
    lines = content.strip().split("\n")
    in_frontmatter = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if stripped.startswith("#"):
            desc = stripped.lstrip("#").strip()
            return desc if desc else "No description"
        if stripped:
            return stripped[:120]
    return "No description"


def discover_skills(project_root: Path | None = None) -> list[Skill]:
    """Find all installed skills across search paths."""
    skills: list[Skill] = []
    seen: set[str] = set()

    for search_dir in _search_paths(project_root):
        if not search_dir.exists():
            continue
        for entry in sorted(search_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name in seen:
                continue

            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                md_files = list(entry.glob("*.md"))
                if not md_files:
                    continue

            seen.add(entry.name)

            source = "project" if str(search_dir).startswith(str(project_root or _project_root())) else "user"

            desc = ""
            if skill_md.exists():
                desc = _extract_description(skill_md.read_text())
            else:
                desc = entry.name

            skills.append(Skill(name=entry.name, path=entry, description=desc, source=source))

    return skills


def get_skill(name: str, project_root: Path | None = None) -> Skill | None:
    for skill in discover_skills(project_root):
        if skill.name == name:
            return skill
    return None


def get_skill_list_text(project_root: Path | None = None) -> str:
    skills = discover_skills(project_root)
    if not skills:
        return "No skills installed."
    lines = ["Available skills:\n"]
    for s in skills:
        lines.append(f"  {s.name} ({s.source}) — {s.description}")
    return "\n".join(lines)


# ── embedding-based skill routing ─────────────────────────────────────────────

_SKILL_COLLECTION = "skills"


class SkillRouter:
    """ChromaDB-backed skill search via embedding similarity."""

    def __init__(
        self,
        chroma_dir: Path,
        embedding_api_base: str,
        embedding_api_key: str,
        embedding_model: str,
    ):
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_SKILL_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_api_base = embedding_api_base
        self._embedding_api_key = embedding_api_key or os.environ.get("SF_API_KEY", "")
        self._embedding_model = embedding_model
        self._indexed: set[str] = set()

    def _get_embedding(self, text: str) -> list[float]:
        import httpx

        response = httpx.post(
            f"{self._embedding_api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self._embedding_api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._embedding_model, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def index_skills(self, skills: list[Skill]) -> None:
        """Add or update skills in the vector index. Skips already-indexed skills."""
        new_skills = [s for s in skills if s.name not in self._indexed]

        # Remove skills that no longer exist on disk
        current_names = {s.name for s in skills}
        for name in list(self._indexed):
            if name not in current_names:
                try:
                    self._collection.delete(ids=[f"skill-{name}"])
                except Exception:
                    pass
                self._indexed.discard(name)

        if not new_skills:
            return

        for s in new_skills:
            embedding = self._get_embedding(s.index_text)
            self._collection.add(
                ids=[f"skill-{s.name}"],
                embeddings=[embedding],
                documents=[s.index_text],
                metadatas=[{"name": s.name, "description": s.description, "source": s.source}],
            )
            self._indexed.add(s.name)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for skills relevant to the query. Returns [{name, description, source, distance}]."""
        if self._collection.count() == 0:
            return []

        embedding = self._get_embedding(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        skills = []
        if results["ids"] and results["ids"][0]:
            for i, _ in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else None
                skills.append({
                    "name": meta.get("name", ""),
                    "description": meta.get("description", ""),
                    "source": meta.get("source", ""),
                    "distance": dist,
                })
        return skills


def format_skills_for_injection(matched: list[dict]) -> str:
    """Format matched skills for injection into the system prompt."""
    if not matched:
        return ""

    lines = ["## Available Skills (auto-matched)"]
    for i, s in enumerate(matched):
        dist = s.get("distance")
        score = f" (score: {1 - dist:.2f})" if dist is not None else ""
        lines.append(f"### {s['name']}{score}")
        lines.append(f"{s['description']}")
        lines.append(f"Use `load_skill(\"{s['name']}\")` to load full instructions.")
        if i < len(matched) - 1:
            lines.append("")
    return "\n".join(lines)


# ── installation ─────────────────────────────────────────────────────────────

def install_skill(source: str, project_root: Path | None = None) -> str:
    """Install a skill from a directory path or git URL."""
    if project_root is None:
        project_root = _project_root()

    target_dir = project_root / ".agent-memory" / "skills"
    target_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source).expanduser().resolve()

    if source_path.is_dir():
        name = source_path.name
        dest = target_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_path, dest)
        return f"Skill '{name}' installed from {source_path}"

    name = source.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = target_dir / name
    if dest.exists():
        shutil.rmtree(dest)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", source, str(dest)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return f"Failed to clone: {result.stderr}"
        return f"Skill '{name}' installed from {source}"
    except FileNotFoundError:
        return "Error: git not available for remote skill installation"
    except subprocess.TimeoutExpired:
        return "Error: skill installation timed out"
    except Exception as e:
        return f"Error installing skill: {e}"


def list_installed_skills(project_root: Path | None = None) -> str:
    lines = ["Installed skills:"]
    for search_dir in _search_paths(project_root):
        if not search_dir.exists():
            continue
        lines.append(f"\n  [{search_dir}]")
        for entry in sorted(search_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                skill_md = entry / "SKILL.md"
                if skill_md.exists() or list(entry.glob("*.md")):
                    lines.append(f"    {entry.name}")
    return "\n".join(lines)
