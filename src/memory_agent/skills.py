"""Skill discovery, loading, and installation.

Skills are markdown files (.md) in skill directories. Each skill directory
contains a SKILL.md with the skill instructions.

Search paths (in order):
  1. <project>/.agent-memory/skills/
  2. <project>/.claude/skills/
  3. ~/.claude/skills/
  4. ~/.memory_agent/skills/
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    path: Path          # directory containing SKILL.md
    description: str    # first non-empty line after frontmatter or first heading
    source: str         # "project", "user", or "installed"

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


_extra_search_paths: list[Path] = []


def add_search_path(path: str) -> None:
    """Add an additional skill search directory."""
    _extra_search_paths.append(Path(path).expanduser().resolve())


def _project_root() -> Path:
    return Path.cwd()


def _search_paths(project_root: Path | None = None) -> list[Path]:
    """Return all directories to search for skills."""
    if project_root is None:
        project_root = _project_root()
    paths = [
        project_root / ".agent-memory" / "skills",
        project_root / ".claude" / "skills",
        Path.home() / ".claude" / "skills",
        Path.home() / ".memory_agent" / "skills",
    ]
    paths.extend(_extra_search_paths)
    return paths


def _extract_description(content: str) -> str:
    """Extract a one-line description from skill content."""
    lines = content.strip().split("\n")
    # Skip YAML frontmatter if present
    in_frontmatter = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if stripped.startswith("#"):
            # First heading
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
                continue  # earlier search paths win

            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                # Also accept directories with any .md file
                md_files = list(entry.glob("*.md"))
                if not md_files:
                    continue

            seen.add(entry.name)

            # Determine source
            if str(search_dir).startswith(str(project_root or _project_root())):
                source = "project"
            else:
                source = "user"

            # Extract description
            desc = ""
            if skill_md.exists():
                desc = _extract_description(skill_md.read_text())
            else:
                desc = entry.name

            skills.append(Skill(
                name=entry.name,
                path=entry,
                description=desc,
                source=source,
            ))

    return skills


def get_skill(name: str, project_root: Path | None = None) -> Skill | None:
    """Find a specific skill by name."""
    for skill in discover_skills(project_root):
        if skill.name == name:
            return skill
    return None


def get_skill_list_text(project_root: Path | None = None) -> str:
    """Return a formatted list of available skills."""
    skills = discover_skills(project_root)
    if not skills:
        return "No skills installed."

    lines = ["Available skills:\n"]
    for s in skills:
        lines.append(f"  {s.name} ({s.source}) — {s.description}")
    return "\n".join(lines)


def install_skill(source: str, project_root: Path | None = None) -> str:
    """Install a skill from a source (directory path or git URL).

    For git URLs: clones into .agent-memory/skills/<name>
    For local directories: copies into .agent-memory/skills/<name>
    """
    if project_root is None:
        project_root = _project_root()

    target_dir = project_root / ".agent-memory" / "skills"
    target_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source).expanduser().resolve()

    if source_path.is_dir():
        # Local directory — copy
        name = source_path.name
        dest = target_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_path, dest)
        return f"Skill '{name}' installed from {source_path}"

    # Try as git URL
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
    """Return a list of installed skill names and their locations."""
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
