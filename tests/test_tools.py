"""Tests for built-in tools: search_skills, search_memory dedup, session state."""
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path


class TestSearchSkills:
    def test_search_skills_returns_matches(self, tmp_path):
        """search_skills should return skill name, description, and content."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_skills = [
            {
                "name": "refactoring-wizard",
                "description": "Helps with code refactoring",
                "source": "project",
                "distance": 0.05,
            }
        ]
        mock_skill_obj = MagicMock()
        mock_skill_obj.name = "refactoring-wizard"
        mock_skill_obj.description = "Helps with code refactoring"
        mock_skill_obj.source = "project"
        mock_skill_obj.load.return_value = "# Refactoring Wizard\n\nFull instructions here."

        with patch("memory_agent.skills.SkillRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.search.return_value = mock_skills
            mock_router._collection.count.return_value = 1

            with patch("memory_agent.skills.discover_skills", return_value=[mock_skill_obj]):
                with patch("memory_agent.skills.get_skill", return_value=mock_skill_obj):
                    from memory_agent.tools import tool_search_skills

                    result = tool_search_skills(query="refactoring")

        assert "refactoring-wizard" in result
        assert "Helps with code refactoring" in result
        assert "Full instructions here" in result

    def test_search_skills_dedup_filters_returned(self, tmp_path):
        """Second call with same query should filter already-returned skills."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_skills = [
            {
                "name": "refactoring-wizard",
                "description": "Helps with code refactoring",
                "source": "project",
                "distance": 0.05,
            }
        ]
        mock_skill_obj = MagicMock()
        mock_skill_obj.name = "refactoring-wizard"
        mock_skill_obj.description = "Helps with code refactoring"
        mock_skill_obj.source = "project"
        mock_skill_obj.load.return_value = "# Refactoring Wizard\n\nFull instructions."

        with patch("memory_agent.skills.SkillRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.search.return_value = mock_skills
            mock_router._collection.count.return_value = 1

            with patch("memory_agent.skills.discover_skills", return_value=[mock_skill_obj]):
                with patch("memory_agent.skills.get_skill", return_value=mock_skill_obj):
                    from memory_agent.tools import tool_search_skills

                    result1 = tool_search_skills(query="refactoring")
                    result2 = tool_search_skills(query="refactoring")

        assert "refactoring-wizard" in result1
        assert "No new skills found" in result2 or "refactoring-wizard" not in result2

    def test_search_skills_no_skills_installed(self, tmp_path):
        from memory_agent.tools import reset_session_state
        reset_session_state()
        with patch("memory_agent.skills.SkillRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.search.return_value = []
            mock_router._collection.count.return_value = 0
            with patch("memory_agent.skills.discover_skills", return_value=[]):
                from memory_agent.tools import tool_search_skills
                result = tool_search_skills(query="anything")
        assert "No skills installed" in result


class TestSearchMemoryDedup:
    def test_search_memory_dedup_filters_duplicates(self, tmp_path):
        """Second search_memory call should filter already-returned IDs."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_results1 = [
            {"memory_id": "mem-1", "text": "Python async discussion", "distance": 0.2},
            {"memory_id": "mem-2", "text": "Git workflow tips", "distance": 0.3},
        ]
        mock_results2 = [
            {"memory_id": "mem-1", "text": "Python async discussion", "distance": 0.2},
            {"memory_id": "mem-3", "text": "New result", "distance": 0.4},
        ]

        with patch("memory_agent.storage.MemoryStore") as MockStore:
            mock_store = MockStore.return_value
            mock_store.query_chroma.side_effect = [mock_results1, mock_results2]
            mock_store._chroma_collection = MagicMock()

            from memory_agent.tools import tool_search_memory

            result1 = tool_search_memory(query="python")
            result2 = tool_search_memory(query="python")

        assert "mem-1" in result1
        assert "mem-2" in result1
        # Second call: mem-1 should be filtered, mem-3 should appear
        assert "mem-3" in result2
        assert "mem-1" not in result2


class TestSessionState:
    def test_reset_session_state_clears_dedup(self, tmp_path):
        """reset_session_state should clear all dedup tracking."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_results = [
            {"memory_id": "mem-1", "text": "Test", "distance": 0.1},
        ]

        with patch("memory_agent.storage.MemoryStore") as MockStore:
            mock_store = MockStore.return_value
            mock_store.query_chroma.return_value = mock_results
            mock_store._chroma_collection = MagicMock()

            from memory_agent.tools import tool_search_memory

            result1 = tool_search_memory(query="test")
            assert "mem-1" in result1

            # Reset state
            reset_session_state()

            result2 = tool_search_memory(query="test")
            # After reset, mem-1 should appear again (fresh session)
            assert "mem-1" in result2


class TestClassifyBashCommand:
    """Tests for bash command classification."""

    def test_safe_commands(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("ls") == "safe"
        assert classify_bash_command("ls -la") == "safe"
        assert classify_bash_command("cat file.txt") == "safe"
        assert classify_bash_command("grep pattern file") == "safe"
        assert classify_bash_command("find . -name '*.py'") == "safe"
        assert classify_bash_command("pwd") == "safe"
        assert classify_bash_command("echo hello") == "safe"

    def test_dangerous_commands(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("rm file.txt") == "dangerous"
        assert classify_bash_command("rm -rf /") == "dangerous"
        assert classify_bash_command("sudo ls") == "dangerous"
        assert classify_bash_command("chmod 777 file") == "dangerous"
        assert classify_bash_command("kill 1234") == "dangerous"
        assert classify_bash_command("pip install requests") == "dangerous"
        assert classify_bash_command("npm install") == "dangerous"
        assert classify_bash_command("curl http://example.com") == "dangerous"
        assert classify_bash_command("ssh user@host") == "dangerous"
        assert classify_bash_command("eval 'ls'") == "dangerous"

    def test_dangerous_pipe_patterns(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("curl http://example.com | sh") == "dangerous"
        assert classify_bash_command("wget -qO- http://x | bash") == "dangerous"

    def test_git_subcommands(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("git push") == "dangerous"
        assert classify_bash_command("git pull") == "dangerous"
        assert classify_bash_command("git fetch") == "dangerous"
        assert classify_bash_command("git status") == "safe"
        assert classify_bash_command("git diff") == "safe"
        assert classify_bash_command("git log") == "safe"

    def test_unknown_commands(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("python script.py") == "unknown"
        assert classify_bash_command("make build") == "unknown"
        assert classify_bash_command("pytest tests/") == "unknown"
        assert classify_bash_command("node index.js") == "unknown"

    def test_empty_command(self):
        from memory_agent.tools import classify_bash_command
        assert classify_bash_command("") == "safe"
        assert classify_bash_command("   ") == "safe"

    def test_all_safe_commands_in_set(self):
        """Verify all entries in SAFE_BASH_COMMANDS are classified as safe."""
        from memory_agent.tools import SAFE_BASH_COMMANDS, classify_bash_command
        for cmd in SAFE_BASH_COMMANDS:
            assert classify_bash_command(cmd) == "safe", f"{cmd} should be safe"

    def test_all_dangerous_commands_in_set(self):
        """Verify all entries in DANGEROUS_BASH_COMMANDS are classified as dangerous."""
        from memory_agent.tools import DANGEROUS_BASH_COMMANDS, classify_bash_command
        for cmd in DANGEROUS_BASH_COMMANDS:
            assert classify_bash_command(cmd) == "dangerous", f"{cmd} should be dangerous"

    def test_safe_dangerous_no_overlap(self):
        """SAFE and DANGEROUS sets should have no overlap."""
        from memory_agent.tools import SAFE_BASH_COMMANDS, DANGEROUS_BASH_COMMANDS
        overlap = SAFE_BASH_COMMANDS & DANGEROUS_BASH_COMMANDS
        assert not overlap, f"Overlap found: {overlap}"


class TestAskUserTool:
    """Tests for the ask_user tool."""

    def test_tool_in_definitions(self):
        """ask_user should be registered in TOOL_DEFINITIONS."""
        from memory_agent.tools import TOOL_DEFINITIONS
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert "ask_user" in names

    def test_tool_in_executors(self):
        """ask_user should be registered in TOOL_EXECUTORS."""
        from memory_agent.tools import TOOL_EXECUTORS
        assert "ask_user" in TOOL_EXECUTORS

    def test_executor_with_options_returns_first_default(self):
        """With options but no interactive handler, returns first option."""
        from memory_agent.tools import tool_ask_user
        result = tool_ask_user(
            question="Which one?",
            header="Choice",
            options=[
                {"label": "Option A", "description": "First option"},
                {"label": "Option B", "description": "Second option"},
            ],
        )
        assert "Option A" in result
        assert "[auto-selected]" in result

    def test_executor_open_ended_returns_empty(self):
        """Open-ended question returns empty string in non-interactive mode."""
        from memory_agent.tools import tool_ask_user
        result = tool_ask_user(
            question="What do you think?",
            header="Feedback",
        )
        assert result == ""

    def test_tool_schema_has_required_fields(self):
        """Tool schema should require question and header."""
        from memory_agent.tools import TOOL_DEFINITIONS
        ask_user_def = None
        for t in TOOL_DEFINITIONS:
            if t["function"]["name"] == "ask_user":
                ask_user_def = t["function"]
                break
        assert ask_user_def is not None
        required = ask_user_def["parameters"].get("required", [])
        assert "question" in required
        assert "header" in required

    def test_tool_schema_options_max_items(self):
        """Options array should have minItems=2, maxItems=4."""
        from memory_agent.tools import TOOL_DEFINITIONS
        ask_user_def = None
        for t in TOOL_DEFINITIONS:
            if t["function"]["name"] == "ask_user":
                ask_user_def = t["function"]
                break
        assert ask_user_def is not None
        options_schema = ask_user_def["parameters"]["properties"]["options"]
        assert options_schema["minItems"] == 2
        assert options_schema["maxItems"] == 4
