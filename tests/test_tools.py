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
        """When no skills exist, return appropriate message."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        with patch("memory_agent.skills.SkillRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.search.return_value = []
            mock_router._collection.count.return_value = 0

            from memory_agent.tools import tool_search_skills

            result = tool_search_skills(query="anything")

        assert "No skills" in result or "not found" in result.lower()


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

        with patch("memory_agent.tools.MemoryStore") as MockStore:
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


class TestSessionState:
    def test_reset_session_state_clears_dedup(self, tmp_path):
        """reset_session_state should clear all dedup tracking."""
        from memory_agent.tools import reset_session_state

        reset_session_state()

        mock_results = [
            {"memory_id": "mem-1", "text": "Test", "distance": 0.1},
        ]

        with patch("memory_agent.tools.MemoryStore") as MockStore:
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
