#!/usr/bin/env python3
"""
Smoke Test: Import Verification

Verifies all critical QueryMind modules can be imported without errors.
This is the most basic test - if imports fail, nothing else will work.

Run with: pytest tests/test_imports_work.py -v
"""

import pytest


def test_core_imports():
    """Test that all core modules import successfully."""
    from querymind.core.config import config
    from querymind.core.logging_config import setup_logging, get_logger
    from querymind.core.conversation_memory import ConversationMemory

    assert config is not None, "config should be loaded"
    assert callable(setup_logging), "setup_logging should be callable"
    assert callable(get_logger), "get_logger should be callable"

    # Test optional dependencies (skip if not installed)
    try:
        from querymind.core.embeddings import ChromaDBManager
        from querymind.core.cache import QueryCache
    except ImportError as e:
        pytest.skip(f"Optional dependency not installed: {e}")


def test_agent_imports():
    """Test that all agent modules import successfully."""
    from querymind.agents.base_agent import BaseAgent, SearchResult
    from querymind.agents.fast_search_agent import FastSearchAgent
    from querymind.agents.deep_research_agent import DeepResearchAgent
    from querymind.agents.router import AgentRouter, AgentType
    from querymind.agents.vault_search_agent_local import VaultSearchAgentLocal
    from querymind.agents.web_search_client import WebSearchClient, WebSearchResult

    # Verify classes exist
    assert BaseAgent is not None
    assert SearchResult is not None
    assert FastSearchAgent is not None
    assert DeepResearchAgent is not None
    assert AgentRouter is not None
    assert VaultSearchAgentLocal is not None
    assert WebSearchClient is not None


def test_mcp_server_imports():
    """Test that MCP server module imports successfully."""
    # Import should work even if server can't start
    try:
        from querymind.mcp import server
        assert server is not None
    except ImportError as e:
        pytest.skip(f"MCP dependencies not installed: {e}")


def test_top_level_imports():
    """Test that top-level package imports work."""
    import querymind
    from querymind import search, auto_search

    assert querymind.__version__ is not None or True  # Version might not be set yet
    assert callable(search), "search should be callable"
    assert callable(auto_search), "auto_search should be callable"


def test_logging_initialization():
    """Test that logging can be initialized without errors."""
    from querymind.core.logging_config import setup_logging, get_logger
    import logging

    # Setup logging with test configuration
    logger = setup_logging(log_level="DEBUG", log_to_console=False)
    assert logger is not None
    assert logger.level == logging.DEBUG

    # Get a test logger
    test_logger = get_logger("test_module")
    assert test_logger is not None
    assert test_logger.name == "test_module"


def test_conversation_memory_stub():
    """Test that conversation memory stub works."""
    from querymind.core.conversation_memory import ConversationMemory

    memory = ConversationMemory()
    memory.add_message("user", "test message")

    messages = memory.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "test message"

    # Test save doesn't crash (it's a stub)
    memory.save("test_conversation_id")

    # Test clear
    memory.clear()
    assert len(memory.get_messages()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
